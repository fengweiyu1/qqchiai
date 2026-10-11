"""Shared analysis adapted from the project's final localization statistics.

DR is recall of injected faults; FDR is the fraction of inspected weights that
are not injected faults. AUC is normalized within each inspected-weight window.
Paired tests use mutants, not individual faulty weights, as observations.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#2878B5', '#D47726', '#8C6BB1', '#B29A26', '#C56386']
BASE_COLORS = COLORS + ['#456B5C', '#555555', '#888888', '#222222']


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def setup(rq, description, extra=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', default='configs/fm.json')
    parser.add_argument('--mode', choices=['all', 'single', 'multi'], default='all')
    parser.add_argument('--results-dir', help='RQ root containing single/ and multi/')
    parser.add_argument('--output-dir', help='Root for figures/ and tables/')
    parser.add_argument('--dry-run', action='store_true')
    if rq != 3:
        parser.add_argument('--windows', default='1,20', help='Maximum inspected-weight percentages')
    if extra:
        extra(parser)
    args = parser.parse_args()
    cfg = read_json(resolve(args.config))
    if cfg.get('dataset') not in ('fm', 'cifar10', 'gtsrb'):
        parser.error('Configuration must identify fm, cifar10, or gtsrb')
    args.dataset = cfg['dataset']
    args.results_dir = resolve(args.results_dir or str(Path(cfg['output_root']) / f'rq{rq}'))
    args.output_dir = resolve(args.output_dir) if args.output_dir else args.results_dir
    args.modes = ['single', 'multi'] if args.mode == 'all' else [args.mode]
    if rq != 3:
        try:
            args.windows = sorted(set(float(x) for x in args.windows.split(',')))
            if not args.windows or any(not math.isfinite(x) or not 0 < x <= 100 for x in args.windows):
                raise ValueError()
        except ValueError:
            parser.error('--windows must contain percentages in (0, 100]')
    if args.dry_run:
        print(json.dumps({key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()}, indent=2))
        return None
    return args


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise ValueError(f'Missing input: {path}; run the corresponding experiment or set --results-dir') from exc


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'No rows to export: {path}')
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_info(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def index_mutants(data, dataset):
    if data.get('dataset') != dataset:
        raise ValueError(f'Expected dataset {dataset}, found {data.get("dataset")}')
    n = data.get('total_weights')
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError('total_weights must be a positive integer')
    result = {}
    for mutant in data.get('mutants', []):
        mid = mutant['mutant_id']
        if mid in result:
            raise ValueError(f'Duplicate mutant ID: {mid}')
        count = mutant.get('n_faulty')
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= n:
            raise ValueError(f'Invalid fault count for mutant {mid}')
        for key, ranks in mutant.get('ranks', {}).items():
            if len(ranks) != count or any(not isinstance(rank, int) or isinstance(rank, bool)
                                         or not 1 <= rank <= n for rank in ranks):
                raise ValueError(f'Invalid ranks: mutant {mid}, method {key}')
            if len(set(ranks)) != len(ranks):
                raise ValueError(f'Duplicate ranks: mutant {mid}, method {key}')
        result[mid] = mutant
    if not result:
        raise ValueError('No localization mutants found')
    return n, result


def require_methods(mutants, methods):
    for mutant in mutants:
        for label, key in methods:
            if key not in mutant['ranks']:
                raise ValueError(f'Mutant {mutant["mutant_id"]} lacks {label} ({key})')


def budget_grid(n, pct):
    import numpy as np
    return np.arange(1, max(1, min(n, int(round(n * pct / 100)))) + 1, dtype=float)


def auc_values(mutants, key, grid):
    import numpy as np
    dr, fdr = [], []
    for mutant in mutants:
        found = np.searchsorted(np.sort(mutant['ranks'][key]), grid, side='right')
        recall = found / mutant['n_faulty']
        false_discovery = (grid - found) / grid
        if len(grid) == 1:
            dr.append(float(recall[0]))
            fdr.append(float(false_discovery[0]))
        else:
            x = (grid - grid[0]) / (grid[-1] - grid[0])
            dr.append(float(np.trapz(recall, x)))
            fdr.append(float(np.trapz(false_discovery, x)))
    return np.array(dr), np.array(fdr)


def paired_test(a, b):
    import numpy as np
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError('Paired observations must have equal shapes')
    valid = np.isfinite(a) & np.isfinite(b)
    diff = a[valid] - b[valid]
    n = len(diff)
    mean = float(diff.mean()) if n else None
    if n < 2:
        p, d = None, None
    elif np.allclose(diff, 0):
        p, d = 1.0, 0.0
    else:
        p = float(wilcoxon(a[valid], b[valid]).pvalue)
        sd = float(diff.std(ddof=1))
        # A constant nonzero difference has an undefined standardized effect.
        d = float(mean / sd) if sd > 0 else None
    return {'n_pairs': n, 'n_excluded': int((~valid).sum()), 'mean_difference': mean,
            'p_raw': p, 'd_paired': d}


def holm_adjust(values):
    order = sorted(range(len(values)), key=lambda i: values[i] if values[i] is not None else 1.0)
    adjusted = [None] * len(values)
    running = 0.0
    for rank, i in enumerate(order):
        p = values[i] if values[i] is not None else 1.0
        running = max(running, (len(values) - rank) * p)
        adjusted[i] = min(1.0, running) if values[i] is not None else None
    return adjusted


def significance(test, p_adjusted, lower=False):
    if p_adjusted is None or p_adjusted >= 0.05:
        return '~'
    d = test['d_paired']
    magnitude = abs(d) if d is not None else 0.0
    # Keep the effect-size bins of the existing paper analysis.
    strength = 1 if magnitude < .2 else (2 if magnitude < .8 else 3)
    better = test['mean_difference'] < 0 if lower else test['mean_difference'] > 0
    return ('+' if better else '-') * strength


def pairwise(arrays, labels, lower=False):
    rows = []
    matrix = [['.' if i == j else '' for j in range(len(labels))] for i in range(len(labels))]
    for i, left in enumerate(labels):
        for j in range(i + 1, len(labels)):
            right = labels[j]
            rows.append({'method_a': left, 'method_b': right, **paired_test(arrays[left], arrays[right])})
    adjusted = holm_adjust([row['p_raw'] for row in rows])
    for row, p in zip(rows, adjusted):
        row.update(p_holm=p, family_size=len(rows), symbol=significance(row, p, lower))
        i, j = labels.index(row['method_a']), labels.index(row['method_b'])
        matrix[i][j] = row['symbol']
        matrix[j][i] = row['symbol'].translate(str.maketrans('+-', '-+'))
    return rows, matrix


def plotting():
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'weight-localization-matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'weight-localization-cache'))
    Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    return plt


def save_figure(fig, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(directory / f'{stem}.{extension}', dpi=180, bbox_inches='tight', facecolor='white')
    plotting().close(fig)


def draw_matrix(matrix, labels, means, title, directory, stem):
    import numpy as np
    plt = plotting()
    from matplotlib.colors import ListedColormap
    size = max(7, len(labels) * .9)
    fig, ax = plt.subplots(figsize=(size, size))
    values = np.array([[0 if item in ('.', '~') else (1 if '+' in item else -1)
                        for item in row] for row in matrix])
    ax.imshow(values, cmap=ListedColormap(['#E4D1BB', '#FAFAFA', '#C5D9E8']), vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(len(labels)), [f'{label} ({mean:.3f})' for label, mean in zip(labels, means)])
    for i, row in enumerate(matrix):
        for j, symbol in enumerate(row):
            ax.text(j, i, symbol, ha='center', va='center', fontsize=11)
    ax.set_title(title, pad=14)
    ax.set_xlabel('Column method')
    ax.set_ylabel('Row method (mean normalized AUC)')
    fig.text(.5, .01, 'Row vs column: + better, - worse, ~ not significant; paired Wilcoxon + Holm (0.05)\n'
             'Repeated signs follow paired effect size: <0.2, <0.8, >=0.8; undefined effect uses one sign.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, 1))
    save_figure(fig, directory, stem)


def localization_outputs(args, mode, mutants, n, methods, rq, variant=False):
    import numpy as np
    plt = plotting()
    from matplotlib.ticker import FuncFormatter
    summary, pairs, curves = [], [], []
    labels = [label for label, _ in methods]
    for window in args.windows:
        grid = budget_grid(n, window)
        arrays = {'dr': {}, 'fdr': {}}
        tag = f'{window:g}pct'
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for i, (label, key) in enumerate(methods):
            dr, fdr = auc_values(mutants, key, grid)
            arrays['dr'][label], arrays['fdr'][label] = dr, fdr
            flattened = np.sort([rank for mutant in mutants for rank in mutant['ranks'][key]])
            found = np.searchsorted(flattened, grid, side='right')
            pooled_dr = found / len(flattened)
            pooled_fdr = 1 - found / (len(mutants) * grid)
            style = '-' if not variant or i < 5 else '--'
            color = COLORS[i % 5] if variant else BASE_COLORS[i % len(BASE_COLORS)]
            ax.plot(grid / n * 100, pooled_dr * 100, label=label, color=color, linestyle=style,
                    marker=['o', 's', '^', 'D', 'v', 'P', 'X'][i % 7], markevery=max(1, len(grid) // 8),
                    markersize=3, linewidth=1.7)
            summary.append({'dataset': args.dataset, 'mode': mode, 'window_pct': window,
                            'method': label, 'result_key': key, 'n_mutants': len(mutants),
                            'n_faults': len(flattened), 'auc_dr_mean': float(dr.mean()),
                            'auc_fdr_mean': float(fdr.mean())})
            for budget, recall, false_discovery in zip(grid, pooled_dr, pooled_fdr):
                curves.append({'dataset': args.dataset, 'mode': mode, 'window_pct': window,
                               'method': label, 'budget_weights': int(budget),
                               'budget_pct': float(budget / n * 100), 'dr': float(recall),
                               'fdr': float(false_discovery)})
        ax.plot(grid / n * 100, grid / n * 100, color='#888888', linestyle=':', label='Random (expected)')
        if len(grid) > 1:
            ax.set_xscale('log')
            ax.set_xlim(grid[0] / n * 100, grid[-1] / n * 100)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.set_ylim(0, 100)
        ax.set_xlabel('Weights inspected (% of layer; log scale)' if len(grid) > 1 else 'Weights inspected (% of layer)')
        ax.set_ylabel('Injected faults found (%)')
        ax.set_title(f'RQ{rq} | {args.dataset} | {mode} | {len(mutants)} mutants | up to {window:g}%')
        ax.grid(axis='y', color='#DDDDDD', linewidth=.6)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
        fig.tight_layout()
        save_figure(fig, args.output_dir / 'figures', f'rq{rq}_{mode}_dr_{tag}')
        for metric, lower in [('dr', False), ('fdr', True)]:
            rows, matrix = pairwise(arrays[metric], labels, lower)
            pairs.extend({'dataset': args.dataset, 'mode': mode, 'window_pct': window,
                          'metric': f'auc_{metric}', **row} for row in rows)
            draw_matrix(matrix, labels, [float(arrays[metric][label].mean()) for label in labels],
                        f'RQ{rq} | {args.dataset} | {mode} | AUC-{metric.upper()} | 0-{window:g}%\n'
                        f'{len(mutants)} paired mutants; {"lower" if lower else "higher"} is better',
                        args.output_dir / 'figures', f'rq{rq}_{mode}_pairwise_{metric}_{tag}')
    return summary, pairs, curves


def export_localization(args, rq, datasets, metadata, variant=False):
    summary, pairs, curves = [], [], []
    for mode, mutants, n, methods in datasets:
        require_methods(mutants, methods)
        result = localization_outputs(args, mode, mutants, n, methods, rq, variant)
        for target, rows in zip((summary, pairs, curves), result):
            target.extend(rows)
    for name, rows in [('summary', summary), ('pairwise', pairs), ('curves', curves)]:
        write_csv(args.output_dir / 'tables' / f'rq{rq}_{name}.csv', rows)
    write_json(args.output_dir / f'rq{rq}_analysis.json', {'dataset': args.dataset,
               'windows_pct': args.windows, 'sources': metadata,
               'summary': summary, 'pairwise': pairs})
    print(f'RQ{rq}: saved figures and tables to {args.output_dir}')
