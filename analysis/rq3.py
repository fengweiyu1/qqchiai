"""Summarize repair outcomes and paired comparisons for RQ3."""
from collections import defaultdict
import math

from common import (setup, read_json, source_info, write_csv, write_json, paired_test,
                    holm_adjust, significance, plotting, save_figure, BASE_COLORS)

LABELS = {'ours': 'Ours', 'arachnevar': 'ArachneVar', 'deepfaultvar': 'DeepFaultVar',
          'guidervar': 'GuiderVar', 'deepfault': 'DeepFault', 'guider': 'Guider',
          'vere': 'VeRe', 'random': 'Random', 'truefault': 'Oracle'}


def ratio(num, den):
    return num / den if den else None


def mean_defined(values):
    defined = [value for value in values if value is not None and math.isfinite(value)]
    return sum(defined) / len(defined) if defined else None


def record_rate(record, metric):
    fields = {'rr': ('n_fixed', 'n_broken'), 'br': ('n_new_broken', 'n_mut_correct')}
    numerator, denominator = fields[metric]
    return ratio(record['E'][numerator], record['E'][denominator])


def repair_groups(data, dataset):
    if data.get('dataset') != dataset:
        raise ValueError(f'Expected {dataset}, found {data.get("dataset")}')
    groups = defaultdict(dict)
    denominators = {}
    for record in data.get('results', []):
        mid, method = record['mutant_id'], record['localizer']
        key = (record['n_fix'], record['alpha'], method)
        if method not in LABELS or record['n_fix'] <= 0 or not math.isfinite(record['alpha']) or record['alpha'] <= 0:
            raise ValueError(f'Invalid repair settings: {key}')
        if mid in groups[key]:
            raise ValueError(f'Duplicate repair observation: {key}, mutant {mid}')
        for split in ('R', 'E'):
            e = record[split]
            for field in ('n_broken', 'n_fixed', 'n_mut_correct', 'n_new_broken'):
                if not isinstance(e[field], int) or isinstance(e[field], bool) or e[field] < 0:
                    raise ValueError(f'Invalid {split} count {field}: mutant {mid}')
            if e['n_fixed'] > e['n_broken'] or e['n_new_broken'] > e['n_mut_correct']:
                raise ValueError(f'Impossible repair counts: mutant {mid}, {split}')
            for field in ('acc_patched', 'acc_mutant'):
                if not isinstance(e[field], (int, float)) or not math.isfinite(e[field]) or not 0 <= e[field] <= 1:
                    raise ValueError(f'Invalid accuracy: mutant {mid}, {split}, {field}')
            expected = (e['n_broken'], e['n_mut_correct'], e['acc_mutant'])
            if denominators.setdefault((mid, split), expected) != expected:
                raise ValueError(f'Inconsistent evaluation population: mutant {mid}, {split}')
        if not isinstance(record['fault_hit'], int) or not 0 <= record['fault_hit'] <= record['n_fix']:
            raise ValueError(f'Invalid fault_hit: mutant {mid}')
        metadata = data.get('mutants', {}).get(str(mid), {})
        if 'idx' in record and 'faulty' in metadata:
            idx = record['idx']
            if len(idx) != record['n_fix'] or len(set(idx)) != len(idx):
                raise ValueError(f'Invalid selected weights: mutant {mid}')
            if len(set(idx) & set(metadata['faulty'])) != record['fault_hit']:
                raise ValueError(f'fault_hit disagrees with selected weights: mutant {mid}')
        groups[key][mid] = record
    if not groups:
        raise ValueError('No repair observations found')
    return groups


def summarize(groups, dataset, mode):
    rows = []
    for (budget, alpha, method), by_mid in sorted(groups.items()):
        records = list(by_mid.values())
        counts = {name: sum(record['E'][name] for record in records)
                  for name in ('n_broken', 'n_fixed', 'n_mut_correct', 'n_new_broken')}
        rr = [record_rate(record, 'rr') for record in records]
        br = [record_rate(record, 'br') for record in records]
        rows.append({'dataset': dataset, 'mode': mode, 'n_fix': budget, 'alpha': alpha,
                     'method': method, 'n_mutants': len(records), **counts,
                     'rr_pooled': ratio(counts['n_fixed'], counts['n_broken']),
                     'rr_mean': mean_defined(rr), 'n_rr_defined': sum(value is not None for value in rr),
                     'br_pooled': ratio(counts['n_new_broken'], counts['n_mut_correct']),
                     'br_mean': mean_defined(br), 'n_br_defined': sum(value is not None for value in br),
                     'r_acc': mean_defined([ratio(r['E']['acc_patched'], r['E']['acc_mutant']) for r in records]),
                     'fault_hit_rate': sum(r['fault_hit'] > 0 for r in records) / len(records),
                     'fault_hit_mean': sum(r['fault_hit'] for r in records) / len(records)})
    return rows


def compare_groups(left, right, metric):
    import numpy as np
    ids = sorted(set(left) | set(right))
    def values(group):
        return np.array([record_rate(group[mid], metric) if mid in group and record_rate(group[mid], metric) is not None
                         else np.nan for mid in ids])
    return paired_test(values(left), values(right))


def comparisons(groups, dataset, mode):
    rows, alpha_rows = [], []
    settings = sorted(set((budget, alpha) for budget, alpha, method in groups))
    for budget, alpha in settings:
        ours = groups.get((budget, alpha, 'ours'))
        if not ours:
            continue
        family = []
        for method in LABELS:
            if method == 'ours' or (budget, alpha, method) not in groups:
                continue
            family.append({'dataset': dataset, 'mode': mode, 'n_fix': budget, 'alpha': alpha,
                           'method_a': 'ours', 'method_b': method, 'metric': 'rr',
                           **compare_groups(ours, groups[budget, alpha, method], 'rr')})
        for row, adjusted in zip(family, holm_adjust([row['p_raw'] for row in family])):
            rows.append({**row, 'p_holm': adjusted, 'family_size': len(family),
                         'symbol': significance(row, adjusted)})
    # Preserve the existing alpha=10 versus alpha=1 paired analysis. Each
    # budget/metric corrects the available method comparisons as one family.
    for budget in sorted(set(key[0] for key in groups)):
        for metric in ('rr', 'br'):
            family = []
            for method in LABELS:
                left, right = groups.get((budget, 10, method)), groups.get((budget, 1, method))
                if not left or not right:
                    continue
                family.append({'dataset': dataset, 'mode': mode, 'n_fix': budget, 'method': method,
                               'alpha_a': 10, 'alpha_b': 1, 'metric': metric,
                               **compare_groups(left, right, metric)})
            for row, adjusted in zip(family, holm_adjust([row['p_raw'] for row in family])):
                alpha_rows.append({**row, 'p_holm': adjusted, 'family_size': len(family),
                                   'symbol': significance(row, adjusted, lower=metric == 'br')})
    return rows, alpha_rows


def repair_figures(args, mode, rows, br_limit):
    import numpy as np
    plt = plotting()
    from matplotlib.ticker import PercentFormatter
    for alpha in sorted(set(row['alpha'] for row in rows)):
        for metric, ylabel in [('rr_pooled', 'Held-out repair rate'), ('br_pooled', 'Held-out breakage rate')]:
            fig, ax = plt.subplots(figsize=(9, 5))
            for i, method in enumerate(LABELS):
                selected = sorted([row for row in rows if row['method'] == method and row['alpha'] == alpha],
                                  key=lambda row: row['n_fix'])
                if not selected:
                    continue
                ax.plot([row['n_fix'] for row in selected],
                        [row[metric] if row[metric] is not None else np.nan for row in selected],
                        label=LABELS[method], color=BASE_COLORS[i], linewidth=1.7,
                        linestyle='--' if i >= 5 else '-', marker=['o', 's', '^', 'D', 'v', 'P', 'X', '*', '+'][i],
                        markersize=5)
            ax.set_xticks(sorted(set(row['n_fix'] for row in rows)))
            ax.set_ylim(0, 1 if metric == 'rr_pooled' else br_limit)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlabel('Repair budget (weights)')
            ax.set_ylabel(ylabel)
            counts = sorted(set(row['n_mutants'] for row in rows if row['alpha'] == alpha))
            ax.set_title(f'RQ3 | {args.dataset} | {mode} | alpha={alpha:g} | mutants per group: {counts}')
            ax.grid(axis='y', color='#DDDDDD', linewidth=.6)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
            fig.tight_layout()
            save_figure(fig, args.output_dir / 'figures', f'rq3_{mode}_{metric}_alpha{alpha:g}')


def main():
    args = setup(3, __doc__)
    if args is None:
        return
    summaries, tests, alpha_tests, sources = [], [], [], []
    for mode in args.modes:
        path = args.results_dir / mode / 'repair_results.json'
        data = read_json(path)
        groups = repair_groups(data, args.dataset)
        summary = summarize(groups, args.dataset, mode)
        paired, alpha = comparisons(groups, args.dataset, mode)
        summaries.extend(summary)
        tests.extend(paired)
        alpha_tests.extend(alpha)
        sources.append({'mode': mode, **source_info(path), 'n_records': len(data['results']),
                        'split_seed': data.get('split_seed'),
                        'group_sizes': [{'n_fix': key[0], 'alpha': key[1], 'method': key[2], 'n_mutants': len(value)}
                                        for key, value in sorted(groups.items())]})
    br_limit = min(1, max(.01, max(row['br_pooled'] or 0 for row in summaries) * 1.1))
    for mode in args.modes:
        repair_figures(args, mode, [row for row in summaries if row['mode'] == mode], br_limit)
    write_csv(args.output_dir / 'tables' / 'rq3_summary.csv', summaries)
    if tests:
        write_csv(args.output_dir / 'tables' / 'rq3_comparisons.csv', tests)
    if alpha_tests:
        write_csv(args.output_dir / 'tables' / 'rq3_alpha_comparisons.csv', alpha_tests)
    write_json(args.output_dir / 'rq3_analysis.json', {'dataset': args.dataset, 'sources': sources,
               'summary': summaries, 'comparisons': tests, 'alpha_comparisons': alpha_tests})
    print(f'RQ3: saved figures and tables to {args.output_dir}')


if __name__ == '__main__':
    main()
