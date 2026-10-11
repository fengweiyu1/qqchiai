"""Compare G_Cos with the six baseline implementations for RQ2."""
from common import (setup, read_json, index_mutants, source_info, export_localization,
                    require_methods, budget_grid)

METHODS = [('Ours', 'G_QQ_cosSim_cos_ochiai'), ('ArachneVar', 'G_QB_cos_ochiai'),
           ('DeepFaultVar', 'G_DFVar_k10'), ('GuiderVar', 'G_GDVar_k10'),
           ('DeepFault', 'BB_wfn'), ('Guider', 'BQ_wfn'), ('VeRe', 'VeRe')]


def merge_results(method_data, vere_data, dataset, selection='best'):
    import numpy as np
    n, methods = index_mutants(method_data, dataset)
    vn, vere = index_mutants(vere_data, dataset)
    if n != vn:
        raise ValueError('Methods and VeRe must have the same total_weights')
    ids = sorted(set(methods) & set(vere))
    if not ids:
        raise ValueError('Methods and VeRe have no matching mutant IDs')
    candidates = [f'VeRe_it{tag}' for tag in (10, 15, 20)] if selection == 'best' else [f'VeRe_{selection}']
    selected = []
    for mid in ids:
        if methods[mid]['n_faulty'] != vere[mid]['n_faulty']:
            raise ValueError(f'Fault counts disagree for mutant {mid}')
        info_a, info_b = methods[mid].get('subset_info', {}), vere[mid].get('subset_info', {})
        for field in ('n_broken', 'n_unchanged', 'n_used'):
            if field in info_a and field in info_b and info_a[field] != info_b[field]:
                raise ValueError(f'Test selection differs for mutant {mid}: {field}')
        mutant = dict(methods[mid], ranks={**methods[mid]['ranks'], **vere[mid]['ranks']})
        selected.append(mutant)
    require_methods(selected, [(key, key) for key in candidates])
    grid = budget_grid(n, 20)
    # Preserve the final plotting scripts' VeRe selection: highest mean DR on
    # integer budgets through 20%, averaged over aligned mutants.
    scores = {key: float(np.mean([np.searchsorted(np.sort(m['ranks'][key]), grid, side='right').mean()
                                 / m['n_faulty'] for m in selected])) for key in candidates}
    best = max(candidates, key=scores.get)
    for mutant in selected:
        mutant['ranks']['VeRe'] = mutant['ranks'][best]
    return n, selected, {'vere_selection': selection, 'vere_key': best, 'vere_mean_dr_20pct': scores,
                         'n_aligned': len(ids), 'excluded_methods_ids': sorted(set(methods) - set(ids)),
                         'excluded_vere_ids': sorted(set(vere) - set(ids))}


def main():
    def extra(parser):
        parser.add_argument('--vere-selection', choices=['best', 'it10', 'it15', 'it20'], default='best',
                            help='best preserves the paper plotting code; fixed multipliers are also available')
    args = setup(2, __doc__, extra)
    if args is None:
        return
    datasets, metadata = [], []
    for mode in args.modes:
        method_path = args.results_dir / mode / 'methods' / 'faulty_ranks.json'
        vere_path = args.results_dir / mode / 'vere' / 'faulty_ranks.json'
        n, mutants, alignment = merge_results(read_json(method_path), read_json(vere_path),
                                             args.dataset, args.vere_selection)
        datasets.append((mode, mutants, n, METHODS))
        metadata.append({'mode': mode, 'methods': source_info(method_path),
                         'vere': source_info(vere_path), **alignment})
    export_localization(args, 2, datasets, metadata)


if __name__ == '__main__':
    main()
