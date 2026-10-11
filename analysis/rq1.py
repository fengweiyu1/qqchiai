"""Compare the ten involvement/result-side configurations for RQ1."""
from common import setup, read_json, index_mutants, source_info, export_localization

SIGNALS = [('Confidence', 'QQ'), ('Deficit', 'QQ_deficit'), ('Loss', 'QQ_loss'),
           ('Cos', 'QQ_cosSim'), ('MaxLog', 'QQ_max_log')]
METHODS = [(f'{prefix}_{label}', f'{prefix}_{key}_cos_ochiai')
           for prefix in ('G', 'FI') for label, key in SIGNALS]


def main():
    args = setup(1, __doc__)
    if args is None:
        return
    datasets, metadata = [], []
    for mode in args.modes:
        path = args.results_dir / mode / 'methods' / 'faulty_ranks.json'
        n, mutants = index_mutants(read_json(path), args.dataset)
        datasets.append((mode, [mutants[mid] for mid in sorted(mutants)], n, METHODS))
        metadata.append({'mode': mode, **source_info(path), 'n_mutants': len(mutants)})
    export_localization(args, 1, datasets, metadata, variant=True)


if __name__ == '__main__':
    main()
