"""Generate single or composed multi-fault models."""
from _common import (parser, load_config, defaults, resolve, dry_run, require_file,
                     prepare_output, save_settings, positive, positive_list)


def main():
    ap = parser(__doc__)
    ap.add_argument('--mode', choices=['single', 'multi'], default='single')
    ap.add_argument('--model-path')
    ap.add_argument('--from-single', help='Directory containing the accepted single-fault pool')
    for flag in ['n-mutants', 'n-bins', 'min-broken', 'max-values-per-position',
                 'max-per-group', 'max-try-factor', 'test-suite-size', 'seed']:
        ap.add_argument('--' + flag, type=int)
    ap.add_argument('--ratio', type=float)
    ap.add_argument('--threshold', type=float)
    ap.add_argument('--n-faulty-list', help='Fault counts for multi mode, e.g. 2,3,4,5')
    args = ap.parse_args()
    cfg = load_config(args)
    defaults(args, cfg['mutation'])
    args.model_path = resolve(args.model_path or cfg['model_path'])
    args.output_dir = resolve(args.output_dir or cfg['output_root'] + '/mutants_' + args.mode)
    args.from_single = resolve(args.from_single or cfg['output_root'] + '/mutants_single')
    args.pct_list = None  # The multi-fault adapter derives percentages from explicit fault counts.
    for name in ['n_mutants', 'n_bins', 'min_broken', 'max_values_per_position', 'max_per_group', 'max_try_factor']:
        positive(name, getattr(args, name))
    if args.ratio < 1 or not 0 <= args.threshold <= 1 or args.test_suite_size < 0:
        raise ValueError('Require ratio >= 1, threshold in [0,1], and test-suite-size >= 0')
    counts = positive_list('n-faulty-list', args.n_faulty_list)
    if counts != sorted(set(counts)) or counts[0] < 2:
        raise ValueError('Multi-fault counts must be strictly increasing and start at 2 or higher')
    if dry_run(args):
        return
    require_file(args.model_path)
    if args.mode == 'multi':
        require_file(args.from_single + '/mutant_pool.json')
    prepare_output(args.output_dir, 'mutant_pool.json')
    save_settings(args)
    if args.mode == 'single':
        from src.mutation.single import gen_single
        gen_single(args)
    else:
        from src.mutation.multi import gen_multi
        gen_multi(args)


if __name__ == '__main__':
    main()
