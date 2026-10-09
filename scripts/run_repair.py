"""Compare localization methods with a shared differential-evolution repair procedure."""
from _common import (parser, load_config, defaults, resolve, dry_run, validate_pool,
                     prepare_output, save_settings, positive, positive_list)


def main():
    ap = parser(__doc__)
    ap.add_argument('--mode', choices=['single', 'multi'], default='single')
    ap.add_argument('--mutant-pool')
    ap.add_argument('--n-mutants', type=int)
    ap.add_argument('--n-fix-list')
    ap.add_argument('--alpha-list')
    ap.add_argument('--localizers')
    ap.add_argument('--max-gen', type=int)
    ap.add_argument('--workers', type=int)
    ap.add_argument('--split-seed', type=int)
    args = ap.parse_args()
    cfg = load_config(args)
    settings = dict(cfg['repair'])
    settings['n_mutants'] = settings['n_mutants_' + args.mode]
    settings['n_fix_list'] = settings['n_fix_list_' + args.mode]
    defaults(args, {k: v for k, v in settings.items() if not k.endswith(('_single', '_multi'))})
    args.mutant_pool = resolve(args.mutant_pool or cfg['output_root'] + '/mutants_' + args.mode + '/mutant_pool.json')
    args.output_dir = resolve(args.output_dir or cfg['output_root'] + '/repair/' + args.mode)
    for name in ['n_mutants', 'max_gen', 'workers']:
        positive(name, getattr(args, name))
    positive_list('n-fix-list', args.n_fix_list)
    positive_list('alpha-list', args.alpha_list, float)
    allowed = {'ours', 'arachnevar', 'deepfaultvar', 'guidervar', 'deepfault', 'guider', 'vere', 'random', 'truefault'}
    locs = [x.strip() for x in args.localizers.split(',')]
    if not locs or len(locs) != len(set(locs)) or not set(locs) <= allowed:
        raise ValueError('Unknown, empty, or duplicate localizer')
    if dry_run(args):
        return
    validate_pool(args.mutant_pool, args.n_mutants)
    prepare_output(args.output_dir, 'repair_results.json')
    save_settings(args)
    from src.repair.differential_evolution import run
    run(args)


if __name__ == '__main__':
    main()
