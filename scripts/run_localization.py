"""Run the localization methods and/or the VeRe baseline."""
from argparse import Namespace
from _common import (parser, load_config, defaults, resolve, dry_run, validate_pool,
                     prepare_output, save_settings, positive)


def main():
    ap = parser(__doc__)
    ap.add_argument('--mode', choices=['single', 'multi'], default='single')
    ap.add_argument('--mutant-pool')
    ap.add_argument('--baseline', choices=['all', 'methods', 'vere'], default='all')
    ap.add_argument('--n-mutants', type=int)
    ap.add_argument('--execution', choices=['both', 'gradient', 'forward_impact'])
    ap.add_argument('--fi-neuron2out', choices=['pc', 'one'])
    args = ap.parse_args()
    cfg = load_config(args)
    defaults(args, cfg['localization'])
    args.mutant_pool = resolve(args.mutant_pool or cfg['output_root'] + '/mutants_' + args.mode + '/mutant_pool.json')
    args.output_dir = resolve(args.output_dir or cfg['output_root'] + '/localization/' + args.mode)
    positive('n-mutants', args.n_mutants)
    if dry_run(args):
        return
    validate_pool(args.mutant_pool, args.n_mutants)
    groups = ['methods', 'vere'] if args.baseline == 'all' else [args.baseline]
    for group in groups:
        prepare_output(args.output_dir + '/' + group, 'faulty_ranks.json')
    for group in groups:
        selected = Namespace(**vars(args))
        selected.output_dir = args.output_dir + '/' + group
        save_settings(selected)
        if group == 'methods':
            from src.localization.runner import run
        else:
            from src.localization.vere_runner import run
        run(selected)


if __name__ == '__main__':
    main()
