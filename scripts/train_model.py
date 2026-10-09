"""Train the configured network using the extracted training routine."""
from _common import parser, load_config, defaults, resolve, dry_run, prepare_output, save_settings, positive


def main():
    ap = parser(__doc__)
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--lr', type=float)
    ap.add_argument('--seed', type=int)
    args = ap.parse_args()
    cfg = load_config(args)
    defaults(args, cfg['training'])
    args.output_dir = resolve(args.output_dir or cfg['output_root'] + '/models')
    positive('epochs', args.epochs)
    positive('lr', args.lr)
    if dry_run(args):
        return

    import random
    from importlib import import_module
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    prepare_output(args.output_dir, f'{args.dataset}_arachne_rq1_best_model.pth')
    save_settings(args)
    model = import_module(f'src.models.{args.dataset}')
    model.train_model(args.data_dir, args.output_dir, args.epochs, args.lr)


if __name__ == '__main__':
    main()
