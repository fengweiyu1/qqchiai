"""Configuration and paths shared by the command-line entry points."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--config', default='configs/fm.json', help='JSON configuration, relative to the artifact root')
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--dry-run', action='store_true', help='Print resolved settings without loading data or running an experiment')
    return ap


def resolve(path):
    path = Path(path).expanduser()
    return str(path if path.is_absolute() else ROOT / path)


def load_config(args):
    with open(resolve(args.config)) as f:
        cfg = json.load(f)
    if cfg.get('dataset') not in ('fm', 'cifar10', 'gtsrb'):
        raise ValueError('Configuration dataset must be fm, cifar10, or gtsrb')
    args.dataset = cfg['dataset']
    args.data_dir = resolve(args.data_dir or cfg['data_dir'])
    return cfg


def defaults(args, values):
    for name, value in values.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)


def dry_run(args):
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return True
    return False


def require_file(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Missing input: {path}')


def validate_pool(path, n_mutants):
    require_file(path)
    with open(path) as f:
        pool = json.load(f)
    mutants = pool.get('mutants', [])[:n_mutants]
    if not mutants:
        raise ValueError('The mutant pool contains no selected mutants')
    for mutant in mutants:
        checkpoint = Path(path).parent / f"mutant_{mutant['mutant_id']:04d}.pth"
        fallback = Path(mutant.get('save_path', ''))
        if not checkpoint.is_file() and not fallback.is_file():
            raise FileNotFoundError(f'Missing mutant checkpoint: {checkpoint}')


def prepare_output(path, filename):
    output = Path(path)
    if (output / filename).exists():
        raise FileExistsError(f'Output already exists: {output / filename}; choose a new --output-dir')
    output.mkdir(parents=True, exist_ok=True)


def save_settings(args):
    """Record effective settings beside newly generated outputs."""
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run_settings.json').write_text(json.dumps(vars(args), indent=2) + '\n')


def positive(name, value):
    if value <= 0:
        raise ValueError(f'{name} must be positive')


def positive_list(name, value, cast=int):
    values = [cast(x) for x in value.split(',')]
    if not values or any(x <= 0 for x in values):
        raise ValueError(f'{name} must contain positive comma-separated values')
    return values
