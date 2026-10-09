"""Offline checks using temporary synthetic inputs; no benchmark data is downloaded."""
import _common  # Makes the artifact root importable from any working directory.
from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.registry import get_model_and_loader
from src.localization import signals, runner, vere_runner
from src.localization.common import restore_original
from src.mutation import single, multi
from src.repair import differential_evolution as repair


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4, bias=False)
        self.fc2 = nn.Linear(4, 4)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.eye(4))
            weights = torch.full((4, 4), -1.)
            weights.fill_diagonal_(2.)
            self.fc2.weight.copy_(weights)
            self.fc2.bias.zero_()

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def check_gradients():
    for dataset, shape, classes in [('fm', (1, 28, 28), 10),
                                    ('cifar10', (3, 32, 32), 10),
                                    ('gtsrb', (3, 48, 48), 43)]:
        Model, _ = get_model_and_loader(dataset)
        model = Model().eval()
        images = torch.randn(3, *shape)
        labels = torch.tensor([0, 1, 2])
        assert model(images).shape == (3, classes)
        error = signals.verify_against_autograd(model, 'fc2', images, labels, 'cpu')
        assert error < 1e-5, (dataset, error)
        impacts, _, _ = signals.grad_involvement(model, 'fc2', images, labels, 'cpu')
        np.testing.assert_allclose(impacts.sum(1), np.ones(3), atol=1e-6)


def check_fitness():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(12, 4))
    W, b = rng.normal(size=(3, 4)), rng.normal(size=3)
    y = np.arange(12) % 3
    rows, cols = np.array([0, 2]), np.array([1, 3])
    negative = np.arange(12) % 2 == 0
    values = np.array([.3, -.7])
    fit = repair.FastFitness(A, A @ W.T + b, y, rows, cols, W[rows, cols], negative, 10.)
    changed = W.copy()
    changed[rows, cols] = values
    z = A @ changed.T + b
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    ce = -np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1 - 1e-7))
    value = np.where(z.argmax(1) == y, 1., 1. / (ce + 1.))
    np.testing.assert_allclose(fit(values), value[~negative].sum() + 10 * value[negative].sum(), rtol=1e-12)


def check_pipeline(folder):
    model = ToyModel().eval()
    # Each diagonal weight controls one class, producing faults that can accumulate.
    # Small positive jitter avoids all-zero activations and tied logits.
    images = torch.eye(4).repeat(40, 1) + torch.rand(160, 4) * .05
    labels = model(images).argmax(1).detach()
    labels[::17] = (labels[::17] + 1) % 4
    data = TensorDataset(images, labels)

    def loaders(data_dir, batch_size=128):
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        return loader, loader

    def registry(dataset):
        return ToyModel, loaders

    original = folder / 'base.pth'
    torch.save({'model_state_dict': model.state_dict()}, original)
    args = Namespace(dataset='fm', model_path=str(original), data_dir=str(folder),
                     output_dir=str(folder / 'single'), n_mutants=4, n_bins=1,
                     ratio=3., min_broken=1, max_values_per_position=500, threshold=0.,
                     test_suite_size=0, seed=0, n_faulty_list='2,3', pct_list=None,
                     max_per_group=2, max_try_factor=20, from_single=str(folder / 'single'))
    with patch.object(single, 'get_model_and_loader', registry), \
         patch.object(multi, 'get_model_and_loader', registry), \
         patch.object(runner, 'get_model_and_loader', registry), \
         patch.object(vere_runner, 'get_model_and_loader', registry), \
         patch('src.registry.get_model_and_loader', registry):
        single.gen_single(args)
        pool = json.loads((folder / 'single/mutant_pool.json').read_text())
        assert len(pool['mutants']) >= 2, 'Fixture did not generate enough accepted single faults'
        for m in pool['mutants']:
            checkpoint = torch.load(folder / 'single' / f"mutant_{m['mutant_id']:04d}.pth", weights_only=False)
            restored = ToyModel()
            restored.load_state_dict(checkpoint['model_state_dict'])
            restore_original(restored, m, checkpoint)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value)
        args.output_dir = str(folder / 'multi')
        multi.gen_multi(args)
        composed = json.loads((folder / 'multi/mutant_pool.json').read_text())
        assert composed['mutants'], 'Fixture did not generate any accepted multi-fault model'
        for mode in ['single', 'multi']:
            loc = Namespace(dataset='fm', data_dir=str(folder),
                            mutant_pool=str(folder / mode / 'mutant_pool.json'),
                            output_dir=str(folder / ('fl_' + mode)), n_mutants=2,
                            execution='both', fi_neuron2out='pc')
            runner.run(loc)
            methods = json.loads((Path(loc.output_dir) / 'faulty_ranks.json').read_text())
            assert methods['mutants']
            loc.output_dir = str(folder / ('vere_' + mode))
            vere_runner.run(loc)
            vere = json.loads((Path(loc.output_dir) / 'faulty_ranks.json').read_text())
            assert [m['mutant_id'] for m in methods['mutants']] == [m['mutant_id'] for m in vere['mutants']]
            for m in methods['mutants']:
                for ranks in m['ranks'].values():
                    assert len(ranks) == m['n_faulty']
                    assert all(1 <= r <= methods['total_weights'] for r in ranks)
        rep = Namespace(dataset='fm', data_dir=str(folder),
                        mutant_pool=str(folder / 'single/mutant_pool.json'),
                        output_dir=str(folder / 'repair'), n_mutants=2, n_fix_list='1,2',
                        alpha_list='1', max_gen=2, split_seed=0, workers=1,
                        localizers=','.join(repair.LOCALIZERS))
        repair.run(rep)
        result = json.loads((folder / 'repair/repair_results.json').read_text())
        assert result['results']
        assert set(r['localizer'] for r in result['results']) == set(repair.LOCALIZERS)
        for r in result['results']:
            e = r['E']
            assert 0 <= e['n_fixed'] <= e['n_broken']
            assert 0 <= e['n_new_broken'] <= e['n_mut_correct']


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    captured = io.StringIO()
    try:
        with redirect_stdout(captured), tempfile.TemporaryDirectory(prefix='artifact-smoke-') as folder:
            check_gradients()
            check_fitness()
            check_pipeline(Path(folder))
    except Exception:
        print(captured.getvalue())
        raise
    print('PASS: model shapes and analytical gradients (3 datasets)')
    print('PASS: repair fitness agrees with full-logit evaluation')
    print('PASS: single/multi mutation, model restoration, localization, VeRe, and repair')
    print('Synthetic inputs only; no benchmark data downloaded or retained.')


if __name__ == '__main__':
    main()
