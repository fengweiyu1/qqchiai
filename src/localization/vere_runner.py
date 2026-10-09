"""VeRe baseline using the same test selection as the localization runner."""

import json
import time
from pathlib import Path
import numpy as np
import torch
from src.registry import get_model_and_loader, DATASET_CONFIG
from . import signals as fl
from .common import rank_of, predict, restore_original
from .vere import vere_significance

VERE_ITS = fl.VERE_INTERVAL_TIMES
VERE_N = fl.VERE_N


def run(args):

    ModelClass, get_loaders = get_model_and_loader(args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = args.data_dir or DATASET_CONFIG[args.dataset]['data_dir']
    _, test_loader = get_loaders(data_dir, batch_size=128)
    imgs, lbls = fl.build_test_suite(test_loader)
    labels = lbls.numpy()
    pool = json.load(open(args.mutant_pool))
    mutants = pool['mutants'][:args.n_mutants]
    pool_dir = Path(args.mutant_pool).parent
    probe = ModelClass()
    W_OUT, W_IN = dict(probe.named_modules())['fc2'].weight.shape
    N = W_OUT * W_IN
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f'VeRe fc2 (B 口径) — {args.dataset}  fc2 {W_OUT}x{W_IN}={N}  mutants={len(mutants)}  '
          f'interval_times={VERE_ITS}  device={device}')

    preds_orig = None
    recs, start = [], time.time()
    for mi, m in enumerate(mutants):
        mid = m['mutant_id']
        pth_path = pool_dir / f"mutant_{mid:04d}.pth"
        if not pth_path.exists():
            pth_path = Path(m['save_path'])
        pth = torch.load(pth_path, map_location='cpu', weights_only=False)
        faulty = np.asarray(m.get('mutated_indices') or pth['mutated_indices'], dtype=np.int64)
        if len(faulty) == 0:
            continue
        model = ModelClass(); model.load_state_dict(pth['model_state_dict'])
        model = model.to(device).eval()
        if preds_orig is None:
            om = ModelClass(); om.load_state_dict(pth['model_state_dict'])
            preds_orig = predict(restore_original(om, m, pth).to(device).eval(), imgs, device)
            del om
        preds_mut = predict(model, imgs, device)

        # ── B 口径,与 run_fl_behavior_oracle_cleanpass.py 逐字相同 ──
        broken = (preds_orig == labels) & (preds_mut != labels)
        unchanged = (preds_orig == preds_mut) & (preds_orig == labels)
        nb, nu = int(broken.sum()), int(unchanged.sum())
        if nb == 0 or nu == 0:
            print(f'  [skip] mutant {mid}: n_broken={nb} n_unchanged={nu}')
            del model; continue
        rng = np.random.default_rng(mid)
        pass_idx = rng.choice(np.flatnonzero(unchanged), size=min(nb, nu), replace=False)
        sub = np.concatenate([np.flatnonzero(broken), pass_idx]); sub.sort()
        sub_t = torch.from_numpy(sub)

        ranks = {}
        for it in VERE_ITS:
            ns = vere_significance(model, 'fc2', imgs[sub_t], lbls[sub_t], device,
                                   interval_times=it, n_fail=VERE_N)
            wr = rank_of(fl.neuron_scores_to_weights(ns, W_OUT))
            ranks[f'VeRe_it{int(round(it * 10))}'] = [int(wr[i]) for i in faulty]
        recs.append({'mutant_id': mid, 'level': m.get('level', 1), 'n_faulty': int(len(faulty)),
                     'ranks': ranks, 'subset_info': {'n_broken': nb, 'n_unchanged': nu,
                                                     'n_used': int(len(sub))}})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (mi + 1) % 25 == 0 or mi + 1 == len(mutants):
            print(f'  {mi+1}/{len(mutants)}  ({time.time()-start:.0f}s)', flush=True)

    json.dump({'dataset': args.dataset, 'total_weights': int(N),
               'oracle_def': 'behavior_broken_vs_correct_unchanged', 'mutants': recs},
              open(out / 'faulty_ranks.json', 'w'))
    print(f'完成 → {out/"faulty_ranks.json"}  ({len(recs)} 个 mutant)')
    for k in recs[0]['ranks'] if recs else []:
        r = np.array([v for rec in recs for v in rec['ranks'][k]])
        print(f'  {k:<10}  @1% {np.mean(r <= N*0.01):6.1%}  @5% {np.mean(r <= N*0.05):6.1%}  '
              f'@20% {np.mean(r <= N*0.20):6.1%}  名次中位 {np.median(r):.0f}')
