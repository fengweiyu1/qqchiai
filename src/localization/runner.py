"""Localization using newly broken tests and correctly unchanged tests (balanced 1:1)."""

import json
import time
from pathlib import Path
import numpy as np
import torch
from src.registry import get_model_and_loader, DATASET_CONFIG
from . import signals as fl
from .common import rank_of, predict, restore_original


def run(args):
    fl.FI_NEURON2OUT = args.fi_neuron2out

    ModelClass, get_loaders = get_model_and_loader(args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = args.data_dir or DATASET_CONFIG[args.dataset]['data_dir']
    _, test_loader = get_loaders(data_dir, batch_size=128)
    imgs, lbls = fl.build_test_suite(test_loader)
    labels = lbls.numpy()
    T = len(labels)

    pool = json.load(open(args.mutant_pool))
    mutants = pool['mutants'][:args.n_mutants]
    pool_dir = Path(args.mutant_pool).parent

    exec_modes = list(fl.INVOLVEMENT_FNS) if args.execution == 'both' else [args.execution]
    PREFIXES = [fl.EXEC_PREFIX[e] for e in exec_modes]
    probe = ModelClass()
    W_OUT, W_IN = dict(probe.named_modules())['fc2'].weight.shape
    N = W_OUT * W_IN

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print('=' * 78)
    print(f'行为口径 fail/pass(通过 = 对→对)— {args.dataset.upper()}  fc2 {W_OUT}x{W_IN}={N}  '
          f'T={T}  device={device}')
    print(f'  failed = broken (对→错),  passed = 对→对 下采样到 1:1,  '
          f'FI_NEURON2OUT={fl.FI_NEURON2OUT}')
    print('=' * 78)

    preds_orig = None
    recs, start = [], time.time()

    for mi, m in enumerate(mutants):
        mid = m['mutant_id']
        pth_path = pool_dir / f"mutant_{mid:04d}.pth"
        if not pth_path.exists():
            pth_path = Path(m['save_path'])
        pth = torch.load(pth_path, map_location='cpu', weights_only=False)

        true_faulty = m.get('mutated_indices') or pth['mutated_indices']
        if len(true_faulty) == 0:
            continue

        model = ModelClass(); model.load_state_dict(pth['model_state_dict'])
        model = model.to(device).eval()

        if preds_orig is None:
            om = ModelClass(); om.load_state_dict(pth['model_state_dict'])
            om = restore_original(om, m, pth).to(device).eval()
            preds_orig = predict(om, imgs, device)
            acc0 = float((preds_orig == labels).mean())
            print(f'原模型还原完成: acc = {acc0:.4f}  (整批 mutant 共用)')
            del om

        impacts = {}
        probs = logits = None
        for ex in exec_modes:
            imp, probs, logits = fl.INVOLVEMENT_FNS[ex](model, 'fc2', imgs, lbls, device)
            impacts[fl.EXEC_PREFIX[ex]] = imp
        preds_mut = probs.argmax(axis=1)

        # ── fail / pass:与原脚本唯一的差别在 unchanged 多了"原来判对"────────
        broken    = (preds_orig == labels) & (preds_mut != labels)                  # 对 → 错
        unchanged = (preds_orig == preds_mut) & (preds_orig == labels)              # 对 → 对
        n_unch_wrong = int(((preds_orig == preds_mut) & (preds_orig != labels)).sum())
        nb = int(broken.sum()); nu = int(unchanged.sum())
        if nb == 0 or nu == 0:
            print(f'  [skip] mutant {mid}: n_broken={nb} n_unchanged={nu}')
            del model; continue

        rng = np.random.default_rng(mid)
        pass_pool = np.flatnonzero(unchanged)
        k = min(nb, nu)
        pass_idx = rng.choice(pass_pool, size=k, replace=False)
        sub = np.concatenate([np.flatnonzero(broken), pass_idx])
        sub.sort()

        failed = broken[sub]
        passed = ~failed
        nf, np_count = int(failed.sum()), int(passed.sum())

        imp_sub = {p: impacts[p][sub] for p in impacts}
        probs_s, logits_s, labels_s = probs[sub], logits[sub], labels[sub]
        confs = np.clip(probs_s.max(axis=1), 0.01, 0.99)
        faulty_arr = np.asarray(true_faulty)
        ranks = {}

        for oracle in fl.ORACLES:
            ew = fl.oracle_weights(oracle, failed, passed, confs, logits_s,
                                   probs_s, labels_s, nf, np_count)
            for prefix, imp in imp_sub.items():
                wr = rank_of(fl.compute_cosine_fl_score(imp, ew))
                ranks[f'{prefix}_{oracle}_cos_ochiai'] = [int(wr[i]) for i in faulty_arr]

        cov_all, _, _ = fl.neuron_coverage(model, 'fc2', imgs, device, act_threshold=0.1)
        ncov = cov_all[sub]
        for bname, weighted in [('BB', False), ('BQ', True)]:
            ns = fl.sbfl_ochiai(ncov, failed.astype(float), passed.astype(float),
                                confs, weighted)
            wr = rank_of(fl.neuron_scores_to_weights(ns, W_OUT))
            ranks[f'{bname}_wfn'] = [int(wr[i]) for i in faulty_arr]

        cn = (confs - confs.min()) / (confs.max() - confs.min() + 1e-12)
        y_cos = {'DFVarCos': failed.astype(np.float64), 'GDVarCos': 1.0 + cn}
        for vprefix in fl.VAR_EXECS:
            if vprefix not in imp_sub:
                continue
            for ratio in fl.VAR_RATIOS:
                cov_w = fl.topk_coverage(imp_sub[vprefix], ratio)
                tag = f'k{int(round(ratio * 100))}'
                for vname, weighted in [('DFVar', False), ('GDVar', True)]:
                    wr = rank_of(fl.sbfl_ochiai(cov_w, failed.astype(float),
                                                passed.astype(float), confs, weighted))
                    ranks[f'{vprefix}_{vname}_{tag}'] = [int(wr[i]) for i in faulty_arr]
                cov_f = cov_w.astype(np.float32)
                for vname, yv in y_cos.items():
                    wr = rank_of(fl.compute_cosine_fl_score(cov_f, yv))
                    ranks[f'{vprefix}_{vname}_{tag}'] = [int(wr[i]) for i in faulty_arr]

        recs.append({'mutant_id': mid, 'level': m.get('level', 1),
                     'n_faulty': len(true_faulty), 'ranks': ranks,
                     'subset_info': {'n_broken': nb, 'n_unchanged': nu,
                                     'n_unchanged_wrong': n_unch_wrong,
                                     'n_excluded': int(T - nb - nu),
                                     'n_used': int(len(sub))}})
        del model, impacts, imp_sub
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (mi + 1) % 10 == 0 or (mi + 1) == len(mutants):
            print(f'  {mi+1}/{len(mutants)}  ({time.time()-start:.0f}s)')

    with open(out / 'faulty_ranks.json', 'w') as f:
        json.dump({'dataset': args.dataset,
                   'oracle_def': 'behavior_broken_vs_correct_unchanged',
                   'total_weights': int(N), 'w_in': int(W_IN), 'mutants': recs}, f)
    print(f'\n完成 → {out/"faulty_ranks.json"}  ({len(recs)} 个 mutant)')

    nb = np.array([r['subset_info']['n_broken'] for r in recs])
    nw = np.array([r['subset_info']['n_unchanged_wrong'] for r in recs])
    print(f'谱的规模: n_broken 中位 {np.median(nb):.0f};新排除的"错→同一种错" 中位 {np.median(nw):.0f} 条')

    def auc(r, maxp=20):
        g = np.arange(1, int(N * maxp / 100) + 1); xx = g / N
        return float(np.trapz([np.mean(r <= s) for s in g], xx) / (xx[-1] - xx[0]))

    KEYS = [f'{p}_{o}' for p in PREFIXES
            for o in ('QB_cos_ochiai', 'QQ_loss_cos_ochiai', 'QQ_cos_diff_cos_ochiai',
                      'DFVar_k10', 'GDVar_k10')] + ['BB_wfn', 'BQ_wfn']
    print(f'\n{"method":<34}{"AUC@20%":>9}{"DR@1%":>8}{"DR@5%":>8}{"名次中位":>10}')
    for k in KEYS:
        r = np.array([v for rec in recs for v in rec['ranks'].get(k, [])])
        if r.size:
            print(f'{k:<34}{auc(r):>9.3f}{np.mean(r<=N*0.01):>8.1%}'
                  f'{np.mean(r<=N*0.05):>8.1%}{np.median(r):>10.0f}')
