"""Weight-stratified single faults with a minimum broken-test count."""

import json
from pathlib import Path
import numpy as np
import torch
from src.registry import get_model_and_loader
from .common import (build_test_suite, predict, gradient_candidates, apply_weights, forward_impact, report_selection_bias, report_fairness)


def fc2_inputs(model, imgs, device, bs=256):
    """整个测试集的 fc2 输入 [T, W_in](故障只在 fc2,所以对原模型和所有 mutant 都一样)。"""
    fc2 = dict(model.named_modules())['fc2']
    acts = []
    h = fc2.register_forward_hook(lambda mod, inp, out: acts.append(inp[0].detach().cpu()))
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            model(imgs[i:i + bs].to(device))
    h.remove()
    return torch.cat(acts).numpy().astype(np.float64)


class SingleWeightPredictor:
    """只改 fc2 一根权重 (o, i) 时,快速算整个测试集的新预测。"""

    def __init__(self, A, W, b):
        self.A, self.Z = A, A @ W.T + b
        self.cache = {}

    def _others(self, o):
        """除类别 o 以外的最大 logit 和它的类别(每个 o 只算一次)。"""
        if o not in self.cache:
            Zx = self.Z.copy(); Zx[:, o] = -np.inf
            self.cache[o] = (Zx.max(axis=1), Zx.argmax(axis=1))
        return self.cache[o]

    def predict(self, o, i, delta):
        mx, am = self._others(o)
        zo = self.Z[:, o] + self.A[:, i] * delta
        return np.where(zo > mx, o, am)


def gen_single(args):
    ModelClass, get_loaders = get_model_and_loader(args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.RandomState(args.seed)

    base = ModelClass()
    ck = torch.load(args.model_path, map_location='cpu', weights_only=False)
    base.load_state_dict(ck['model_state_dict'] if 'model_state_dict' in ck else ck)
    base = base.to(device).eval()
    _, test_loader = get_loaders(args.data_dir, batch_size=128)
    imgs, lbls = build_test_suite(test_loader, args.test_suite_size)
    labels = lbls.numpy()
    orig_pred = predict(base, imgs, device)
    orig_correct = (orig_pred == labels)
    orig_acc = float(orig_correct.mean() * 100)

    fc2 = dict(base.named_modules())['fc2']
    W = fc2.weight.detach().cpu().numpy().astype(np.float64)
    b = fc2.bias.detach().cpu().numpy().astype(np.float64)
    W_out, W_in = W.shape
    W_flat = W.ravel().copy()
    N = len(W_flat)
    A = fc2_inputs(base, imgs, device)
    swp = SingleWeightPredictor(A, W, b)
    mism = int((swp.Z.argmax(1) != orig_pred).sum())
    assert mism <= 2, f'用 fc2 输入重算的预测和模型不一致({mism} 条)'

    absW = np.abs(W_flat)
    asc = np.argsort(absW)
    abs_sorted = absW[asc]
    bin_of = np.empty(N, dtype=np.int64)
    bin_of[np.argsort(-absW, kind='stable')] = np.arange(N) * args.n_bins // N     # 档 0 = 最大的 1/n_bins
    top_of = lambda v: 1.0 - np.searchsorted(abs_sorted, np.abs(v), side='left') / N

    def try_position(idx, w_old):
        """把窗口里的新值随机打乱依次试;返回 (新值, broken 条数, 行为改变率, 试了几个) 或 (None, 最好的 broken, …)。"""
        a = abs(w_old)
        lo = np.searchsorted(abs_sorted, a / args.ratio, side='left')
        hi = np.searchsorted(abs_sorted, a * args.ratio, side='right')
        if hi <= lo:
            return None, 0, 0.0, 0
        pool = asc[lo:hi][rng.permutation(hi - lo)][:args.max_values_per_position]
        o, i = divmod(idx, W_in)
        best = 0
        for t, j in enumerate(pool, 1):
            v = float(W_flat[j])
            pred = swp.predict(o, i, v - w_old)
            nb = int((orig_correct & (pred != labels)).sum())
            rate = float((pred != orig_pred).mean())
            best = max(best, nb)
            if nb >= args.min_broken and rate >= args.threshold:
                return v, nb, rate, t
        return None, best, 0.0, len(pool)

    print('=' * 78)
    print(f'Step1 group-by-|w| + min_broken / SINGLE — {args.dataset.upper()}   fc2 {W_out}x{W_in}={N}   '
          f'原始准确率 {orig_acc:.2f}%   T={len(labels)}')
    print(f'选位置:按原值 |w| 分 {args.n_bins} 档,每档接受 {args.n_mutants}/{args.n_bins} 个')
    print(f'新值:从本层抄,|新值| ∈ [|原值|/{args.ratio:g}, {args.ratio:g}·|原值|],每位置最多试 '
          f'{args.max_values_per_position} 个')
    print(f'接受:broken(对→错)≥ {args.min_broken} 且行为改变率 ≥ {args.threshold:.3%}')
    print('=' * 78)

    cand = gradient_candidates(base, imgs, lbls, device)
    fi = forward_impact(base, imgs, device)
    quota = [args.n_mutants // args.n_bins + (1 if k < args.n_mutants % args.n_bins else 0)
             for k in range(args.n_bins)]
    by_bin = {k: rng.permutation(cand[bin_of[cand] == k]) for k in range(args.n_bins)}
    print('候选池按档分布(档 1 = 全层最大的那一档):',
          ' '.join(f'{k+1}:{len(by_bin[k])}' for k in range(args.n_bins)))

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    mutants, pairs, accepted, stats, best_miss = [], [], [], [], []
    for k in range(args.n_bins):
        got, visited = 0, 0
        for idx in by_bin[k]:
            if got >= quota[k]:
                break
            idx = int(idx); visited += 1
            w_old = float(W_flat[idx])
            v, nb, rate, tried = try_position(idx, w_old)
            if v is None:
                best_miss.append(nb)
                continue
            m = apply_weights(base, None, [idx], [v]).to(device)
            pred = predict(m, imgs, device)
            nb_real = int((orig_correct & (pred != labels)).sum())
            assert abs(nb_real - nb) <= 2, f'快速预测和真实前向的 broken 不一致: {nb} vs {nb_real}'
            acc = float((pred == labels).mean() * 100)
            mid = len(mutants)
            rec = {'mutant_id': mid, 'level': 1, 'pct': 1, 'mutation_rate': 0.0,
                   'num_mutated_weights': 1, 'mutated_indices': [idx], 'new_values': [v],
                   'original_values': [w_old], 'accuracy': acc, 'acc_drop': orig_acc - acc,
                   'behaviour_change': float((pred != orig_pred).mean()), 'new_failures': nb_real,
                   'n_broken': nb_real, 'values_tried': int(tried),
                   'w_bin': int(k), 'ratio': abs(v) / max(abs(w_old), 1e-12),
                   'save_path': str(out / f"mutant_{mid:04d}.pth")}
            keep = ('mutated_indices', 'new_values', 'original_values', 'pct', 'mutation_rate',
                    'num_mutated_weights', 'accuracy', 'acc_drop', 'behaviour_change', 'new_failures')
            torch.save(dict({kk: rec[kk] for kk in keep}, model_state_dict=m.state_dict()),
                       out / f"mutant_{mid:04d}.pth")
            mutants.append(rec); pairs.append((v, w_old)); accepted.append(idx)
            got += 1
            del m
        stats.append((k, len(by_bin[k]), visited, got, quota[k]))
        print(f'  档 {k+1:>2}: 候选 {len(by_bin[k]):5d}  访问 {visited:5d}  接受 {got:3d}/{quota[k]}'
              f'  接受率 {100*got/max(1,visited):5.1f}%' + ('' if got >= quota[k] else '   [!] 候选不够,缺额'),
              flush=True)

    fair = report_fairness(pairs, W_flat)
    bias = report_selection_bias(accepted, cand, fi, W_flat)
    print(f'\n  生成 {len(mutants)} 个单故障')
    if mutants:
        new_top = top_of(np.array([m['new_values'][0] for m in mutants]))
        old_top = top_of(np.array([m['original_values'][0] for m in mutants]))
        nbs = np.array([m['n_broken'] for m in mutants])
        print(f'  broken:中位 {np.median(nbs):.0f}  25% {np.percentile(nbs, 25):.0f}  75% {np.percentile(nbs, 75):.0f}'
              f'  最大 {nbs.max()}   每个收下的位置平均试了 {np.mean([m["values_tried"] for m in mutants]):.1f} 个新值')
        if best_miss:
            print(f'  没收下的位置:{len(best_miss)} 个,它们试过的新值里最多弄坏 中位 {np.median(best_miss):.0f} 条')
        print(f'  泄漏检查(无泄漏时:前 30% 约 30%;只看 |w| 前 1/5/20% 约 1/5/20%)')
        print(f'    原值落在全层 |w| 前 30% 的:{100*np.mean(old_top<=0.3):.0f}%')
        print(f'    新值落在全层 |w| 前 30% 的:{100*np.mean(new_top<=0.3):.0f}%   只看 |w|:'
              f'前1% {100*np.mean(new_top<=0.01):.1f}%  前5% {100*np.mean(new_top<=0.05):.1f}%  前20% {100*np.mean(new_top<=0.2):.1f}%')
        r = np.array([m['ratio'] for m in mutants])
        print(f'    |新值|/|原值|:中位 {np.median(r):.2f}  范围 {r.min():.2f}~{r.max():.2f}')

    json.dump({'original_accuracy': orig_acc, 'total_mutants': len(mutants),
               'config': {'mode': 'single', 'injection': 'switch_weight_group_by_w_minbroken',
                          'n_bins': args.n_bins, 'ratio': args.ratio, 'min_broken': args.min_broken,
                          'max_values_per_position': args.max_values_per_position,
                          'total_fc2_weights': int(N), 'behaviour_threshold': args.threshold,
                          'accept': f'n_broken >= {args.min_broken} and behaviour_change >= threshold',
                          'bin_stats': [{'bin': k, 'candidates': c, 'visited': v, 'accepted': g, 'quota': q}
                                        for k, c, v, g, q in stats],
                          'fairness': fair, 'selection_bias': bias},
               'mutants': mutants}, open(out / 'mutant_pool.json', 'w'), indent=2)
    print(f'  已保存: {out/"mutant_pool.json"}')
