"""Arachne-style differential evolution with a fixed repair/evaluation split. See THIRD_PARTY.md."""

import json
import time
import multiprocessing as mp
from pathlib import Path
import numpy as np
from src.registry import DATASET_CONFIG
from src.metrics.repair import evaluate

CE_EPS = 1e-7
N_FIX = {'fm': 8, 'cifar10': 12, 'gtsrb': 14}
LOCALIZERS = ['ours', 'arachnevar', 'deepfaultvar', 'guidervar', 'deepfault', 'guider', 'vere', 'random', 'truefault']
_G = {}


class FastFitness:
    """只改 fc2 的 N 根权重时,快速算 Arachne 的适应度。

    A [n, W_in] fc2 输入;Z [n, C] 当前(mutant)logits;y [n] 真值;
    pos(o_k, i_k) 被改的权重;w0 [N] 它们当前的值;is_neg [n] 属于 I_neg(否则属于 I_pos)。
    """

    def __init__(self, A, Z, y, rows, cols, w0, is_neg, alpha):
        n, C = Z.shape
        self.S = np.unique(rows)                                # 被改动的类别
        U = np.setdiff1d(np.arange(C), self.S)
        self.M = A[:, cols].astype(np.float64)                  # [n, N]
        self.P = (rows[:, None] == self.S[None, :]).astype(np.float64)   # [N, |S|] 权重 → 类别
        self.ZS = Z[:, self.S].astype(np.float64)
        self.w0 = w0.astype(np.float64)
        self.y = y
        ZU = Z[:, U].astype(np.float64)
        # 未改动类别:logsumexp、以及"除真值类以外"的最大值
        if len(U):
            mU = ZU.max(axis=1)
            self.lse_U = mU + np.log(np.exp(ZU - mU[:, None]).sum(axis=1))
        else:
            self.lse_U = np.full(n, -np.inf)
        ZU_ex = ZU.copy()
        inU = np.isin(y, U)
        if len(U):
            posU = np.searchsorted(U, y)
            ZU_ex[inU, posU[inU]] = -np.inf
        self.max_U_ex = ZU_ex.max(axis=1) if len(U) else np.full(n, -np.inf)
        self.y_in_S = ~inU
        self.posS = np.searchsorted(self.S, y)
        self.posS_c = np.minimum(self.posS, len(self.S) - 1)       # 真值类不在 S 时随便取一格(会被 where 丢掉)
        self.ar_S = np.flatnonzero(self.y_in_S)                     # 真值类在 S 里的样本
        self.posS_S = self.posS[self.ar_S]
        self.z_true_U = np.where(inU, Z[np.arange(n), y], 0.0)
        self.is_neg = is_neg
        self.alpha = alpha
        self.ar = np.arange(n)

    def logits_S(self, x):
        return self.ZS + (self.M * (x - self.w0)) @ self.P       # [n, |S|]

    def correct_and_ce(self, x):
        zs = self.logits_S(x)
        # logsumexp(未改动类别, 改动类别):减掉最大值再求和(np.logaddexp.reduce 太慢)
        m = np.maximum(self.lse_U, zs.max(axis=1))
        lse = m + np.log(np.exp(self.lse_U - m) + np.exp(zs - m[:, None]).sum(axis=1))
        z_true = np.where(self.y_in_S, zs[self.ar, self.posS_c], self.z_true_U)
        # 除真值类以外的最大 logit:真值类在改动类别里时,把它那一格临时设成 −inf
        zs[self.ar_S, self.posS_S] = -np.inf
        other = np.maximum(self.max_U_ex, zs.max(axis=1))
        correct = z_true > other
        p = np.clip(np.exp(z_true - lse), CE_EPS, 1 - CE_EPS)   # Keras categorical_crossentropy 的截断
        return correct, -np.log(p)

    def __call__(self, x):
        correct, ce = self.correct_and_ce(x)
        s = np.where(correct, 1.0, 1.0 / (ce + 1.0))
        return float(s[~self.is_neg].sum() + self.alpha * s[self.is_neg].sum())


def de_search(fit, w_layer, n_dim, rng, pop_size=100, max_gen=100, cr=0.7, f_range=(0.5, 1.0)):
    """照 Arachne de_vk.py 的 DE。返回 (最好的解, 它的适应度, 跑了几代)。"""
    mean, std = float(np.mean(w_layer)), float(np.std(w_layer))
    lo, hi = float(np.min(w_layer)), float(np.max(w_layer))
    lo = lo * 2 if lo < 0 else lo / 2
    hi = hi * 2 if hi > 0 else hi / 2
    pop = rng.normal(mean, std, size=(pop_size, n_dim)).astype(np.float32)   # 初始值不截断(同官方)
    fits = np.array([fit(x) for x in pop])
    max_unchanged = max(int(max_gen / 10), 10)
    best_perf, n_unchanged, gen = None, 0, 0
    for gen in range(1, max_gen + 1):
        F = rng.uniform(*f_range)                                # 每代抽一次
        for i in range(pop_size):
            others = np.delete(np.arange(pop_size), i)
            a, b, c = rng.choice(others, size=3, replace=False)
            y = pop[i].copy()
            mask = rng.random(n_dim) < cr
            mask[rng.integers(n_dim)] = True                    # 至少换一维
            y[mask] = np.clip(pop[a][mask] + F * (pop[b][mask] - pop[c][mask]), lo, hi)
            fy = fit(y)
            if fy >= fits[i]:                                    # 不比旧的差就替换(原地,同代可见)
                pop[i], fits[i] = y, fy
        cur = fits.max()
        if best_perf is None:
            best_perf = cur
        elif np.float32(cur) == np.float32(best_perf):
            n_unchanged += 1
        else:
            n_unchanged = 0
            best_perf = max(best_perf, cur)
        if n_unchanged > max_unchanged:
            break
    k = int(np.argmax(fits))
    return pop[k].astype(np.float64), float(fits[k]), gen


def init_worker(A, y, orig_correct, Ridx, Eidx, b):
    _G.update(A=A, y=y, orig_correct=orig_correct, Ridx=Ridx, Eidx=Eidx, b=b)


def repair_task(t):
    """子进程:一个 mutant × 一个定位方法。t 只带这个 mutant 自己的东西(W、要修的下标、判对标记)。"""
    A, y, oc, Ridx, Eidx, b = (_G[k] for k in ('A', 'y', 'orig_correct', 'Ridx', 'Eidx', 'b'))
    rng = np.random.default_rng(t['seed'])
    W, idx, mc = t['W'], t['idx'], t['mut_correct']
    broken = oc & ~mc
    W_in = W.shape[1]
    rows, cols = idx // W_in, idx % W_in
    # DE 用的测试:I_neg = R 的 broken,I_pos = R 里 mutant 判对的
    bR, mcR = broken[Ridx], mc[Ridx]
    use = Ridx[bR | mcR]
    A_use = A[use]
    fit = FastFitness(A_use, A_use @ W.T + b, y[use], rows, cols, W.ravel()[idx], broken[use], t['alpha'])
    t0 = time.time()
    x, f_best, n_gen = de_search(fit, W, len(idx), rng, max_gen=t['max_gen'])
    W_new = W.copy(); W_new.ravel()[idx] = x
    return {'mutant_id': t['mid'], 'localizer': t['loc'], 'n_fix': int(len(idx)),
            'alpha': t['alpha'], 'idx': [int(i) for i in idx],   # 选中的权重,用来分析改了哪些位置
            'fault_hit': t['hit'], 'fitness': f_best, 'n_gen': n_gen,
            'fitness_before': fit(W.ravel()[idx]), 'de_sec': round(time.time() - t0, 1),
            'E': evaluate(W_new, b, A[Eidx], y[Eidx], broken[Eidx], mc[Eidx], oc[Eidx]),
            'R': evaluate(W_new, b, A[Ridx], y[Ridx], bR, mcR, oc[Ridx])}


def localize(model, imgs_s, lbls_s, failed, device, W_OUT, fl):
    """返回 {定位方法: fc2 每根权重的分数}(ours / 四个 Var / BB / BQ / VeRe)。"""
    from src.localization.vere import vere_significance
    passed = ~failed
    fl_f, fl_p = failed.astype(float), passed.astype(float)
    imp, probs, logits = fl.grad_involvement(model, 'fc2', imgs_s, lbls_s, device)
    labels_s = lbls_s.numpy()
    confs = np.clip(probs.max(axis=1), 0.01, 0.99)
    nf, npass = int(failed.sum()), int(passed.sum())
    sc = {}
    for name, orc in (('ours', 'QQ_cosSim'), ('arachnevar', 'QB')):
        ew = fl.oracle_weights(orc, failed, passed, confs, logits, probs, labels_s, nf, npass)
        sc[name] = fl.compute_cosine_fl_score(imp, ew)
    cov_w = fl.topk_coverage(imp, 0.1)
    sc['deepfaultvar'] = fl.sbfl_ochiai(cov_w, fl_f, fl_p, confs, False)
    sc['guidervar'] = fl.sbfl_ochiai(cov_w, fl_f, fl_p, confs, True)
    ncov, _, _ = fl.neuron_coverage(model, 'fc2', imgs_s, device, act_threshold=0.1)
    sc['deepfault'] = fl.neuron_scores_to_weights(fl.sbfl_ochiai(ncov, fl_f, fl_p, confs, False), W_OUT)
    sc['guider'] = fl.neuron_scores_to_weights(fl.sbfl_ochiai(ncov, fl_f, fl_p, confs, True), W_OUT)
    ns = vere_significance(model, 'fc2', imgs_s, lbls_s, device, interval_times=1.0,
                           n_fail=getattr(fl, 'VERE_N', 0))
    sc['vere'] = fl.neuron_scores_to_weights(ns, W_OUT)
    return {k: np.asarray(v, dtype=np.float64).ravel() for k, v in sc.items()}


def rank_order(scores, rng):
    """按分数从高到低的权重下标;同分随机打破(先随机打乱,再稳定排序)。
    扫多个预算 N 时对同一个方法只算一次,前 N 名就是它的前缀,保证 N=1 的那根一定也在 N=8 里。"""
    perm = rng.permutation(len(scores))
    return perm[np.argsort(-scores[perm], kind='stable')]


def run(args):

    import torch
    from src.registry import get_model_and_loader
    from src.localization import signals as fl
    from src.localization.common import restore_original

    locs = [s.strip() for s in args.localizers.split(',') if s.strip()]
    n_fix_list = ([int(s) for s in args.n_fix_list.split(',')] if args.n_fix_list
                  else [N_FIX[args.dataset]])
    alpha_list = [float(s) for s in args.alpha_list.split(',')]
    ModelClass, get_loaders = get_model_and_loader(args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = args.data_dir or DATASET_CONFIG[args.dataset]['data_dir']
    _, test_loader = get_loaders(data_dir, batch_size=128)
    imgs, lbls = fl.build_test_suite(test_loader)
    labels = lbls.numpy(); T = len(labels)

    # 测试集对半分:R 修复用,E 评估用(所有 mutant 相同)
    perm = np.random.default_rng(args.split_seed).permutation(T)
    Ridx, Eidx = np.sort(perm[:T // 2]), np.sort(perm[T // 2:])

    pool = json.load(open(args.mutant_pool))
    mutants = pool['mutants'][:args.n_mutants]
    pool_dir = Path(args.mutant_pool).parent
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # fc2 输入:故障只在 fc2,所以原模型和所有 mutant 一样,只算一次
    A_all = W_OUT = b = orig_correct = None
    tasks, meta = [], {}
    t_start = time.time()
    for mi, m in enumerate(mutants):
        mid = m['mutant_id']
        pth_path = pool_dir / f'mutant_{mid:04d}.pth'
        if not pth_path.exists():
            pth_path = Path(m['save_path'])
        pth = torch.load(pth_path, map_location='cpu', weights_only=False)
        faulty = np.asarray(m.get('mutated_indices') or pth['mutated_indices'], dtype=np.int64)
        model = ModelClass(); model.load_state_dict(pth['model_state_dict']); model = model.to(device).eval()
        fc2 = dict(model.named_modules())['fc2']
        W = fc2.weight.detach().cpu().numpy().astype(np.float64)
        if A_all is None:
            W_OUT = W.shape[0]
            b = fc2.bias.detach().cpu().numpy().astype(np.float64)
            cov, _, z = fl.neuron_coverage(model, 'fc2', imgs, device, act_threshold=0.1)
            del cov
            acts = []
            h = fc2.register_forward_hook(lambda mod, inp, o: acts.append(inp[0].detach().cpu()))
            with torch.no_grad():
                for i in range(0, T, 256):
                    model(imgs[i:i + 256].to(device))
            h.remove()
            A_all = torch.cat(acts).numpy().astype(np.float64)
            err = np.abs(A_all @ W.T + b - z).max()
            assert err < 1e-3, f'用 fc2 输入重算的 logits 和模型输出不一致(最大差 {err})'
            om = ModelClass(); om.load_state_dict(pth['model_state_dict'])
            W_orig = dict(restore_original(om, m, pth).named_modules())['fc2'].weight.detach().numpy().astype(np.float64)
            orig_correct = (A_all @ W_orig.T + b).argmax(1) == labels
            print(f'{args.dataset}: fc2 {W.shape[0]}x{W.shape[1]}  T={T}  R={len(Ridx)} E={len(Eidx)}  '
                  f'原模型 acc={orig_correct.mean():.4f}  logits 自检最大差 {err:.1e}  '
                  f'N={n_fix_list}  α={alpha_list}  定位方法={locs}  workers={args.workers}', flush=True)
        assert np.allclose(fc2.bias.detach().cpu().numpy(), b), '不同 mutant 的 fc2 bias 不同'
        mut_correct = (A_all @ W.T + b).argmax(1) == labels
        broken = orig_correct & ~mut_correct
        clean = orig_correct & mut_correct

        # ── R 上的 B 口径抽样(同 step2:1:1,种子 = mutant_id)──
        bR, cR = broken[Ridx], clean[Ridx]
        nb, nu = int(bR.sum()), int(cR.sum())
        if nb == 0 or nu == 0:
            print(f'  [skip] mutant {mid}: R 上 n_broken={nb} n_clean={nu}')
            continue
        rng = np.random.default_rng(mid)
        pass_idx = rng.choice(np.flatnonzero(cR), size=min(nb, nu), replace=False)
        sub = np.sort(np.concatenate([np.flatnonzero(bR), pass_idx]))
        failed = bR[sub]
        sub_t = torch.from_numpy(Ridx[sub])
        scores = localize(model, imgs[sub_t], lbls[sub_t], failed, device, W_OUT, fl)
        N = W.size
        trng = np.random.default_rng(10_000 + mid)
        n_max = max(n_fix_list)
        order = {}                                  # 每个方法的完整名次,只算一次;前 N 名 = 前缀
        for loc in locs:
            if loc in scores:
                order[loc] = rank_order(scores[loc], trng)
            elif loc == 'random':
                order[loc] = trng.choice(N, size=n_max, replace=False)
            elif loc == 'truefault':
                pad = trng.choice(np.setdiff1d(np.arange(N), faulty), size=max(0, n_max - len(faulty)),
                                  replace=False)
                order[loc] = np.concatenate([faulty, pad])       # 真故障排在最前,不够再随机补
            else:
                raise ValueError(loc)
        for li, loc in enumerate(locs):
            for ni, n_fix in enumerate(n_fix_list):
                idx = np.asarray(order[loc][:n_fix], dtype=np.int64)
                for ai, alpha in enumerate(alpha_list):
                    tasks.append({'mid': mid, 'loc': loc, 'idx': idx, 'W': W,
                                  'mut_correct': mut_correct, 'alpha': alpha, 'max_gen': args.max_gen,
                                  'seed': ((mid * 100 + li) * 10 + ni) * 10 + ai,
                                  'hit': int(np.isin(faulty, idx).sum())})
        meta[mid] = {'faulty': faulty.tolist(), 'n_broken_R': nb, 'n_broken_E': int(broken[Eidx].sum()),
                     'n_used_loc': int(len(sub))}
        del model
        if (mi + 1) % 10 == 0 or mi + 1 == len(mutants):
            print(f'  定位 {mi + 1}/{len(mutants)}  ({time.time() - t_start:.0f}s)', flush=True)

    print(f'DE 任务 {len(tasks)} 个,{args.workers} 个进程并行', flush=True)
    results, t1 = [], time.time()
    if args.workers == 1:
        init_worker(A_all, labels, orig_correct, Ridx, Eidx, b)
        for k, task in enumerate(tasks):
            results.append(repair_task(task))
            if (k + 1) % 20 == 0 or k + 1 == len(tasks):
                print(f'  DE {k + 1}/{len(tasks)}  ({time.time() - t1:.0f}s)', flush=True)
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.workers, initializer=init_worker,
                      initargs=(A_all, labels, orig_correct, Ridx, Eidx, b)) as p:
            for k, r in enumerate(p.imap_unordered(repair_task, tasks)):
                results.append(r)
                if (k + 1) % 20 == 0 or k + 1 == len(tasks):
                    print(f'  DE {k + 1}/{len(tasks)}  ({time.time() - t1:.0f}s)', flush=True)

    results.sort(key=lambda r: (r['mutant_id'], locs.index(r['localizer']), r['n_fix'], r['alpha']))
    json.dump({'dataset': args.dataset, 'n_fix_list': n_fix_list, 'alpha_list': alpha_list,
               'max_gen': args.max_gen, 'split_seed': args.split_seed, 'localizers': locs,
               'mutants': meta, 'results': results}, open(out / 'repair_results.json', 'w'))
    print(f'\n完成 → {out / "repair_results.json"}  ({len(meta)} 个 mutant)')

    # 汇总(E 上,汇总口径:所有 mutant 的条数先加起来再算比例)
    for alpha in alpha_list:
        for n_fix in n_fix_list:
            print(f'\n── α={alpha:g}  N={n_fix} ' + '─' * 52)
            print(f'{"定位方法":<14}{"命中故障":>8}{"修复率":>9}{"误伤率":>9}{"净修好":>9}'
                  f'{"每mutant修好":>13}{"代数":>6}{"秒":>7}')
            for loc in locs:
                rs = [r for r in results if r['localizer'] == loc and r['n_fix'] == n_fix
                      and r['alpha'] == alpha]
                if not rs:
                    continue
                nb = sum(r['E']['n_broken'] for r in rs)
                nf = sum(r['E']['n_fixed'] for r in rs)
                nmc = sum(r['E']['n_mut_correct'] for r in rs)
                nnb = sum(r['E']['n_new_broken'] for r in rs)
                hit = np.mean([r['fault_hit'] > 0 for r in rs])
                print(f'{loc:<14}{hit:>8.0%}{nf/max(nb,1):>9.1%}{nnb/max(nmc,1):>9.2%}'
                      f'{(nf-nnb)/len(rs):>9.1f}{nf/len(rs):>13.1f}'
                      f'{np.mean([r["n_gen"] for r in rs]):>6.0f}{np.mean([r["de_sec"] for r in rs]):>7.1f}')
    print('\n修复率 = 所有 mutant 的 broken 汇总后被改回判对的比例;误伤率 = mutant 判对的测试被改错的比例;'
          '\n净修好 = 每个 mutant 平均(修好条数 − 新弄坏条数),为负说明整体准确率反而降了。')
