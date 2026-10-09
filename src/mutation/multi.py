"""Compose accepted single faults into cumulative multi-fault models."""

import json
from pathlib import Path
import numpy as np
import torch
from src.registry import get_model_and_loader
from .common import build_test_suite, predict, apply_weights, temp_weights, report_fairness


def compose_pool(args):
    """分级递归构建多故障基准:第 j 档继承第 j-1 档的一个结果,再补足故障数差额。

    对标 SpectAcle 5.2.2 的 SetMuts^k(它每次加 1 个故障,逐级构建到 5)。我们的档位是
    密度百分比,最高档故障数可达数百,逐个加需要构建同样多的中间层级,故把步长由
    "1 个"放宽为"一批",其余保持一致:每一步都要检验新加入的故障确实在累积致错。

    接受条件(均来自 SpectAcle 5.2.2):
      (1) acc 低于父档、且低于新增各单故障 —— 保证故障累积而非互相抵消。若某故障把
          别的故障的效果抵消掉,它仍被算作 ground truth 却不对失败负责,任何定位方法
          都找不到它,只会无差别拉低所有方法的分数。
      (2) acc 高于随机猜的水平 —— 模型被彻底打死后通过的测试所剩无几,SBFL 失去对照。
    """
    ModelClass, get_loaders = get_model_and_loader(args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.RandomState(args.seed)

    src = json.load(open(Path(args.from_single) / 'mutant_pool.json'))
    singles = src['mutants']
    if not singles:
        print("单故障池为空"); return

    base = ModelClass()
    ck = torch.load(args.model_path, map_location='cpu', weights_only=False)
    base.load_state_dict(ck['model_state_dict'] if 'model_state_dict' in ck else ck)
    base = base.to(device).eval()
    _, test_loader = get_loaders(args.data_dir, batch_size=128)
    imgs, lbls = build_test_suite(test_loader, args.test_suite_size)
    orig_acc = float((predict(base, imgs, device) == lbls.numpy()).mean() * 100)
    W = dict(base.named_modules())['fc2'].weight
    W_flat = W.data.view(-1).cpu().numpy()
    total = int(W.numel())
    n_classes = dict(base.named_modules())['fc2'].out_features
    random_level = 100.0 / n_classes        # 随机猜的准确率,acc 跌到这以下模型已被打死

    pct_list = sorted(float(x) for x in args.pct_list.split(','))
    n_list = [max(1, int(round(total * p / 100))) for p in pct_list]
    print("=" * 74)
    print(f"Step1 noise / MULTI — {args.dataset.upper()}   单故障池 {len(singles)} 个")
    print(f"原始准确率 {orig_acc:.2f}%   levels={pct_list}  → 故障数 {n_list}")
    print(f"每档 {args.max_per_group} 个;分级递归:第 j 档继承第 j-1 档并补足差额")
    print(f"接受条件: acc < 父档 acc, acc < min(新增各单故障 acc), acc > {random_level:.2f}% (随机水平)")
    print("=" * 74)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    mutants, pairs, gid = [], [], 0

    def evaluate(idxs, vals):
        with temp_weights(base, idxs, vals):
            return float((predict(base, imgs, device) == lbls.numpy()).mean() * 100)

    def save(sel, acc, pct):
        """sel = 单故障池下标列表。落盘 + 登记。"""
        nonlocal gid
        idxs = [singles[i]['mutated_indices'][0] for i in sel]
        vals = [singles[i]['new_values'][0] for i in sel]
        olds = [singles[i]['original_values'][0] for i in sel]
        m = apply_weights(base, ModelClass, idxs, vals).to(device)   # 只在接受时才复制
        torch.save({'model_state_dict': m.state_dict(),
                    'mutated_indices': idxs, 'new_values': vals,
                    'original_values': olds, 'pct': pct,
                    'num_mutated_weights': len(sel),
                    'accuracy': acc, 'acc_drop': orig_acc - acc},
                   out / f"mutant_{gid:04d}.pth")
        mutants.append({'mutant_id': gid, 'level': pct, 'pct': pct,
                        'num_mutated_weights': len(sel),
                        'mutated_indices': idxs, 'new_values': vals, 'original_values': olds,
                        'accuracy': acc, 'acc_drop': orig_acc - acc,
                        'save_path': str(out / f"mutant_{gid:04d}.pth")})
        pairs.extend(zip(vals, olds))
        gid += 1
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # POOL[j] = 第 j 档已接受的 mutant,元素为 (单故障下标集合, acc)
    POOL = []
    for j, (pct, n_faulty) in enumerate(zip(pct_list, n_list)):
        if n_faulty > len(singles):
            print(f"[{pct}%] 需要 {n_faulty} 个故障,单故障池只有 {len(singles)} 个 —— 跳过")
            POOL.append([]); continue
        if j > 0 and not POOL[j - 1]:
            print(f"[{pct}%] 上一档为空,无法递归 —— 跳过")
            POOL.append([]); continue

        cur, tries = [], 0
        n_cancel = n_dead = n_dup = 0
        seen = set()
        limit = args.max_per_group * args.max_try_factor
        while len(cur) < args.max_per_group and tries < limit:
            tries += 1
            if j == 0:
                # 第一档没有父档,直接抽;抵消判据退化为"比每个成分都差"
                sel = [int(i) for i in rng.choice(len(singles), size=n_faulty, replace=False)]
                base_acc = min(singles[i]['accuracy'] for i in sel)
            else:
                parent_sel, parent_acc = POOL[j - 1][int(rng.randint(len(POOL[j - 1])))]
                pool_rest = [i for i in range(len(singles)) if i not in set(parent_sel)]
                need = n_faulty - len(parent_sel)
                if need <= 0:
                    break                    # 该档故障数与上档相同(取整所致),没有增量可加
                if need > len(pool_rest):
                    print(f"[{pct}%] 单故障池剩余 {len(pool_rest)} 个,不足以补足 {need} 个 —— 提前停止")
                    break
                add = [int(i) for i in rng.choice(pool_rest, size=need, replace=False)]
                sel = list(parent_sel) + add
                # 既要比父档更差,也要比新增的每个单故障更差(SpectAcle 5.2.2)
                base_acc = min(parent_acc, min(singles[i]['accuracy'] for i in add))

            key = frozenset(sel)
            if key in seen:
                n_dup += 1; continue         # 这组故障已经生成过,换一组
            seen.add(key)
            idxs = [singles[i]['mutated_indices'][0] for i in sel]
            vals = [singles[i]['new_values'][0] for i in sel]
            acc = evaluate(idxs, vals)
            # 抵消判据只在存在两个及以上故障时才有意义:单个故障的组合就是它自己,
            # acc 必然等于成分 acc,该条件恒不成立(fm 的 0.1% 档正好是 1 个故障)。
            if len(sel) > 1 and acc >= base_acc:
                n_cancel += 1; continue      # 新增的故障没让模型更差 —— 抵消了
            if acc <= random_level:
                n_dead += 1; continue        # 模型被打死,SBFL 失去对照
            save(sel, acc, pct)
            cur.append((sel, acc))

        POOL.append(cur)
        d = [x['acc_drop'] for x in mutants if x['level'] == pct]
        print(f"[{pct}%] {n_faulty} 个故障 → 保留 {len(cur)} 个 (尝试 {tries}; "
              f"抵消拒 {n_cancel}, 打死拒 {n_dead}, 重复拒 {n_dup})"
              + (f"  acc_drop 中位数 {np.median(d):.2f}" if d else ""))

    fair = report_fairness(pairs, W_flat)
    json.dump({'original_accuracy': orig_acc, 'total_mutants': len(mutants),
               'config': {'mode': 'multi', 'injection': 'gaussian_noise_fixed_scale',
                          'from_single': str(args.from_single),
                          'total_fc2_weights': total, 'pct_list': pct_list,
                          'n_faulty_list': n_list, 'random_level': random_level,
                          'build': 'staged recursive: level j inherits level j-1 and adds the difference',
                          'accept': 'acc < parent acc and acc < min(added component acc) and acc > random_level',
                          'fairness': fair},
               'mutants': mutants}, open(out / 'mutant_pool.json', 'w'), indent=2)
    print(f"\n  共 {len(mutants)} 个多故障 mutant。已保存: {out/'mutant_pool.json'}")


def gen_multi(args):
    """Convert explicit fault counts to percentages and compose a single-fault pool.

    The injection description is inherited from the source pool.
    """
    if args.n_faulty_list:
        ModelClass, _ = get_model_and_loader(args.dataset)
        total = int(dict(ModelClass().named_modules())['fc2'].weight.numel())
        ns = [int(x) for x in args.n_faulty_list.split(',')]
        args.pct_list = ','.join(repr(n * 100.0 / total) for n in ns)
        print(f'故障根数 {ns} → 占 fc2({total} 根)的百分比 {args.pct_list}')
    src = json.load(open(Path(args.from_single) / 'mutant_pool.json'))
    compose_pool(args)
    p = Path(args.output_dir) / 'mutant_pool.json'
    if p.exists():
        d = json.load(open(p))
        d['config']['injection'] = f"{src['config'].get('injection')} (composed from single pool)"
        if args.n_faulty_list:
            d['config']['n_faulty_requested'] = args.n_faulty_list
        json.dump(d, open(p, 'w'), indent=2)
