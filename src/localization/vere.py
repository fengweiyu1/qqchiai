"""VeRe repair significance specialized to a linear fc2 suffix. See THIRD_PARTY.md for scope and attribution; no general CROWN solver is included."""

import numpy as np
import torch


def _forward_cache(model, layer_name, test_images, device, batch=256):
    """一次前向拿到 target layer 的输入激活和 logits。

    Returns h [T, Np] (post-ReLU, fc2 的输入), logits [T, C]。
    """
    model.eval()
    layer = dict(model.named_modules())[layer_name]
    acts = []

    def hook(m, inp, out):
        acts.append(inp[0].detach())

    h = layer.register_forward_hook(hook)
    all_a, all_z = [], []
    with torch.no_grad():
        for i in range(0, len(test_images), batch):
            acts.clear()
            out = model(test_images[i:i + batch].to(device))
            all_z.append(out.cpu())
            all_a.append(acts[0].cpu())
    h.remove()
    return torch.cat(all_a).numpy(), torch.cat(all_z).numpy()


def vere_significance(model, layer_name, test_images, test_labels, device,
                      interval_times=2.0, interval_num=10, n_fail=0,
                      act_bound='all', return_debug=False):
    """VeRe repair significance,神经元级。

    Args:
        model:        待定位的模型(mutant)
        layer_name:   故障所在层名,如 'fc2'。定位的神经元 = 该层的输入。
        test_images / test_labels: 测试套件(与其它方法同一套)
        interval_times: 区间宽度倍率,官方按模型取 1 / 1.5 / 2。
                        amin=0 时 it=1 → [−amax, 0](只允许抑制,backdoor 设定);
                        it=2 → [−amax, amax](可增可减)。
        interval_num: 子区间数。本实现下为 no-op(见模块 docstring),仅接口对齐。
        n_fail:       只用前 n_fail 条失败测试(VeRe 原设 N=1000);<=0 表示全用。
        act_bound:    激活范围的统计口径。'all' = 全体测试(对齐官方
                      analyze_neuron_action:incor + normal 一起统计)。
        return_debug: 额外返回中间量,给交叉验证脚本用。

    Returns:
        score [Np]  越大越可疑。若没有失败测试则返回全 0。
    """
    layer = dict(model.named_modules())[layer_name]
    W = layer.weight.detach().cpu().numpy()          # [C, Np]
    n_out, n_p = W.shape

    h, logits = _forward_cache(model, layer_name, test_images, device)   # [T, Np], [T, C]
    labels = test_labels.numpy() if torch.is_tensor(test_labels) else np.asarray(test_labels)
    preds = logits.argmax(axis=1)
    fail_idx = np.flatnonzero(preds != labels)
    if len(fail_idx) == 0:
        return (np.zeros(n_p), None) if return_debug else np.zeros(n_p)
    if n_fail > 0:
        fail_idx = fail_idx[:n_fail]

    # --- 区间: 官方 analyze_neuron_action + eps_record 的构造 ---
    if act_bound == 'all':
        amin, amax = h.min(axis=0), h.max(axis=0)
    else:                                    # 'fail': 只用失败测试统计(消融用)
        amin, amax = h[fail_idx].min(axis=0), h[fail_idx].max(axis=0)
    rng = amax - amin
    lo = amin - rng
    hi = lo + rng * interval_times                        # [Np]

    # --- coef[t, n] = W[y_t, n] − W[ŷ_t, n]  (C 矩阵 = +1 真值, −1 预测) ---
    hf = h[fail_idx]                                      # [F, Np]
    coef = W[labels[fail_idx]] - W[preds[fail_idx]]       # [F, Np]

    # bound(v) − ori = coef · (v − h);  v 的极值必在区间端点取到。
    d_lo = coef * (lo[None, :] - hf)
    d_hi = coef * (hi[None, :] - hf)
    effect = np.maximum(np.maximum(d_lo, d_hi), 0.0)      # [F, Np]  max(0, ·)

    # --- 逐失败测试对全体神经元 min-max 归一化(官方 all_neuron_effect[:, i]) ---
    e_min = effect.min(axis=1, keepdims=True)
    e_max = effect.max(axis=1, keepdims=True)
    span = e_max - e_min
    effect = np.where(span > 0, (effect - e_min) / np.where(span > 0, span, 1.0), 0.0)

    score = effect.sum(axis=0)                            # [Np]
    if return_debug:
        return score, {'h': h, 'logits': logits, 'fail_idx': fail_idx,
                       'lo': lo, 'hi': hi, 'coef': coef, 'effect': effect}
    return score
