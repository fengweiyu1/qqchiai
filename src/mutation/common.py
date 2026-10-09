"""Shared mutation and benchmark diagnostics."""

import copy
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn


def build_test_suite(test_loader, n=0):
    """n <= 0 表示用整个测试集(默认)。

    取子集会同时损伤两件事:(a) SBFL 的 pass/fail 谱由更少的失败测试撑起,分数方差变大;
    (b) test_loader 的 shuffle=False,子集取的是**前缀**而非随机样本,测试集本身若按类别
    有序就只覆盖到部分类别。此外行为改变阈值是比例判据 —— 1000 条时 0.1% 恰好等于 1 条,
    与"至少 1 个新增失败"重合而失效,须在全集上(10000 条 → 10 条)才恢复 Arachne 3.2.1 的原意。
    """
    imgs, lbls = [], []
    for x, y in test_loader:
        imgs.append(x); lbls.append(y)
        if 0 < n <= sum(t.size(0) for t in imgs):
            break
    imgs, lbls = torch.cat(imgs), torch.cat(lbls)
    return (imgs, lbls) if n <= 0 else (imgs[:n], lbls[:n])


def predict(model, images, device, bs=256):
    """返回每条测试的预测类别 [T]。"""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), bs):
            out.append(model(images[i:i+bs].to(device)).argmax(1).cpu())
    return torch.cat(out).numpy()


def gradient_candidates(model, images, labels, device, layer='fc2', n_batch=10, bs=128):
    """|grad| > 均值 的权重位置(展平下标)。与现有 step1 一致(Arachne 3.2.1 也如此选)。"""
    model.to(device).train()
    crit = nn.CrossEntropyLoss()
    W = dict(model.named_modules())[layer].weight
    acc = torch.zeros(W.numel())
    for i in range(min(n_batch, (len(images) + bs - 1) // bs)):
        x, y = images[i*bs:(i+1)*bs].to(device), labels[i*bs:(i+1)*bs].to(device)
        if len(x) == 0:
            break
        model.zero_grad()
        crit(model(x), y).backward()
        acc += W.grad.abs().detach().cpu().view(-1)
    model.eval()
    cand = torch.where(acc > acc.mean())[0].numpy()
    print(f"  梯度候选权重: {len(cand)}/{W.numel()} ({100*len(cand)/W.numel():.1f}%)")
    return cand


def apply_weights(base_model, ModelClass, idx_list, val_list, layer='fc2'):
    """复制一份模型,把指定下标的权重改成给定值。"""
    m = copy.deepcopy(base_model)
    w = dict(m.named_modules())[layer].weight
    flat = w.data.view(-1)
    for i, v in zip(idx_list, val_list):
        flat[i] = v
    w.data = flat.view(w.shape)
    return m


@contextmanager
def temp_weights(model, idx_list, val_list, layer='fc2'):
    """临时把指定下标的权重改成给定值,退出时复原。

    生成一个 mutant 平均要试多次(接受率约 30~40%),gtsrb 高 level 需要上千个故障,
    每次尝试都 deepcopy 整个模型代价过高 —— 原地改再复原,只在最终接受时才复制。
    """
    flat = dict(model.named_modules())[layer].weight.data.view(-1)
    old = [flat[i].item() for i in idx_list]
    try:
        for i, v in zip(idx_list, val_list):
            flat[i] = v
        yield model
    finally:
        for i, o in zip(idx_list, old):
            flat[i] = o


def forward_impact(model, images, device, layer='fc2', bs=256):
    """|w_ij| x 输入神经元 j 的平均激活。仅用于诊断注入协议的选择偏差。"""
    mod = dict(model.named_modules())[layer]
    total = torch.zeros(mod.in_features)

    def _hook(m, inp):
        # forward_pre_hook 只要返回非 None 就会替换该层的输入,故此处必须无返回值。
        total.add_(inp[0].detach().sum(0).cpu())

    h = mod.register_forward_pre_hook(_hook)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(images), bs):
            model(images[i:i + bs].to(device))
    h.remove()
    act = total / max(1, len(images))
    return (mod.weight.data.abs().cpu() * act.unsqueeze(0)).view(-1).numpy()


def report_selection_bias(accepted, cand, fi, W_flat):
    """被接受位置 vs 全体候选位置的 fi / |w| 中位数比值。比值 ->1 表示协议不偏袒任何位置。"""
    if len(accepted) == 0:
        return {}
    a = np.asarray(accepted, dtype=int)
    r_fi = float(np.median(fi[a]) / np.median(fi[cand]))
    r_w = float(np.median(np.abs(W_flat[a])) / np.median(np.abs(W_flat[cand])))
    cov = len(a) / len(cand)
    print(f"  [选择偏差] 被接受位置 fi  / 全体候选 = {r_fi:.2f}   (1.00 = 无偏差; K=1 时实测 1.33)")
    print(f"  [选择偏差] 被接受位置 |w| / 全体候选 = {r_w:.2f}")
    print(f"  [选择偏差] 候选池覆盖率 = {len(a)}/{len(cand)} = {cov:.0%}")
    if r_fi > 1.15:
        print(f"  [!] fi 比值偏高 —— 调大 --retry_per_position 或放宽 --threshold")
    return {'fi_ratio': r_fi, 'w_ratio': r_w, 'coverage': cov}


def report_fairness(all_new_vals, W_flat, tag=""):
    """打印公平性指标:放大倍数、变号率(对照 Arachne 官方 1.1~2.8 / 69~74%)。"""
    if not all_new_vals:
        return {}
    new = np.abs(np.array([v for v, _ in all_new_vals]))
    old = np.array([o for _, o in all_new_vals])
    ratio = float(np.median(new) / np.median(np.abs(W_flat)))
    flip = float(np.mean(np.sign(old) != np.sign([v for v, _ in all_new_vals])))
    print(f"\n  [公平性{tag}] 放大倍数 = {ratio:.2f}  (Arachne 官方 1.1~2.8, 乘性注入 16)")
    print(f"  [公平性{tag}] 变号率   = {flip:.0%}   (Arachne 官方 69~74%, 乘性注入 0%)")
    return {'magnitude_ratio': ratio, 'sign_flip_rate': flip}
