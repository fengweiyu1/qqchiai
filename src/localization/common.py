"""Ranking, prediction, and restoration of an unmutated model."""

import numpy as np
import torch


def rank_of(scores):
    order = np.argsort(scores)[::-1]
    r = np.empty(len(scores), dtype=np.int64)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def predict(model, imgs, device, bs=256):
    """返回 argmax 预测 [T]。"""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            out.append(model(imgs[i:i + bs].to(device)).argmax(1).cpu())
    return torch.cat(out).numpy()


def restore_original(model, mutant_info, pth, layer='fc2'):
    """把 original_values 写回 mutated_indices,原地还原成注入前的模型。"""
    idx = mutant_info.get('mutated_indices') or pth['mutated_indices']
    orig = mutant_info.get('original_values')
    if orig is None:
        orig = pth['original_values']
    W = dict(model.named_modules())[layer].weight.data
    flat = W.view(-1)
    for i, v in zip(idx, orig):
        flat[int(i)] = float(v)
    return model
