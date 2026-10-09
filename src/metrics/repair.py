"""Repair and breakage counts for an evaluation split."""

import numpy as np


def evaluate(W_new, b, A, y, broken, mut_correct, orig_correct):
    """比例 + 条数都记下来:汇总口径(把所有 mutant 的 broken 合起来算)要用条数。"""
    pred = (A @ W_new.T + b).argmax(1)
    ok = pred == y
    return {'repair': float(ok[broken].mean()) if broken.any() else None,
            'break': float((~ok[mut_correct]).mean()) if mut_correct.any() else None,
            'acc_patched': float(ok.mean()), 'acc_mutant': float(mut_correct.mean()),
            'acc_orig': float(orig_correct.mean()), 'n_broken': int(broken.sum()),
            'n_fixed': int(ok[broken].sum()), 'n_mut_correct': int(mut_correct.sum()),
            'n_new_broken': int((~ok[mut_correct]).sum())}
