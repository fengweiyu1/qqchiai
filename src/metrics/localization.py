"""Localization metrics without plotting dependencies."""

import numpy as np


def compute_top_n_accuracy(ranked_weights, true_faulty_weights, n_values,
                           total_weights=None):
    """
    计算 Precision@K：检查 top-K 个权重中，有多少比例是真正的故障权重

    Args:
        ranked_weights: 全部权重按可疑度降序排列的索引列表
        true_faulty_weights: 真实的故障权重索引列表
        n_values: top-K 的绝对数量列表，如 [10, 50, 100, 200, 500]
        total_weights: 暂不使用，保留兼容性

    Returns:
        precision: 字典，{k: precision_value}
                   e.g. {100: 0.5} 表示 top-100 中 50% 是故障权重
    """
    true_faulty_set = set(true_faulty_weights)
    precision = {}

    for k in n_values:
        top_k = min(k, len(ranked_weights))
        top_set = set(ranked_weights[:top_k])
        found = len(top_set & true_faulty_set)
        precision[k] = found / top_k if top_k > 0 else 0.0

    return precision


def compute_recall_at_k(ranked_weights, true_faulty_weights, n_values):
    """
    计算 Recall@K：top-K 个权重里，找回了真实故障权重的百分之几。

    Recall@K = |top_K ∩ faulty| / |faulty|     （分母是故障权重总数）

    与 Precision@K 的区别只在分母：Precision 除以 K，Recall 除以故障总数。
    Returns: {k: recall_value}
    """
    true_faulty_set = set(true_faulty_weights)
    n_faulty = len(true_faulty_set)
    recall = {}
    for k in n_values:
        top_k = min(k, len(ranked_weights))
        found = len(set(ranked_weights[:top_k]) & true_faulty_set)
        recall[k] = found / n_faulty if n_faulty > 0 else 0.0
    return recall


def compute_recall_at_pct(ranked_weights, true_faulty_weights, pct_values, total_weights):
    """
    计算 Recall@K%：K 由百分比动态确定，K = round(pct * total_weights / 100)。

    跨数据集可比：随机基准线 ≈ pct%，与 total_weights 无关。

    Args:
        ranked_weights:      全部权重按可疑度降序排列的索引列表
        true_faulty_weights: 真实故障权重索引列表
        pct_values:          百分比列表，如 [0.1, 0.5, 1.0, 5.0, 10.0]
        total_weights:       fc2 权重总数（用于把百分比转成绝对 K）

    Returns:
        {pct: recall_value}  key 是百分比数值
    """
    true_faulty_set = set(true_faulty_weights)
    n_faulty = len(true_faulty_set)
    recall = {}
    for pct in pct_values:
        k = max(1, round(pct * total_weights / 100))
        top_k = min(k, len(ranked_weights))
        found = len(set(ranked_weights[:top_k]) & true_faulty_set)
        recall[pct] = found / n_faulty if n_faulty > 0 else 0.0
    return recall


def compute_recall_at_k_relative(ranked_weights, true_faulty_weights, multipliers):
    """
    Recall@(m×|faulty|): K = m × n_faulty，使得不同 level 的 K 具有相同难度比例。

    例如 multipliers=[1,5,10,50,100]：
      level 0.1% (n=1)  → K = 1,  5,  10,  50,  100
      level 0.3% (n=3)  → K = 3,  15, 30,  150, 300
      level 10%  (n=100)→ K = 100,500,1000,5000,10000 (capped at total)

    Returns: {multiplier: recall_value}
    """
    true_faulty_set = set(true_faulty_weights)
    n_faulty = len(true_faulty_set)
    recall = {}
    for m in multipliers:
        k = max(1, m * n_faulty)
        top_k = min(k, len(ranked_weights))
        found = len(set(ranked_weights[:top_k]) & true_faulty_set)
        recall[m] = found / n_faulty if n_faulty > 0 else 0.0
    return recall


def compute_average_top_n(results_list, n_values):
    """
    计算平均Top-N准确率

    Args:
        results_list: Top-N准确率结果列表
        n_values: N值列表

    Returns:
        avg_top_n: 字典，{n: average_accuracy}
    """
    avg_top_n = {}

    for n in n_values:
        accuracies = [result[n] for result in results_list]
        avg_top_n[n] = np.mean(accuracies)

    return avg_top_n
