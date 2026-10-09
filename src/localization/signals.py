"""Last-layer involvement, result weights, and baseline scores. Gradient involvement is abs(gradient), normalized per test."""

import numpy as np
import torch
import torch.nn.functional as F

FORMULAS = ['cos_ochiai']
ORACLES = ['QB', 'QQ', 'QQ_sq', 'QQ_1p', 'QQ_margin', 'QQ_deficit', 'QQ_loss', 'QQ_logit_relu', 'QQ_logit_sub', 'QQ_logit_exp', 'QQ_logit_margin', 'QQ_cosSim', 'QQ_max', 'QQ_max_pow', 'QQ_max_log', 'QQ_max_exp', 'QQ_max_softmax_loss', 'QQ_cos_diff']
EXEC_PREFIX = {'gradient': 'G', 'forward_impact': 'FI'}
FI_NEURON2OUT = 'pc'
VAR_EXECS = ('FI', 'G')
VAR_RATIOS = (0.1, 0.2, 0.3)
VERE_INTERVAL_TIMES = (1.0, 1.5, 2.0)
VERE_N = 0


def build_methods(prefix):
    return [f'{prefix}_{o}_{f}' for o in ORACLES for f in FORMULAS]


def build_test_suite(test_loader, num_samples=0):
    """num_samples <= 0 表示用整个测试集(默认)。

    子集会同时损伤两件事:(a) SBFL 的 pass/fail 谱由更少的失败测试撑起,分数方差变大;
    (b) test_loader 的 shuffle=False,子集取的是**前缀**而非随机样本,测试集本身若按类别
    有序就只覆盖到部分类别。与 step1 注入时的测试集保持一致。
    """
    all_images, all_labels = [], []
    for images, labels in test_loader:
        all_images.append(images); all_labels.append(labels)
        if 0 < num_samples <= sum(x.size(0) for x in all_images):
            break
    imgs, lbls = torch.cat(all_images), torch.cat(all_labels)
    return (imgs, lbls) if num_samples <= 0 else (imgs[:num_samples], lbls[:num_samples])


def grad_involvement(model, layer_name, test_images, test_labels, device):
    """
    Closed-form per-test gradient involvement for the last linear layer.
    Returns impacts [T, out*in], per-test sum-normalized (每条测试展平后和=1)。

      raw_grad[t, j, i] = (p[t,j] - y[t,j]) * a[t, i]
      默认 impact = |grad|,flatten,per-test sum-norm。
      (不对 grad 做逐行 L2 归一化,否则 diff 会被约掉;
       也不再乘 |a_norm| —— |grad| 链式法则里已含一个 a_i,再乘等于 a^2,无第一性解释)
    """
    model.eval()
    layer = dict(model.named_modules())[layer_name]
    W_out, W_in = layer.weight.shape

    acts = []
    def hook(m, inp, out): acts.append(inp[0].detach())
    h = layer.register_forward_hook(hook)

    all_p, all_a, all_z = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_images), 256):
            acts.clear()
            out = model(test_images[i:i+256].to(device))   # raw logits
            all_z.append(out.cpu()) # logits
            all_p.append(F.softmax(out, dim=1).cpu())
            all_a.append(acts[0].cpu())
    h.remove()
    p = torch.cat(all_p, dim=0)          # [T, out]
    a = torch.cat(all_a, dim=0)          # [T, in]
    logits_np = torch.cat(all_z, dim=0).numpy()   # [T, out] raw logits
    labels = test_labels.long()
    y = F.one_hot(labels, num_classes=W_out).float()   # [T, out]
    diff = p - y                          # [T, out]   = dL/dz

    T = len(test_images)
    impacts = np.zeros((T, W_out * W_in), dtype=np.float32)
    # process in chunks to bound memory (grad tensor is T*out*in)
    a_np = a.numpy(); diff_np = diff.numpy()
    for s in range(0, T, 256):
        e = min(s + 256, T)
        # grad[b, j, i] = diff[b, j] * a[b, i]
        g = diff_np[s:e][:, :, None] * a_np[s:e][:, None, :]   # g就是梯度[b, out, in]
        # NOTE: 不做逐行 L2 归一化 —— g 的第 j 行是 a 的标量倍(标量=diff[j]),
        #       逐行归一化会把 diff[j] 约掉,梯度幅度信息全丢, impact 退化成 â_i^2。
        impact = np.abs(g)                                     # 版本A(默认): |grad|
        # 版本B(消融, a^2): 再乘 L2 归一化激活 → impact = |grad| * |â|
        # an = a_np[s:e] / (np.linalg.norm(a_np[s:e], axis=1, keepdims=True) + 1e-10)
        # impact = np.abs(g) * np.abs(an)[:, None, :]
        flat = impact.reshape(e - s, -1)
        flat = flat / (flat.sum(axis=1, keepdims=True) + 1e-10)
        impacts[s:e] = flat
    return impacts, p.numpy(), logits_np  # impacts, softmax probs, raw logits


def forward_impact_involvement(model, layer_name, test_images, test_labels, device):
    """
    Forward impact (Arachne 定义,无需标签): fi = weight2Neuron × neuron2Out。

      weight2Neuron[t,j,i] = |a_i·w_ji| / Σ_l |a_l·w_jl|      沿 in 维行归一化
      neuron2Out[t,j]      = |∂O/∂z_j|                        目标神经元对输出的梯度

    第一段是占比,逐行归一化是定义的一部分(w_j· 各元素不同,不是标量倍,信息不会
    被约掉)。第二段由 FI_NEURON2OUT 选:

      'pc'   O = 预测类概率 p_c  →  ∂p_c/∂z_j = p_c·(1[j=c] − p_j)   【默认】
      'one'  O = z_j 自己        →  ∂z_j/∂z_j ≡ 1                    Arachne 原式

    取 'one' 时 fi 退化成 weight2Neuron 本身,后果是 fc2 上的 FI 完全不带类别信息:
    每个输出行的总质量恒为 1(行和恒 = W_out),且余弦按列归一化会把与 t 无关的
    |w_ji| 约掉,分数化简成 cos(|a_i(t)|/D_j(t), y) —— 退化为神经元级信号,与 BB
    的 np.tile 同构。'pc' 补上的这一项随测试变、随类别变,不会被余弦约掉,是唯一
    能把类别信息注入 fc2 FI 的通道。fc1 版(step2_mutant_in_fc1.fi_chunk)同口径。

    O 取模型输出而非 loss:Arachne 把 forward impact 与 gradient loss 列为两个相互
    竞争的目标,用 loss 会与后者重复;取预测类概率也不需要标签。

    签名与 grad_involvement 一致以便无缝替换(test_labels 不使用,仅为对齐接口)。
    Returns impacts [T, out*in] (per-test sum 归一化), softmax probs, raw logits。
    注:'one' 下逐测试归一化是除以常数 W_out(空转),'pc' 下行和 = 2·p_c(1−p_c)
    随测试变,这一步才真正生效。
    """
    model.eval()
    layer = dict(model.named_modules())[layer_name]
    W = layer.weight.detach().cpu().numpy()          # [out, in]
    abs_W = np.abs(W)
    W_out, W_in = W.shape

    acts = []
    def hook(m, inp, out): acts.append(inp[0].detach())
    h = layer.register_forward_hook(hook)

    all_p, all_a, all_z = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_images), 256):
            acts.clear()
            out = model(test_images[i:i+256].to(device))
            all_z.append(out.cpu())
            all_p.append(F.softmax(out, dim=1).cpu())
            all_a.append(acts[0].cpu())
    h.remove()
    p = torch.cat(all_p, dim=0).numpy()              # [T, out]
    a = np.abs(torch.cat(all_a, dim=0).numpy())      # [T, in]
    logits_np = torch.cat(all_z, dim=0).numpy()

    T = len(test_images)
    impacts = np.zeros((T, W_out * W_in), dtype=np.float32)
    for s in range(0, T, 256):
        e = min(s + 256, T)
        # |a_i * w_ji|  → [b, out, in]
        contrib = a[s:e][:, None, :] * abs_W[None, :, :]
        # weight2Neuron: 每个 output 行沿 i 归一化
        row_sum = contrib.sum(axis=2, keepdims=True) + 1e-10
        w2n = contrib / row_sum
        if FI_NEURON2OUT == 'pc':
            # neuron2Out: O = 预测类概率 p_c,目标神经元 = softmax 前的 z_j
            #   ∂p_c/∂z_j = p_c·(1[j=c] − p_j)
            pb = p[s:e]
            idx = np.arange(e - s)
            c = pb.argmax(axis=1)
            oh = np.zeros_like(pb); oh[idx, c] = 1.0
            n2o = np.abs(pb[idx, c][:, None] * (oh - pb))        # [b, out]
            fi = w2n * n2o[:, :, None]
        else:                                        # 'one': neuron2Out ≡ 1,Arachne 原式
            fi = w2n
        flat = fi.reshape(e - s, -1)
        flat = flat / (flat.sum(axis=1, keepdims=True) + 1e-10)  # per-test sum-norm
        impacts[s:e] = flat
    return impacts, p, logits_np


def neuron_coverage(model, layer_name, test_images, device, act_threshold=0.1):
    """
    BB/BQ 执行侧: fc2 输入侧隐藏神经元的二元激活覆盖(忠于 GUIDER/DeepFault)。
    覆盖 = 原始 ReLU 激活 > 阈值(不归一化,照 GUIDER 代码 x>threshold)。
    这些隐藏神经元 = torch.relu(fc1(x)) 的输出 = fc2 的输入(W_in 个,ReLU≥0)。
    Returns:
      cov   [T, W_in] bool  —— 每个 test 是否覆盖每个隐藏神经元 (a[t,i] > 阈值)
      probs [T, W_out]      —— softmax(算 pass/fail 和 confidence 用)
      logits[T, W_out]
    """
    model.eval()
    layer = dict(model.named_modules())[layer_name]
    acts = []
    def hook(m, inp, out): acts.append(inp[0].detach())
    h = layer.register_forward_hook(hook)

    all_p, all_a, all_z = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_images), 256):
            acts.clear()
            out = model(test_images[i:i+256].to(device))
            all_z.append(out.cpu())
            all_p.append(F.softmax(out, dim=1).cpu())
            all_a.append(acts[0].cpu())              # fc2 输入 = ReLU 隐藏激活 [b, W_in]
    h.remove()
    a = torch.cat(all_a, dim=0).numpy()              # [T, W_in]  原始 ReLU (>=0)
    p = torch.cat(all_p, dim=0).numpy()
    z = torch.cat(all_z, dim=0).numpy()
    cov = a > act_threshold                          # 原始激活 > 0.1,不归一化
    return cov, p, z


def sbfl_ochiai(cov, failed, passed, confs, weighted):
    """
    传统 Ochiai(GUIDER 的 weighted-size 形式,Table 2)。粒度由 cov 的第二维决定:
      - BB / BQ       : cov[T, W_in]           → 每个隐藏神经元一个分(再列展开成权重)
      - DFVar / GDVar : cov[T, total_weights]  → 直接每个权重一个分,无需展开
      每个 test 权重 w:  DeepFault 系(weighted=False) → 1(纯计数)
                         GUIDER 系(weighted=True)     → 1 + minmax(confidence)(Eq.2/5)
      对单元 u:
        ef = Σ_t (cov[t,u] 且 failed[t]) * w[t]      # 覆盖 u 的失败测试(加权)
        ep = Σ_t (cov[t,u] 且 passed[t]) * w[t]      # 覆盖 u 的通过测试
        WS_totalfail = Σ_t failed[t] * w[t]
        score[u] = ef / sqrt(WS_totalfail * (ef + ep))
    Returns score[cov.shape[1]] —— 每个单元一个可疑度(越大越可疑)。
    """
    eps = 1e-12
    if weighted:
        c = confs.astype(np.float64)
        cn = (c - c.min()) / (c.max() - c.min() + eps)   # min-max 归一化到[0,1](GUIDER Eq.2)
        w = 1.0 + cn                                      # WS 的 (1 + W(Ti))
    else:
        w = np.ones(len(confs), dtype=np.float64)         # BB: 每个 test 计 1
    wf = w * failed                                       # [T] 失败测试的权重(failed 是 0/1)
    wp = w * passed
    ef = cov.T.astype(np.float64) @ wf                    # [W_in] 覆盖i的失败加权和
    ep = cov.T.astype(np.float64) @ wp                    # [W_in]
    ws_totalfail = wf.sum()
    return ef / (np.sqrt(ws_totalfail * (ef + ep)) + eps)


def neuron_scores_to_weights(neuron_score, W_out):
    """
    列展开(WeightsFromNeurons): 隐藏神经元 i 的分数 → 它连出去的所有 fc2 权重。
    权重展平序是 C-order [W_out, W_in],即 idx = j*W_in + i,权重(j,i)属于隐藏神经元 i。
    所以 weight_score[j*W_in + i] = neuron_score[i] → 就是把 neuron_score 平铺 W_out 次。
    Returns weight_score[W_out * W_in]。
    """
    return np.tile(neuron_score, W_out)


def topk_coverage(impacts, ratio):
    """
    DFVar/GDVar 执行侧:把连续 impact 二值化成"权重执行了没有"(SpectAcle 4.3 的 top-k 判据)。
    每条测试独立:
      N_p = 该测试上 impact > 0 的权重数   —— 先滤掉从未参与执行的权重(SpectAcle 5.3)
      k   = round(ratio × N_p)            —— impact 最大的 k 个算 executed
    对 gradient / forward_impact 都适用(两者的 impacts 口径一致)。
    Returns cov[T, total_weights] bool。
    """
    T, N = impacts.shape
    cov = np.zeros((T, N), dtype=bool)
    for t in range(T):
        row = impacts[t]
        n_pos = int((row > 1e-12).sum())
        if n_pos == 0:
            continue
        k = min(max(1, int(round(ratio * n_pos))), N)
        idx = np.argpartition(row, -k)[-k:]   # top-k,O(N) 比全排序快
        cov[t, idx] = True
    return cov


def verify_against_autograd(model, layer_name, test_images, test_labels, device, n=3):
    """Compare closed-form raw gradient vs autograd on first n tests."""
    layer = dict(model.named_modules())[layer_name]
    W_out = layer.weight.shape[0]
    acts = []
    def hook(m, inp, out): acts.append(inp[0].detach())
    h = layer.register_forward_hook(hook)
    max_err = 0.0
    for t in range(n):
        acts.clear()
        model.zero_grad()
        out = model(test_images[t:t+1].to(device))
        loss = F.cross_entropy(out, test_labels[t:t+1].to(device))
        loss.backward()
        g_auto = layer.weight.grad.detach().cpu().numpy()         # [out, in]
        p = F.softmax(out, dim=1).detach().cpu().numpy()[0]       # [out]
        y = np.eye(W_out)[int(test_labels[t])]
        a = acts[0].cpu().numpy()[0]                              # [in]
        g_cf = (p - y)[:, None] * a[None, :]
        max_err = max(max_err, np.abs(g_auto - g_cf).max())
    h.remove()
    return max_err


def compute_cosine_fl_score(impacts, fw):
    """余弦相似度 FL 分数: 每个 weight 的 involvement 向量和 fw 的余弦相似度"""
    eps = 1e-10
    imp_T = impacts.T  # [1280, T]
    return (imp_T @ fw) / (np.linalg.norm(imp_T, axis=1) * np.linalg.norm(fw) + eps)


def oracle_weights(oracle, failed, passed, confs, logits, probs, labels, 
                   nf, np_count):
    """Return ew[T] — one un-normalized result-side weight per test.

    和错误的关联
    """
    eps = 1e-12
    # norm: 把 w 里 mask 为 True 的部分缩放到总和等于 target（用于归一化，让不同 oracle 量级可比）
    # def norm(w, mask, target):
    #     s = w[mask].sum()
    #     if s > 0:
    #         w = w.copy(); w[mask] = w[mask] / s * target
    #     return w
    # 当前版本不用 norm，raw sum 直接传给 adapted Ochiai 分母
    idx = np.arange(len(labels))
    p_true = probs[idx, labels]                  # 每个测试:真值类的概率
    logit_max = logits.max(axis=1)               # 预测类的原始 logit(可能为负)
    top2_p = np.sort(probs,  axis=1)[:, -2:]     # 概率最大的两个
    top2_l = np.sort(logits, axis=1)[:, -2:]     # logit 最大的两个

    if oracle == 'QB':                     # 基线:二元,不加权(s=1 → 经典 ochiai)
        s = np.ones(len(labels))
    # ---- confidence(softmax 后):三形态 + margin ----
    elif oracle == 'QQ':              s = confs                  # 置信度本身
    elif oracle == 'QQ_sq':           s = confs ** 2             # 平方,放大高置信测试区分
    elif oracle == 'QQ_1p':           s = 1.0 + confs            # 1+置信度,保底权重再叠加
    elif oracle == 'QQ_margin':       s = top2_p[:, 1] - top2_p[:, 0]   # 概率 top1-top2:多决断
    # ---- 基于真值的信号(注:通过测试上≈0,会压垮 ep,可能退化,留作对照)----
    elif oracle == 'QQ_deficit':      s = 1.0 - p_true           # 1-真值概率
    elif oracle == 'QQ_loss':
        # 失败、通过都用交叉熵 −log(真值类概率),直接返回,不走下面"通过用 1−置信度"的统一规则。
        # (2026-09-12 改;此前通过测试是 1−置信度:高置信通过测试上两者几乎相同,
        #  0.95 → 0.051 vs 0.050,低置信或原本就判错的通过测试上差别较大)
        return -np.log(p_true + eps)
    # ---- logit(softmax 前):logit 可能为负,s 必须≥0,故用三种"挪正"办法各一个 ----
    elif oracle == 'QQ_logit_relu':   s = np.maximum(logit_max, 0.0)        # 挪正1:relu 砍负,最简单
    elif oracle == 'QQ_logit_sub':    s = logit_max - logits.min(axis=1)    # 挪正2:减每测试最小,保留相对差距
    elif oracle == 'QQ_logit_exp':    s = np.exp(logit_max)                 # 挪正3:exp 天然全正,可能数值偏大
    elif oracle == 'QQ_logit_margin': s = top2_l[:, 1] - top2_l[:, 0]       # logit top1-top2:天然≥0,压扁前的决断度
    elif oracle in ('QQ_max', 'QQ_max_pow', 'QQ_max_log', 'QQ_max_exp'):
        probs_masked = probs.copy()
        probs_masked[idx, labels] = -np.inf
        s = probs_masked.max(axis=1) - p_true             # max over i≠true of (prob[i] - p_true)
        v = (s + 1) / 2                                   # 线性映射到[0,1]
        if oracle == 'QQ_max':      return v
        if oracle == 'QQ_max_pow':  return v ** 2                      # 幂次放大
        if oracle == 'QQ_max_log':  return -np.log(1 - v + eps)       # -log(1-v) 放大
        if oracle == 'QQ_max_exp':  return np.exp(v)                   # exp 放大
    elif oracle == 'QQ_max_softmax_loss':
        d = probs - p_true[:, None]                            # [T, n], d[:, true]=0
        v = (d + 1) / 2                                        # 映射到[0,1]
        v_exp = np.exp(v - v.max(axis=1, keepdims=True))      # 数值稳定
        softmax_v = v_exp / v_exp.sum(axis=1, keepdims=True)
        return -np.log(softmax_v[idx, labels] + eps)
    elif oracle == 'QQ_cos_diff':
        d = probs - p_true[:, None]                            # [T, n], d[:, true]=0
        d_ideal = np.full_like(probs, -1.0)                   # (0, -1, -1, ...)
        d_ideal[idx, labels] = 0.0
        dot = (d * d_ideal).sum(axis=1)                        # [T]
        norm_d = np.linalg.norm(d, axis=1)
        norm_ideal = np.linalg.norm(d_ideal, axis=1)
        cos_sim = dot / (norm_d * norm_ideal + eps)
        return 1.0 - cos_sim
    elif oracle == 'QQ_cosSim':
        y = np.eye(probs.shape[1])[labels]                          # one-hot [T, 10]
        cos_sim = (probs * y).sum(axis=1) / (
            np.linalg.norm(probs, axis=1) * np.linalg.norm(y, axis=1) + eps)  # [T]
        return 1.0 - cos_sim                                        # 不分failed/passed，统一用1-相似度
    else:
        raise ValueError(oracle)

    # confidence类oracle: failed用s，passed用1-confs
    return np.where(failed, s, 1.0 - confs)

INVOLVEMENT_FNS = {'gradient': grad_involvement, 'forward_impact': forward_impact_involvement}
