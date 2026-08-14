import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# def cohesion_early_fusion(behavior_feat, visual_feat, text_feat, epsilon=1e-8):
#     visual_sign = torch.sign(visual_feat)
#     text_sign = torch.sign(text_feat)
#
#     visual_magnitude = torch.sqrt(0.5 * (torch.square(visual_feat) +
#                                          torch.square(behavior_feat)) + epsilon)
#
#     text_magnitude = torch.sqrt(0.5 * (torch.square(text_feat) +
#                                        torch.square(behavior_feat)) + epsilon)
#
#     refined_visual_feat = visual_sign * visual_magnitude
#     refined_text_feat = text_sign * text_magnitude
#
#     return behavior_feat, refined_visual_feat, refined_text_feat

def cohesion_early_fusion_2(behavior_feat, visual_feat, text_feat, epsilon=1e-8):
    behavior_sign = torch.sign(behavior_feat)

    visual_sign = torch.sign(visual_feat)
    text_sign = torch.sign(text_feat)

    mask = (behavior_sign != 0) & (visual_sign != 0) & (text_sign != 0)

    refined_visual_feat = visual_feat.clone()
    refined_text_feat = text_feat.clone()

    if mask.any():
        behavior_masked = behavior_feat[mask]
        visual_masked = visual_feat[mask]
        text_masked = text_feat[mask]

        visual_magnitude = torch.sqrt(0.5 * (
                torch.square(visual_masked) +
                torch.square(behavior_masked)
        ) + epsilon)

        text_magnitude = torch.sqrt(0.5 * (
                torch.square(text_masked) +
                torch.square(behavior_masked)
        ) + epsilon)

        refined_visual_feat[mask] = visual_sign[mask] * visual_magnitude
        refined_text_feat[mask] = text_sign[mask] * text_magnitude

    return behavior_feat, refined_visual_feat, refined_text_feat

def cohesion_early_fusion(behavior_feat, visual_feat, text_feat, epsilon=1e-8):
    behavior_sign = torch.sign(behavior_feat)

    visual_sign = torch.sign(visual_feat)
    text_sign = torch.sign(text_feat)

    visual_sign = torch.where(visual_sign == 0, behavior_sign, visual_sign)

    text_sign = torch.where(text_sign == 0, behavior_sign, text_sign)

    visual_magnitude = torch.sqrt(0.5 * (torch.square(visual_feat) +
                                         torch.square(behavior_feat)) + epsilon)

    text_magnitude = torch.sqrt(0.5 * (torch.square(text_feat) +
                                       torch.square(behavior_feat)) + epsilon)

    refined_visual_feat = visual_sign * visual_magnitude
    refined_text_feat = text_sign * text_magnitude

    return behavior_feat, refined_visual_feat, refined_text_feat


class GatedAffineFusion(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )
        self.beta_net = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, behavior_feat, modality_feat):
        gamma = self.gamma_net(behavior_feat)
        beta = self.beta_net(behavior_feat)

        return gamma * modality_feat + beta


# def signed_harmonic_mean_refinement(behavior_feat, modality_feat, epsilon=1e-8):
#     """
#
#     Args:
#
#     Returns:
#     """
#     original_sign = torch.sign(modality_feat)
#
#     abs_behavior = torch.abs(behavior_feat)
#     abs_modality = torch.abs(modality_feat)
#
#     numerator = 2 * abs_behavior * abs_modality
#     denominator = abs_behavior + abs_modality + epsilon
#
#     harmonic_magnitude = numerator / denominator
#
#     refined_feat = original_sign * harmonic_magnitude
#
#     return refined_feat


def signed_harmonic_fusion(behavior_feat, visual_feat, text_feat, epsilon=1e-8):

    fusion1 = GatedAffineFusion(32).to("cuda")

    refined_visual = fusion1(behavior_feat, visual_feat)

    fusion2 = GatedAffineFusion(32).to("cuda")
    refined_text = fusion2(behavior_feat, text_feat)



    # refined_visual = signed_harmonic_mean_refinement(behavior_feat, visual_feat, epsilon)
    #
    # refined_text = signed_harmonic_mean_refinement(behavior_feat, text_feat, epsilon)

    return behavior_feat, refined_visual, refined_text


# def stabilized_harmonic_refinement(behavior_feat, modality_feat, epsilon=1e-6, clamp_value=10.0):
#     """
#
#     Args:
#
#     Returns:
#     """
#     numerator = 2 * behavior_feat * modality_feat
#
#     denominator = behavior_feat + modality_feat
#
#     sign = torch.sign(denominator)
#     denominator_stable = denominator + epsilon * sign
#
#     mask_small = torch.abs(denominator) < epsilon
#     backup_value = torch.sign(numerator) * torch.sqrt(torch.abs(numerator) + epsilon)
#
#     harmonic_mean = numerator / denominator_stable
#
#     harmonic_mean = torch.where(mask_small, backup_value, harmonic_mean)
#
#     harmonic_mean = torch.clamp(harmonic_mean, -clamp_value, clamp_value)
#
#     harmonic_mean = torch.nan_to_num(harmonic_mean, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
#
#     return harmonic_mean
#
#
# def robust_harmonic_fusion(behavior_feat, visual_feat, text_feat, epsilon=1e-6):
#     """
#     """
#     refined_visual = stabilized_harmonic_refinement(behavior_feat, visual_feat, epsilon)
#
#     refined_text = stabilized_harmonic_refinement(behavior_feat, text_feat, epsilon)
#
#     return behavior_feat, refined_visual, refined_text

class StaticDominanceCalculator(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2),
            # nn.Sigmoid()
            # nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            # nn.Sigmoid()
            # nn.ReLU()
        )
    def forward(self, metric, log, trace):
        self.proj = self.proj.to(self.device)

        proj_metric = self.proj(metric)
        proj_log = self.proj(log)
        proj_trace = self.proj(trace)

        self.fusion = self.fusion.to(self.device)
        fused = self.fusion(torch.cat([metric, log, trace], dim=1))
        proj_fused = self.proj(fused)

        sim_metric = F.cosine_similarity(proj_metric, proj_fused, dim=-1)
        sim_log = F.cosine_similarity(proj_log, proj_fused, dim=-1)
        sim_trace = F.cosine_similarity(proj_trace, proj_fused, dim=-1)

        similarities = torch.stack([sim_metric, sim_log, sim_trace], dim=1)
        dominant_idx = torch.argmax(similarities, dim=1)

        modality_names = ['metric', 'log', "trace"]
        return [modality_names[idx] for idx in dominant_idx]

def inplace_swap1(metric, log, trace, dominant_list):

    for i, mod in enumerate(dominant_list):
        if mod == 'log':
            temp = metric[i].clone()
            metric[i] = log[i].clone()
            log[i] = temp
        elif mod == 'trace':
            temp = metric[i].clone()
            metric[i] = trace[i].clone()
            trace[i] = temp

    metric, log, trace= signed_harmonic_fusion(metric, log, trace)

    return metric, log, trace

def inplace_swap(metric, log, trace, entropy_batch):
    result = entropy_batch.clone()
    batch_size = result.shape[0]

    idx = torch.argmax(entropy_batch, dim=1)
    modality = ['metric', 'log', 'trace']
    dominant_list = [modality[i] for i in idx]

    for i, mod in enumerate(dominant_list):
        if mod == 'log':
            temp = metric[i].clone()
            metric[i] = log[i].clone()
            log[i] = temp
        elif mod == 'trace':
            temp = metric[i].clone()
            metric[i] = trace[i].clone()
            trace[i] = temp

    rows = torch.arange(batch_size)
    max_values = result[rows, idx]
    first_col = result[:, 0].clone()
    result[rows, 0] = max_values
    result[rows, idx] = first_col
    result = result / result[:, 0:1]

    metric, log, trace = signed_harmonic_fusion(metric, log, trace)

    return metric, log, trace, result

class OneShotCrossFusion(nn.Module):
    def __init__(self, hidden=32, num_heads=4, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.router = nn.Sequential(
            nn.Linear(hidden * 3, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)
        )

    def forward(self, a, b, c):

        router_in = torch.cat([a, b, c], dim=1)  # (96,)
        weight = F.softmax(self.router(router_in), dim=0)      # (3,)

        dominant_idx = torch.argmax(weight, dim=1)

        modality_names = ['metric', 'log', "trace"]
        return [modality_names[idx] for idx in dominant_idx]

def entropy_vec(a, b, c, bins=32):
    B, D = a.shape
    device = a.device
    ent = torch.zeros(B, 3, device=device)

    vec_min = torch.min(a.min(dim=1)[0], torch.min(b.min(dim=1)[0], c.min(dim=1)[0]))
    vec_max = torch.max(a.max(dim=1)[0], torch.max(b.max(dim=1)[0], c.max(dim=1)[0]))
    vec_min = vec_min.view(B, 1)
    vec_max = vec_max.view(B, 1) + 1e-6

    for i, vec in enumerate([a, b, c]):
        idx = ((vec - vec_min) / (vec_max - vec_min) * (bins - 1)).long().clamp(0, bins - 1)
        hist = torch.zeros(B, bins, device=device, dtype=torch.float)
        hist.scatter_add_(dim=1, index=idx, src=torch.ones_like(idx, dtype=torch.float))
        prob = hist / (D * 1.0) # (B, bins)
        prob = prob.masked_fill(prob == 0, 1.0)
        ent[:, i] = -torch.sum(prob * torch.log2(prob), dim=1)

    res = F.softmax(ent, dim=-1)
    # print(res)

    return res


def batch_joint_entropy(h_m: torch.Tensor,
                        h_l: torch.Tensor,
                        h_t: torch.Tensor,
                        eps: float = 1e-9) -> torch.Tensor:
    B, d = h_m.shape
    h_cat = torch.cat([h_m, h_l, h_t], dim=1)
    p_cat = F.softmax(h_cat, dim=1)
    p_m, p_l, p_t = p_cat.split(d, dim=1)
    entropy = lambda p: -(p * (p + eps).log()).sum(dim=1)
    e_m = entropy(p_m)
    e_l = entropy(p_l)
    e_t = entropy(p_t)

    res = F.softmax(torch.stack([e_m, e_l, e_t], dim=1), dim=-1)
    # print(res)

    return res

    # return torch.stack([e_m, e_l, e_t], dim=1)

def feature_statistics(importance):
    _, max_indices = torch.max(importance, dim=1)

    max_counts = torch.bincount(max_indices, minlength=3)

    importance_sums = torch.sum(importance, dim=0)

    # print(max_counts, importance_sums)

    max_count_value, max_count_idx = torch.max(max_counts, dim=0)
    features = ['metric', 'log', 'trace']
    dominant_feature = features[max_count_idx.item()]

    max_sum_value, _ = torch.max(importance_sums, dim=0)
    total_sum = torch.sum(importance_sums)
    max_ratio = max_sum_value / total_sum

    return dominant_feature, max_ratio.item()

def compute_entropy_batch_parallel(
        input_tensor: torch.Tensor,
        bins: int = 20,
        eps: float = 1e-6,
        ) -> torch.Tensor:
    min_vals = input_tensor.min(dim=1).values.unsqueeze(1)  # (batch, 1)
    max_vals = input_tensor.max(dim=1).values.unsqueeze(1)  # (batch, 1)

    input_norm = (input_tensor - min_vals) / (max_vals - min_vals)
    input_norm = torch.clamp(input_norm, 0, 1)
    bin_indices = (input_norm * bins).long()
    bin_indices = torch.clamp(bin_indices, 0, bins - 1)

    one_hot = torch.nn.functional.one_hot(bin_indices, num_classes=bins)
    histograms = one_hot.sum(dim=1).float()  # (batch_size, bins)

    probs = histograms / histograms.sum(dim=1, keepdim=True)
    probs = torch.clamp(probs, min=eps)

    entropies = -torch.sum(probs * torch.log2(probs), dim=1)

    return entropies

def geometric_weighted_fusion(MUC, PCC, alpha=0.5, eps=1e-6):

    if alpha < 0 or alpha > 1:
        raise ValueError("error")

    if alpha == 0:
        scores = PCC
    elif alpha == 1:
        scores = MUC
    else:

        MUC_safe = MUC.clone()
        PCC_safe = PCC.clone()

        MUC_safe[MUC_safe <= 0] = eps
        PCC_safe[PCC_safe <= 0] = eps

        log_MUC = torch.log(MUC_safe)
        log_PCC = torch.log(PCC_safe)

        log_scores = alpha * log_MUC + (1 - alpha) * log_PCC

        scores = torch.exp(log_scores)

    pi = F.softmax(scores, dim=1)

    return pi


def process_columns(tensor_data):

    columns = ['metric', 'log', 'trace']

    column_sums = torch.sum(tensor_data, dim=0)
    total_sum = torch.sum(column_sums)
    normalized_values = column_sums / total_sum
    feature_names = []
    normalized_list = []
    threshold = 1 / 3

    for i, col in enumerate(columns):
        if normalized_values[i] > threshold:
            feature_names.append(col)
            normalized_list.append(normalized_values[i].item())

    return feature_names, normalized_list