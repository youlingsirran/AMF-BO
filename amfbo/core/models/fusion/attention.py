import math

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SSP(nn.Softplus):
    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__(beta, threshold)

    def forward(self, input):
        sp0 = F.softplus(torch.zeros(1), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):

        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)


        self.linear_trace_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_trace_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        
        self.linear_log_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_log_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        
        self.linear_metric_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_metric_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)


        self.softmax = nn.Softmax(dim=-1)
        self.dropout_metric = nn.Dropout(dropout)
        self.dropout_trace = nn.Dropout(dropout)
        self.dropout_log = nn.Dropout(dropout)

        model_num = 3
        self.final_linear = nn.Linear(model_dim * model_num, model_dim)

    def forward(self, metric, trace, log):
        print(metric.shape, trace.shape, log.shape)
        print("%" * 20)

        query = metric.unsqueeze(1)
        print("query", query.shape)
        print("&" * 40)
        metric_key = query
        metric_value = query

        trace = trace.unsqueeze(1)
        trace_key = trace
        trace_value = trace

        batch_size = query.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        query_projected = self.linear_query(query)
        query_shaped = shape(query_projected)

        metric_key_projected = self.linear_metric_keys(metric_key)
        metric_value_projected = self.linear_metric_values(metric_value)
        metric_key_shaped = shape(metric_key_projected)
        metric_value_shaped = shape(metric_value_projected)
        metric_query_shaped = query_shaped / math.sqrt(dim_per_head)
        scores = torch.matmul(metric_query_shaped, metric_key_shaped.transpose(2, 3))
        attn = torch.sigmoid(scores)
        drop_attn = self.dropout_metric(attn)
        context = torch.matmul(drop_attn, metric_value_shaped)
        metric_context = unshape(context)

        query_len = query_shaped.size(2)
        metric_key_len = metric_key_shaped.size(2)
        top_score = scores.view(batch_size, scores.shape[1], query_len, metric_key_len)[:, 0, :, :].contiguous()

        trace_key_projected = self.linear_trace_keys(trace_key)
        trace_value_projected = self.linear_trace_values(trace_value)
        trace_key_shaped = shape(trace_key_projected)
        trace_value_shaped = shape(trace_value_projected)
        metric_query_shaped = query_shaped / math.sqrt(dim_per_head)
        scores = torch.matmul(metric_query_shaped, trace_key_shaped.transpose(2, 3))
        attn = torch.sigmoid(scores)
        drop_attn = self.dropout_trace(attn)
        context = torch.matmul(drop_attn, trace_value_shaped)
        trace_context = unshape(context)

        log = log.unsqueeze(1)
        log_key = log
        log_value = log
        log_key_projected = self.linear_log_keys(log_key)
        log_value_projected = self.linear_log_values(log_value)
        log_key_shaped = shape(log_key_projected)
        log_value_shaped = shape(log_value_projected)
        metric_query_shaped = query_shaped / math.sqrt(dim_per_head)
        scores = torch.matmul(metric_query_shaped, log_key_shaped.transpose(2, 3))
        # print("+" * 30)
        attn = torch.sigmoid(scores)
        drop_attn = self.dropout_log(attn)
        context = torch.matmul(drop_attn, log_value_shaped)
        log_context = unshape(context)

        context = torch.cat([metric_context, trace_context], dim=-1)
        context = torch.cat([context, log_context], dim=-1)
        output = self.final_linear(context)

        return output, top_score

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

import torch
import torch.nn as nn


class TripleFusion(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.2):
        super().__init__()
        self.emb_size = emb_size
        self.merge = nn.Linear(3 * emb_size, emb_size)
        self.norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        self.refine = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size, 32)
        )
        self.final_norm = nn.LayerNorm(32)

    def forward(self, x1, x2, x3):
        # 1. concat
        concat = torch.cat([x1, x2, x3], dim=-1)          # [B, 3*emb_size]
        fused = self.merge(concat)                        # [B, emb_size]
        fused = self.norm(self.dropout(fused) + x1)
        out = self.refine(fused)
        return self.final_norm(out)


class TripleCrossAttnFusion(nn.Module):
    def __init__(self, emb_size: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert emb_size % n_heads == 0
        self.emb_size = emb_size
        self.n_heads  = n_heads
        self.head_dim = emb_size // n_heads
        self.scale    = self.head_dim ** -0.5

        self.q_proj = nn.Linear(emb_size, emb_size)
        self.k_proj = nn.Linear(emb_size, emb_size)
        self.v_proj = nn.Linear(emb_size, emb_size)
        # self.k_proj2 = nn.Linear(emb_size, emb_size)
        # self.v_proj2 = nn.Linear(emb_size, emb_size)

        self.out_proj1 = nn.Linear(emb_size, emb_size)
        self.out_proj2 = nn.Linear(emb_size, emb_size)
        self.merge    = nn.Linear(2 * emb_size, emb_size)

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.ffb = nn.Sequential(
            nn.Linear(emb_size, 2 * emb_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * emb_size, emb_size),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        q1 = rearrange(self.q_proj(x1), 'b (h d) -> b h 1 d', h=self.n_heads)
        k2 = rearrange(self.k_proj(x2), 'b (h d) -> b h 1 d', h=self.n_heads)
        v2 = rearrange(self.v_proj(x2), 'b (h d) -> b h 1 d', h=self.n_heads)
        k3 = rearrange(self.k_proj(x3), 'b (h d) -> b h 1 d', h=self.n_heads)
        v3 = rearrange(self.v_proj(x3), 'b (h d) -> b h 1 d', h=self.n_heads)

        energy12 = torch.einsum('bhqd, bhkd -> bhqk', q1, k2) * self.scale
        attn12   = F.softmax(energy12, dim=-1)
        out12    = torch.einsum('bhqk, bhvd -> bhqd', attn12, v2)   # [B, h, 1, d]
        out12    = rearrange(out12, 'b h 1 d -> b (h d)')          # [B, E]
        out12    = self.out_proj1(out12)
        out12 = self.drop1(out12)

        energy13 = torch.einsum('bhqd, bhkd -> bhqk', q1, k3) * self.scale
        attn13   = F.softmax(energy13, dim=-1)
        out13    = torch.einsum('bhqk, bhvd -> bhqd', attn13, v3)
        out13    = rearrange(out13, 'b h 1 d -> b (h d)')
        out13    = self.out_proj2(out13)
        out13 = self.drop2(out13)

        # fused = torch.cat([out12, out13], dim=-1)        # [B, 2E]
        fused = torch.cat([ out12 + x1, out13 + x1], dim=-1)  # [B, 2E]
        fused = self.merge(fused)                        # [B, E]

        x1 = self.norm1(self.dropout(fused) + x1)

        ff = self.ffb(x1)

        out = self.norm2(ff + x1)  # [B, E]
        return out

class GuidedSelfAttn25_4Head(nn.Module):
    def __init__(self, emb_size: int, guidance_num: int = 25, n_heads: int = 4):
        super().__init__()
        self.emb_size = emb_size
        self.guidance_num = guidance_num
        self.n_heads = n_heads
        assert emb_size % n_heads == 0, "error"
        self.head_dim = emb_size // n_heads
        self.scale = self.head_dim ** -0.5
        self.guidance_generator = nn.Sequential(
            nn.Linear(emb_size, emb_size * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(emb_size * 2, guidance_num * emb_size)
        )
        self.q_proj = nn.Linear(emb_size, emb_size)
        self.k_proj = nn.Linear(emb_size, emb_size)
        self.v_proj = nn.Linear(emb_size, emb_size)
        self.out_proj = nn.Linear(emb_size, emb_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.guidance_generator[0].weight, std=0.01)
        nn.init.normal_(self.guidance_generator[3].weight, std=0.01)

    def forward(self, q_x: torch.Tensor, kv_x: torch.Tensor):
        B, E = q_x.shape
        H = self.n_heads
        d = self.head_dim

        guidance = self.guidance_generator(kv_x)  # [B, guidance_num * E]
        guidance = guidance.view(B, self.guidance_num, E)  # [B, guidance_num, E]

        q = rearrange(self.q_proj(q_x), 'b (h d) -> b h 1 d', h=H)  # [B, H, 1, d]
        k = rearrange(self.k_proj(guidance), 'b n (h d) -> b h n d', h=H)  # [B, H, guidance_num, d]
        v = rearrange(self.v_proj(guidance), 'b n (h d) -> b h n d', h=H)  # [B, H, guidance_num, d]

        attn = torch.softmax(torch.einsum('bhqd, bhkd -> bhqk', q, k) * self.scale, dim=-1)  # [B, H, 1, guidance_num]
        out = torch.einsum('bhqk, bhvd -> bhqd', attn, v)  # [B, H, 1, d]
        out = rearrange(out, 'b h 1 d -> b (h d)')  # [B, E]
        out = self.out_proj(out)  # [B, E]

        return out

class TripleGuidedFusion4Head(nn.Module):
    def __init__(self, emb_size: int, guidance_num: int = 25, n_heads: int = 4):
        super().__init__()
        self.guided12 = GuidedSelfAttn25_4Head(emb_size, guidance_num, n_heads)
        self.guided13 = GuidedSelfAttn25_4Head(emb_size, guidance_num, n_heads)
        self.merge = nn.Linear(2 * emb_size, emb_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(0.2)
        self.ffb = nn.Sequential(
            nn.Linear(emb_size, 2 * emb_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2 * emb_size, emb_size),
        )
        self.norm2 = nn.LayerNorm(emb_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        out12 = self.guided12(x1, x2)
        out13 = self.guided13(x1, x3)

        fused = torch.cat([out12, out13], dim=-1)  # [B, 2E]
        merged = self.merge(fused)  # [B, E]

        residual = self.dropout(merged) + x1
        normalized = self.norm1(residual)

        ff = self.ffb(normalized)
        out = self.norm2(ff + normalized)  # [B, E]

        return out

class DualCrossAttnFusion(nn.Module):
    def __init__(self, emb_size: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert emb_size % n_heads == 0
        self.emb_size = emb_size
        self.n_heads  = n_heads
        self.head_dim = emb_size // n_heads
        self.scale    = self.head_dim ** -0.5

        self.q_proj = nn.Linear(emb_size, emb_size)
        self.k_proj = nn.Linear(emb_size, emb_size)
        self.v_proj = nn.Linear(emb_size, emb_size)

        self.out_proj = nn.Linear(emb_size, emb_size)

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ffb = nn.Sequential(
            nn.Linear(emb_size, 2 * emb_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * emb_size, emb_size),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        q1 = rearrange(self.q_proj(x1), 'b (h d) -> b h 1 d', h=self.n_heads)
        k2 = rearrange(self.k_proj(x2), 'b (h d) -> b h 1 d', h=self.n_heads)
        v2 = rearrange(self.v_proj(x2), 'b (h d) -> b h 1 d', h=self.n_heads)

        energy12 = torch.einsum('bhqd, bhkd -> bhqk', q1, k2) * self.scale
        attn12   = F.softmax(energy12, dim=-1)
        out12    = torch.einsum('bhqk, bhvd -> bhqd', attn12, v2)   # [B, h, 1, d]
        out12    = rearrange(out12, 'b h 1 d -> b (h d)')          # [B, E]
        out12    = self.out_proj(out12)
        # out12 = self.drop1(out12) # sockshop

        output = out12 + x1
        out = self.norm1(output)

        return out

import torch
import torch.nn as nn


class AdaptiveFusionTopology(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(AdaptiveFusionTopology, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.projection_net = nn.Linear(dim, dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        # self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0]))
        self.project = nn.Linear(2 * dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.cross_ATTN = DualCrossAttnFusion(32)

    def forward(self, A, B, C, E):
        modalities = torch.stack([A, B, C], dim=1)  # [B, 3, D]
        fused_output = torch.zeros_like(A)
        # sorted_vals: [B, 3], sorted_indices: [B, 3]
        sorted_vals, sorted_indices = torch.sort(E, dim=1, descending=True)
        diff1 = sorted_vals[:, 0] - sorted_vals[:, 1]
        diff2 = sorted_vals[:, 1] - sorted_vals[:, 2]
        diff3 = sorted_vals[:, 0] - sorted_vals[:, 2]
        mask_A = (diff1 > 0.35) & (diff2 < 0.15)
        mask_C = (diff3 < 0.25)
        mask_B = ~(mask_A | mask_C)
        idx_A = mask_A.nonzero(as_tuple=True)[0]
        idx_B = mask_B.nonzero(as_tuple=True)[0]
        idx_C = mask_C.nonzero(as_tuple=True)[0]

        # ==========================================
        # ==========================================
        if len(idx_A) > 0:
            print("a")
            dom_idx = sorted_indices[idx_A, 0]
            weak_idx_1 = sorted_indices[idx_A, 1]
            weak_idx_2 = sorted_indices[idx_A, 2]

            F_dominant = modalities[idx_A, dom_idx, :]  # [N_A, D]
            F_weak_1 = modalities[idx_A, weak_idx_1, :]
            F_weak_2 = modalities[idx_A, weak_idx_2, :]
            # F_dominant, F_weak_1, F_weak_2 = signed_harmonic_fusion(F_dominant, F_weak_1, F_weak_2)

            scores_dominant = E[idx_A, dom_idx].unsqueeze(1)  # [N_A, 1]
            scores_w1 = E[idx_A, weak_idx_1].unsqueeze(1)
            scores_w2 = E[idx_A, weak_idx_2].unsqueeze(1)

            out_1 = self.cross_ATTN(F_dominant, F_weak_1).squeeze(1)
            out_2 = self.cross_ATTN(F_dominant, F_weak_2).squeeze(1)

            # sockshop
            alpha_w1_prime = (scores_dominant + scores_w1) / (2 * scores_dominant + scores_w1 + scores_w2)
            alpha_w2_prime = (scores_dominant + scores_w2) / (2 * scores_dominant + scores_w1 + scores_w2)
            out_A = alpha_w1_prime * out_1 + alpha_w2_prime * out_2

            # # gaia
            # alpha_dom = scores_dominant / (scores_dominant + scores_w1 + scores_w2)
            # sum_weak = scores_w1 + scores_w2
            # alpha_w1_prime = scores_w1 / (sum_weak)
            # alpha_w2_prime = scores_w2 / (sum_weak)
            # fused_weak_branch = alpha_w1_prime * out_1 + alpha_w2_prime * out_2
            # out_A = alpha_dom * F_dominant + (1 - alpha_dom) * fused_weak_branch

            fused_output[idx_A] = out_A

        # ==========================================
        # ==========================================
        if len(idx_B) > 0:
            print("bb")
            strong_idx_1 = sorted_indices[idx_B, 0]
            strong_idx_2 = sorted_indices[idx_B, 1]
            weak_idx = sorted_indices[idx_B, 2]

            F_s1 = modalities[idx_B, strong_idx_1, :]
            F_s2 = modalities[idx_B, strong_idx_2, :]
            F_w = modalities[idx_B, weak_idx, :]

            scores_strong1 = E[idx_B, strong_idx_1].unsqueeze(1)  # [N_A, 1]
            scores_strong2 = E[idx_B, strong_idx_2].unsqueeze(1)
            scores_weak = E[idx_B, weak_idx].unsqueeze(1)

            # sockshop
            F_1w = self.cross_ATTN(F_s1, F_w).squeeze(1)
            F_2w = self.cross_ATTN(F_s2, F_w).squeeze(1)
            alpha_w1 = (scores_weak + scores_strong1) / (2 * scores_weak + scores_strong1 + scores_strong2)
            alpha_w2 = (scores_weak + scores_strong2) / (2 * scores_weak + scores_strong1 + scores_strong2)
            out_B = alpha_w1 * F_1w + alpha_w2 * F_2w

            # # gaia
            # F_12 = self.cross_ATTN(F_s1, F_s2).squeeze(1)  # [N_B, D]
            # residual = self.cross_ATTN(F_12, F_w).squeeze(1)
            # out_B = (scores_strong1 + scores_strong2) * F_12 + scores_weak * residual
            fused_output[idx_B] = out_B

        # ==========================================
        # ==========================================
        if len(idx_C) > 0:
            strong_1 = sorted_indices[idx_C, 0]
            strong_2 = sorted_indices[idx_C, 1]
            strong_3 = sorted_indices[idx_C, 2]

            F1 = modalities[idx_C, strong_1, :]
            F2 = modalities[idx_C, strong_2, :]
            F3 = modalities[idx_C, strong_3, :]

            scores_strong1 = E[idx_C, strong_1].unsqueeze(1)
            scores_strong2 = E[idx_C, strong_2].unsqueeze(1)
            scores_strong3 = E[idx_C, strong_3].unsqueeze(1)

            stacked = torch.stack([F1, F2, F3], dim=0)
            global_feature, _ = torch.max(stacked, dim=0)
            out_1 = self.cross_ATTN(global_feature, F1).squeeze(1)
            out_2 = self.cross_ATTN(global_feature, F2).squeeze(1)
            out_3 = self.cross_ATTN(global_feature, F3).squeeze(1)

            out = scores_strong1 * out_1 + scores_strong2 * out_2 + scores_strong3 * out_3
            out_C = self.project(torch.cat([global_feature, out], dim=1))

            fused_output[idx_C] = out_C

        return self.layer_norm(fused_output)


def cal_rcl_loss(root_logit, batch_graphs):
    num_nodes_list = batch_graphs.batch_num_nodes()
    total_loss = None

    all_anomaly_scores = []

    start_idx = 0
    for idx, num_nodes in enumerate(num_nodes_list):
        end_idx = start_idx + num_nodes

        node_logits = root_logit[start_idx: end_idx].reshape(1, -1)

        node_probs = F.softmax(node_logits, dim=1)

        anomaly_scores = node_probs.squeeze(0)
        all_anomaly_scores.append(anomaly_scores)

        root = batch_graphs.ndata["root"][start_idx: end_idx].tolist().index(1)

        loss = F.cross_entropy(node_logits, torch.LongTensor([root]).view(1).to('cuda'))

        if total_loss is None:
            total_loss = loss
        else:
            total_loss += loss

        start_idx += num_nodes

    l_rcl = total_loss / len(num_nodes_list)

    all_anomaly_scores = torch.cat(all_anomaly_scores, dim=0)

    return l_rcl, all_anomaly_scores


def anomaly_topk_pooling(batch_graphs, emb, all_anomaly_scores, K=5):
    num_nodes_list = batch_graphs.batch_num_nodes()
    if len(emb.shape) > 1:
        feat_dim = emb.shape[1]
    else:
        feat_dim = 1
    pooled_features_list = []
    start_idx = 0
    for i, num_nodes in enumerate(num_nodes_list):
        end_idx = start_idx + num_nodes
        graph_emb = emb[start_idx:end_idx]
        graph_scores = all_anomaly_scores[start_idx:end_idx]
        actual_K = min(K, num_nodes)
        if actual_K > 0:
            topk_indices = torch.topk(graph_scores, actual_K, largest=True).indices
            topk_features = graph_emb[topk_indices]
            pooled_feature = torch.max(topk_features, dim=0).values
        else:
            pooled_feature = torch.zeros(feat_dim, device=emb.device)
        pooled_features_list.append(pooled_feature)
        start_idx += num_nodes
    pooled_features = torch.stack(pooled_features_list, dim=0)
    return pooled_features