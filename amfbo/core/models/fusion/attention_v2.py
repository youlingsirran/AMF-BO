import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ChunkBasedSequenceFusion(nn.Module):

    def __init__(self, emb_size: int, seq_len: int = 4, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert emb_size % seq_len == 0, f"the embedding size {emb_size} must be divisible by the sequence length {seq_len}"
        assert emb_size % n_heads == 0, f"the embedding size {emb_size} must be divisible by the sequence length {n_heads}"

        self.emb_size = emb_size
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.seq_dim = emb_size // seq_len
        self.head_dim = self.seq_dim // n_heads
        self.scale = self.head_dim ** -0.5


        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, self.seq_dim))

        self.q_proj = nn.Linear(self.seq_dim, self.seq_dim)
        self.k_proj = nn.Linear(self.seq_dim, self.seq_dim)
        self.v_proj = nn.Linear(self.seq_dim, self.seq_dim)

        self.out_proj1 = nn.Linear(self.seq_dim, self.seq_dim)
        self.out_proj2 = nn.Linear(self.seq_dim, self.seq_dim)

        self.merge = nn.Linear(2 * self.seq_dim, self.seq_dim)

        self.norm1 = nn.LayerNorm(self.seq_dim)
        self.norm2 = nn.LayerNorm(self.seq_dim)
        self.dropout = nn.Dropout(dropout)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.ffb = nn.Sequential(
            nn.Linear(self.seq_dim, 2 * self.seq_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.seq_dim, self.seq_dim),
        )

    def chunk_features(self, x: torch.Tensor) -> torch.Tensor:
        B, E = x.shape
        x_seq = x.view(B, self.seq_len, self.seq_dim)
        x_seq = x_seq + self.pos_encoding
        return x_seq

    def compute_attention2(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, entropy_rate):
        B, seq_len,E = x1.shape
        _, weight_x2, weight_x3 = entropy_rate.split(1, dim=1)

        weight_x2 = weight_x2.view(B, 1, 1)
        weight_x3 = weight_x3.view(B, 1, 1)

        q1 = rearrange(self.q_proj(x1), 'b s (h d) -> b h s d', h=self.n_heads)
        # k1 = rearrange(self.k_proj1(x1), 'b (h d) -> b h 1 d', h=self.n_heads)
        # v1 = rearrange(self.v_proj1(x1), 'b (h d) -> b h 1 d', h=self.n_heads)
        k2 = rearrange(self.k_proj(x2), 'b s (h d) -> b h s d', h=self.n_heads)
        v2 = rearrange(self.v_proj(x2), 'b s (h d) -> b h s d', h=self.n_heads)
        k3 = rearrange(self.k_proj(x3), 'b s (h d) -> b h s d', h=self.n_heads)
        v3 = rearrange(self.v_proj(x3), 'b s (h d) -> b h s d', h=self.n_heads)

        # ---------- 2. cross-attn：x1 ← x2 ----------
        energy12 = torch.einsum('bhqd, bhkd -> bhqk', q1, k2) * self.scale
        attn12   = F.softmax(energy12, dim=-1)
        out12    = torch.einsum('bhqk, bhvd -> bhqd', attn12, v2)   # [B, h, 1, d]
        out12    = rearrange(out12, 'b h s d -> b s (h d)')          # [B, E]
        out12    = self.out_proj1(out12)
        out12 = self.drop1(out12) * weight_x2

        # ---------- 3. cross-attn：x1 ← x3 ----------
        energy13 = torch.einsum('bhqd, bhkd -> bhqk', q1, k3) * self.scale
        attn13   = F.softmax(energy13, dim=-1)
        out13    = torch.einsum('bhqk, bhvd -> bhqd', attn13, v3)
        out13    = rearrange(out13, 'b h s d -> b s (h d)')
        out13    = self.out_proj2(out13)
        out13 = self.drop2(out13) * weight_x3

        # fused = torch.cat([out12, out13], dim=-1)        # [B, 2E]
        fused = torch.cat([ out12 + x1, out13 + x1], dim=-1)  # [B, 2E]
        fused = self.merge(fused)                        # [B, E]

        x1 = self.norm1(self.dropout(fused) + x1)

        ff = self.ffb(x1)

        # out = self.norm2(self.dropout(ff) + x1)                     # [B, E]
        out = self.norm2(ff + x1)  # [B, E]
        return out

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, entropy_rate):
        B, E = x1.shape

        x1_seq = self.chunk_features(x1)  # [B, seq_len, seq_dim]
        x2_seq = self.chunk_features(x2)  # [B, seq_len, seq_dim]
        x3_seq = self.chunk_features(x3)  # [B, seq_len, seq_dim]


        output = self.compute_attention2(x1_seq, x2_seq, x3_seq, entropy_rate)

        output = output.contiguous().view(B, -1)  # [B, seq_len * seq_dim] = [B, E]

        return output

class EnhancedChunkFusionWithWeights(ChunkBasedSequenceFusion):

    def __init__(self, emb_size: int, seq_len: int = 4, n_heads: int = 4, dropout: float = 0.2):
        super().__init__(emb_size, seq_len, n_heads, dropout)

        self.weight_gate = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Sigmoid()
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor,
                weight_b: torch.Tensor = None, weight_c: torch.Tensor = None):
        B, E = x1.shape

        x1_seq = self.chunk_features(x1)
        x2_seq = self.chunk_features(x2)
        x3_seq = self.chunk_features(x3)

        out12_seq, attn_weights12 = self.compute_attention(x1_seq, x2_seq, x2_seq, self.out_proj1, self.drop1)

        out13_seq, attn_weights13 = self.compute_attention(x1_seq, x3_seq, x3_seq, self.out_proj2, self.drop2)

        if weight_b is not None and weight_c is not None:
            weight_b_seq = weight_b.view(B, 1, 1)  # [B, 1, 1]
            weight_c_seq = weight_c.view(B, 1, 1)  # [B, 1, 1]

            out12_seq = out12_seq * weight_b_seq
            out13_seq = out13_seq * weight_c_seq

        fused_seq = torch.cat([out12_seq, out13_seq], dim=-1)
        fused_seq = self.merge(fused_seq)

        x1_seq = self.norm1(self.dropout(fused_seq) + x1_seq)
        ff_seq = self.ffb(x1_seq)
        output_seq = self.norm2(self.dropout(ff_seq) + x1_seq)
        output = output_seq.contiguous().view(B, -1)

        return output


def analyze_attention_patterns(attn_weights: torch.Tensor, description: str):
    B, n_heads, seq_len, _ = attn_weights.shape

    sample_attn = attn_weights[0]  # [n_heads, seq_len, seq_len]

    for head in range(min(2, n_heads)):
        for i in range(seq_len):
            row = sample_attn[head, i].detach().numpy()
            row_str = " ".join([f"{v:.3f}" for v in row])

    attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
    avg_entropy = attn_entropy.mean().item()

def demo_chunk_based_fusion():
    B, E = 3, 64
    seq_len = 8
    n_heads = 4

    x1 = torch.randn(B, E)
    x2 = torch.randn(B, E)
    x3 = torch.randn(B, E)

    model = ChunkBasedSequenceFusion(
        emb_size=E,
        seq_len=seq_len,
        n_heads=n_heads
    )

    output, attn_weights12, attn_weights13 = model(x1, x2, x3)

    analyze_attention_patterns(attn_weights12, "Query x2's attention pattern")
    analyze_attention_patterns(attn_weights13, "x1 查询 x3 的注意力模式")

    weight_b = torch.sigmoid(torch.randn(B, 1))
    weight_c = torch.sigmoid(torch.randn(B, 1))

    weighted_model = EnhancedChunkFusionWithWeights(
        emb_size=E,
        seq_len=seq_len,
        n_heads=n_heads
    )

    weighted_output, w_attn12, w_attn13 = weighted_model(x1, x2, x3, weight_b, weight_c)

    return output, attn_weights12, attn_weights13

if __name__ == "__main__":
    output, attn12, attn13 = demo_chunk_based_fusion()
