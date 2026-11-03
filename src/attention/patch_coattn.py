"""
Patch-based cross/co-attention model — faithful to Zhang et al. (2026, PDF 1):
"Integrative stock price trend prediction via hierarchical LLM text processing
and patch-based transformer with co-attention".

Pipeline (our data uses FinBERT/Marketaux headline embeddings in place of the
paper's hierarchical-LLM daily summary embedding):

  1. Embedding & Alignment: cross-attention with price as Query and text as
     Key/Value (paper Eq. 2-3) -> text-aligned price stream X_tp.
  2. Patch partitioning: split the length-L sequences into N overlapping patches
     of length P, stride S (Eq. 4), with a linear patch projection.
  3. Self-attention enhancement: a self-attention layer per stream (Eq. 6-7).
  4. Co-Attention Fusion: bidirectional (text->price and price->text), concat,
     linear projection, residual + LayerNorm, then FFN (Eq. 8-13).
  5. Classification: pool + FC -> single logit for up/down movement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention.modules import MultiHeadCrossAttention, CoAttention


class PatchCoAttentionNet(nn.Module):
    def __init__(self, price_dim: int, news_dim: int, d_model: int = 64, num_heads: int = 4,
                 patch_len: int = 4, stride: int = 2, dropout: float = 0.1, out_dim: int = 1):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.price_proj = nn.Linear(price_dim, d_model)
        self.news_proj = nn.Linear(news_dim, d_model)
        self.align = MultiHeadCrossAttention(d_model, num_heads, dropout)   # price=Q, text=K/V
        self.patch_proj = nn.Linear(patch_len * d_model, d_model)
        self.self_price = nn.TransformerEncoderLayer(d_model, num_heads, d_model * 2,
                                                     dropout, batch_first=True)
        self.self_text = nn.TransformerEncoderLayer(d_model, num_heads, d_model * 2,
                                                    dropout, batch_first=True)
        self.co = CoAttention(d_model, num_heads, dropout)
        self.proj = nn.Linear(2 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, out_dim))

    def _patch(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, d) -> (B, N, d) via overlapping patches of length P, stride S."""
        B, L, d = x.shape
        segs, i = [], 0
        while i < L:
            seg = x[:, i:i + self.patch_len]
            if seg.shape[1] < self.patch_len:
                seg = F.pad(seg, (0, 0, 0, self.patch_len - seg.shape[1]))
            segs.append(seg.reshape(B, 1, self.patch_len * d))
            if i + self.patch_len >= L:
                break
            i += self.stride
        patches = torch.cat(segs, dim=1)                 # (B, N, P*d)
        return self.patch_proj(patches)                  # (B, N, d)

    def forward(self, price, news, news_mask=None, numeric_mask=None):
        p = self.price_proj(price)                       # (B, L, d)
        w = self.news_proj(news)                         # (B, Lw, d)
        x_tp, _ = self.align(p, w, kv_mask=news_mask)    # cross-attention alignment
        tp_tok = self.self_text(self._patch(x_tp))       # text-aligned stream, patched + self-attn
        p_tok = self.self_price(self._patch(p))          # price stream, patched + self-attn
        tp_ctx, p_ctx = self.co(tp_tok, p_tok)           # bidirectional co-attention
        fused = self.proj(torch.cat([tp_ctx, p_ctx], dim=-1))
        z1 = self.norm1(fused + p_tok)                   # residual + LN
        z = self.norm2(z1 + self.ffn(z1))                # FFN + residual + LN
        return self.head(z.mean(dim=1))                  # pool patches -> logit(s)
