"""
Attention primitives for fusing numerical features with news-text features.

Two fusion mechanisms are provided:

* ``MultiHeadCrossAttention`` -- one modality (queries) attends to another
  (keys/values). This is standard scaled dot-product multi-head attention with
  support for a key-padding mask (news days often have a variable number of
  headlines, padded to a common length).

* ``CoAttention`` -- *parallel* co-attention: the numerical tokens attend to the
  news tokens and, simultaneously, the news tokens attend to the numerical
  tokens. Both attended streams are pooled and returned, so a downstream model
  sees each modality contextualised by the other. This follows the parallel
  co-attention idea of Lu et al. (2016), adapted to (numeric, news) pairs.
"""

import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor = None, dim: int = 1) -> torch.Tensor:
    """
    Mean-pool ``x`` (B, L, D) over length dim, ignoring padded positions.

    ``mask`` is (B, L) with True for *valid* tokens. If ``mask`` is None a plain
    mean is returned. Rows with no valid tokens pool to zeros.
    """
    if mask is None:
        return x.mean(dim=dim)
    mask_f = mask.unsqueeze(-1).to(x.dtype)            # (B, L, 1)
    summed = (x * mask_f).sum(dim=dim)                 # (B, D)
    count = mask_f.sum(dim=dim).clamp_min(1.0)         # (B, 1)
    return summed / count


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention: ``query`` attends to ``key_value``.

    query:      (B, Lq, embed_dim)
    key_value:  (B, Lk, embed_dim)
    kv_mask:    (B, Lk) bool, True for valid key/value positions (optional)

    Returns the attended queries (B, Lq, embed_dim) and the attention weights.
    Built on ``nn.MultiheadAttention`` (batch_first) for numerical stability,
    with a residual connection and layer norm.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, kv_mask: torch.Tensor = None):
        # nn.MultiheadAttention expects key_padding_mask with True = *ignore*.
        key_padding_mask = None
        if kv_mask is not None:
            key_padding_mask = ~kv_mask
            # A fully-padded row would make softmax produce NaNs; let such rows
            # attend to everything, then the caller masks them out on pooling.
            all_padded = key_padding_mask.all(dim=1)
            if all_padded.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_padded] = False
        attended, weights = self.attn(
            query, key_value, key_value,
            key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True,
        )
        return self.norm(query + self.dropout(attended)), weights


class CoAttention(nn.Module):
    """
    Parallel co-attention between numerical tokens and news tokens.

    numeric:    (B, Ln, embed_dim)
    news:       (B, Lw, embed_dim)
    news_mask:  (B, Lw) bool, True for valid headlines (optional)
    numeric_mask: (B, Ln) bool (optional; numeric tokens are usually all valid)

    Returns ``(numeric_ctx, news_ctx)`` -- each stream after being attended by
    the other, same shapes as the inputs.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_attends_news = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.news_attends_num = MultiHeadCrossAttention(embed_dim, num_heads, dropout)

    def forward(self, numeric: torch.Tensor, news: torch.Tensor,
                news_mask: torch.Tensor = None, numeric_mask: torch.Tensor = None):
        numeric_ctx, _ = self.num_attends_news(numeric, news, kv_mask=news_mask)
        news_ctx, _ = self.news_attends_num(news, numeric, kv_mask=numeric_mask)
        return numeric_ctx, news_ctx
