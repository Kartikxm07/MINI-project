"""
Prediction models that fuse a numeric sequence with news-text embeddings.

All three models share the same interface so they can be compared head-to-head
on an identical task and tuned with the same search space:

    model(numeric, news, news_mask) -> (B, out_dim)

    numeric:   (B, Ln, num_feat_dim)   numeric token sequence
               (price/OHLCV window for stocks; contract features for options)
    news:      (B, Lw, news_emb_dim)   per-sample headline embeddings
    news_mask: (B, Lw) bool            True for real headlines, False for padding

* ``NumericOnlyModel``   -- baseline; ignores news entirely.
* ``CrossAttentionModel``-- numeric tokens attend to news (one-directional).
* ``CoAttentionModel``   -- parallel co-attention (numeric<->news).

A GRU encodes the numeric sequence into contextual tokens first, so temporal
order is captured before attention.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.attention.modules import MultiHeadCrossAttention, CoAttention, masked_mean


@dataclass
class ModelConfig:
    d_model: int = 64
    num_heads: int = 4
    gru_layers: int = 1
    mlp_hidden: int = 64
    dropout: float = 0.1
    out_dim: int = 1


class _NumericEncoder(nn.Module):
    """GRU encoder turning a numeric sequence into contextual tokens."""
    def __init__(self, in_dim: int, d_model: int, gru_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            in_dim, d_model, num_layers=gru_layers, batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)          # (B, Ln, d_model)
        return self.norm(out)


class _NewsProjector(nn.Module):
    """Projects raw news embeddings (e.g. FinBERT 768-d) into the model space."""
    def __init__(self, news_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(news_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, news: torch.Tensor) -> torch.Tensor:
        return self.proj(news)


class _Head(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NumericOnlyModel(nn.Module):
    """Baseline: numeric sequence only, no news (for the comparison)."""
    kind = "numeric"

    def __init__(self, num_feat_dim: int, news_emb_dim: int, cfg: ModelConfig):
        super().__init__()
        self.encoder = _NumericEncoder(num_feat_dim, cfg.d_model, cfg.gru_layers, cfg.dropout)
        self.head = _Head(cfg.d_model, cfg.mlp_hidden, cfg.out_dim, cfg.dropout)

    def forward(self, numeric, news=None, news_mask=None, numeric_mask=None):
        tokens = self.encoder(numeric)
        pooled = masked_mean(tokens, numeric_mask)
        return self.head(pooled)


class CrossAttentionModel(nn.Module):
    """Numeric tokens (queries) cross-attend to news tokens."""
    kind = "cross"

    def __init__(self, num_feat_dim: int, news_emb_dim: int, cfg: ModelConfig):
        super().__init__()
        self.encoder = _NumericEncoder(num_feat_dim, cfg.d_model, cfg.gru_layers, cfg.dropout)
        self.news_proj = _NewsProjector(news_emb_dim, cfg.d_model, cfg.dropout)
        self.cross = MultiHeadCrossAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.head = _Head(cfg.d_model, cfg.mlp_hidden, cfg.out_dim, cfg.dropout)

    def forward(self, numeric, news, news_mask=None, numeric_mask=None):
        tokens = self.encoder(numeric)
        news_tok = self.news_proj(news)
        ctx, _ = self.cross(tokens, news_tok, kv_mask=news_mask)
        pooled = masked_mean(ctx, numeric_mask)
        return self.head(pooled)


class CoAttentionModel(nn.Module):
    """Parallel co-attention: numeric<->news, both streams pooled and fused."""
    kind = "co"

    def __init__(self, num_feat_dim: int, news_emb_dim: int, cfg: ModelConfig):
        super().__init__()
        self.encoder = _NumericEncoder(num_feat_dim, cfg.d_model, cfg.gru_layers, cfg.dropout)
        self.news_proj = _NewsProjector(news_emb_dim, cfg.d_model, cfg.dropout)
        self.co = CoAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.head = _Head(2 * cfg.d_model, cfg.mlp_hidden, cfg.out_dim, cfg.dropout)

    def forward(self, numeric, news, news_mask=None, numeric_mask=None):
        tokens = self.encoder(numeric)
        news_tok = self.news_proj(news)
        num_ctx, news_ctx = self.co(tokens, news_tok, news_mask=news_mask, numeric_mask=numeric_mask)
        num_pooled = masked_mean(num_ctx, numeric_mask)
        news_pooled = masked_mean(news_ctx, news_mask)
        fused = torch.cat([num_pooled, news_pooled], dim=-1)
        return self.head(fused)


class CoCrossAttentionModel(nn.Module):
    """
    Two-stage fusion: co-attention (both ways) THEN cross-attention.

    Stage 1 -- co-attention: numeric attends to news AND news attends to numeric,
               giving each stream contextualised by the other (n_co, w_co).
    Stage 2 -- cross-attention: the two co-attended streams are cross-attended
               against each other (n_co queries w_co, and w_co queries n_co).

    Both final streams are pooled and fused. This matches "co-attend text and
    numeric either way, then cross-attend them".
    """
    kind = "cocross"

    def __init__(self, num_feat_dim: int, news_emb_dim: int, cfg: ModelConfig):
        super().__init__()
        self.encoder = _NumericEncoder(num_feat_dim, cfg.d_model, cfg.gru_layers, cfg.dropout)
        self.news_proj = _NewsProjector(news_emb_dim, cfg.d_model, cfg.dropout)
        self.co = CoAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.cross_num = MultiHeadCrossAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.cross_news = MultiHeadCrossAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.head = _Head(2 * cfg.d_model, cfg.mlp_hidden, cfg.out_dim, cfg.dropout)

    def forward(self, numeric, news, news_mask=None, numeric_mask=None):
        tokens = self.encoder(numeric)
        news_tok = self.news_proj(news)
        # Stage 1: co-attention both ways.
        num_co, news_co = self.co(tokens, news_tok, news_mask=news_mask, numeric_mask=numeric_mask)
        # Stage 2: cross-attend the two co-attended streams.
        num_cx, _ = self.cross_num(num_co, news_co, kv_mask=news_mask)
        news_cx, _ = self.cross_news(news_co, num_co, kv_mask=numeric_mask)
        num_pooled = masked_mean(num_cx, numeric_mask)
        news_pooled = masked_mean(news_cx, news_mask)
        fused = torch.cat([num_pooled, news_pooled], dim=-1)
        return self.head(fused)


_REGISTRY = {
    "numeric": NumericOnlyModel,
    "cross": CrossAttentionModel,
    "co": CoAttentionModel,
    "cocross": CoCrossAttentionModel,
}


def build_model(kind: str, num_feat_dim: int, news_emb_dim: int, cfg: ModelConfig = None):
    """Factory: ``kind`` in {'numeric', 'cross', 'co'}."""
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown model kind '{kind}'. Choose from {list(_REGISTRY)}.")
    return _REGISTRY[kind](num_feat_dim, news_emb_dim, cfg or ModelConfig())
