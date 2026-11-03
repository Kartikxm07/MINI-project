"""
Align news with numeric samples and build padded embedding tensors.

For each sample date we gather the headlines published within a trailing window
ending on that date (``[date - lookback_days + 1, date]``) so no future text
leaks into the features, embed them, and pad/truncate to a fixed number of
headlines per sample. The result is ready for the attention models:

    news_emb  : (N, max_headlines, dim) float32
    news_mask : (N, max_headlines)      bool   (True = real headline)
"""

from collections import defaultdict

import numpy as np
import pandas as pd


def group_by_date(items) -> dict:
    """Group ``NewsItem`` objects into ``{date: [headline, ...]}``."""
    by_date = defaultdict(list)
    for it in items:
        by_date[pd.to_datetime(it.date).normalize()].append(it.headline)
    return dict(by_date)


def build_news_tensor(sample_dates, news_by_date, embedder,
                      max_headlines: int = 8, lookback_days: int = 3):
    """
    Build padded news-embedding tensors aligned to ``sample_dates``.

    ``news_by_date`` maps a normalized Timestamp to a list of headline strings.
    Only headlines on/before each sample date (within ``lookback_days``) are used.
    """
    sample_dates = [pd.to_datetime(d).normalize() for d in sample_dates]
    norm_by_date = {pd.to_datetime(k).normalize(): v for k, v in news_by_date.items()}

    # Encode every unique headline once.
    unique = sorted({h for v in norm_by_date.values() for h in v})
    lookup = {}
    if unique:
        vecs = embedder.encode(unique)
        lookup = {h: vecs[i] for i, h in enumerate(unique)}

    dim = embedder.dim
    n = len(sample_dates)
    news_emb = np.zeros((n, max_headlines, dim), dtype=np.float32)
    news_mask = np.zeros((n, max_headlines), dtype=bool)

    all_days = sorted(norm_by_date.keys())
    all_days_arr = np.array([d.value for d in all_days]) if all_days else np.array([])

    for i, d in enumerate(sample_dates):
        if all_days_arr.size == 0:
            continue
        lo = (d - pd.Timedelta(days=lookback_days - 1)).value
        hi = d.value
        # Collect headlines in the trailing window, most recent first.
        window_headlines = []
        for day in all_days:
            if lo <= day.value <= hi:
                window_headlines.extend([(day, h) for h in norm_by_date[day]])
        window_headlines.sort(key=lambda x: x[0].value, reverse=True)
        for slot, (_, h) in enumerate(window_headlines[:max_headlines]):
            vec = lookup.get(h)
            if vec is not None:
                news_emb[i, slot] = vec
                news_mask[i, slot] = True
    return news_emb, news_mask


_POS_TEMPLATES = [
    "{s} surges as quarterly profit beats estimates",
    "Analysts upgrade {s} on strong demand outlook",
    "{s} rallies after record revenue guidance",
    "Investors cheer {s} expansion plans",
]
_NEG_TEMPLATES = [
    "{s} slides on disappointing earnings",
    "Analysts cut {s} target amid margin pressure",
    "{s} falls as regulator opens probe",
    "Weak demand clouds {s} outlook",
]
_NEU_TEMPLATES = [
    "{s} trades flat ahead of results",
    "{s} in focus as sector awaits policy cues",
    "{s} holds steady in choppy session",
]


def synthetic_news_embeddings(signal, dim: int = 768, max_headlines: int = 8,
                              strength: float = 1.0, noise: float = 1.0,
                              coverage: float = 0.8, seed: int = 42, symbol: str = "STOCK"):
    """
    Fabricate news embeddings that correlate with ``signal`` (e.g. the target).

    Used for the offline / no-API-key demo: a single embedding axis encodes a
    (standardized) sentiment proportional to ``signal`` plus Gaussian noise, so
    news-aware attention models *can* exploit the text signal while numeric-only
    models cannot. ``coverage`` is the fraction of samples that have any news
    (the rest are left empty to mimic the sparsity of real feeds). Returns
    ``(news_emb, news_mask, headlines)`` where ``headlines`` is a list (per
    sample) of the generated headline strings, for inspection.

    NOTE: in synthetic mode the embedding is generated directly; the headline
    strings are cosmetic. Real mode (a keyed provider + FinBERT) embeds actual text.
    """
    rng = np.random.default_rng(seed)
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = len(signal)
    std = signal.std()
    z = (signal - signal.mean()) / std if std > 1e-12 else np.zeros_like(signal)

    news_emb = np.zeros((n, max_headlines, dim), dtype=np.float32)
    news_mask = np.zeros((n, max_headlines), dtype=bool)
    headlines = [[] for _ in range(n)]

    for i in range(n):
        if rng.random() > coverage:
            continue  # no news this sample
        k = int(rng.integers(1, max_headlines + 1))
        for slot in range(k):
            vec = rng.standard_normal(dim).astype(np.float32) * noise
            # Axis 0 carries the sentiment signal (+ per-headline jitter).
            sentiment = strength * z[i] + 0.25 * rng.standard_normal()
            vec[0] = sentiment
            news_emb[i, slot] = vec
            news_mask[i, slot] = True
            if sentiment > 0.3:
                tmpl = _POS_TEMPLATES[rng.integers(len(_POS_TEMPLATES))]
            elif sentiment < -0.3:
                tmpl = _NEG_TEMPLATES[rng.integers(len(_NEG_TEMPLATES))]
            else:
                tmpl = _NEU_TEMPLATES[rng.integers(len(_NEU_TEMPLATES))]
            headlines[i].append(tmpl.format(s=symbol))
    return news_emb, news_mask, headlines
