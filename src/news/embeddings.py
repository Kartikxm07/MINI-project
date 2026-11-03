"""
Turn news headlines into fixed-length vectors for the attention models.

``FinBERTEmbedder`` uses the pretrained ProsusAI/finbert transformer (financial
BERT) and mean-pools its last hidden state. Embeddings are cached on disk keyed
by a hash of the text, so the model is loaded once and each unique headline is
encoded once across runs.

``HashingEmbedder`` is a dependency-free, offline fallback (deterministic hashed
bag-of-words). It carries no real semantics but lets tests and no-internet
environments exercise the full pipeline. Both expose the same interface:

    embedder.dim                      -> int
    embedder.encode(list[str])        -> np.ndarray (n, dim)
"""

import hashlib
import os
from pathlib import Path

import numpy as np


class TextEmbedder:
    dim = 0

    def encode(self, texts):
        raise NotImplementedError


class HashingEmbedder(TextEmbedder):
    """Deterministic hashed-token embedding. Offline, no model download."""

    def __init__(self, dim: int = 64):
        self.dim = int(dim)

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in str(text)).split()
        if not tokens:
            return vec
        for tok in tokens:
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "little")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts):
        return np.stack([self._encode_one(t) for t in texts]).astype(np.float32)


class FinBERTEmbedder(TextEmbedder):
    """Pretrained FinBERT sentence embeddings with on-disk caching."""

    dim = 768

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = None,
                 cache_dir: str = None, max_length: int = 64, batch_size: int = 16):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir or (Path.home() / ".cache" / "stocksight_news_emb"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = None
        self._model = None
        self._device = device

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModel
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self._device).eval()

    def _cache_path(self, text: str) -> Path:
        key = hashlib.sha1(f"{self.model_name}:{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.npy"

    def encode(self, texts):
        texts = [str(t) for t in texts]
        out = [None] * len(texts)
        missing_idx, missing_txt = [], []
        for i, t in enumerate(texts):
            p = self._cache_path(t)
            if p.exists():
                try:
                    out[i] = np.load(p)
                    continue
                except Exception:
                    pass
            missing_idx.append(i)
            missing_txt.append(t)

        if missing_txt:
            import torch
            self._ensure_model()
            for b in range(0, len(missing_txt), self.batch_size):
                batch = missing_txt[b:b + self.batch_size]
                enc = self._tokenizer(batch, padding=True, truncation=True,
                                      max_length=self.max_length, return_tensors="pt").to(self._device)
                with torch.no_grad():
                    hidden = self._model(**enc).last_hidden_state      # (B, L, 768)
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
                pooled = pooled.cpu().numpy().astype(np.float32)
                for j, vec in enumerate(pooled):
                    gi = missing_idx[b + j]
                    out[gi] = vec
                    try:
                        np.save(self._cache_path(missing_txt[b + j]), vec)
                    except Exception:
                        pass
        return np.stack(out).astype(np.float32)


def get_embedder(kind: str = "finbert", **kwargs) -> TextEmbedder:
    """``kind`` in {'finbert', 'hashing'}."""
    kind = (kind or "finbert").lower()
    if kind == "finbert":
        return FinBERTEmbedder(**kwargs)
    if kind == "hashing":
        return HashingEmbedder(**kwargs)
    raise ValueError(f"Unknown embedder '{kind}'. Choose 'finbert' or 'hashing'.")
