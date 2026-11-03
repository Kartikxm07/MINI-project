"""
Tests for the news + cross/co-attention stack: attention primitives, fusion
models, the news layer (providers/embeddings/alignment), and the shared trainer.

All tests are offline: they use the dependency-free HashingEmbedder and never
hit the network or download FinBERT.
"""

import os
import unittest

import numpy as np
import pandas as pd
import torch

from src.attention.modules import masked_mean, MultiHeadCrossAttention, CoAttention
from src.attention.fusion_models import build_model, ModelConfig
from src.attention.trainer import set_seed, train_model, predict, evaluate_regression
from src.news.providers import NewsItem, get_provider, YFinanceNewsProvider, FinnhubNewsProvider
from src.news.embeddings import get_embedder, HashingEmbedder
from src.news.aggregate import group_by_date, build_news_tensor, synthetic_news_embeddings


class TestAttentionModules(unittest.TestCase):
    def test_masked_mean_ignores_padding(self):
        x = torch.tensor([[[1.0], [3.0], [100.0]]])        # (1, 3, 1)
        mask = torch.tensor([[True, True, False]])
        self.assertAlmostEqual(float(masked_mean(x, mask)), 2.0, places=5)

    def test_masked_mean_all_padded_is_zero(self):
        x = torch.randn(2, 4, 5)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        out = masked_mean(x, mask)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_cross_attention_shapes_and_mask(self):
        attn = MultiHeadCrossAttention(embed_dim=16, num_heads=4)
        q = torch.randn(3, 5, 16)
        kv = torch.randn(3, 7, 16)
        mask = torch.ones(3, 7, dtype=torch.bool)
        mask[0] = False                                    # fully-padded row must not NaN
        out, weights = attn(q, kv, kv_mask=mask)
        self.assertEqual(out.shape, (3, 5, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_co_attention_returns_both_streams(self):
        co = CoAttention(embed_dim=16, num_heads=4)
        numeric = torch.randn(2, 6, 16)
        news = torch.randn(2, 4, 16)
        news_mask = torch.ones(2, 4, dtype=torch.bool)
        n_ctx, w_ctx = co(numeric, news, news_mask=news_mask)
        self.assertEqual(n_ctx.shape, numeric.shape)
        self.assertEqual(w_ctx.shape, news.shape)


class TestFusionModels(unittest.TestCase):
    def test_all_models_forward_backward(self):
        B, Ln, Fn, Lw, Fw = 8, 12, 4, 5, 32
        numeric = torch.randn(B, Ln, Fn)
        news = torch.randn(B, Lw, Fw)
        mask = torch.ones(B, Lw, dtype=torch.bool)
        mask[0] = False
        cfg = ModelConfig(d_model=32, num_heads=4, mlp_hidden=32)
        for kind in ["numeric", "cross", "co", "cocross"]:
            model = build_model(kind, Fn, Fw, cfg)
            out = model(numeric, news, mask)
            self.assertEqual(out.shape, (B, 1))
            out.pow(2).mean().backward()
            self.assertTrue(torch.isfinite(out).all())

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            build_model("bogus", 4, 8)

    def test_patch_coattention_forward_backward(self):
        from src.attention.patch_coattn import PatchCoAttentionNet
        m = PatchCoAttentionNet(price_dim=3, news_dim=32, d_model=32, num_heads=4,
                                patch_len=4, stride=2)
        price = torch.randn(6, 20, 3)
        news = torch.randn(6, 5, 32)
        mask = torch.ones(6, 5, dtype=torch.bool)
        mask[0] = False                                  # a no-news sample must not NaN
        out = m(price, news, mask)
        self.assertEqual(out.shape, (6, 1))
        out.pow(2).mean().backward()
        self.assertTrue(torch.isfinite(out).all())


class TestNewsLayer(unittest.TestCase):
    def test_provider_selection_defaults_to_yfinance(self):
        for var in ("FINNHUB_API_KEY", "ALPHAVANTAGE_API_KEY", "MARKETAUX_API_KEY"):
            os.environ.pop(var, None)
        self.assertIsInstance(get_provider(), YFinanceNewsProvider)

    def test_provider_selection_prefers_key(self):
        os.environ["FINNHUB_API_KEY"] = "dummy"
        try:
            self.assertIsInstance(get_provider(), FinnhubNewsProvider)
        finally:
            os.environ.pop("FINNHUB_API_KEY", None)

    def test_keyless_finnhub_returns_empty(self):
        # No key -> graceful empty, no network call.
        self.assertEqual(FinnhubNewsProvider(api_key="").fetch("TCS", "2024-01-01", "2024-02-01"), [])

    def test_news_item_normalizes_timestamp(self):
        it = NewsItem(pd.Timestamp("2024-03-04 09:30"), "headline")
        self.assertEqual(str(it.date), "2024-03-04")

    def test_hashing_embedder_is_deterministic(self):
        emb = get_embedder("hashing", dim=48)
        a = emb.encode(["profit beats estimates", "regulator opens probe"])
        b = emb.encode(["profit beats estimates", "regulator opens probe"])
        self.assertEqual(a.shape, (2, 48))
        self.assertTrue(np.allclose(a, b))

    def test_build_news_tensor_respects_lookback_no_leakage(self):
        emb = HashingEmbedder(dim=16)
        items = [NewsItem(pd.Timestamp("2024-01-02"), "a"),
                 NewsItem(pd.Timestamp("2024-01-02"), "b"),
                 NewsItem(pd.Timestamp("2024-01-10"), "c")]
        by_date = group_by_date(items)
        dates = pd.to_datetime(["2024-01-03", "2024-01-20"])
        ne, nm = build_news_tensor(dates, by_date, emb, max_headlines=4, lookback_days=3)
        self.assertEqual(ne.shape, (2, 4, 16))
        self.assertEqual(int(nm[0].sum()), 2)    # 01-02 headlines are within 3 days of 01-03
        self.assertEqual(int(nm[1].sum()), 0)    # nothing in the trailing window of 01-20

    def test_synthetic_news_correlates_with_signal(self):
        sig = np.random.default_rng(0).standard_normal(400)
        ne, nm, heads = synthetic_news_embeddings(sig, dim=16, max_headlines=5,
                                                  strength=1.0, noise=1.0, coverage=0.9, seed=0)
        axis0 = np.array([ne[i, nm[i], 0].mean() if nm[i].any() else 0.0 for i in range(len(sig))])
        have = nm.any(1)
        corr = np.corrcoef(sig[have], axis0[have])[0, 1]
        self.assertGreater(corr, 0.5)
        self.assertEqual(len(heads), len(sig))


class TestTrainer(unittest.TestCase):
    def test_training_reduces_val_loss_and_predicts(self):
        set_seed(0)
        rng = np.random.default_rng(0)
        N, Ln, Fn, Lw, Fw = 240, 8, 3, 4, 16
        numeric = rng.standard_normal((N, Ln, Fn)).astype(np.float32)
        news = rng.standard_normal((N, Lw, Fw)).astype(np.float32)
        mask = np.ones((N, Lw), dtype=bool)
        # Target is a learnable function of the numeric features.
        y = (numeric[:, -1, 0] * 2.0 - numeric[:, -1, 1]).astype(np.float32).reshape(-1, 1)

        def pack(sl):
            return {"numeric": numeric[sl], "news_emb": news[sl],
                    "news_mask": mask[sl], "y": y[sl]}
        train_d, val_d, test_d = pack(slice(0, 160)), pack(slice(160, 200)), pack(slice(200, N))

        model = build_model("co", Fn, Fw, ModelConfig(d_model=32, num_heads=4, mlp_hidden=32))
        model, hist, best_val = train_model(model, train_d, val_d, epochs=20, lr=5e-3, batch_size=32)
        # Training must improve on the starting point; the trainer restores the
        # best-validation checkpoint, so compare that (robust to a noisy last epoch).
        self.assertLess(best_val, hist["val"][0])
        self.assertLessEqual(best_val, min(hist["val"]) + 1e-9)
        yhat = predict(model, test_d)
        self.assertEqual(yhat.shape, (len(y[200:N]), 1))
        self.assertTrue(np.isfinite(yhat).all())

    def test_evaluate_regression_keys(self):
        m = evaluate_regression([0.1, -0.2, 0.3], [0.1, -0.1, 0.2])
        for k in ["RMSE", "MAE", "R2", "DirAcc"]:
            self.assertIn(k, m)


if __name__ == "__main__":
    unittest.main()
