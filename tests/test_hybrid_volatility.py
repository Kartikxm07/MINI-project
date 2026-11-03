"""
Tests for the Deep-Learning Volatility model (Objective 2) and the Hybrid
Black-Scholes pricer (Objective 3).
"""

import unittest

import numpy as np
import torch

from src.pricing import black_scholes_call
from model import (VolatilityNet, black_scholes_call_torch,
                   HybridShvimerSubModel1, HybridShvimerSubModel2)


class TestDifferentiableBlackScholes(unittest.TestCase):
    def test_matches_numpy_black_scholes(self):
        # The torch BS must agree with the analytic numpy implementation.
        S = torch.tensor([100.0, 105.0, 90.0])
        K = torch.tensor([100.0, 100.0, 95.0])
        T = torch.tensor([1.0, 0.5, 0.25])
        r = torch.tensor([0.05, 0.03, 0.02])
        sigma = torch.tensor([0.20, 0.25, 0.30])

        torch_price = black_scholes_call_torch(S, K, T, r, sigma).numpy()
        numpy_price = black_scholes_call(S.numpy(), K.numpy(), T.numpy(), r.numpy(), sigma.numpy())
        np.testing.assert_allclose(torch_price, numpy_price, rtol=1e-5, atol=1e-4)

    def test_known_value(self):
        price = black_scholes_call_torch(
            torch.tensor(100.0), torch.tensor(100.0),
            torch.tensor(1.0), torch.tensor(0.05), torch.tensor(0.20)
        )
        self.assertAlmostEqual(float(price), 10.4506, places=3)

    def test_gradient_flows_into_sigma(self):
        sigma = torch.tensor([0.2], requires_grad=True)
        price = black_scholes_call_torch(
            torch.tensor([100.0]), torch.tensor([100.0]),
            torch.tensor([1.0]), torch.tensor([0.05]), sigma
        )
        price.sum().backward()
        self.assertIsNotNone(sigma.grad)
        # Vega is strictly positive, so the price increases with volatility.
        self.assertGreater(float(sigma.grad), 0.0)


class TestVolatilityNet(unittest.TestCase):
    def test_output_is_bounded_and_positive(self):
        net = VolatilityNet(input_dim=4, sigma_min=0.01, sigma_max=1.5)
        x = torch.randn(32, 4) * 5.0  # large inputs to probe the bounds
        sigma = net(x)
        self.assertEqual(sigma.shape, (32, 1))
        self.assertTrue(torch.all(sigma > 0.01 - 1e-6))
        self.assertTrue(torch.all(sigma < 1.5 + 1e-6))

    def test_hybrid_price_is_finite_and_nonneg(self):
        net = VolatilityNet(input_dim=4)
        x = torch.randn(16, 4)
        S = torch.full((16,), 100.0)
        K = torch.full((16,), 100.0)
        T = torch.full((16,), 0.5)
        r = torch.full((16,), 0.03)
        price = net.hybrid_price(x, S, K, T, r)
        self.assertEqual(price.shape, (16,))
        self.assertTrue(torch.all(torch.isfinite(price)))
        self.assertTrue(torch.all(price >= 0.0))

    def test_end_to_end_training_reduces_price_error(self):
        # One contract family where true vol (0.35) differs from any constant
        # guess; training through BS should recover a better fit than the
        # randomly initialised network.
        torch.manual_seed(0)
        n = 256
        S = torch.full((n,), 100.0)
        K = 100.0 + torch.linspace(-10, 10, n)
        T = torch.full((n,), 0.5)
        r = torch.full((n,), 0.03)
        true_sigma = torch.full((n,), 0.35)
        target = black_scholes_call_torch(S, K, T, r, true_sigma).detach()

        # Features: [S/K, T, r, rough historical vol guess]
        x = torch.stack([S / K, T, r, torch.full((n,), 0.20)], dim=1)

        net = VolatilityNet(input_dim=4)
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        with torch.no_grad():
            start = loss_fn(net.hybrid_price(x, S, K, T, r) / K, target / K).item()

        for _ in range(200):
            opt.zero_grad()
            pred = net.hybrid_price(x, S, K, T, r)
            loss = loss_fn(pred / K, target / K)
            loss.backward()
            opt.step()

        with torch.no_grad():
            end = loss_fn(net.hybrid_price(x, S, K, T, r) / K, target / K).item()
            mean_sigma = net(x).mean().item()

        self.assertLess(end, start * 0.5)      # error at least halved
        self.assertAlmostEqual(mean_sigma, 0.35, delta=0.05)  # recovered true vol


class TestShvimerHybrids(unittest.TestCase):
    def test_submodel1_shape(self):
        m = HybridShvimerSubModel1(input_dim=5)
        out = m(torch.randn(10, 5))
        self.assertEqual(out.shape, (10, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_submodel2_fuses_bs_price(self):
        m = HybridShvimerSubModel2(input_dim=4)
        x = torch.randn(10, 4)
        c_bs = torch.rand(10, 1)
        out = m(x, c_bs)
        self.assertEqual(out.shape, (10, 1))
        out.pow(2).mean().backward()
        self.assertTrue(torch.isfinite(out).all())


if __name__ == '__main__':
    unittest.main()
