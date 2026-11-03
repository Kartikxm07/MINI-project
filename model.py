"""
Neural Network Architectures for Option Pricing & Error Correction.
Includes:
1. SimpleOptionANN: Dense -> ReLU -> Dense -> ReLU -> Output (as specified by Member 3).
2. ResidualOptionANN: Dedicated network learning the Black-Scholes error E = C_market - C_bs.
3. OptionLSTM: Sequence-aware LSTM combining historical price dynamics with contract features.
4. HybridShvimerSubModel1: Exact architecture from Shvimer & Zhu (2024) research paper.
5. VolatilityNet: Deep-learning dynamic volatility model sigma_DL = f_theta(X) (Objective 2),
   combined with a differentiable Black-Scholes to form the Hybrid pricer
   C_hybrid = BS(S, K, T, r, sigma_DL) (Objective 3).
"""

import math

import torch
import torch.nn as nn


class SimpleOptionANN(nn.Module):
    """
    Simple Feedforward Artificial Neural Network (ANN) as requested:
    
    Input features (e.g., S, K, T, r, sigma, or S/K, T, r, sigma)
          |
        Dense (hidden_dim1, default 64)
          |
        ReLU
          |
        Dense (hidden_dim2, default 32)
          |
        ReLU
          |
        Output Dense (1 unit)
    
    Can be trained to predict:
    1. Direct Market Price: X -> C_market
    2. Residual Error:      X -> E
    """
    def __init__(self, input_dim: int = 5, hidden_dim1: int = 64, hidden_dim2: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualOptionANN(nn.Module):
    """
    Residual Option Pricing Network:
    Directly models the pricing discrepancy of Black-Scholes:
        E = C_market - C_bs
        E_hat = f_theta(X)
        C_pred = C_bs + E_hat
    """
    def __init__(self, input_dim: int = 5, hidden_dim1: int = 64, hidden_dim2: int = 32):
        super().__init__()
        self.error_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts the pricing error E_hat."""
        return self.error_net(x)

    def predict_price(self, x: torch.Tensor, c_bs: torch.Tensor) -> torch.Tensor:
        """Computes C_pred = C_bs + E_hat, enforcing non-negativity."""
        e_hat = self.forward(x)
        if c_bs.ndim == 1:
            c_bs = c_bs.unsqueeze(-1)
        if c_bs.shape != e_hat.shape:
            raise ValueError("c_bs must contain one baseline price per input row")
        c_pred = c_bs + e_hat
        return torch.clamp(c_pred, min=0.0)


class OptionLSTM(nn.Module):
    """
    Sequence-Aware LSTM for Option Pricing.
    Processes rolling temporal window of underlying stock movements (seq_len, seq_dim)
    and merges temporal representations with static contract parameters (static_dim).
    """
    def __init__(
        self,
        seq_input_dim: int = 2,
        static_input_dim: int = 5,
        lstm_hidden: int = 64,
        dense_hidden: int = 32,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        # LSTM processes past price & return trajectories
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout_prob
        )
        
        # Merge LSTM output with static contract features [S/K, T, r, sigma_hist, C_bs/K]
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + static_input_dim, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(dense_hidden, 1)
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        # x_seq: (batch_size, seq_len, seq_input_dim)
        lstm_out, (h_n, _) = self.lstm(x_seq)
        # Take the final hidden state of the top LSTM layer
        last_hidden = h_n[-1]  # (batch_size, lstm_hidden)
        
        # Concatenate temporal embedding with static contract terms
        merged = torch.cat([last_hidden, x_static], dim=-1)
        output = self.head(merged)
        return output


class HybridShvimerSubModel1(nn.Module):
    """
    Exact Topology of Sub-model 1 from Shvimer & Zhu (2024):
    - Input: [S/K, r, sigma, T, C_bs/K]
    - Layer 1: Dense (40 units) + Sigmoid
    - Dropout: 10%
    - Layer 2: Dense (40 units) + Sigmoid
    - Dropout: 10%
    - Output: Dense (1 unit) + Linear activation
    - Scaled target: C_market / K
    """
    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.Sigmoid(),
            nn.Dropout(0.10),
            nn.Linear(40, 40),
            nn.Sigmoid(),
            nn.Dropout(0.10),
            nn.Linear(40, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def black_scholes_call_torch(S, K, T, r, sigma, eps_T: float = 1e-6, eps_sigma: float = 1e-6):
    """
    Differentiable Black-Scholes European call price in PyTorch.

    Mirrors ``src.pricing.black_scholes_call`` but keeps the full autograd graph,
    so gradients can flow back into a *predicted* volatility. This is what lets
    VolatilityNet be trained end-to-end on observed/market prices (the hybrid
    model of Objective 3). The standard normal CDF is evaluated with ``torch.erf``:
        N(x) = 0.5 * (1 + erf(x / sqrt(2)))
    All inputs broadcast and may be scalars or tensors.
    """
    S = torch.clamp(S, min=1e-8)
    K = torch.clamp(K, min=1e-8)
    T = torch.clamp(T, min=eps_T)
    sigma = torch.clamp(sigma, min=eps_sigma)

    sqrt_T = torch.sqrt(T)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    n_d1 = 0.5 * (1.0 + torch.erf(d1 * inv_sqrt2))
    n_d2 = 0.5 * (1.0 + torch.erf(d2 * inv_sqrt2))
    price = S * n_d1 - K * torch.exp(-r * T) * n_d2
    # European call can never be worth less than its discounted intrinsic value.
    intrinsic = torch.clamp(S - K * torch.exp(-r * T), min=0.0)
    return torch.maximum(price, intrinsic)


class VolatilityNet(nn.Module):
    """
    Deep-Learning Dynamic Volatility Model (Objective 2): sigma_DL = f_theta(X).

    Estimates a contract-specific volatility from historical/market features X
    (e.g. [S/K, T, r, sigma_hist]) instead of assuming the single constant
    volatility of classical Black-Scholes. The raw network output is squashed
    into a strictly positive, bounded range so Black-Scholes stays well defined:

        sigma_DL = sigma_min + (sigma_max - sigma_min) * sigmoid(raw_output)

    Paired with ``black_scholes_call_torch`` via :meth:`hybrid_price`, it forms
    the Hybrid pricer (Objective 3):

        C_hybrid = BS(S, K, T, r, sigma_DL)

    which relaxes Black-Scholes' constant-volatility assumption while keeping the
    interpretable, no-arbitrage structure of the closed-form formula.
    """
    def __init__(self, input_dim: int = 4, hidden_dim1: int = 64, hidden_dim2: int = 32,
                 sigma_min: float = 0.01, sigma_max: float = 1.5):
        super().__init__()
        if not (0.0 <= sigma_min < sigma_max):
            raise ValueError("Require 0 <= sigma_min < sigma_max")
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the predicted volatility sigma_DL, shape (batch, 1)."""
        raw = self.network(x)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw)

    def hybrid_price(self, x: torch.Tensor, S: torch.Tensor, K: torch.Tensor,
                     T: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Objective 3 forward pass: predict sigma_DL from x, then price the contract
        with differentiable Black-Scholes. Returns the hybrid call price, shape (batch,).
        S, K, T, r are the *raw* (unscaled) contract parameters.
        """
        sigma = self.forward(x).squeeze(-1)
        return black_scholes_call_torch(S, K, T, r, sigma)
