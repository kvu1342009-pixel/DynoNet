# ==============================================================================
# MODULE: WorkerNet - Supreme Worker
# ==============================================================================
# @context: Execution network that obeys Supreme ControlNet
# @goal: Dynamic gating, FiLM, and adaptive dropout
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class AdaptiveDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x: torch.Tensor, dropout_rate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.training:
            return x
        if dropout_rate is None:
            return F.dropout(x, p=0.1, training=True)

        # Per-sample adaptive dropout
        keep_prob = 1.0 - dropout_rate

        # Broadcast keep_prob to match x: (batch, 1) -> (batch, 1, 1) -> (batch, seq, hidden)
        if x.dim() == 3:
            keep_prob = keep_prob.unsqueeze(1)

        mask = torch.bernoulli(keep_prob.expand_as(x))
        return x * mask * (1.0 / (keep_prob + 1e-8))


class WorkerNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        pred_len: int = 96,
        max_seq_len: int = 96,  # For positional encoding
    ):
        super().__init__()

        # PURE SPECIALIST: No Decomp, No Trend.
        # Just focuses on the difficult Residuals.

        # Learnable Positional Encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Residual Branch (The "Artist" - GRU)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.grus = nn.ModuleList(
            [
                nn.GRU(hidden_dim, hidden_dim, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.lns = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.adaptive_dropout = AdaptiveDropout()

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim * pred_len),
        )

        self.pred_len = pred_len
        self.output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        film_params: List[Tuple[torch.Tensor, torch.Tensor]],
        context: torch.Tensor,
        dropout_rate: torch.Tensor,
        gate_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        # x: Residual Input (Batch, Seq, InputDim)

        # 1. Input Project + Positional Encoding
        h = self.input_proj(x)  # (B, Seq, Hidden)
        h = h + self.pos_embedding[:, : x.size(1), :]  # Add position info

        h = h * gate_mask.unsqueeze(1)

        # 2. GRU Layers with FiLM & Adaptive Dropout
        for i, (gru, ln) in enumerate(zip(self.grus, self.lns)):
            h_out, _ = gru(h)

            # FiLM
            gamma, beta = film_params[i]
            h_out = h_out * gamma.unsqueeze(1) + beta.unsqueeze(1)

            # LayerNorm + Dropout
            h_out = ln(h_out)
            h_out = self.adaptive_dropout(h_out, dropout_rate)

            # Residual Connection
            if i > 0:
                h_out = h_out + h
            h = h_out

        # 3. Output Fusion
        last_h = h[:, -1, :]
        combined = torch.cat([last_h, context], dim=-1)
        res_out = self.output_proj(combined).view(
            x.size(0), self.pred_len, self.output_dim
        )

        return res_out
