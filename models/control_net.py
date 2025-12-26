# ==============================================================================
# MODULE: ControlNet - Supreme Controller
# ==============================================================================
# @context: Full control over WorkerNet's architecture and training dynamics
# @goal: Generate modulation for weights, LR, dropout, weight decay, and gating
# @constraint: Keep meta-parameters bounded for training stability
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class ControlGRU(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return output[:, -1, :]  # Global context


class SupremeSignalGenerator(nn.Module):
    """Generates a specific control signal bounded in a range."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        min_val: float,
        max_val: float,
        init_val: float,
    ):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.gen = nn.Linear(input_dim, output_dim)

        # Initialize to specific value
        # sigmoid(x) * (max - min) + min = init
        # x = logit((init - min) / (max - min))
        target_logit = torch.logit(
            torch.tensor((init_val - min_val) / (max_val - min_val))
        )
        nn.init.zeros_(self.gen.weight)
        nn.init.constant_(self.gen.bias, target_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_val + (self.max_val - self.min_val) * torch.sigmoid(self.gen(x))


class ControlNet(nn.Module):
    """
    The Brain - Now controls 7 Independent Workers.
    """

    def __init__(
        self,
        input_dim: int = 7,
        control_hidden: int = 32,
        worker_hidden: int = 16,  # Per-worker hidden dim
        num_worker_layers: int = 1,
        dropout: float = 0.2,
        num_channels: int = 7,
    ):
        super().__init__()

        self.brain = ControlGRU(input_dim, control_hidden, dropout=dropout)
        self.num_channels = num_channels
        self.worker_hidden = worker_hidden
        self.num_layers = num_worker_layers

        # 1. Architecture Modulation (FiLM) - Independent for each channel
        # We need to generate controls for ALL workers
        # Output dim = num_channels * (worker_hidden * 2)
        self.film_gens = nn.ModuleList(
            [
                nn.Linear(control_hidden, num_channels * worker_hidden * 2)
                for _ in range(num_worker_layers)
            ]
        )
        for g in self.film_gens:
            nn.init.zeros_(g.weight)
            nn.init.zeros_(g.bias)

        # 2. Training Dynamics Control (Personalized Management)
        # These are now individualized policies for each channel (Batch, Channels)
        self.lr_ctrl = SupremeSignalGenerator(
            control_hidden, num_channels, 0.01, 5.0, 1.0
        )
        self.wd_ctrl = SupremeSignalGenerator(
            control_hidden, num_channels, 0.1, 10.0, 1.0
        )
        self.dropout_ctrl = SupremeSignalGenerator(
            control_hidden, num_channels, 0.0, 0.7, 0.1
        )
        self.grad_clip_ctrl = SupremeSignalGenerator(
            control_hidden, num_channels, 0.1, 2.0, 1.0
        )

        # New: Loss Attention Weight ("Can thiệp 1 chút")
        # Gives controller ability to focus on specific channels
        self.loss_weight_ctrl = SupremeSignalGenerator(
            control_hidden, num_channels, 0.5, 1.5, 1.0
        )

        # New: Worker Freezing Control (Stop Learning)
        # 1.0 = Freeze completely, 0.0 = Learn normally
        self.freeze_ctrl = SupremeSignalGenerator(
            control_hidden, num_channels, 0.0, 1.0, 0.0
        )

        # 5. Learned Data Augmentation (Noise Injection)
        # Global noise pattern
        self.noise_ctrl = nn.Linear(control_hidden, input_dim)
        nn.init.zeros_(self.noise_ctrl.weight)
        nn.init.zeros_(self.noise_ctrl.bias)

        # 3. Dynamic Feature Gating (Sparsity) - Independent for each channel
        # Output: (batch, num_channels * worker_hidden)
        self.gate_ctrl = nn.Sequential(
            nn.Linear(control_hidden, num_channels * worker_hidden), nn.Sigmoid()
        )

        # 4. Context Projection - Shared global context
        self.context_proj = nn.Linear(control_hidden, worker_hidden)

    def forward(self, x: torch.Tensor) -> Dict[str, any]:
        ctx = self.brain(x)

        # Generate FiLM: List[List[(gamma, beta)]]
        # Outer list: Layers
        # Inner list: Channels
        film_params_per_channel = [[] for _ in range(self.num_channels)]

        for layer_gen in self.film_gens:
            # (Batch, Channels * Hidden * 2)
            p_all = layer_gen(ctx)
            p_all = p_all.view(x.size(0), self.num_channels, self.worker_hidden * 2)

            for c in range(self.num_channels):
                p_c = p_all[:, c, :]  # (Batch, Hidden*2)
                gamma = 1.0 + 0.1 * torch.tanh(p_c[:, : self.worker_hidden])
                beta = 0.1 * torch.tanh(p_c[:, self.worker_hidden :])
                film_params_per_channel[c].append((gamma, beta))

        gate_masks = self.gate_ctrl(ctx).view(
            x.size(0), self.num_channels, self.worker_hidden
        )

        return {
            "film_params_list": film_params_per_channel,
            "lr_scale": self.lr_ctrl(ctx),
            "wd_scale": self.wd_ctrl(ctx),
            "loss_weight": self.loss_weight_ctrl(ctx),
            "dropout_rate": self.dropout_ctrl(ctx),
            "grad_clip": self.grad_clip_ctrl(ctx),
            "freeze_prob": self.freeze_ctrl(ctx),
            "input_noise": torch.tanh(self.noise_ctrl(ctx)) * 0.5,
            "gate_masks": gate_masks,
            "context": self.context_proj(ctx),
        }
