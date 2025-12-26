# ==============================================================================
# MODULE: DynoNet - Supreme Orchestrator
# ==============================================================================
# @context: Orchestrates Supreme ControlNet and WorkerNet
# @goal: Full control over every aspect of model execution and training
# ==============================================================================

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

from .revin import RevIN
from .control_net import ControlNet
from .distributed_worker import DistributedWorker


class DynoNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        control_hidden: int = 32,
        worker_hidden: int = 16,  # Tiny worker dim
        output_dim: int = 1,
        num_layers: int = 1,  # Tiny depth
        dropout: float = 0.1,
        pred_len: int = 96,
        seq_len: int = 336,
        worker_seq_len: int = 96,
    ):
        super().__init__()

        # 1. Reversible Instance Normalization (The "Anti-Shift" Shield)
        self.revin = RevIN(num_features=input_dim, affine=True)

        # Controller needs regularization too!
        self.control_net = ControlNet(
            input_dim=input_dim,
            control_hidden=control_hidden,
            worker_hidden=worker_hidden,
            num_worker_layers=num_layers,
            dropout=0.2,
            num_channels=input_dim,  # Same as number of workers
        )

        # Distributed Workers (Channel Independent)
        self.worker_net = DistributedWorker(
            num_channels=input_dim,
            hidden_dim=worker_hidden,
            num_layers=num_layers,
            dropout=dropout,
            pred_len=pred_len,
            seq_len=seq_len,
            worker_seq_len=worker_seq_len,
        )

        # Initialize noise generator logic
        self._last_signals = None

    def forward(self, x: torch.Tensor, return_info: bool = False):
        # 0. Normalize Input (RevIN)
        x_norm = self.revin(x, "norm")

        # 1. Controller Brain (sees normalized data)
        ctrl = self.control_net(x_norm)
        self._last_signals = ctrl

        # 2. Logic: Learned Data Augmentation
        x_worker = x_norm
        if self.training:
            noise = ctrl["input_noise"].unsqueeze(1)
            x_worker = x_norm + noise

        # 3. Worker Muscles
        predictions = self.worker_net(
            x_worker,
            film_params_list=ctrl["film_params_list"],
            context=ctrl["context"],
            dropout_rate=ctrl["dropout_rate"],
            gate_masks=ctrl["gate_masks"],
        )

        # 4. Denormalize Prediction (RevIN)
        predictions = self.revin(predictions, "denorm")

        info = None
        if return_info:
            info = {
                "lr_scale_mean": ctrl["lr_scale"].mean().item(),
                "loss_weight_mean": ctrl["loss_weight"].mean().item(),
                "dropout_rate_mean": ctrl["dropout_rate"].mean().item(),
                "wd_scale_mean": ctrl["wd_scale"].mean().item(),
                "grad_clip_mean": ctrl["grad_clip"].mean().item(),
                "freeze_prob_mean": ctrl["freeze_prob"].mean().item(),
                "noise_std": ctrl["input_noise"].std().item(),
                "gate_sparsity": (ctrl["gate_masks"] < 0.5).float().mean().item(),
            }

        return predictions, info

    def get_last_control_signals(self):
        return self._last_signals

    def get_controller_info(self):
        ctrl_params = sum(p.numel() for p in self.control_net.parameters())
        worker_params = sum(p.numel() for p in self.worker_net.parameters())
        return {
            "controller_params": ctrl_params,
            "base_params": worker_params,
            "total_params": ctrl_params + worker_params,
            "controller_ratio": ctrl_params / (ctrl_params + worker_params),
        }
