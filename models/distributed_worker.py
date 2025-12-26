import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from .worker_net import WorkerNet


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        x: [Batch, Seq, Channel]
        """
        # Padding to keep length same
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)

        x_uv = x_pad.permute(0, 2, 1)  # (B, C, Seq)
        x_trend = self.moving_avg(x_uv).permute(0, 2, 1)  # (B, Seq, C)
        x_res = x - x_trend
        return x_res, x_trend


class DistributedWorker(nn.Module):
    def __init__(
        self,
        num_channels: int = 7,  # 7 Features
        hidden_dim: int = 16,  # Tiny hidden dim per worker
        num_layers: int = 1,
        dropout: float = 0.1,
        pred_len: int = 96,
        seq_len: int = 336,  # Long lookback for Trend
        worker_seq_len: int = 96,  # Short lookback for GRU
    ):
        super().__init__()
        self.num_channels = num_channels
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.worker_seq_len = worker_seq_len

        # 1. Decomposition (Shared Logic)
        self.decomp = SeriesDecomp(kernel_size=25)

        # 2. Shared Trend Backbone (DLinear Style)
        # One Linear Layer SHARED across all channels
        # Input: (B, C, Seq) -> Output: (B, C, Pred)
        # Using Full Long Lookback
        self.shared_trend_linear = nn.Linear(seq_len, pred_len)

        # 3. Independent Residual Specialists
        # 7 independent tiny workers dealing with residuals
        # Using Short Lookback
        self.workers = nn.ModuleList(
            [
                WorkerNet(
                    input_dim=1,
                    hidden_dim=hidden_dim,
                    output_dim=1,
                    num_layers=num_layers,
                    dropout=dropout,
                    pred_len=pred_len,
                )
                for _ in range(num_channels)
            ]
        )

        # 4. Deeper Channel Mixer - Learn complex cross-channel interactions
        # MLP with expansion, GELU, and residual connection
        mixer_expansion = 4  # Expand channels by 4x
        self.channel_mixer = nn.Sequential(
            nn.Linear(num_channels, num_channels * mixer_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels * mixer_expansion, num_channels),
        )
        # Initialize last layer close to zero for residual-friendly start
        nn.init.zeros_(self.channel_mixer[-1].weight)
        nn.init.zeros_(self.channel_mixer[-1].bias)

    def forward(
        self,
        x: torch.Tensor,  # (Batch, Seq, 7)
        film_params_list: List[
            List[Tuple[torch.Tensor, torch.Tensor]]
        ],  # List of 7 film params sets
        context: torch.Tensor,  # Global context (Batch, GlobalHidden)
        dropout_rate: torch.Tensor,
        gate_masks: torch.Tensor,  # (Batch, 7, Hidden)
        **kwargs
    ) -> torch.Tensor:

        # 1. Decompose Input (Full Sequence)
        x_res, x_trend = self.decomp(x)

        # 2. Process Trend (Shared Backbone - Long Lookback)
        # x_trend: (B, Seq, C) -> Permute -> (B, C, Seq)
        trend_out = self.shared_trend_linear(x_trend.permute(0, 2, 1))  # (B, C, Pred)
        trend_out = trend_out.permute(0, 2, 1)  # (B, Pred, C)

        # 3. Process Residuals (Independent Specialists - Short Lookback)
        # Slice the last 96 steps for GRU workers
        x_res_short = x_res[:, -self.worker_seq_len :, :]  # (Batch, 96, 7)

        res_outputs = []
        for i, worker in enumerate(self.workers):
            # Slice residual for this channel
            res_i = x_res_short[:, :, i : i + 1]  # (Batch, 96, 1)

            # Get controls
            worker_film = film_params_list[i]
            mask_i = gate_masks[:, i, :]
            drop_i = dropout_rate[:, i : i + 1]

            # Forward Specialist
            out_i = worker(
                res_i,
                film_params=worker_film,
                context=context,
                dropout_rate=drop_i,
                gate_mask=mask_i,
            )
            res_outputs.append(out_i)

        res_out_cat = torch.cat(res_outputs, dim=-1)  # (B, Pred, C)

        # 4. Final Fusion
        combined = trend_out + res_out_cat  # (B, Pred, C)

        # 5. Channel Mixer - Cross-channel interaction with residual
        # (B, Pred, C) -> MLP on last dim -> (B, Pred, C)
        output = combined + self.channel_mixer(combined)  # Residual connection

        return output
