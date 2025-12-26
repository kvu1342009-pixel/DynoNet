#!/usr/bin/env python3
# ==============================================================================
# MODULE: Main - DynoNet Training on ETTh1
# ==============================================================================
# @context: Train and evaluate DynoNet (ControlNet + WorkerNet) on ETTh1
# @goal: Test dynamic network architecture on H96 prediction task
# ==============================================================================

import torch
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import DynoNet
from data import get_ett_dataloaders
from utils import Trainer, count_parameters, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="DynoNet on ETTh1")

    # Data
    parser.add_argument(
        "--seq_len",
        type=int,
        default=336,
        help="Input sequence length (Long for Trend)",
    )
    parser.add_argument(
        "--worker_seq_len",
        type=int,
        default=96,
        help="Worker input length (Short for GRU)",
    )
    parser.add_argument(
        "--pred_len", type=int, default=96, help="Prediction length (H96)"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")

    # Model
    parser.add_argument(
        "--worker_hidden", type=int, default=8, help="Worker GRU hidden dim"
    )
    parser.add_argument(
        "--control_hidden", type=int, default=64, help="Control GRU hidden dim"
    )
    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of GRU layers"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Max training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )
    parser.add_argument(
        "--target_val_mse",
        type=float,
        default=None,
        help="Target val MSE for early stopping",
    )

    # Misc
    parser.add_argument(
        "--seed", type=int, default=12345, help="Random seed (SOTA: 12345)"
    )

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np

    np.random.seed(seed)


def run_experiment(args):
    print("\n" + "=" * 70)
    print("  DynoNet (ControlNet + WorkerNet) - ETTh1 H96")
    print("=" * 70)

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Load data
    print("\n--- Loading ETTh1 Dataset ---")
    train_loader, val_loader, test_loader = get_ett_dataloaders(
        root_dir="./data",
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Create model
    print("\n--- Creating DynoNet ---")
    model = DynoNet(
        input_dim=7,
        control_hidden=args.control_hidden,
        worker_hidden=args.worker_hidden,
        output_dim=7,  # Multivariate Output
        num_layers=args.num_layers,
        dropout=args.dropout,
        pred_len=args.pred_len,
        seq_len=args.seq_len,
        worker_seq_len=args.worker_seq_len,
    )

    # Print architecture
    ctrl_info = model.get_controller_info()
    print(f"Total Parameters: {ctrl_info['total_params']:,}")
    print(
        f"  ControlNet: {ctrl_info['controller_params']:,} ({ctrl_info['controller_ratio']:.1%})"
    )
    print(f"  WorkerNet: {ctrl_info['base_params']:,}")

    # Create optimizers for Bi-level optimization
    worker_opt = torch.optim.AdamW(
        model.worker_net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ControlNet gets stronger regularization to prevent "gaming" the validation set
    control_opt = torch.optim.AdamW(
        model.control_net.parameters(),
        lr=args.lr * 0.05,  # Slower learning
        weight_decay=args.weight_decay * 10.0,  # Heavy penalty
    )

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device,
        model_name="Supreme_DynoNet",
        control_optimizer=control_opt,
        target_val_mse=args.target_val_mse,
    )
    trainer.optimizer = worker_opt

    history = trainer.train()

    return history


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
