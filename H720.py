#!/usr/bin/env python3
# ==============================================================================
# MODULE: H720 - DynoNet Training for Horizon 720
# ==============================================================================
# @context: Train DynoNet on ETTh1 with prediction horizon = 720 steps
# @goal: Benchmark long-term forecasting performance (~1 month)
# ==============================================================================

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import DynoNet
from data import get_ett_dataloaders
from utils import Trainer, get_device

# === Configuration for H720 ===
CONFIG = {
    # Data
    "seq_len": 720,          # Longer lookback for long-term
    "worker_seq_len": 96,    # Worker GRU lookback
    "pred_len": 720,         # Prediction horizon (long-term)
    "batch_size": 128,       # Smaller batch for memory
    
    # Model - Larger capacity for long horizon
    "control_hidden": 128,   # Increased controller
    "worker_hidden": 32,     # Increased workers
    "num_layers": 2,         # Deeper network
    "dropout": 0.2,          # More regularization
    
    # Training
    "lr": 5e-4,              # Lower LR for long-term stability
    "weight_decay": 1e-3,    # Stronger regularization
    "max_epochs": 150,       # More epochs for harder task
    "patience": 30,          # More patience
    
    # Misc
    "seed": 12345,
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def main():
    print("\n" + "=" * 70)
    print("  DynoNet - ETTh1 Horizon 720 (Long-term Forecasting)")
    print("=" * 70)
    
    set_seed(CONFIG["seed"])
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: {CONFIG}")
    
    # Load data
    print("\n--- Loading ETTh1 Dataset ---")
    train_loader, val_loader, test_loader = get_ett_dataloaders(
        root_dir="./data",
        seq_len=CONFIG["seq_len"],
        pred_len=CONFIG["pred_len"],
        batch_size=CONFIG["batch_size"],
        num_workers=4,
    )
    
    # Create model
    print("\n--- Creating DynoNet ---")
    model = DynoNet(
        input_dim=7,
        control_hidden=CONFIG["control_hidden"],
        worker_hidden=CONFIG["worker_hidden"],
        output_dim=7,
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        pred_len=CONFIG["pred_len"],
        seq_len=CONFIG["seq_len"],
        worker_seq_len=CONFIG["worker_seq_len"],
    )
    
    # Print architecture
    ctrl_info = model.get_controller_info()
    print(f"Total Parameters: {ctrl_info['total_params']:,}")
    print(f"  ControlNet: {ctrl_info['controller_params']:,} ({ctrl_info['controller_ratio']:.1%})")
    print(f"  WorkerNet: {ctrl_info['base_params']:,}")
    
    # Create optimizers (Bi-level)
    worker_opt = torch.optim.AdamW(
        model.worker_net.parameters(), 
        lr=CONFIG["lr"], 
        weight_decay=CONFIG["weight_decay"]
    )
    control_opt = torch.optim.AdamW(
        model.control_net.parameters(),
        lr=CONFIG["lr"] * 0.1,  # Even slower for long-term
        weight_decay=CONFIG["weight_decay"] * 5.0,
    )
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        max_epochs=CONFIG["max_epochs"],
        patience=CONFIG["patience"],
        device=device,
        model_name="DynoNet_H720",
        control_optimizer=control_opt,
    )
    trainer.optimizer = worker_opt
    
    history = trainer.train()
    return history


if __name__ == "__main__":
    main()
