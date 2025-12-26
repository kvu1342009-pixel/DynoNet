#!/usr/bin/env python3
# ==============================================================================
# MODULE: H96 - DynoNet Training for Horizon 96
# ==============================================================================
# @context: Train DynoNet on ETTh1 with prediction horizon = 96 steps
# @goal: Benchmark short-term forecasting performance
# ==============================================================================

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import DynoNet
from data import get_ett_dataloaders
from utils import Trainer, get_device

# === Configuration for H96 ===
CONFIG = {
    # Data
    "seq_len": 512,          # Input sequence length
    "worker_seq_len": 96,    # Worker GRU lookback
    "pred_len": 96,          # Prediction horizon
    "batch_size": 512,
    
    # Model
    "control_hidden": 64,
    "worker_hidden": 8,
    "num_layers": 1,
    "dropout": 0.1,
    
    # Training
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "max_epochs": 100,
    "patience": 20,
    
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
    print("  DynoNet - ETTh1 Horizon 96 (Short-term Forecasting)")
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
        lr=CONFIG["lr"] * 0.05,
        weight_decay=CONFIG["weight_decay"] * 10.0,
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
        model_name="DynoNet_H96",
        control_optimizer=control_opt,
    )
    trainer.optimizer = worker_opt
    
    history = trainer.train()
    return history


if __name__ == "__main__":
    main()
