#!/usr/bin/env python3
# ==============================================================================
# MODULE: H96 - DynoNet Training for Horizon 96 with Benchmark Logging
# ==============================================================================
# @context: Train DynoNet on ETTh1 with prediction horizon = 96 steps
# @goal: Benchmark short-term forecasting with detailed logs and charts
# ==============================================================================

import torch
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

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

# === SOTA Benchmark Data (ETTh1 H96 M→M) ===
SOTA_BENCHMARK = {
    "TimesNet": {"mse": 0.384, "mae": 0.402, "params": 500},
    "iTransformer": {"mse": 0.386, "mae": 0.405, "params": 500},
    "PatchTST": {"mse": 0.414, "mae": 0.419, "params": 550},
    "FEDformer": {"mse": 0.376, "mae": 0.419, "params": 500},
    "DLinear": {"mse": 0.456, "mae": 0.452, "params": 10},
    "Autoformer": {"mse": 0.449, "mae": 0.459, "params": 500},
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def generate_benchmark_chart(dynonet_mse: float, dynonet_mae: float, dynonet_params: int):
    """Generate SOTA comparison chart."""
    # Prepare data
    models = list(SOTA_BENCHMARK.keys()) + ["DynoNet (Ours)"]
    mse_vals = [v["mse"] for v in SOTA_BENCHMARK.values()] + [dynonet_mse]
    mae_vals = [v["mae"] for v in SOTA_BENCHMARK.values()] + [dynonet_mae]
    params_k = [v["params"] for v in SOTA_BENCHMARK.values()] + [dynonet_params // 1000]
    
    # Sort by MSE
    sorted_idx = np.argsort(mse_vals)
    models = [models[i] for i in sorted_idx]
    mse_vals = [mse_vals[i] for i in sorted_idx]
    mae_vals = [mae_vals[i] for i in sorted_idx]
    params_k = [params_k[i] for i in sorted_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chart 1: MSE/MAE Comparison
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.35
    
    colors = ['#2196F3' if 'DynoNet' not in m else '#E91E63' for m in models]
    bars1 = ax1.bar(x - width/2, mse_vals, width, label='MSE', color=colors, alpha=0.8)
    bars2 = ax1.bar(x + width/2, mae_vals, width, label='MAE', color=colors, alpha=0.5)
    
    ax1.set_ylabel('Error (Lower is Better)', fontweight='bold')
    ax1.set_title('ETTh1 H96 Benchmark: MSE & MAE', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0.35, 0.50)
    ax1.grid(axis='y', alpha=0.3)
    
    # Highlight DynoNet
    dynonet_idx = models.index("DynoNet (Ours)")
    ax1.annotate('★ Ours', xy=(dynonet_idx, mse_vals[dynonet_idx]), 
                xytext=(dynonet_idx + 0.5, mse_vals[dynonet_idx] + 0.02),
                fontsize=10, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # Chart 2: Efficiency (MSE vs Params)
    ax2 = axes[1]
    for i, (model, mse, params) in enumerate(zip(models, mse_vals, params_k)):
        color = '#E91E63' if 'DynoNet' in model else '#2196F3'
        size = 300 if 'DynoNet' in model else 150
        ax2.scatter(params, mse, s=size, c=color, alpha=0.7, edgecolors='black', linewidths=1)
        ax2.annotate(model.replace(' (Ours)', ''), (params, mse), 
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    ax2.set_xlabel('Parameters (K)', fontweight='bold')
    ax2.set_ylabel('MSE (Lower is Better)', fontweight='bold')
    ax2.set_title('Efficiency: MSE vs Model Size', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 600)
    
    plt.tight_layout()
    plt.savefig('png/H96_benchmark.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved benchmark chart: png/H96_benchmark.png")


def generate_training_curves(history: dict):
    """Generate training loss, MSE, MAE curves over epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Chart 1: Training Loss
    ax1 = axes[0]
    ax1.plot(epochs, history["train_loss"], 'b-', linewidth=2, label='Train Loss')
    ax1.plot(epochs, history["val_loss"], 'r--', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Chart 2: Validation MSE
    ax2 = axes[1]
    ax2.plot(epochs, history["val_mse"], 'g-', linewidth=2, marker='o', markersize=3)
    best_epoch = history["val_mse"].index(min(history["val_mse"])) + 1
    best_mse = min(history["val_mse"])
    ax2.axhline(y=best_mse, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_mse:.4f}')
    ax2.scatter([best_epoch], [best_mse], color='red', s=100, zorder=5, marker='*')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('MSE', fontweight='bold')
    ax2.set_title('Validation MSE', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Chart 3: Validation MAE
    ax3 = axes[2]
    ax3.plot(epochs, history["val_mae"], 'm-', linewidth=2, marker='s', markersize=3)
    best_mae = min(history["val_mae"])
    ax3.axhline(y=best_mae, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_mae:.4f}')
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('MAE', fontweight='bold')
    ax3.set_title('Validation MAE', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('png/H96_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved training curves: png/H96_training_curves.png")


def generate_channel_metrics(model, test_loader, device):
    """Generate per-channel MSE/MAE bar chart."""
    # ETTh1 feature names
    FEATURE_NAMES = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y = batch["y_full"].to(device)
            pred, _ = model(x)
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
    
    preds = torch.cat(all_preds, dim=0)    # (N, 96, 7)
    targets = torch.cat(all_targets, dim=0)  # (N, 96, 7)
    
    # Per-channel metrics
    channel_mse = []
    channel_mae = []
    for c in range(7):
        mse_c = ((preds[:, :, c] - targets[:, :, c]) ** 2).mean().item()
        mae_c = (preds[:, :, c] - targets[:, :, c]).abs().mean().item()
        channel_mse.append(mse_c)
        channel_mae.append(mae_c)
    
    # Create chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(7)
    width = 0.6
    
    # MSE per channel
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 7))
    bars1 = ax1.bar(x, channel_mse, width, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Feature', fontweight='bold')
    ax1.set_ylabel('MSE', fontweight='bold')
    ax1.set_title('Per-Channel MSE (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, channel_mse):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # MAE per channel
    ax2 = axes[1]
    bars2 = ax2.bar(x, channel_mae, width, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Feature', fontweight='bold')
    ax2.set_ylabel('MAE', fontweight='bold')
    ax2.set_title('Per-Channel MAE (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, channel_mae):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('png/H96_channel_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved channel metrics: png/H96_channel_metrics.png")
    
    # Print summary
    print("\n" + "-" * 50)
    print("  PER-CHANNEL METRICS")
    print("-" * 50)
    print(f"  {'Feature':<8} {'MSE':>10} {'MAE':>10}")
    print("-" * 50)
    for name, mse, mae in zip(FEATURE_NAMES, channel_mse, channel_mae):
        print(f"  {name:<8} {mse:>10.4f} {mae:>10.4f}")
    print("-" * 50)
    print(f"  {'AVERAGE':<8} {np.mean(channel_mse):>10.4f} {np.mean(channel_mae):>10.4f}")
    print("-" * 50)
    
    return dict(zip(FEATURE_NAMES, zip(channel_mse, channel_mae)))


def save_benchmark_log(history: dict, test_mse: float, test_mae: float, duration: float):
    """Save detailed benchmark log."""
    log = {
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
        "results": {
            "test_mse": test_mse,
            "test_mae": test_mae,
            "best_val_mse": min(history["val_mse"]),
            "best_epoch": history["val_mse"].index(min(history["val_mse"])) + 1,
            "total_epochs": len(history["train_loss"]),
            "training_time_minutes": duration,
        },
        "sota_comparison": {},
        "history": history,
    }
    
    # Compare with SOTA
    for model, metrics in SOTA_BENCHMARK.items():
        delta_mse = test_mse - metrics["mse"]
        log["sota_comparison"][model] = {
            "their_mse": metrics["mse"],
            "our_mse": test_mse,
            "delta": delta_mse,
            "status": "BETTER" if delta_mse < 0 else ("SAME" if delta_mse == 0 else "WORSE"),
        }
    
    # Rank
    all_mse = [v["mse"] for v in SOTA_BENCHMARK.values()] + [test_mse]
    rank = sorted(all_mse).index(test_mse) + 1
    log["results"]["rank"] = f"{rank}/{len(all_mse)}"
    
    # Save
    Path("logs").mkdir(exist_ok=True)
    log_path = f"logs/H96_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"✓ Saved benchmark log: {log_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Test MSE: {test_mse:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Rank: {log['results']['rank']}")
    print(f"  Training Time: {duration:.1f} minutes")
    print("-" * 60)
    print("  Comparison with SOTA:")
    for model, comp in log["sota_comparison"].items():
        status_icon = "✓" if comp["status"] == "BETTER" else ("=" if comp["status"] == "SAME" else "✗")
        print(f"    {status_icon} vs {model}: {comp['delta']:+.4f}")
    print("=" * 60)
    
    return log


def main():
    import time
    start_time = time.time()
    
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
    
    # Calculate duration
    duration = (time.time() - start_time) / 60
    
    # Get test results from history file
    with open("checkpoints/DynoNet_H96_history.json", "r") as f:
        results = json.load(f)
    
    test_mse = results["test_mse"]
    test_mae = results["test_mae"]
    
    # Generate benchmark chart
    Path("png").mkdir(exist_ok=True)
    generate_benchmark_chart(test_mse, test_mae, ctrl_info['total_params'])
    
    # Generate training curves
    generate_training_curves(history)
    
    # Generate per-channel metrics
    generate_channel_metrics(model, test_loader, device)
    
    # Save detailed log
    save_benchmark_log(history, test_mse, test_mae, duration)
    
    return history


if __name__ == "__main__":
    main()

