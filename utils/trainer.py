# ==============================================================================
# MODULE: Training Utilities
# ==============================================================================
# @context: Training loop and evaluation utilities for GRU models
# @goal: Clean, reusable training code with proper logging
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from typing import Dict, Tuple, Optional, Callable
import time
from pathlib import Path
import json
from tqdm import tqdm


class Trainer:
    """
    Trainer class for GRU models.

    # @logic:
    #   1. Train model with early stopping
    #   2. Evaluate on validation set each epoch
    #   3. Save best checkpoint
    #   4. Log metrics
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 10,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        model_name: str = "model",
        target_val_mse: Optional[float] = None,
        **kwargs,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.model_name = model_name
        self.target_val_mse = target_val_mse

        # Checkpoint dir
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Bi-level optimization support
        self.control_optimizer = kwargs.get("control_optimizer", None)

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=lr * 0.01,
        )

        # Loss
        self.criterion = nn.MSELoss()

        # Tracking
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mse": [],
            "val_mae": [],
            "lr": [],
        }

    def train_epoch(self) -> float:
        """
        Meta-Training Epoch.
        @logic:
          1. WorkerNet update: Train on training batch.
          2. ControlNet update: Train on validation batch to minimize Val Loss.
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Iterate through train loader, and cycle through val loader for meta-updates
        val_iter = iter(self.val_loader)

        pbar = tqdm(self.train_loader, desc="Meta-Training", leave=False)
        for batch_train in pbar:
            # ~~~ Step 1: Worker Update (on Train Data) ~~~
            x_t = batch_train["x"].to(self.device)
            y_t = batch_train["y_full"].to(self.device)

            self.optimizer.zero_grad()

            # Forward (Worker uses current Controller signals)
            pred_t, info_t = self.model(x_t, return_info=True)
            ctrl_t = self.model.get_last_control_signals()

            # 1. Apply Adaptive LR scaling via Loss
            # pred_t, y_t: (B, Seq, 7)
            # lr_scale: (B, 7) -> unsqueeze to (B, 1, 7)
            batch_loss_t = F.mse_loss(pred_t, y_t, reduction="none")
            if ctrl_t is not None and "lr_scale" in ctrl_t:
                lr_weights = ctrl_t["lr_scale"].unsqueeze(1)
                batch_loss_t = batch_loss_t * lr_weights

            # Apply Adaptive Loss Weighting (Attention Strategy)
            if ctrl_t is not None and "loss_weight" in ctrl_t:
                loss_weights = ctrl_t["loss_weight"].unsqueeze(1)
                batch_loss_t = batch_loss_t * loss_weights

            loss_t = batch_loss_t.mean()

            # 2. Apply Adaptive Weight Decay (Personalized per Worker)
            if (
                ctrl_t is not None
                and "wd_scale" in ctrl_t
                and hasattr(self.model.worker_net, "workers")
            ):
                wd_penalty = 0
                base_wd = self.optimizer.param_groups[0]["weight_decay"]
                wd_scales = ctrl_t["wd_scale"].mean(dim=0)  # Average over batch -> (7,)

                # Iterate over each worker to apply specific penalty
                for i, worker in enumerate(self.model.worker_net.workers):
                    worker_norm = 0
                    for param in worker.parameters():
                        worker_norm += torch.norm(param, 2) ** 2
                    wd_penalty += 0.5 * base_wd * wd_scales[i] * worker_norm

                loss_t = loss_t + wd_penalty
            elif ctrl_t is not None and "wd_scale" in ctrl_t:
                # Fallback for non-distributed worker (should not happen now but safe to keep)
                wd_penalty = 0
                wd_scale_mean = ctrl_t["wd_scale"].mean()
                for param in self.model.worker_net.parameters():
                    wd_penalty += torch.norm(param, 2) ** 2
                loss_t = (
                    loss_t
                    + 0.5
                    * self.optimizer.param_groups[0]["weight_decay"]
                    * wd_scale_mean
                    * wd_penalty
                )

            loss_t.backward()

            # 3. Apply Adaptive Gradient Clipping (Personalized per Worker)
            if ctrl_t is not None and "grad_clip" in ctrl_t:
                clip_vals = ctrl_t["grad_clip"].mean(dim=0)  # (7,)
                if hasattr(self.model.worker_net, "workers"):
                    for i, worker in enumerate(self.model.worker_net.workers):
                        torch.nn.utils.clip_grad_norm_(
                            worker.parameters(), max_norm=clip_vals[i].item()
                        )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.worker_net.parameters(),
                        max_norm=clip_vals.mean().item(),
                    )
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.worker_net.parameters(), max_norm=1.0
                )

            # 4. Adaptive Worker Freezing (Automated Curriculum)
            if (
                ctrl_t is not None
                and "freeze_prob" in ctrl_t
                and hasattr(self.model.worker_net, "workers")
            ):
                freeze_probs = ctrl_t["freeze_prob"].mean(dim=0)  # (7,)

                will_freeze = torch.bernoulli(freeze_probs).bool()

                for i, worker in enumerate(self.model.worker_net.workers):
                    if will_freeze[i]:
                        for param in worker.parameters():
                            if param.grad is not None:
                                param.grad = None

            self.optimizer.step()

            # ~~~ Step 2: Controller Meta-Update (on Validation Data) ~~~
            # We want to optimize Controller to minimize Val Loss
            if hasattr(self, "control_optimizer"):
                try:
                    batch_v = next(val_iter)
                except StopIteration:
                    val_iter = iter(self.val_loader)
                    batch_v = next(val_iter)

                x_v = batch_v["x"].to(self.device)
                y_v = batch_v["y_full"].to(self.device)

                self.control_optimizer.zero_grad()

                # Forward on Val data
                pred_v, info_v = self.model(x_v, return_info=True)
                loss_v = F.mse_loss(pred_v, y_v)

                # Backprop through the whole graph
                loss_v.backward()
                self.control_optimizer.step()

                display_loss = loss_v.item()
            else:
                display_loss = loss_t.item()

            total_loss += display_loss
            num_batches += 1

            # Update progress bar
            stats = {"v_loss": f"{display_loss:.4f}"}
            if info_t is not None:
                if "lr_scale_mean" in info_t:
                    stats["lr_s"] = f"{info_t['lr_scale_mean']:.2f}"
                if "dropout_rate_mean" in info_t:
                    stats["drop"] = f"{info_t['dropout_rate_mean']:.2f}"

            pbar.set_postfix(stats)

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        """Evaluate on a data loader."""
        self.model.eval()

        all_preds = []
        all_targets = []
        total_loss = 0
        num_batches = 0

        for batch in loader:
            x = batch["x"].to(self.device)
            y = batch["y_full"].to(self.device)

            # Forward
            if hasattr(self.model, "get_controller_info"):
                pred, _ = self.model(x)
            else:
                pred, _ = self.model(x)

            loss = self.criterion(pred, y)

            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
            total_loss += loss.item()
            num_batches += 1

        # Concatenate
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        # Metrics
        mse = ((preds - targets) ** 2).mean().item()
        mae = (preds - targets).abs().mean().item()

        return {
            "loss": total_loss / num_batches,
            "mse": mse,
            "mae": mae,
        }

    def train(self) -> Dict[str, list]:
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
        print(f"Max epochs: {self.max_epochs}")
        print(f"Patience: {self.patience}")
        if self.target_val_mse is not None:
            print(f"Target Val MSE: {self.target_val_mse:.4f}")
        print(f"{'='*60}\n")

        start_time = time.time()

        # Init logic for Active Early Stopping
        prev_val_loss = float("inf")
        consecutive_increase_count = 0

        for epoch in range(self.max_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_metrics = self.evaluate(self.val_loader)

            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_mse"].append(val_metrics["mse"])
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["lr"].append(current_lr)

            # Check improvement
            val_loss = val_metrics["loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_metrics)
                improved = "✓ NEW BEST"
                consecutive_increase_count = (
                    0  # Reset strict increase count on new best
                )
            else:
                self.epochs_without_improvement += 1
                improved = ""

                # Check for Consecutive Non-Best (Strict Patience of 5)
                # User Request: "Nếu không phải ✓ NEW BEST tự động dừng" (after 5 times)
                if val_loss >= self.best_val_loss:
                    consecutive_increase_count += 1
                else:
                    consecutive_increase_count = 0

            prev_val_loss = val_loss

            # Print
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch+1:3d}/{self.max_epochs} | Train: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val MAE: {val_metrics['mae']:.4f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s {improved}"
            )

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(
                    f"\nEarly stopping at epoch {epoch+1} (Patience {self.patience} exceeded)"
                )
                break

            if consecutive_increase_count >= 5:
                print(
                    f"\nActive Early Stopping at epoch {epoch+1} (No NEW BEST for 5 consecutive epochs!)"
                )
                break

            # Target MSE Early Stopping
            if (
                self.target_val_mse is not None
                and val_metrics["mse"] <= self.target_val_mse
            ):
                print(
                    f"\n🎯 TARGET REACHED at epoch {epoch+1}! Val MSE: {val_metrics['mse']:.4f} <= Target: {self.target_val_mse:.4f}"
                )
                self._save_checkpoint(epoch, val_metrics)  # Save this as best
                break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(
            f"Best epoch: {self.best_epoch + 1} with val loss: {self.best_val_loss:.4f}"
        )

        # Load best model and test
        self._load_best_checkpoint()
        test_metrics = self.evaluate(self.test_loader)

        print(f"\n{'='*60}")
        print(f"TEST RESULTS - {self.model_name}")
        print(f"{'='*60}")
        print(f"MSE: {test_metrics['mse']:.4f}")
        print(f"MAE: {test_metrics['mae']:.4f}")
        print(f"{'='*60}")

        # Save history
        self._save_history(test_metrics)

        return self.history

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )

    def _load_best_checkpoint(self):
        """Load best checkpoint."""
        path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

    def _save_history(self, test_metrics: Dict):
        """Save training history."""
        result = {
            "model_name": self.model_name,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
            "history": self.history,
        }

        path = self.checkpoint_dir / f"{self.model_name}_history.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> str:
    """Get available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
