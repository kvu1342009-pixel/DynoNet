# ==============================================================================
# MODULE: ETT Dataset
# ==============================================================================
# @context: Load and preprocess ETTh1 dataset for time series forecasting
# @goal: Provide clean DataLoaders for train/val/test splits
# @constraint: Must handle missing data and normalize properly
# ==============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import os
from pathlib import Path


class ETTh1Dataset(Dataset):
    """
    ETTh1 Dataset for time series forecasting.

    # @logic:
    #   1. Load CSV data from file or download
    #   2. Split into train/val/test based on ETT standard splits
    #   3. Normalize using training statistics
    #   4. Create sliding window samples

    # @param:
    #   - Standard split: 12 months train / 4 months val / 4 months test
    #   - Total: ~17420 hourly samples

    # @invariant: Normalization uses only training data statistics
    """

    # Standard ETT split lengths (hours)
    TRAIN_LEN = 12 * 30 * 24  # 8640 samples (12 months)
    VAL_LEN = 4 * 30 * 24  # 2880 samples (4 months)
    TEST_LEN = 4 * 30 * 24  # 2880 samples (4 months)

    def __init__(
        self,
        root_dir: str = "./data",
        split: str = "train",
        seq_len: int = 96,
        pred_len: int = 96,
        target_col: str = "OT",
        scale: bool = True,
        download: bool = True,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = split
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        self.scale = scale

        # Load data
        self.data, self.target_idx = self._load_data(download)

        # Precise paper splits
        # Train: [0, 8640]
        # Val: [8640, 8640 + 2880]
        # Test: [8640 + 2880, 8640 + 2880 + 2880]

        # Statistics computed ONLY on Train set
        train_data = self.data[: self.TRAIN_LEN]
        self.mean = train_data.mean(axis=0)
        self.std = train_data.std(axis=0)
        self.std = np.where(self.std == 0, 1, self.std)

        # Get split data
        if self.split == "train":
            self.data_split = self.data[: self.TRAIN_LEN]
        elif self.split == "val":
            # Note: We take some overlap from previous split to form the first window
            start = self.TRAIN_LEN - self.seq_len
            end = self.TRAIN_LEN + self.VAL_LEN
            self.data_split = self.data[start:end]
        else:  # test
            start = self.TRAIN_LEN + self.VAL_LEN - self.seq_len
            end = self.TRAIN_LEN + self.VAL_LEN + self.TEST_LEN
            self.data_split = self.data[start:end]

        # Create samples
        self.samples = self._create_samples()

    def _load_data(self, download: bool) -> Tuple[np.ndarray, int]:
        """Load ETTh1 data from CSV."""
        data_path = self.root_dir / "ETTh1.csv"

        if not data_path.exists():
            if download:
                self._download_data()
            else:
                raise FileNotFoundError(f"Data not found at {data_path}")

        # Load CSV
        df = pd.read_csv(data_path)

        # Drop date column, keep features
        # Columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
        feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        data = df[feature_cols].values.astype(np.float32)

        # Get target column index
        target_idx = feature_cols.index(self.target_col)

        print(f"Loaded ETTh1: {len(data)} samples, {len(feature_cols)} features")

        return data, target_idx

    def _download_data(self):
        """Download ETTh1 dataset."""
        import urllib.request

        url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"

        self.root_dir.mkdir(parents=True, exist_ok=True)
        data_path = self.root_dir / "ETTh1.csv"

        print(f"Downloading ETTh1 dataset to {data_path}...")
        urllib.request.urlretrieve(url, data_path)
        print("Download complete!")

    def _create_samples(self) -> list:
        """Create sliding window samples."""
        samples = []
        total_len = self.seq_len + self.pred_len

        for i in range(len(self.data_split) - total_len + 1):
            samples.append(i)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.

        Returns:
            Dict with:
                - x: Input sequence (seq_len, num_features)
                - y: Target sequence (pred_len, 1) - only target column
                - y_full: Full target sequence (pred_len, num_features)
        """
        start_idx = self.samples[idx]

        # Get sequences
        x = self.data_split[start_idx : start_idx + self.seq_len]
        y_full = self.data_split[
            start_idx + self.seq_len : start_idx + self.seq_len + self.pred_len
        ]
        y = y_full[:, self.target_idx : self.target_idx + 1]

        # Normalize if needed
        if self.scale:
            x = (x - self.mean) / self.std
            y_full = (y_full - self.mean) / self.std
            y = (y - self.mean[self.target_idx]) / self.std[self.target_idx]

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "y_full": torch.from_numpy(y_full),
        }

    def inverse_transform(
        self, data: torch.Tensor, target_only: bool = True
    ) -> torch.Tensor:
        """Inverse normalization."""
        if target_only:
            return data * self.std[self.target_idx] + self.mean[self.target_idx]
        else:
            return data * torch.from_numpy(self.std) + torch.from_numpy(self.mean)


def get_ett_dataloaders(
    root_dir: str = "./data",
    seq_len: int = 96,
    pred_len: int = 96,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, val, test DataLoaders for ETTh1.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = ETTh1Dataset(root_dir, "train", seq_len, pred_len)
    val_ds = ETTh1Dataset(root_dir, "val", seq_len, pred_len)
    test_ds = ETTh1Dataset(root_dir, "test", seq_len, pred_len)

    # Share normalization stats
    val_ds.mean, val_ds.std = train_ds.mean, train_ds.std
    test_ds.mean, test_ds.std = train_ds.mean, train_ds.std

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"DataLoaders created:")
    print(f"  Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_ds)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_ds)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# --- Test Block ---
if __name__ == "__main__":
    print("Testing ETTh1 Dataset...")

    train_loader, val_loader, test_loader = get_ett_dataloaders(
        seq_len=96,
        pred_len=96,
        batch_size=32,
    )

    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  x: {batch['x'].shape}")
    print(f"  y: {batch['y'].shape}")
    print(f"  y_full: {batch['y_full'].shape}")

    print("\n✓ ETTh1 Dataset test passed!")
