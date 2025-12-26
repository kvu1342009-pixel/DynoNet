<div align="center">

# DynoNet

### Dynamic Controller-Worker Networks for Efficient Time Series Forecasting

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat&logo=python&logoColor=white)](https://python.org/)

*A lightweight meta-learning architecture that achieves near state-of-the-art performance with only **94K parameters** — 5x fewer than Transformer-based models.*

</div>

---

## Table of Contents

- [Abstract](#abstract)
- [Architecture](#architecture)
- [Key Innovations](#key-innovations)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Abstract

Time Series Forecasting has recently been dominated by large Transformer models (PatchTST, iTransformer) which, despite achieving strong results, are computationally expensive and require extensive hyperparameter tuning.

**DynoNet** challenges this paradigm by combining:

1. **Recurrent Neural Networks (GRU)** — for efficient sequential modeling
2. **Meta-Learning principles** — for dynamic adaptation to non-stationary data
3. **Controller-Worker architecture** — where a small "brain" network modulates the behavior of lightweight "worker" networks

Instead of static weights, DynoNet learns to *generate* optimal hyperparameters (learning rate, dropout, weight decay) and architectural modulations (FiLM layers, gates) on-the-fly. This enables instant adaptation without retraining.

---

## Architecture

<p align="center">
  <img src="png/architecture_overview.png" width="90%" alt="DynoNet Architecture">
</p>

DynoNet operates on a **Controller-Worker** paradigm where the Controller dynamically modulates Worker networks based on input context.

### Component Details

| Component | Description | Parameters |
|-----------|-------------|------------|
| **RevIN** | Reversible Instance Normalization to handle distribution shift | ~14 |
| **Controller GRU** | Analyzes global context, generates modulation signals | ~25K |
| **Series Decomposition** | Moving average to separate Trend and Residual | 0 |
| **Shared Trend Linear** | Simple linear projection for trend (DLinear-style) | ~32K |
| **7 Independent Workers** | Tiny GRU specialists, one per channel (8 hidden each) | ~37K |
| **Channel Mixer** | Cross-channel interaction MLP with residual connection | ~200 |

**Total: ~94K parameters**

---

## Key Innovations

### 1. Dynamic FiLM Modulation

<p align="center">
  <img src="png/series_decomposition.png" width="70%" alt="Series Decomposition">
</p>

The Controller generates **Feature-wise Linear Modulation (FiLM)** parameters that scale and shift the hidden states of each Worker:

```
h_out = γ · h_in + β

where γ, β are generated dynamically based on input context
```

### 2. Bi-Level Meta-Learning Training

<p align="center">
  <img src="png/bilevel_training.png" width="80%" alt="Bi-Level Training">
</p>

Training alternates between two optimization levels:

```python
# Level 1: Worker learns to forecast (on Train data)
θ_worker ← θ_worker - α · ∇L_train

# Level 2: Controller learns to optimize Worker (on Val data)  
θ_controller ← θ_controller - β · ∇L_val
```

This ensures the Controller learns signals that truly improve **generalization**, not just training loss.

### 3. Adaptive Training Dynamics

The Controller doesn't just modulate architecture — it also controls training:

| Signal | Range | Purpose |
|--------|-------|---------|
| `lr_scale` | [0.01, 5.0] | Per-channel learning rate multiplier |
| `wd_scale` | [0.1, 10.0] | Per-channel weight decay multiplier |
| `dropout_rate` | [0.0, 0.7] | Dynamic dropout per sample |
| `grad_clip` | [0.1, 2.0] | Adaptive gradient clipping |
| `freeze_prob` | [0.0, 1.0] | Probability to freeze worker (curriculum) |

---

## Results

### Benchmark: ETTh1 Dataset (Horizon = 96)

| Rank | Model | MSE ↓ | MAE ↓ | Params | Type |
|:----:|:------|:-----:|:-----:|:------:|:-----|
| 1 | **TSMixer** | 0.361 | 0.395 | ~500K | MLP |
| 2 | **DynoNet (Ours)** | **0.386** | **0.415** | **94K** | Dynamic RNN |
| 3 | PatchTST | 0.388 | 0.400 | >550K | Transformer |
| 4 | DLinear | 0.390 | 0.405 | 10K | Linear |
| 5 | Crossformer | 0.395 | 0.410 | >1M | Transformer |
| 6 | iTransformer | 0.487 | 0.458 | >500K | Transformer |

### Efficiency Comparison

<p align="center">
  <img src="png/paper_efficiency_plot.png" width="70%" alt="Efficiency Plot">
</p>

### Forecast Visualization

<p align="center">
  <img src="png/paper_forecast_all_features.png" width="90%" alt="Forecast All Features">
</p>

### Internal Dynamics

<p align="center">
  <img src="png/paper_dynamics_heatmap.png" width="70%" alt="Dynamics Heatmap">
</p>

*The Controller dynamically adjusts gate activations based on input patterns.*

---

## Installation

```bash
# Clone repository
git clone https://github.com/kvu1342009-pixel/DynoNet.git
cd DynoNet

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
numpy
pandas
tqdm
matplotlib
```

---

## Usage

### Quick Start

```bash
python main.py
```

### Reproduce Paper Results

```bash
python main.py \
  --seq_len 512 \
  --pred_len 96 \
  --batch_size 512 \
  --lr 0.002 \
  --weight_decay 1e-4 \
  --control_hidden 64 \
  --worker_hidden 8 \
  --seed 12345
```

### Custom Training

```python
from models import DynoNet
from data import get_ett_dataloaders
from utils import Trainer

# Load data
train_loader, val_loader, test_loader = get_ett_dataloaders(
    seq_len=336, pred_len=96, batch_size=512
)

# Create model
model = DynoNet(
    input_dim=7,
    control_hidden=64,
    worker_hidden=8,
    pred_len=96,
    seq_len=336,
)

# Train with bi-level optimization
trainer = Trainer(model, train_loader, val_loader, test_loader)
history = trainer.train()
```

---

## Project Structure

```
DynoNet/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── README.md              
│
├── models/
│   ├── __init__.py
│   ├── dyno_net.py         # Main DynoNet orchestrator
│   ├── control_net.py      # Controller (meta-network)
│   ├── distributed_worker.py # Worker coordinator + decomposition
│   ├── worker_net.py       # Individual GRU worker
│   └── revin.py            # Reversible Instance Normalization
│
├── data/
│   ├── __init__.py
│   ├── ett_dataset.py      # ETTh1 dataset loader
│   └── ETTh1.csv           # Dataset (auto-downloaded)
│
├── utils/
│   ├── __init__.py
│   └── trainer.py          # Training loop with bi-level optimization
│
└── png/                    # Visualization assets
```

---

## Key Design Decisions

### Why GRU over LSTM?
- Fewer parameters (2 gates vs 3)
- Comparable performance on shorter sequences
- Faster training and inference

### Why Channel-Independent Workers?
- Each time series channel (HUFL, HULL, etc.) has different characteristics
- Independent specialists can learn channel-specific patterns
- Reduces cross-channel interference

### Why Bi-Level Training?
- Standard training optimizes for training loss only
- Controller trained on validation data learns to prevent overfitting
- Results in better generalization without explicit regularization

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{DynoNet2025,
  title   = {DynoNet: Dynamic Controller-Worker Networks for Efficient Time Series Forecasting},
  author  = {Khanh Vu},
  journal = {GitHub Repository},
  year    = {2025},
  url     = {https://github.com/kvu1342009-pixel/DynoNet}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, suggestions, or collaborations:

- **Email:** [kvu1342009@gmail.com](mailto:kvu1342009@gmail.com)
- **GitHub:** [@kvu1342009-pixel](https://github.com/kvu1342009-pixel)

---

<div align="center">

**Made with ❤️ for the Time Series community**

</div>
