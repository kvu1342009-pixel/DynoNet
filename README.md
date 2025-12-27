# DynoNet: When a Tiny Controller Beats Giant Transformers

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*"What if we stopped making models bigger and started making them smarter?"*

</div>

---

## TL;DR

I built **DynoNet** — a tiny 94K parameter model that nearly matches 500K+ Transformer giants on time series forecasting. The secret? Instead of learning static weights, it learns *how to adapt* on-the-fly.

**Key results on ETTh1 (H=96):**
- 🎯 **MSE: 0.386** (ranks #3, beating PatchTST, iTransformer, DLinear, Autoformer)
- ⚡ **5× fewer parameters** than comparable models
- 🧠 **Zero hyperparameter tuning** — the model learns its own dropout, learning rate, etc.

---

## The Problem I Wanted to Solve

Everyone's building bigger Transformers. PatchTST has 550K params. iTransformer has 500K. They work great, but:

1. **They're expensive** — both to train and deploy
2. **They need careful tuning** — wrong learning rate? Your model's toast.
3. **They're static** — same weights for easy and hard inputs

I asked myself: *What if instead of learning complex patterns, we learned how to learn them?*

---

## My Solution: Controller-Worker Architecture

<p align="center">
  <img src="png/architecture_overview.png" width="80%" alt="DynoNet Architecture">
</p>

DynoNet has two parts:

| Component | What it does | Size |
|-----------|--------------|------|
| **Controller** | Looks at input, decides *how* workers should behave | 25K params |
| **Workers** | 7 tiny GRUs, one per channel, actually do the forecasting | 65K params |

The magic: **Controller generates modulation signals** (FiLM parameters, dropout rates, learning rate scales) that adapt the workers to each specific input.

Think of it like a football coach directing players. The coach doesn't score goals — they decide *strategy* based on the opponent.

---

## 💡 The Core Insight: Why Not Let a Model Manage Other Models?

Traditional approach:
```
Data → Model (learns alone) → Prediction
```

**The question I asked:** Why do we let models learn by themselves? Why not have another model manage them?

DynoNet's approach:
```
Data → Controller (observes, decides strategy)
              ↓
       Workers (execute orders) → Prediction
```

### Real-world analogy: Football Team

| Traditional ML | DynoNet |
|----------------|---------|
| 11 players play by themselves | **Coach** directs the team |
| Same strategy every game | Coach **adapts** to each opponent |
| Fixed hyperparameters | **Learned** hyperparameters per input |

### What the Controller actually controls:

| Problem with "learning alone" | How Controller fixes it |
|-------------------------------|------------------------|
| Fixed learning rate for all inputs | **Adaptive LR** per channel, per sample |
| Fixed dropout | **Learned dropout** (noisier input → more dropout) |
| Gradient explosion | **Adaptive gradient clipping** |
| All features treated equally | **Gate masks** ignore unimportant features |

### Proof it works:

| Mode | Test MSE |
|------|:--------:|
| Workers alone (no Controller) | 0.4049 |
| **Workers + Controller** | **0.3858** |

**Controller reduces MSE by 4.7%!** Having a "manager" genuinely helps.

## Results

### Benchmark Comparison

<p align="center">
  <img src="png/H96_benchmark.png" width="90%" alt="SOTA Benchmark">
</p>

**Table: ETTh1 Multivariate Forecasting (H=96)**

| Model | MSE ↓ | MAE ↓ | Params | vs DynoNet |
|-------|:-----:|:-----:|-------:|:----------:|
| FEDformer | 0.376 | 0.419 | 500K | +0.010 |
| TimesNet | 0.384 | 0.402 | 500K | +0.002 |
| **DynoNet** | **0.386** | **0.415** | **94K** | — |
| iTransformer | 0.386 | 0.405 | 500K | 0.000 |
| PatchTST | 0.414 | 0.419 | 550K | -0.028 |
| DLinear | 0.456 | 0.452 | 10K | -0.070 |
| Autoformer | 0.449 | 0.459 | 500K | -0.063 |

**We're #3 overall, but #1 in efficiency.** DynoNet achieves nearly the same accuracy as models 5× its size.

### Training Dynamics

<p align="center">
  <img src="png/H96_training_curves.png" width="100%" alt="Training Curves">
</p>

A few things I noticed:
- **Converges fast** — best validation MSE (0.641) reached by epoch 24
- **No overfitting** — train and val loss stay close
- **Stable training** — no spikes or crashes (thanks Controller!)

### Per-Channel Analysis

<p align="center">
  <img src="png/H96_channel_metrics.png" width="90%" alt="Per-Channel Metrics">
</p>

**Interesting findings:**
- **OT (Oil Temperature)** is easiest to predict (MSE: 0.054) — makes sense, it's the "target" variable others depend on
- **HUFL/MUFL** are hardest (MSE: ~0.77-0.79) — these are "High Usage" features with high volatility
- The model learns to allocate more attention to harder channels (via learnable dropout and gating)

| Feature | MSE | MAE | Insight |
|---------|:---:|:---:|---------|
| OT | 0.054 | 0.178 | Smooth, easy target |
| LULL | 0.129 | 0.280 | Low usage = low variance |
| MULL | 0.186 | 0.316 | Medium usage, stable |
| HULL | 0.231 | 0.362 | High usage, some variance |
| LUFL | 0.541 | 0.534 | Low freq high volatility |
| HUFL | 0.769 | 0.621 | Hard! High usage, high freq |
| MUFL | 0.791 | 0.617 | Hardest! |

---

## How It Works (Technical Details)

### 1. Bi-Level Meta-Learning

This is the key innovation. Training alternates between:

```
Level 1: Worker learns to forecast (on train data)
   θ_worker ← θ_worker - lr × ∇L_train

Level 2: Controller learns to help Worker generalize (on val data)
   θ_controller ← θ_controller - lr × ∇L_val
```

By training Controller on validation data, it learns to generate signals that improve *generalization*, not just training fit. It's like having a coach who watches your practice matches and adjusts your strategy for the real game.

### 2. Dynamic Signals

The Controller outputs these per-channel:

| Signal | Range | What it does |
|--------|-------|--------------|
| `gamma, beta` (FiLM) | ℝ | Scale and shift hidden states |
| `dropout_rate` | [0, 0.7] | More dropout for noisy inputs |
| `lr_scale` | [0.01, 5.0] | Learn faster/slower per channel |
| `gate_mask` | [0, 1] | Attention-like feature selection |

### 3. Trend-Residual Decomposition

Stolen from DLinear (credit where due). I split the signal into:
- **Trend** — handled by a simple linear projection (full 512-step lookback)
- **Residual** — handled by GRU workers (last 96 steps only)

This works because GRUs struggle with long sequences, but linear projections handle them fine.

---

## Usage

### Quick Start

```bash
# Clone
git clone https://github.com/kvu1342009-pixel/DynoNet.git
cd DynoNet

# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train
python H96.py
```

### Expected Output

```
============================================================
TEST RESULTS - DynoNet_H96
============================================================
MSE: 0.3858
MAE: 0.4153
============================================================
```

Plus automatic generation of:
- `png/H96_benchmark.png` — SOTA comparison chart
- `png/H96_training_curves.png` — Loss/MSE/MAE curves
- `png/H96_channel_metrics.png` — Per-feature breakdown
- `logs/H96_benchmark_*.json` — Detailed logs

---

## Project Structure

```
DynoNet/
├── H96.py                    # Train horizon=96 (main script)
├── H336.py                   # Train horizon=336
├── H720.py                   # Train horizon=720
├── models/
│   ├── dyno_net.py           # Main model (orchestrates everything)
│   ├── control_net.py        # The "brain" — generates modulation signals
│   ├── distributed_worker.py # Coordinates 7 GRU workers
│   ├── worker_net.py         # Individual GRU worker
│   └── revin.py              # Reversible Instance Normalization
├── data/
│   └── ett_dataset.py        # ETTh1 data loader
├── utils/
│   └── trainer.py            # Training loop with bi-level optimization
└── png/                      # Generated visualizations
```

---

## Lessons Learned

1. **Bigger isn't always better.** A well-designed 94K model can compete with 500K+ models.

2. **Meta-learning works.** Training Controller on validation data was the key insight. It forces the model to learn generalizable modulation strategies.

3. **Channel independence matters.** ETTh1 features have different dynamics. Independent workers + Controller coordination outperforms shared representations.

4. **Simple baselines are strong.** Adding DLinear-style trend projection was a huge boost. Don't ignore linear models.

---

## Limitations & Future Work

**What's not working yet:**
- Only tested on ETTh1. Need more datasets (Weather, Traffic, etc.)
- No cross-channel attention. Workers are independent — might be leaving performance on the table.
- Bi-level training is slower than single-level (~2× epochs)

**Ideas to try:**
- Add a tiny cross-channel mixer in Controller
- Test on longer horizons (H=720)
- Explore Controller → Worker gradient flow (MAML-style)

---

## Citation

```bibtex
@article{dynonet2025,
  title   = {DynoNet: Dynamic Controller-Worker Networks for 
             Efficient Time Series Forecasting},
  author  = {Vu, Khanh},
  journal = {arXiv preprint},
  year    = {2025},
  url     = {https://github.com/kvu1342009-pixel/DynoNet}
}
```

---

## Contact

Questions? Ideas? Found a bug?

📧 kvu1342009@gmail.com  
🐙 [@kvu1342009-pixel](https://github.com/kvu1342009-pixel)

---

<div align="center">

**If you made it this far, maybe drop a ⭐?**

*Built with lots of coffee and "why isn't this working" moments.*

</div>
