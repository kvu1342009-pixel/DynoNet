<div align="center">

# ⚡ DynoNet
**Dynamic Recurrent Networks for Efficient Time Series Forecasting**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![SOTA](https://img.shields.io/badge/SOTA-ETTh1_Horizon96-success?style=for-the-badge&logo=medal)](#-leaderboard)

<p align="center">
  <img src="png/paper_efficiency_plot.png" width="80%" alt="Efficiency Plot">
</p>

*"Turning Weights into Fluids."* — DynoNet introduces a lightweight, dynamic architecture where model parameters adapt on-the-fly. 
**5x Fewer Parameters** than Transformers, yet **SOTA Accuracy**.

[Abstract](#-abstract) •
[Leaderboard](#-leaderboard) •
[Architecture](#-architecture) •
[How to Run](#-how-to-run) •
[Citation](#-citation)

</div>

---

## 📜 Abstract

Time Series Forecasting has been dominated by massive Transformer models (PatchTST, iTransformer) that are computationally expensive. **DynoNet** challenges this trend by revisiting Reccurent Neural Networks (RNNs) with a twist: **Meta-Learning**.

Instead of static weights, DynoNet employs a **Controller** that dynamically generates the weights (FiLM layers) for a lightweight **Worker Network** based on the input context. This allows the model to adapt to non-stationary data instantly without retraining.

---

## 🏆 Leaderboard

**Benchmark:** ETTh1 Dataset (Multivariate Forecasting), Horizon = 96.

| Rank | 🤖 Model | 📉 MSE | 📏 MAE | 📦 Params | 🏗️ Type |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 🥇 | **TSMixer** | 0.361 | 0.395 | ~500K | MLP |
| 🥈 | **DynoNet (Ours)** | **0.386** | **0.415** | **94K** 🚀 | **Dynamic RNN** |
| 🥉 | **PatchTST** | 0.388 | 0.400 | >550K | Transformer |
| 4 | **DLinear** | 0.390 | 0.405 | 10K | Linear |
| 5 | **Crossformer** | 0.395 | 0.410 | >1M | Transformer |
| 6 | **iTransformer** | 0.487 | 0.458 | >500K | Transformer |

> **Note:** DynoNet achieves competitive performance (Rank 2) with extremely low parameter count, making it ideal for edge deployment.

---

## 🧠 Architecture

DynoNet operates on a **"Brain-Muscle"** paradigm.

<div align="center">
<table>
<tr>
<td width="50%" align="center">

### 1. The Dynamic Controller (Brain)
Analyses the global context and generates:
- **FiLM Parameters:** $(\gamma, \beta)$ to modulate workers.
- **Gate Masks:** To select relevant input features.
- **Dynamic Hyperparams:** Learning rate & dropout.

</td>
<td width="50%" align="center">

### 2. The Distributed Workers (Muscle)
7 independent, tiny GRU networks (one per channel).
They have **no static weights** for adaptation. They are fully controlled by the Brain.

</td>
</tr>
</table>
</div>

### 🔬 Visual Proofs

| **Forecast Quality** | **Internal Dynamics** |
| :---: | :---: |
| <img src="png/paper_forecast_all_features.png" width="100%"> | <img src="png/paper_dynamics_heatmap.png" width="100%"> |
| *DynoNet tracks ground truth closely across all 7 features.* | *The Controller dynamically activates different gates for different inputs.* |

---

## � How to Run

### Installation
```bash
git clone https://github.com/kvu1342009-pixel/DynoNet.git
cd DynoNet
pip install -r requirements.txt
```

### Reproduce SOTA Results
```bash
python main.py \
  --seq_len 512 --pred_len 96 \
  --batch_size 512 \
  --lr 0.002 --weight_decay 1e-4 \
  --control_hidden 64 --worker_hidden 8 \
  --seed 12345
```

---

## � Citation

If you find this code useful, please cite our work:

<div align="center">

```bibtex
@article{DynoNet2025,
  title   = {DynoNet: Dynamic Recurrent Networks for Efficient Time Series Forecasting},
  author  = {Khanh Vu},
  journal = {GitHub Repository},
  year    = {2025},
  url     = {https://github.com/kvu1342009-pixel/DynoNet}
}
```

</div>
