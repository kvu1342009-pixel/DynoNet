<div align="center">

# DynoNet
**Dynamic Recurrent Networks for Efficient Time Series Forecasting**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat)](LICENSE)
[![SOTA](https://img.shields.io/badge/SOTA-ETTh1_Horizon96-success?style=flat)](https://github.com/kvu1342009-pixel/DynoNet)

<p align="center">
  <img src="png/paper_efficiency_plot.png" width="80%" alt="Efficiency Plot">
</p>

*DynoNet introduces a lightweight, dynamic architecture where model parameters adapt on-the-fly. It achieves state-of-the-art accuracy with 5x fewer parameters compared to Transformer-based models.*

[Abstract](#abstract) •
[Leaderboard](#leaderboard) •
[Architecture](#architecture) •
[Usage](#usage) •
[Citation](#citation)

</div>

---

## Abstract

Time Series Forecasting has recently been dominated by massive Transformer models (e.g., PatchTST, iTransformer) which are computationally expensive. **DynoNet** challenges this trend by revisiting Recurrent Neural Networks (RNNs) combined with **Meta-Learning principles**.

Instead of relying on static weights, DynoNet employs a **Controller** that dynamically generates the parameters (FiLM layers) for a lightweight **Worker Network** based on the input context. This allows the model to adapt to non-stationary data instantly without retraining, offering a superior trade-off between efficiency and accuracy.

---

## Leaderboard

**Benchmark:** ETTh1 Dataset (Multivariate Forecasting), Horizon = 96.

| Rank | Model | MSE | MAE | Params | Type |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | **TSMixer** | 0.361 | 0.395 | ~500K | MLP |
| 2 | **DynoNet (Ours)** | **0.386** | **0.415** | **94K** | **Dynamic RNN** |
| 3 | **PatchTST** | 0.388 | 0.400 | >550K | Transformer |
| 4 | **DLinear** | 0.390 | 0.405 | 10K | Linear |
| 5 | **Crossformer** | 0.395 | 0.410 | >1M | Transformer |
| 6 | **iTransformer** | 0.487 | 0.458 | >500K | Transformer |

> **Note:** DynoNet achieves Rank 2 performance with significantly fewer parameters than top-tier Transformer models, demonstrating exceptional efficiency.

---

## Architecture

DynoNet operates on a **Controller-Worker** paradigm.

<div align="center">
<table>
<tr>
<td width="50%" valign="top">

### 1. The Dynamic Controller
Analyses the global context and generates:
- **FiLM Parameters:** $(\gamma, \beta)$ to modulate workers.
- **Gate Masks:** To select relevant input features.
- **Dynamic Hyperparams:** Adaptive learning rate & dropout.

</td>
<td width="50%" valign="top">

### 2. The Distributed Workers
7 independent, tiny GRU networks (one per channel).
They possess **no static weights** for adaptation and are fully modulated by the signals from the Controller.

</td>
</tr>
</table>
</div>

### Visual Analysis

| Forecast Quality | Internal Dynamics |
| :---: | :---: |
| <img src="png/paper_forecast_all_features.png" width="100%"> | <img src="png/paper_dynamics_heatmap.png" width="100%"> |
| *DynoNet tracks ground truth closely across all 7 features.* | *The Controller dynamically activates different gates for different inputs.* |

---

## Usage

### Installation
```bash
git clone https://github.com/kvu1342009-pixel/DynoNet.git
cd DynoNet
pip install -r requirements.txt
```

### Reproduce Results
Run the training script with the exact configuration used to achieve the reported results:

```bash
python main.py \
  --seq_len 512 --pred_len 96 \
  --batch_size 512 \
  --lr 0.002 --weight_decay 1e-4 \
  --control_hidden 64 --worker_hidden 8 \
  --seed 12345
```

---

## Citation

If you find this code useful for your research, please cite:

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
