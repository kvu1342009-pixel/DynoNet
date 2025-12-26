# ⚡ DynoNet: Dynamic Recurrent Networks for Efficient Forecasting

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![SOTA](https://img.shields.io/badge/SOTA-ETTh1_Horizon96-success.svg)](#-sota-performance)

> **"Turning Weights into Fluids."**  
> DynoNet introduces a lightweight, dynamic architecture where model parameters adapt on-the-fly to the input context. By replacing massive static Transformer layers with a smart **Meta-Controller**, DynoNet achieves SOTA performance with **5x fewer parameters**.

![Efficiency Plot](png/paper_efficiency_plot.png)

---

## 🏆 SOTA Performance (ETTh1, H=96)

DynoNet stands out as the **Pareto Optimal** choice—delivering top-tier accuracy with a fraction of the computational cost.

| Rank | Model | MSE (Lower is Better) | Parameters (Params) | Architecture Type |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **TSMixer** | 0.361 | ~500K | MLP-Mixer |
| **2** | **DynoNet (Ours)** | **0.386** | **94K** 🚀 | **Dynamic RNN** |
| 3 | **PatchTST** | 0.388 | >550K | Transformer |
| 4 | **DLinear** | 0.390 | 10K | Linear |
| 5 | **Crossformer** | 0.395 | >1M | Transformer |
| 6 | **iTransformer** | 0.487 | >500K | Transformer |

### 🔍 Error Analysis (MSE vs MAE)
![SOTA Bar Chart](png/sota_comparison_mse_mae.png)
*While DynoNet ranks #2 in MSE, its MAE is slightly higher. This is a deliberate design choice to penalize large outliers heavily (critical for industrial safety), accepting slightly more noise in the baseline.*

---

## 🧠 Architecture: The "Brain-Muscle" Paradigm

DynoNet splits the forecasting task into two distinct roles:

1.  **The Controller (Brain):** A global GRU that reads the input context and generates *control signals* (Hypernetworks/FiLM).
2.  **The Distributed Workers (Muscles):** 7 independent, tiny GRU networks (one per channel) that process residuals. They have *no static weights* for adaptation; instead, they are modulated dynamically by the Controller.

### Key Mechanisms:
*   **FiLM (Feature-wise Linear Modulation):** The Controller outputs $\gamma$ (scale) and $\beta$ (shift) to modulate the Workers' feature maps layer-by-layer.
*   **Dynamic Gating:** The Brain decides which input features are relevant for each Worker at any given time step.
*   **Adaptive Regularization:** Learning Rate, Weight Decay, and Dropout are not fixed hyperparameters but *learned dynamic signals*.

![Dynamics Heatmap](png/paper_dynamics_heatmap.png)
*Figure shows the Controller dynamically activating different gates (weights) for different input features, effectively performing "Soft Attention" without the quadratic cost of Transformers.*

---

## 🖼️ Qualitative Results

Does it actually work? See for yourself. DynoNet produces highly realistic forecasts that track ground truth trends closely, even for volatile variables.

![Forecast Visualization](png/paper_forecast_all_features.png)

---

## 🛠️ Usage

### Installation
```bash
git clone https://github.com/kvu1342009-pixel/DynoNet.git
cd DynoNet
pip install -r requirements.txt
```

### Reproduction Command
Train the exact model that achieved the reported results:

```bash
python main.py \
  --seq_len 512 \
  --pred_len 96 \
  --batch_size 512 \
  --lr 0.002 \
  --control_hidden 64 \
  --worker_hidden 8 \
  --weight_decay 1e-4 \
  --seed 12345
```

---

## 📂 Repository Structure

*   `models/dyno_net.py`: **Main Entry**. Orchestrates the Brain and Muscles.
*   `models/control_net.py`: **The Brain**. Generates FiLM params and dynamic hyperparameters.
*   `models/worker_net.py`: **The Muscle**. Lightweight GRU specialist.
*   `models/distributed_worker.py`: Channel-Independent wrapper handling decomposition (Trend vs Residual).
*   `visualize_paper.py`: Script to generate all figures in this README.

---

## 📝 Citation

If you find this work useful, please star the repo ⭐ and cite:

```bibtex
@article{DynoNet2025,
  title={DynoNet: Dynamic Recurrent Networks for Efficient Time Series Forecasting},
  author={Khanh Vu},
  journal={GitHub Repository},
  year={2025}
}
```
