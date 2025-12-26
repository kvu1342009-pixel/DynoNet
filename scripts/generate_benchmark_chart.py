#!/usr/bin/env python3
# ==============================================================================
# MODULE: Generate SOTA Benchmark Chart
# ==============================================================================
# @context: Create publication-quality benchmark comparison chart
# @goal: Visualize MSE, MAE, and Model Size for ETTh1 M→M forecasting
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0

def create_sota_benchmark_chart():
    """
    Create a grouped bar chart comparing SOTA models.
    Metrics: MSE, MAE, and Parameters (in green)
    """
    # Data from benchmark
    models = ['TimesNet', 'DynoNet\n(Ours)', 'iTransformer', 'PatchTST', 
              'Crossformer', 'DLinear', 'Autoformer']
    
    mse = [0.384, 0.386, 0.386, 0.414, 0.423, 0.456, 0.449]
    mae = [0.402, 0.415, 0.405, 0.419, 0.448, 0.452, 0.459]
    params_k = [500, 94, 500, 550, 1000, 10, 500]  # in thousands
    
    # Normalize params for visualization (scale to similar range as MSE/MAE)
    params_normalized = [p / 1000 for p in params_k]  # Scale to 0-1 range
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=150)
    
    # Colors
    color_mse = '#2196F3'     # Blue
    color_mae = '#FF9800'     # Orange
    color_params = '#4CAF50'  # Green
    
    # Bar plots for MSE and MAE
    bars1 = ax1.bar(x - width, mse, width, label='MSE', color=color_mse, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x, mae, width, label='MAE', color=color_mae, edgecolor='black', linewidth=0.5)
    
    ax1.set_ylabel('MSE / MAE', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.3, 0.55)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    
    # Create second y-axis for Parameters
    ax2 = ax1.twinx()
    bars3 = ax2.bar(x + width, params_k, width, label='Parameters (K)', color=color_params, 
                    edgecolor='black', linewidth=0.5, alpha=0.8)
    ax2.set_ylabel('Parameters (×1000)', fontsize=12, fontweight='bold', color=color_params)
    ax2.set_ylim(0, 1200)
    ax2.tick_params(axis='y', labelcolor=color_params)
    
    # Highlight DynoNet (index 1)
    for bar_group in [bars1, bars2, bars3]:
        bar_group[1].set_edgecolor('red')
        bar_group[1].set_linewidth(2.5)
    
    # Add value labels on bars
    def add_labels(bars, ax, fmt='{:.3f}', offset=0.01, fontsize=8):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(fmt.format(height),
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=fontsize)
    
    add_labels(bars1, ax1, fmt='{:.3f}', fontsize=8)
    add_labels(bars2, ax1, fmt='{:.3f}', fontsize=8)
    add_labels(bars3, ax2, fmt='{:.0f}K', fontsize=8)
    
    # Title
    ax1.set_title('ETTh1 Multivariate Forecasting (M→M, Horizon=96)\nModel Comparison: MSE, MAE, and Parameters', 
                  fontsize=14, fontweight='bold', pad=15)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add annotation for DynoNet
    ax1.annotate('★ Best Efficiency', 
                xy=(1, 0.386), xytext=(1.5, 0.50),
                fontsize=10, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    # Grid
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('png/sota_benchmark_chart.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Saved: png/sota_benchmark_chart.png")


def create_efficiency_scatter():
    """
    Create a scatter plot showing MSE vs Parameters trade-off.
    """
    models = ['TimesNet', 'DynoNet', 'iTransformer', 'PatchTST', 
              'Crossformer', 'DLinear', 'Autoformer', 'FEDformer']
    
    mse = [0.384, 0.386, 0.386, 0.414, 0.423, 0.456, 0.449, 0.376]
    params_k = [500, 94, 500, 550, 1000, 10, 500, 500]
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    
    # Scatter points
    colors = ['#2196F3'] * len(models)
    colors[1] = '#E91E63'  # DynoNet in pink/red
    
    sizes = [100] * len(models)
    sizes[1] = 200  # DynoNet larger
    
    for i, (model, x, y) in enumerate(zip(models, params_k, mse)):
        ax.scatter(x, y, s=sizes[i], c=colors[i], edgecolors='black', 
                  linewidths=1.5, zorder=5, alpha=0.8)
        
        # Label offset
        offset = (15, 5) if model != 'DynoNet' else (-60, -15)
        ax.annotate(model, (x, y), textcoords="offset points", 
                   xytext=offset, fontsize=9,
                   fontweight='bold' if model == 'DynoNet' else 'normal')
    
    # Highlight DynoNet region
    circle = plt.Circle((94, 0.386), 30, color='#E91E63', alpha=0.1)
    ax.add_patch(circle)
    
    ax.set_xlabel('Parameters (×1000)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSE ↓', fontsize=12, fontweight='bold')
    ax.set_title('Efficiency vs Accuracy Trade-off\nETTh1 M→M Horizon=96', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Add quadrant lines
    ax.axhline(y=0.40, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=300, color='gray', linestyle='--', alpha=0.5)
    
    # Annotations for quadrants
    ax.text(50, 0.35, 'Best: Low Params\nLow Error', fontsize=9, 
           color='green', ha='center', style='italic')
    ax.text(800, 0.35, 'High Params\nLow Error', fontsize=9, 
           color='orange', ha='center', style='italic')
    
    ax.set_xlim(-50, 1100)
    ax.set_ylim(0.35, 0.48)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('png/efficiency_vs_accuracy.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Saved: png/efficiency_vs_accuracy.png")


if __name__ == "__main__":
    print("Generating SOTA benchmark charts...")
    create_sota_benchmark_chart()
    create_efficiency_scatter()
    print("\n✅ All charts generated!")
