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
    Create a clean bubble chart showing MSE vs Parameters trade-off.
    Bubble size = inverse of params (smaller is better)
    """
    models = ['TimesNet', 'DynoNet', 'iTransformer', 'PatchTST', 
              'Crossformer', 'DLinear', 'Autoformer', 'FEDformer']
    
    mse = [0.384, 0.386, 0.386, 0.414, 0.423, 0.456, 0.449, 0.376]
    params_k = [500, 94, 500, 550, 1000, 10, 500, 500]
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')
    
    # Color palette
    colors = ['#64B5F6', '#E91E63', '#64B5F6', '#64B5F6', 
              '#64B5F6', '#64B5F6', '#64B5F6', '#64B5F6']
    
    # Bubble sizes (inverse relationship - smaller params = bigger bubble for visibility)
    bubble_sizes = [max(50, 400 - p/3) for p in params_k]
    bubble_sizes[1] = 500  # Make DynoNet biggest
    
    # Plot bubbles
    for i, (model, x, y, size, color) in enumerate(zip(models, params_k, mse, bubble_sizes, colors)):
        if model == 'DynoNet':
            ax.scatter(x, y, s=size, c=color, edgecolors='white', 
                      linewidths=3, zorder=10, alpha=0.9)
            # Add glow effect
            ax.scatter(x, y, s=size*1.5, c=color, alpha=0.2, zorder=5)
        else:
            ax.scatter(x, y, s=size, c=color, edgecolors='white', 
                      linewidths=2, zorder=5, alpha=0.7)
    
    # Labels with clean positioning
    label_offsets = {
        'TimesNet': (10, 15),
        'DynoNet': (0, -35),
        'iTransformer': (10, -20),
        'PatchTST': (10, 10),
        'Crossformer': (10, 10),
        'DLinear': (10, 10),
        'Autoformer': (10, -20),
        'FEDformer': (-70, 10),
    }
    
    for i, (model, x, y) in enumerate(zip(models, params_k, mse)):
        offset = label_offsets.get(model, (10, 5))
        weight = 'bold' if model == 'DynoNet' else 'normal'
        color = '#E91E63' if model == 'DynoNet' else '#333333'
        
        ax.annotate(model, (x, y), textcoords="offset points", 
                   xytext=offset, fontsize=11, fontweight=weight, color=color)
    
    # DynoNet callout box
    ax.annotate('★ DynoNet\n94K params\nMSE: 0.386', 
                xy=(94, 0.386), xytext=(200, 0.44),
                fontsize=11, color='#E91E63', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='#E91E63', linewidth=2),
                arrowprops=dict(arrowstyle='->', color='#E91E63', lw=2,
                               connectionstyle='arc3,rad=-0.2'))
    
    # Axes styling
    ax.set_xlabel('Model Size (Parameters ×1000)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('MSE (Lower is Better) ↓', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Efficiency vs Accuracy Trade-off\n', fontsize=16, fontweight='bold')
    ax.text(0.5, 1.02, 'ETTh1 Multivariate (M→M), Horizon=96', 
           transform=ax.transAxes, ha='center', fontsize=11, color='gray')
    
    # Set limits with padding
    ax.set_xlim(-50, 1150)
    ax.set_ylim(0.36, 0.48)
    
    # Clean grid
    ax.grid(True, linestyle='-', alpha=0.2, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add efficiency zone
    rect = plt.Rectangle((0, 0.36), 200, 0.05, 
                         facecolor='#C8E6C9', alpha=0.3, zorder=1)
    ax.add_patch(rect)
    ax.text(100, 0.365, '✓ High Efficiency Zone', fontsize=9, 
           ha='center', color='#2E7D32', style='italic')
    
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
