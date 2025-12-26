#!/usr/bin/env python3
# ==============================================================================
# MODULE: Generate Paper-Quality Diagrams for DynoNet
# ==============================================================================
# @context: Create clean, scientific diagrams for README and papers
# @goal: Professional architecture visualization
# ==============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

# Set paper style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5


def draw_architecture_diagram():
    """
    Draw DynoNet architecture - Clean paper style
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Colors (professional palette)
    C_INPUT = '#E3F2FD'   # Light blue
    C_CTRL = '#FFF3E0'    # Light orange  
    C_WORKER = '#E8F5E9'  # Light green
    C_OUTPUT = '#F3E5F5'  # Light purple
    C_BORDER = '#37474F'  # Dark gray
    C_ARROW = '#455A64'   # Medium gray
    
    # === TITLE ===
    ax.text(7, 7.6, 'DynoNet Architecture', fontsize=16, fontweight='bold', 
            ha='center', va='center')
    ax.text(7, 7.2, 'Dynamic Controller-Worker Paradigm for Time Series Forecasting', 
            fontsize=10, ha='center', va='center', style='italic', color='gray')
    
    # === INPUT BLOCK ===
    input_box = FancyBboxPatch((0.5, 3), 2, 1.5, boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor=C_INPUT, edgecolor=C_BORDER, linewidth=1.5)
    ax.add_patch(input_box)
    ax.text(1.5, 4.0, 'Input', fontsize=11, fontweight='bold', ha='center')
    ax.text(1.5, 3.5, r'$X \in \mathbb{R}^{B \times T \times 7}$', fontsize=9, ha='center')
    ax.text(1.5, 3.2, '(336 timesteps)', fontsize=8, ha='center', color='gray')
    
    # === REVIN NORMALIZE ===
    revin_box = FancyBboxPatch((3, 3), 1.8, 1.5, boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor='#ECEFF1', edgecolor=C_BORDER, linewidth=1.5)
    ax.add_patch(revin_box)
    ax.text(3.9, 3.9, 'RevIN', fontsize=10, fontweight='bold', ha='center')
    ax.text(3.9, 3.5, 'Normalize', fontsize=9, ha='center')
    
    # Arrow: Input -> RevIN
    ax.annotate('', xy=(3, 3.75), xytext=(2.5, 3.75),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))
    
    # === CONTROLLER (Top Path) ===
    ctrl_box = FancyBboxPatch((5.5, 5), 3.5, 2, boxstyle="round,pad=0.05,rounding_size=0.15",
                               facecolor=C_CTRL, edgecolor='#E65100', linewidth=2)
    ax.add_patch(ctrl_box)
    ax.text(7.25, 6.5, '🧠 Controller', fontsize=12, fontweight='bold', ha='center')
    ax.text(7.25, 6.0, 'ControlGRU (64 hidden)', fontsize=9, ha='center')
    
    # Controller outputs
    signals = ['FiLM (γ, β)', 'Gate Masks', 'LR Scale', 'Dropout']
    for i, sig in enumerate(signals):
        ax.text(5.7 + i*0.9, 5.3, sig, fontsize=7, ha='center', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#E65100', lw=0.5))
    
    # Arrow: RevIN -> Controller
    ax.annotate('', xy=(5.5, 6), xytext=(4.8, 4.2),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5, 
                              connectionstyle='arc3,rad=0.2'))
    
    # === SERIES DECOMPOSITION ===
    decomp_box = FancyBboxPatch((5.5, 1.5), 2, 1.2, boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor='#FFF8E1', edgecolor=C_BORDER, linewidth=1.5)
    ax.add_patch(decomp_box)
    ax.text(6.5, 2.3, 'Decomposition', fontsize=10, fontweight='bold', ha='center')
    ax.text(6.5, 1.9, 'Trend + Residual', fontsize=8, ha='center')
    
    # Arrow: RevIN -> Decomp
    ax.annotate('', xy=(5.5, 2.1), xytext=(4.8, 3.5),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5,
                              connectionstyle='arc3,rad=-0.2'))
    
    # === TREND BRANCH (Simple) ===
    trend_box = FancyBboxPatch((8, 0.5), 1.8, 1, boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor='#E0E0E0', edgecolor=C_BORDER, linewidth=1)
    ax.add_patch(trend_box)
    ax.text(8.9, 1.1, 'Linear', fontsize=9, fontweight='bold', ha='center')
    ax.text(8.9, 0.75, '(Shared)', fontsize=8, ha='center', color='gray')
    
    # === WORKERS (Bottom Path) ===
    worker_box = FancyBboxPatch((8, 2.5), 3.5, 2, boxstyle="round,pad=0.05,rounding_size=0.15",
                                 facecolor=C_WORKER, edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(worker_box)
    ax.text(9.75, 4.0, '💪 Distributed Workers', fontsize=11, fontweight='bold', ha='center')
    
    # Draw 7 mini GRU boxes
    for i in range(7):
        mini_box = FancyBboxPatch((8.2 + i*0.45, 2.7), 0.4, 0.9, 
                                   boxstyle="round,pad=0.02,rounding_size=0.05",
                                   facecolor='white', edgecolor='#2E7D32', linewidth=0.5)
        ax.add_patch(mini_box)
        ax.text(8.4 + i*0.45, 3.0, f'W{i+1}', fontsize=6, ha='center')
    
    ax.text(9.75, 3.7, '7 Independent GRU (8 hidden each)', fontsize=8, ha='center', color='gray')
    
    # Arrow: Decomp -> Workers (Residual)
    ax.annotate('', xy=(8, 3.5), xytext=(7.5, 2.3),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))
    ax.text(7.6, 2.9, 'Residual', fontsize=7, color='gray', rotation=45)
    
    # Arrow: Decomp -> Trend Linear
    ax.annotate('', xy=(8, 1), xytext=(7.5, 1.8),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))
    ax.text(7.5, 1.3, 'Trend', fontsize=7, color='gray', rotation=-30)
    
    # === CONTROLLER -> WORKERS (Modulation) ===
    ax.annotate('', xy=(9.75, 4.5), xytext=(7.25, 5),
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=2, ls='--'))
    ax.text(8.5, 4.9, 'Modulates', fontsize=8, color='#E65100', style='italic')
    
    # === CHANNEL MIXER ===
    mixer_box = FancyBboxPatch((10.5, 1.5), 1.5, 1.2, boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor='#E1F5FE', edgecolor=C_BORDER, linewidth=1.5)
    ax.add_patch(mixer_box)
    ax.text(11.25, 2.3, 'Channel', fontsize=9, fontweight='bold', ha='center')
    ax.text(11.25, 1.9, 'Mixer', fontsize=9, ha='center')
    
    # Arrows to Mixer
    ax.annotate('', xy=(10.5, 2.1), xytext=(9.8, 1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))
    ax.annotate('', xy=(10.5, 2.1), xytext=(11.5, 3.5),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))
    
    # === OUTPUT ===
    output_box = FancyBboxPatch((12.5, 2.5), 1.3, 1.5, boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor=C_OUTPUT, edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(output_box)
    ax.text(13.15, 3.5, 'Output', fontsize=11, fontweight='bold', ha='center')
    ax.text(13.15, 3.1, r'$\hat{Y}$', fontsize=10, ha='center')
    ax.text(13.15, 2.75, '(96 steps)', fontsize=8, ha='center', color='gray')
    
    # Arrow: Mixer -> Output
    ax.annotate('', xy=(12.5, 3.25), xytext=(12, 2.1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))
    
    # === LEGEND ===
    legend_elements = [
        mpatches.Patch(facecolor=C_CTRL, edgecolor='#E65100', label='Controller (Meta-Network)'),
        mpatches.Patch(facecolor=C_WORKER, edgecolor='#2E7D32', label='Workers (Base Network)'),
        plt.Line2D([0], [0], color='#E65100', linestyle='--', lw=2, label='Dynamic Modulation'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('png/architecture_diagram.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Saved: png/architecture_diagram.png")


def draw_training_strategy():
    """
    Draw Bi-Level Training Strategy - Paper style
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Colors
    C_TRAIN = '#BBDEFB'
    C_VAL = '#FFCCBC'
    C_BORDER = '#37474F'
    
    # Title
    ax.text(6, 5.6, 'Bi-Level Meta-Learning Training Strategy', 
            fontsize=14, fontweight='bold', ha='center')
    
    # === LEVEL 1: Worker Training ===
    level1_box = FancyBboxPatch((0.5, 2.5), 5, 2.5, boxstyle="round,pad=0.1,rounding_size=0.2",
                                 facecolor=C_TRAIN, edgecolor='#1565C0', linewidth=2)
    ax.add_patch(level1_box)
    ax.text(3, 4.7, 'Level 1: Base Training', fontsize=12, fontweight='bold', ha='center')
    
    # Train data
    ax.text(1.5, 4.0, '📊 Train Data', fontsize=10, ha='center')
    ax.text(1.5, 3.5, 'Batch Xₜ, Yₜ', fontsize=9, ha='center', color='gray')
    
    # Arrow
    ax.annotate('', xy=(3.5, 3.7), xytext=(2.3, 3.7),
                arrowprops=dict(arrowstyle='->', color=C_BORDER, lw=1.5))
    
    # Worker update
    ax.text(4.2, 4.0, '💪 Worker', fontsize=10, ha='center', fontweight='bold')
    ax.text(4.2, 3.5, 'θ_worker -= α∇L_train', fontsize=8, ha='center', 
            family='monospace', color='#1565C0')
    ax.text(4.2, 3.0, '+ Adaptive LR, WD, Dropout', fontsize=7, ha='center', color='gray')
    
    # === LEVEL 2: Controller Training ===
    level2_box = FancyBboxPatch((6.5, 2.5), 5, 2.5, boxstyle="round,pad=0.1,rounding_size=0.2",
                                 facecolor=C_VAL, edgecolor='#D84315', linewidth=2)
    ax.add_patch(level2_box)
    ax.text(9, 4.7, 'Level 2: Meta Training', fontsize=12, fontweight='bold', ha='center')
    
    # Val data
    ax.text(7.5, 4.0, '📊 Val Data', fontsize=10, ha='center')
    ax.text(7.5, 3.5, 'Batch Xᵥ, Yᵥ', fontsize=9, ha='center', color='gray')
    
    # Arrow
    ax.annotate('', xy=(9.5, 3.7), xytext=(8.3, 3.7),
                arrowprops=dict(arrowstyle='->', color=C_BORDER, lw=1.5))
    
    # Controller update
    ax.text(10.2, 4.0, '🧠 Controller', fontsize=10, ha='center', fontweight='bold')
    ax.text(10.2, 3.5, 'θ_ctrl -= β∇L_val', fontsize=8, ha='center', 
            family='monospace', color='#D84315')
    ax.text(10.2, 3.0, 'Learns optimal signals', fontsize=7, ha='center', color='gray')
    
    # === CONNECTION: Controller -> Worker ===
    ax.annotate('', xy=(5.5, 3.5), xytext=(6.5, 3.5),
                arrowprops=dict(arrowstyle='<->', color='#6A1B9A', lw=2, ls='--'))
    ax.text(6, 3.2, 'Modulates', fontsize=8, ha='center', color='#6A1B9A', style='italic')
    
    # === TIMELINE ===
    ax.plot([1, 11], [1.2, 1.2], color=C_BORDER, lw=2)
    
    # Timeline points
    points = [2, 4, 6, 8, 10]
    labels = ['Train\nBatch', 'Update\nWorker', 'Val\nBatch', 'Update\nCtrl', 'Repeat\n→']
    colors = ['#1565C0', '#1565C0', '#D84315', '#D84315', 'gray']
    
    for i, (x, label, color) in enumerate(zip(points, labels, colors)):
        ax.plot(x, 1.2, 'o', markersize=10, color=color)
        ax.text(x, 0.6, label, fontsize=8, ha='center', color=color)
    
    # Arrows between points
    for i in range(len(points)-1):
        ax.annotate('', xy=(points[i+1]-0.3, 1.2), xytext=(points[i]+0.3, 1.2),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    
    ax.text(6, 1.8, 'Training Loop (Each Epoch)', fontsize=9, ha='center', 
            style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig('png/training_strategy.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Saved: png/training_strategy.png")


def draw_component_diagram():
    """
    Draw detailed component breakdown
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    
    for ax in axes:
        ax.axis('off')
    
    # === Panel 1: RevIN ===
    ax1 = axes[0]
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)
    ax1.set_title('(a) RevIN Layer', fontsize=12, fontweight='bold', pad=10)
    
    # Input
    ax1.add_patch(FancyBboxPatch((0.5, 3.5), 1.5, 1, boxstyle="round,pad=0.05",
                                  facecolor='#E3F2FD', edgecolor='#37474F'))
    ax1.text(1.25, 4, 'Input X', fontsize=9, ha='center')
    
    # Normalize
    ax1.add_patch(FancyBboxPatch((0.5, 2), 1.5, 1, boxstyle="round,pad=0.05",
                                  facecolor='#FFF3E0', edgecolor='#37474F'))
    ax1.text(1.25, 2.5, 'Normalize', fontsize=9, ha='center')
    ax1.text(1.25, 2.2, r'$\frac{x-\mu}{\sigma}$', fontsize=8, ha='center')
    
    # Affine
    ax1.add_patch(FancyBboxPatch((2.5, 2), 1.5, 1, boxstyle="round,pad=0.05",
                                  facecolor='#E8F5E9', edgecolor='#37474F'))
    ax1.text(3.25, 2.5, 'Affine', fontsize=9, ha='center')
    ax1.text(3.25, 2.2, r'$\gamma x + \beta$', fontsize=8, ha='center')
    
    # Denorm
    ax1.add_patch(FancyBboxPatch((2.5, 3.5), 1.5, 1, boxstyle="round,pad=0.05",
                                  facecolor='#F3E5F5', edgecolor='#37474F'))
    ax1.text(3.25, 4, 'Denorm', fontsize=9, ha='center')
    ax1.text(3.25, 3.7, r'$x \cdot \sigma + \mu$', fontsize=8, ha='center')
    
    # Arrows
    ax1.annotate('', xy=(1.25, 3.5), xytext=(1.25, 3),
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.annotate('', xy=(2.5, 2.5), xytext=(2, 2.5),
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.annotate('', xy=(3.25, 3.5), xytext=(3.25, 3),
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax1.text(2.25, 1, 'Reversible Instance\nNormalization', fontsize=8, 
             ha='center', style='italic', color='gray')
    
    # === Panel 2: FiLM Modulation ===
    ax2 = axes[1]
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 5)
    ax2.set_title('(b) FiLM Modulation', fontsize=12, fontweight='bold', pad=10)
    
    # Context
    ax2.add_patch(FancyBboxPatch((1.5, 3.5), 2, 1, boxstyle="round,pad=0.05",
                                  facecolor='#FFF3E0', edgecolor='#E65100'))
    ax2.text(2.5, 4, 'Context', fontsize=9, ha='center')
    ax2.text(2.5, 3.7, 'from Controller', fontsize=7, ha='center', color='gray')
    
    # Gamma, Beta
    ax2.add_patch(FancyBboxPatch((0.5, 2), 1.5, 1, boxstyle="round,pad=0.05",
                                  facecolor='#FFECB3', edgecolor='#E65100'))
    ax2.text(1.25, 2.5, r'$\gamma$', fontsize=12, ha='center')
    
    ax2.add_patch(FancyBboxPatch((3, 2), 1.5, 1, boxstyle="round,pad=0.05",
                                  facecolor='#FFECB3', edgecolor='#E65100'))
    ax2.text(3.75, 2.5, r'$\beta$', fontsize=12, ha='center')
    
    # Formula
    ax2.text(2.5, 1.2, r'$h_{out} = \gamma \cdot h_{in} + \beta$', fontsize=11, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
    
    ax2.text(2.5, 0.5, 'Feature-wise Linear\nModulation', fontsize=8, 
             ha='center', style='italic', color='gray')
    
    # Arrows
    ax2.annotate('', xy=(1.25, 3), xytext=(1.8, 3.5),
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax2.annotate('', xy=(3.75, 3), xytext=(3.2, 3.5),
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    # === Panel 3: Series Decomposition ===
    ax3 = axes[2]
    ax3.set_xlim(0, 5)
    ax3.set_ylim(0, 5)
    ax3.set_title('(c) Series Decomposition', fontsize=12, fontweight='bold', pad=10)
    
    # Input series
    t = np.linspace(0, 4, 100)
    trend = 0.3 * np.sin(0.5*t) + 2.5
    noise = 0.15 * np.sin(5*t + np.random.randn(100)*0.5)
    series = trend + noise
    
    ax3.plot(t + 0.5, series + 1.5, 'b-', lw=1.5, label='Input')
    ax3.plot(t + 0.5, trend, 'g--', lw=1.5, label='Trend')
    ax3.plot(t + 0.5, noise + 1.5, 'r:', lw=1, label='Residual')
    
    ax3.text(2.5, 4.5, 'Moving Average\nDecomposition', fontsize=9, ha='center')
    ax3.legend(loc='lower right', fontsize=7)
    
    ax3.text(2.5, 0.5, r'$X = X_{trend} + X_{residual}$', fontsize=10, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig('png/components_detail.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Saved: png/components_detail.png")


if __name__ == "__main__":
    print("Generating paper-quality diagrams...")
    draw_architecture_diagram()
    draw_training_strategy()
    draw_component_diagram()
    print("\n✅ All diagrams generated successfully!")
