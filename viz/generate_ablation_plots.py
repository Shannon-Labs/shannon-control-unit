#!/usr/bin/env python3
"""Generate ablation study plots from CSV data."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def load_ablation_data():
    """Load all ablation study CSVs."""
    data = {}
    
    # Load PI control
    pi_df = pd.read_csv('ablations/pi_control.csv')
    data['PI Control'] = pi_df
    
    # Load fixed lambda studies
    for lambda_val in [1.0, 5.0]:
        path = f'ablations/fixed_{lambda_val}.csv'
        if Path(path).exists():
            df = pd.read_csv(path)
            data[f'Fixed λ={lambda_val}'] = df
    
    return data

def plot_ablation_comparison():
    """Create comprehensive ablation study comparison plot."""
    data = load_ablation_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Ablation Study: PI Control vs Fixed Lambda', fontsize=14, fontweight='bold')
    
    colors = {'PI Control': '#0052E0', 'Fixed λ=1.0': '#FF6B6B', 'Fixed λ=5.0': '#4ECDC4'}
    
    for label, df in data.items():
        color = colors.get(label, '#888888')
        steps = df['step'].values[:150]  # First 150 steps
        
        # Plot 1: Data BPT
        axes[0, 0].plot(steps, df['data_bpt'].values[:150], label=label, 
                       color=color, linewidth=1.5, alpha=0.8)
        
        # Plot 2: S ratio
        axes[0, 1].plot(steps, df['S'].values[:150] * 100, label=label,
                       color=color, linewidth=1.5, alpha=0.8)
        
        # Plot 3: Lambda evolution
        if 'lambda' in df.columns:
            axes[1, 0].plot(steps, df['lambda'].values[:150], label=label,
                           color=color, linewidth=1.5, alpha=0.8)
        
        # Plot 4: Param BPT
        axes[1, 1].plot(steps, df['param_bpt'].values[:150], label=label,
                       color=color, linewidth=1.5, alpha=0.8)
    
    # Add target band for S ratio
    axes[0, 1].axhspan(0.8, 1.2, alpha=0.1, color='green', label='Target: 1.0% ± 0.2pp')
    axes[0, 1].axhline(y=1.0, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Configure axes
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Data BPT')
    axes[0, 0].set_title('Data BPT Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('S (%)')
    axes[0, 1].set_title('Information Ratio S(t)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Lambda (λ)')
    axes[1, 0].set_title('Lambda Adaptation')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Param BPT')
    axes[1, 1].set_title('Parameter BPT Evolution')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('assets/figures/ablation_study.png', bbox_inches='tight', facecolor='white')
    plt.savefig('assets/figures/ablation_study.svg', bbox_inches='tight', facecolor='white')
    print("✓ Generated ablation_study.png/svg")
    
    return fig

def plot_convergence_analysis():
    """Plot convergence analysis for different methods."""
    data = load_ablation_data()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'PI Control': '#0052E0', 'Fixed λ=1.0': '#FF6B6B', 'Fixed λ=5.0': '#4ECDC4'}
    
    for label, df in data.items():
        color = colors.get(label, '#888888')
        steps = df['step'].values[:150]
        
        # Calculate moving average of S ratio distance from target
        target = 0.01  # 1%
        s_error = np.abs(df['S'].values[:150] - target)
        
        # Moving average with window of 10
        window = 10
        s_error_smooth = np.convolve(s_error, np.ones(window)/window, mode='valid')
        steps_smooth = steps[:len(s_error_smooth)]
        
        ax.plot(steps_smooth, s_error_smooth * 100, label=label, 
                color=color, linewidth=2)
    
    ax.axhline(y=0.2, color='green', linestyle='--', linewidth=1, 
               alpha=0.5, label='Tolerance: ±0.2pp')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('|S - S*| (%)', fontsize=12)
    ax.set_title('Convergence Analysis: Distance from Target S*=1%', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('assets/figures/convergence_analysis.png', bbox_inches='tight', facecolor='white')
    plt.savefig('assets/figures/convergence_analysis.svg', bbox_inches='tight', facecolor='white')
    print("✓ Generated convergence_analysis.png/svg")
    
    return fig

if __name__ == '__main__':
    print("\nGenerating ablation study visualizations...")
    print("=" * 60)
    
    # Generate plots
    fig1 = plot_ablation_comparison()
    fig2 = plot_convergence_analysis()
    
    print("\n✅ Successfully generated ablation plots!")
    print("Files saved to assets/figures/")