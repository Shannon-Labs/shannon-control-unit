#!/usr/bin/env python3
"""
Professional Ablation Analysis Generator
Creates investor-grade visualizations and CSV exports from SCU ablation data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple

# Professional styling
plt.style.use('default')
sns.set_palette("husl")

# Create consistent color scheme
COLORS = {
    'PI_control': '#2E86AB',    # Professional blue
    'fixed_0.5': '#A23B72',     # Purple
    'fixed_1.0': '#F18F01',     # Orange
    'fixed_2.0': '#C73E1D',     # Red
    'target_band': '#E8E8E8'    # Light gray for target bands
}

def load_ablation_data() -> Dict[str, pd.DataFrame]:
    """Load all ablation CSV files."""
    ablation_dir = Path('ablations')
    data = {}
    
    # Load PI control data
    pi_file = ablation_dir / 'pi_control.csv'
    if pi_file.exists():
        data['PI_control'] = pd.read_csv(pi_file)
        print(f"✓ Loaded PI control data: {len(data['PI_control'])} steps")
    
    # Load fixed lambda data
    for lambda_val in [0.5, 1.0, 2.0, 5.0]:
        fixed_file = ablation_dir / f'fixed_{lambda_val}.csv'
        if fixed_file.exists():
            df = pd.read_csv(fixed_file)
            if len(df) > 1:  # Only include if has meaningful data
                data[f'fixed_{lambda_val}'] = df
                print(f"✓ Loaded fixed λ={lambda_val} data: {len(df)} steps")
            else:
                print(f"⚠️  Skipped fixed λ={lambda_val}: insufficient data ({len(df)} steps)")
    
    return data

def create_s_tracking_comparison(data: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Create S-tracking comparison plot."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Target band for PI control (1% ± 0.2pp)
    target = 0.01
    tolerance = 0.002
    ax.axhspan(target - tolerance, target + tolerance, 
               color=COLORS['target_band'], alpha=0.3, label='Target band (1.0% ± 0.2pp)')
    ax.axhline(target, color='black', linestyle='--', alpha=0.5)
    
    # Plot each ablation
    for name, df in data.items():
        if 'S' in df.columns and len(df) > 10:
            color = COLORS.get(name, '#333333')
            if name == 'PI_control':
                ax.plot(df['step'], df['S'] * 100, color=color, linewidth=2.5, 
                       label='PI Control (Adaptive)', alpha=0.9)
            else:
                lambda_val = name.split('_')[1]
                ax.plot(df['step'], df['S'] * 100, color=color, linewidth=1.5,
                       label=f'Fixed λ={lambda_val}', alpha=0.7)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Information Ratio S (%)', fontsize=12)
    ax.set_title('Ablation Study: S-Tracking Performance\n(PI Control vs Fixed Lambda)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Make it look professional
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    # Save in multiple formats
    fig.savefig(output_dir / 'ablation_s_tracking.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'ablation_s_tracking.svg', bbox_inches='tight')
    plt.close()
    print("✓ Generated S-tracking comparison plot")

def create_lambda_evolution_plot(data: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Create lambda evolution comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot each configuration
    for name, df in data.items():
        if 'lambda' in df.columns and len(df) > 10:
            color = COLORS.get(name, '#333333')
            if name == 'PI_control':
                ax.semilogy(df['step'], df['lambda'], color=color, linewidth=2.5,
                           label='PI Control (Adaptive)', alpha=0.9)
            else:
                lambda_val = name.split('_')[1]
                ax.semilogy(df['step'], df['lambda'], color=color, linewidth=1.5,
                           label=f'Fixed λ={lambda_val}', alpha=0.7)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Lambda (log scale)', fontsize=12)
    ax.set_title('Ablation Study: Lambda Evolution\n(Adaptive vs Fixed Regularization)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    fig.savefig(output_dir / 'ablation_lambda_evolution.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'ablation_lambda_evolution.svg', bbox_inches='tight')
    plt.close()
    print("✓ Generated lambda evolution plot")

def create_final_performance_comparison(data: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Create final BPT performance comparison."""
    results = []
    
    # Calculate final performance for each configuration
    for name, df in data.items():
        if len(df) > 10:
            # Take average of last 20% of training
            final_portion = df.tail(max(1, len(df) // 5))
            final_data_bpt = final_portion['data_bpt'].mean()
            final_s = final_portion['S'].mean() * 100
            
            if name == 'PI_control':
                display_name = 'PI Control'
                category = 'Adaptive'
            else:
                lambda_val = name.split('_')[1]
                display_name = f'Fixed λ={lambda_val}'
                category = 'Fixed'
            
            results.append({
                'name': display_name,
                'category': category,
                'data_bpt': final_data_bpt,
                'final_s': final_s,
                'config': name
            })
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Data BPT comparison
    df_results = pd.DataFrame(results)
    categories = df_results['category'].unique()
    x_pos = np.arange(len(df_results))
    
    colors = [COLORS.get(r['config'], '#333333') for r in results]
    bars1 = ax1.bar(x_pos, df_results['data_bpt'], color=colors, alpha=0.8)
    ax1.set_xlabel('Configuration', fontsize=12)
    ax1.set_ylabel('Final Data BPT (bits/token)', fontsize=12)
    ax1.set_title('Final Performance: Data BPT', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([r['name'] for r in results], rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # S tracking accuracy
    bars2 = ax2.bar(x_pos, df_results['final_s'], color=colors, alpha=0.8)
    ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Target (1.0%)')
    ax2.axhspan(0.8, 1.2, color=COLORS['target_band'], alpha=0.3, label='Target band')
    ax2.set_xlabel('Configuration', fontsize=12)
    ax2.set_ylabel('Final S (%)', fontsize=12)
    ax2.set_title('Final Performance: S Tracking', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([r['name'] for r in results], rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'ablation_final_performance.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'ablation_final_performance.svg', bbox_inches='tight')
    plt.close()
    print("✓ Generated final performance comparison")
    
    return results

def export_analysis_csvs(data: Dict[str, pd.DataFrame], results: List[Dict], output_dir: Path) -> None:
    """Export analysis data to CSV files."""
    
    # Create data directory
    data_dir = output_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Export summary results
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(data_dir / 'ablation_summary.csv', index=False)
    print(f"✓ Exported ablation summary: {len(summary_df)} configurations")
    
    # Export individual run data with metadata
    for name, df in data.items():
        if len(df) > 1:
            # Add metadata columns
            df_export = df.copy()
            df_export['configuration'] = name
            df_export['lambda_type'] = 'adaptive' if name == 'PI_control' else 'fixed'
            
            if name != 'PI_control':
                lambda_val = name.split('_')[1]
                df_export['lambda_value'] = float(lambda_val)
            else:
                df_export['lambda_value'] = 'adaptive'
            
            # Export
            df_export.to_csv(data_dir / f'{name}_detailed.csv', index=False)
            print(f"✓ Exported detailed data for {name}: {len(df_export)} steps")

def create_validation_comparison_plot(output_dir: Path) -> None:
    """Create validation results comparison using the updated 3B results."""
    
    # Load validation data
    with open('results/3b_validation_results.json', 'r') as f:
        validation_data = json.load(f)
    
    results = validation_data['results']
    
    # Extract data
    models = []
    bpt_values = []
    perplexity_values = []
    colors = []
    
    for result in results:
        model_name = result['model']
        if model_name == 'base_model':
            models.append('Baseline')
            colors.append('#666666')
        elif model_name == '3b-scu':
            models.append('SCU (PI)')
            colors.append(COLORS['PI_control'])
        elif model_name == '3b-fixed':
            models.append('Fixed λ=0.5')
            colors.append(COLORS['fixed_0.5'])
        
        bpt_values.append(result['avg_bpt'])
        perplexity_values.append(result['avg_perplexity'])
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # BPT comparison
    bars1 = ax1.bar(models, bpt_values, color=colors, alpha=0.8)
    ax1.set_ylabel('Average BPT (bits/token)', fontsize=12)
    ax1.set_title('3B Model Validation: BPT Performance', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add improvement annotations
    baseline_bpt = bpt_values[0]
    for i, (bar, bpt) in enumerate(zip(bars1, bpt_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        if i > 0:  # Not baseline
            improvement = (baseline_bpt - bpt) / baseline_bpt * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{improvement:+.1f}%', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
    
    # Perplexity comparison
    bars2 = ax2.bar(models, perplexity_values, color=colors, alpha=0.8)
    ax2.set_ylabel('Average Perplexity', fontsize=12)
    ax2.set_title('3B Model Validation: Perplexity', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    baseline_ppl = perplexity_values[0]
    for i, (bar, ppl) in enumerate(zip(bars2, perplexity_values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        if i > 0:
            improvement = (baseline_ppl - ppl) / baseline_ppl * 100
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{improvement:+.1f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
    
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'validation_3b_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'validation_3b_comparison.svg', bbox_inches='tight')
    plt.close()
    print("✓ Generated 3B validation comparison plot")

def main():
    """Generate all ablation analysis materials."""
    print("🚀 Shannon Control Unit - Professional Ablation Analysis")
    print("=" * 60)
    
    # Setup directories
    output_dir = Path('assets/figures')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n📊 Loading ablation data...")
    data = load_ablation_data()
    
    if not data:
        print("❌ No ablation data found!")
        return 1
    
    print(f"\n✓ Loaded {len(data)} ablation configurations")
    
    # Generate visualizations
    print("\n🎨 Generating professional visualizations...")
    
    # Ablation plots
    if len(data) > 1:
        create_s_tracking_comparison(data, output_dir)
        create_lambda_evolution_plot(data, output_dir)
        results = create_final_performance_comparison(data, output_dir)
        
        # Export analysis data
        print("\n📈 Exporting analysis data...")
        export_analysis_csvs(data, results, output_dir)
    else:
        print("⚠️  Need multiple configurations for comparison plots")
    
    # Validation comparison
    print("\n🔬 Generating validation comparison...")
    create_validation_comparison_plot(output_dir)
    
    print("\n✅ Ablation analysis complete!")
    print(f"📁 Output directory: {output_dir}/")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('ablation_*.png')):
        print(f"  - {file.name}")
    for file in sorted(output_dir.glob('validation_*.png')):
        print(f"  - {file.name}")
    
    data_dir = output_dir / 'data'
    if data_dir.exists():
        print("\nData exports:")
        for file in sorted(data_dir.glob('*.csv')):
            print(f"  - data/{file.name}")
    
    return 0

if __name__ == '__main__':
    exit(main())
