#!/usr/bin/env python3
"""
Train Qwen3 with Shannon Control Unit (SCU) v1.0
Uses the proven PI controller from the original paper.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from scripts.train_scu import main
from argparse import Namespace
import torch

def setup_qwen_config(use_scu=True, experiment_name="qwen_scu"):
    """Setup configuration for Qwen3 training"""
    
    # Detect device and optimize settings
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        use_4bit = True
        batch_size = 4
        gradient_accumulation = 8
        print("🚀 Using CUDA with 4-bit quantization")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        use_4bit = False
        batch_size = 2
        gradient_accumulation = 8
        print("🍎 Using Apple Silicon MPS")
    else:
        device = "cpu"
        dtype = torch.float32
        use_4bit = False
        batch_size = 1
        gradient_accumulation = 16
        print("⚠️  Using CPU (will be slow)")
    
    print(f"Device: {device}, Batch size: {batch_size}, Accumulation: {gradient_accumulation}")
    
    config = Namespace(
        # Model
        base_model="./Qwen3-1.7B-Base",
        adapter_out=f"adapters/{experiment_name}",
        
        # SCU Control (proven values from paper)
        target_s=0.01,      # 1% target S-ratio
        kp=0.8,             # Proportional gain
        ki=0.15,            # Integral gain
        deadband=0.002,     # ±0.2% deadband
        lambda_init=1.0,    # Initial lambda
        lambda_min=1e-4,    # Min lambda (anti-windup)
        lambda_max=2.0,     # Max lambda (safety)
        
        # Training
        prior_sigma=0.01,   # Prior std dev
        epochs=1,           # 1 epoch
        steps=None,         # Use epochs
        batch_size=batch_size,
        lr=5e-5,            # Learning rate
        block_size=1024,    # Context length (Qwen supports up to 32K)
        gradient_accumulation_steps=gradient_accumulation,
        fp16=(device == "cuda"),  # Only use fp16 on CUDA
        seed=42,
        
        # Data - use train.txt for quick experiments, slimpajama for full
        train_data="data/train.txt",
        max_texts=2000 if use_scu else 1000,  # Smaller for baseline
        
        # Logging
        log_csv=f"logs/{experiment_name}.csv",
        quickstart=False
    )
    
    return config

def run_qwen_scu_training():
    """Run Qwen3 training with SCU"""
    
    print("=" * 80)
    print("QWEN3 + SHANNON CONTROL UNIT v1.0")
    print("=" * 80)
    print()
    print("Model: Qwen3-1.7B-Base (1.7B parameters)")
    print("Control: PI controller with adaptive regularization")
    print("Target: Maintain S-ratio at 1% throughout training")
    print()
    
    # Run with SCU
    print("🔬 Running: Qwen3 with SCU adaptive regularization")
    print("-" * 80)
    
    config_scu = setup_qwen_config(use_scu=True, experiment_name="qwen3_scu_v1")
    
    print(f"Configuration:")
    print(f"  Model: {config_scu.base_model}")
    print(f"  Output: {config_scu.adapter_out}")
    print(f"  Target S: {config_scu.target_s:.1%}")
    print(f"  Control: Kp={config_scu.kp}, Ki={config_scu.ki}")
    print(f"  Training: {config_scu.epochs} epoch, batch_size={config_scu.batch_size}")
    print(f"  Data: {config_scu.train_data} (max {config_scu.max_texts} texts)")
    print()
    
    try:
        main(config_scu)
        print()
        print("✅ Qwen3 + SCU training completed successfully!")
        print()
        
        # Show results
        print("Results:")
        print(f"  📁 Adapter saved: {config_scu.adapter_out}")
        print(f"  📊 Training log: {config_scu.log_csv}")
        print(f"  ℹ️  Metadata: {config_scu.adapter_out}/metadata.json")
        print()
        
        # Quick analysis
        import pandas as pd
        try:
            df = pd.read_csv(config_scu.log_csv)
            if not df.empty:
                print("Training Summary:")
                print(f"  Steps: {len(df)}")
                print(f"  Final S-ratio: {df['S'].iloc[-1]:.2%}")
                print(f"  Final lambda: {df['lambda'].iloc[-1]:.3f}")
                print(f"  Data BPT: {df['data_bpt'].iloc[-1]:.3f}")
                print(f"  Param BPT: {df['param_bpt'].iloc[-1]:.4f}")
                print()
                
                # Control performance
                s_mean = df['S'].mean()
                s_std = df['S'].std()
                print(f"Control Performance:")
                print(f"  Mean S-ratio: {s_mean:.2%} (target: {config_scu.target_s:.1%})")
                print(f"  Std dev: {s_std:.3%}")
                print(f"  Tracking error: {abs(s_mean - config_scu.target_s):.3%}")
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_qwen_baseline():
    """Run Qwen3 baseline without SCU for comparison"""
    
    print("=" * 80)
    print("QWEN3 BASELINE (No SCU)")
    print("=" * 80)
    print()
    print("🔬 Running: Qwen3 without SCU (fixed lambda = 0)")
    print("-" * 80)
    
    config_baseline = setup_qwen_config(use_scu=False, experiment_name="qwen3_baseline")
    
    # Disable SCU by setting lambda bounds to 0
    config_baseline.lambda_init = 0
    config_baseline.lambda_min = 0
    config_baseline.lambda_max = 0
    
    print(f"Configuration:")
    print(f"  Model: {config_baseline.base_model}")
    print(f"  Output: {config_baseline.adapter_out}")
    print(f"  SCU: DISABLED (lambda = 0)")
    print(f"  Training: {config_baseline.epochs} epoch, batch_size={config_baseline.batch_size}")
    print()
    
    try:
        main(config_baseline)
        print()
        print("✅ Qwen3 baseline training completed!")
        print(f"  📁 Adapter saved: {config_baseline.adapter_out}")
        print(f"  📊 Training log: {config_baseline.log_csv}")
        return True
    except Exception as e:
        print(f"❌ Baseline training failed: {e}")
        return False

def compare_results():
    """Compare SCU vs baseline results"""
    
    print("=" * 80)
    print("COMPARISON: SCU vs BASELINE")
    print("=" * 80)
    print()
    
    try:
        import pandas as pd
        
        # Load SCU results
        scu_log = "logs/qwen3_scu_v1.csv"
        baseline_log = "logs/qwen3_baseline.csv"
        
        scu_df = pd.read_csv(scu_log)
        baseline_df = pd.read_csv(baseline_log)
        
        print("Final Metrics:")
        print(f"{'Metric':<20} {'SCU':<15} {'Baseline':<15} {'Difference':<15}")
        print("-" * 80)
        
        # Compare final values
        scu_final = scu_df.iloc[-1]
        baseline_final = baseline_df.iloc[-1]
        
        metrics = ['data_bpt', 'param_bpt', 'total_bpt', 'S', 'lambda']
        
        for metric in metrics:
            if metric in scu_df.columns and metric in baseline_df.columns:
                scu_val = scu_final[metric]
                baseline_val = baseline_final[metric]
                diff = scu_val - baseline_val
                
                if metric == 'S':
                    print(f"{metric:<20} {scu_val:<15.2%} {baseline_val:<15.2%} {diff:<15.2%}")
                elif metric == 'lambda':
                    print(f"{metric:<20} {scu_val:<15.3f} {baseline_val:<15.3f} {diff:<15.3f}")
                else:
                    print(f"{metric:<20} {scu_val:<15.4f} {baseline_val:<15.4f} {diff:<15.4f}")
        
        print()
        
        # BPT improvement
        scu_total_bpt = scu_final['total_bpt']
        baseline_total_bpt = baseline_final['total_bpt']
        improvement = (baseline_total_bpt - scu_total_bpt) / baseline_total_bpt * 100
        
        print(f"BPT Improvement: {improvement:.1f}%")
        print()
        
        if improvement > 0:
            print("✅ SCU improved BPT!")
        else:
            print("⚠️  No improvement or negative result")
        
    except Exception as e:
        print(f"Could not compare results: {e}")
        print("Make sure both training runs completed successfully.")

def main():
    """Main execution"""
    
    import argparse
    parser = argparse.ArgumentParser(description="Train Qwen3 with SCU")
    parser.add_argument("--mode", choices=["scu", "baseline", "both", "compare"], 
                       default="scu", help="Training mode")
    parser.add_argument("--compare", action="store_true", help="Compare SCU vs baseline")
    
    args = parser.parse_args()
    
    if args.mode == "scu" or args.mode == "both":
        success = run_qwen_scu_training()
        if not success:
            sys.exit(1)
    
    if args.mode == "baseline" or args.mode == "both":
        success = run_qwen_baseline()
        if not success:
            sys.exit(1)
    
    if args.mode == "compare" or args.compare:
        compare_results()

if __name__ == "__main__":
    main()