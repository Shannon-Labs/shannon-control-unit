#!/usr/bin/env python3
"""
Run the proven SCU v1.0 implementation with proper configuration.
This uses the original, tested PI controller from scu/control.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from scripts.train_scu import main
from argparse import Namespace

print("=" * 70)
print("SHANNON CONTROL UNIT v1.0 - PROVEN IMPLEMENTATION")
print("=" * 70)
print()
print("This uses the original PI controller that achieved:")
print("  • 6.2% BPT improvement (1B model)")
print("  • 10.6% BPT improvement (3B model)")
print("  • Patent-pending adaptive regularization")
print()

# Configuration for proper training
args = Namespace(
    # Model
    base_model="./Qwen3-1.7B-Base",
    adapter_out="adapters/scu_qwen3_1.7b_proven",
    
    # SCU Control Parameters (the proven values from paper)
    target_s=0.01,      # Target 1% S-ratio (proven optimal)
    kp=0.8,             # Proportional gain (proven)
    ki=0.15,            # Integral gain (proven)
    deadband=0.002,     # ±0.2pp deadband (prevents oscillation)
    lambda_init=1.0,    # Start at lambda=1.0
    lambda_min=1e-4,    # Minimum lambda (anti-windup)
    lambda_max=2.0,     # Maximum lambda (safety)
    
    # Training Parameters
    prior_sigma=0.01,   # Prior std dev for ParamBPT
    epochs=1,           # 1 epoch
    steps=None,         # Use epochs instead of steps
    batch_size=2,       # Small batch for MPS
    lr=5e-5,            # Standard fine-tuning LR
    block_size=1024,    # Context length
    gradient_accumulation_steps=4,  # Effective batch size = 8
    fp16=False,         # MPS doesn't need fp16
    seed=42,            # Reproducibility
    
    # Data
    train_data="data/train.txt",  # Use train.txt (10k lines)
    max_texts=1000,     # Use 1000 texts for proper statistics
    
    # Logging
    log_csv="logs/proven_scu_training.csv",
    quickstart=False
)

print("Configuration:")
print(f"  Model: {args.base_model}")
print(f"  Data: {args.train_data} (max {args.max_texts} texts)")
print(f"  Target S: {args.target_s:.1%}")
print(f"  Control: Kp={args.kp}, Ki={args.ki}, deadband=±{args.deadband:.1%}")
print(f"  Training: {args.epochs} epoch, batch_size={args.batch_size}, lr={args.lr}")
print()
print("Starting training...")
print("=" * 70)

try:
    main(args)
    print()
    print("=" * 70)
    print("✅ SUCCESS! Proven SCU v1.0 training completed!")
    print("=" * 70)
    print()
    print("Results saved:")
    print(f"  • Adapter: {args.adapter_out}")
    print(f"  • Training log: {args.log_csv}")
    print(f"  • Metadata: {args.adapter_out}/metadata.json")
    print()
    print("Next steps:")
    print("  1. Check the CSV log to see lambda and S-ratio over time")
    print("  2. Evaluate with: python scripts/eval_bpt.py --model-path adapters/scu_qwen3_1.7b_proven")
    print("  3. Compare against baseline training without SCU")
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)