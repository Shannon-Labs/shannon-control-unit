#!/usr/bin/env python3
"""Simple test of SCU v1.0 training"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from scripts.train_scu import main
from argparse import Namespace

args = Namespace(
    base_model="./Qwen3-1.7B-Base",
    adapter_out="adapters/test_scu_adapter",
    target_s=0.01,
    kp=0.8,
    ki=0.15,
    deadband=0.002,
    lambda_init=1.0,
    lambda_min=1e-4,
    lambda_max=2.0,
    prior_sigma=0.01,
    epochs=1,
    steps=50,
    batch_size=1,
    lr=5e-5,
    block_size=512,
    gradient_accumulation_steps=4,
    fp16=False,
    seed=42,
    train_data="data/train.txt",  # Use the smaller train.txt for faster test
    max_texts=50,
    log_csv="logs/test_scu.csv",
    quickstart=False
)

print("Testing SCU v1.0 with simplified settings...")
print(f"Model: {args.base_model}")
print(f"Data: {args.train_data}")
print(f"Steps: {args.steps}")
print(f"Target S: {args.target_s:.1%}")

try:
    main(args)
    print("\n✅ SUCCESS! SCU v1.0 training completed!")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()