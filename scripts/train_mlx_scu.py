#!/usr/bin/env python3
"""
MLX-SCU Training Script

Native Apple MLX training with Shannon Control Unit integration.
Trains DeepSeek-R1-Distill-Qwen-1.5B (or other MLX models) with adaptive
regularization controlled by PI feedback.

Usage:
    python scripts/train_mlx_scu.py --train-data data/train.jsonl --steps 1000

    # With custom model
    python scripts/train_mlx_scu.py \\
        --model mlx-community/Llama-3.2-1B-Instruct-4bit \\
        --train-data data/train.jsonl \\
        --target-s 0.01 \\
        --steps 500

    # Production run with all options
    python scripts/train_mlx_scu.py \\
        --model mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit \\
        --train-data data/train.jsonl \\
        --val-data data/val.jsonl \\
        --target-s 0.01 \\
        --steps 1000 \\
        --batch-size 2 \\
        --lr 2e-4 \\
        --lora-r 16 \\
        --adapter-out adapters/deepseek_scu \\
        --log-dir logs/deepseek_run

Requirements:
    pip install mlx mlx-lm
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Train an MLX model with Shannon Control Unit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic training:
    python train_mlx_scu.py --train-data data/train.jsonl --steps 100

  Full production run:
    python train_mlx_scu.py \\
        --model mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit \\
        --train-data data/train.jsonl \\
        --steps 1000 \\
        --target-s 0.01 \\
        --adapter-out adapters/my_adapter
        """
    )

    # Model configuration
    parser.add_argument(
        "--model", "-m",
        default="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
        help="MLX model ID from HuggingFace (default: DeepSeek-R1-Distill-Qwen-1.5B-4bit)"
    )

    # Data configuration
    parser.add_argument(
        "--train-data", "-t",
        required=True,
        help="Path to training data (.jsonl or .txt)"
    )
    parser.add_argument(
        "--val-data", "-v",
        default=None,
        help="Path to validation data (optional)"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)"
    )

    # Training hyperparameters
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=1000,
        help="Number of training steps (default: 1000)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=2,
        help="Batch size (default: 2)"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # LoRA configuration
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)"
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to apply LoRA (default: 16)"
    )
    parser.add_argument(
        "--full-params",
        action="store_true",
        help="Train full parameters instead of LoRA (experimental)"
    )

    # SCU control parameters
    parser.add_argument(
        "--target-s",
        type=float,
        default=0.01,
        help="Target S ratio (default: 0.01)"
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=0.8,
        help="PI proportional gain (default: 0.8)"
    )
    parser.add_argument(
        "--ki",
        type=float,
        default=0.15,
        help="PI integral gain (default: 0.15)"
    )
    parser.add_argument(
        "--deadband",
        type=float,
        default=0.002,
        help="Control deadband (default: 0.002)"
    )
    parser.add_argument(
        "--lambda-init",
        type=float,
        default=1.0,
        help="Initial lambda (default: 1.0)"
    )
    parser.add_argument(
        "--lambda-min",
        type=float,
        default=1e-4,
        help="Minimum lambda (default: 1e-4)"
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=2.0,
        help="Maximum lambda (default: 2.0)"
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=50,
        help="SCU control frequency in steps (default: 50)"
    )
    parser.add_argument(
        "--param-bpt-frequency",
        type=int,
        default=None,
        help="ParamBPT update interval (default: every step for LoRA, control frequency for full params)"
    )

    # Output configuration
    parser.add_argument(
        "--adapter-out", "-o",
        default="adapters/mlx_scu_adapter",
        help="Output directory for adapter (default: adapters/mlx_scu_adapter)"
    )
    parser.add_argument(
        "--log-dir",
        default="logs/mlx_scu",
        help="Directory for logs (default: logs/mlx_scu)"
    )
    parser.add_argument(
        "--log-csv",
        default=None,
        help="Path for CSV metrics log (optional)"
    )

    # Other options
    parser.add_argument(
        "--no-power-monitoring",
        action="store_true",
        help="Disable Apple Silicon power monitoring"
    )
    parser.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="Disable gradient checkpointing"
    )
    parser.add_argument(
        "--report-steps",
        type=int,
        default=10,
        help="Report metrics every N steps (default: 10)"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Run validation every N steps (default: 100)"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)"
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Custom job ID (auto-generated if not provided)"
    )

    args = parser.parse_args()

    # Check MLX availability
    try:
        import mlx.core
        import mlx_lm
    except ImportError:
        print("Error: MLX is not available.")
        print("This script requires Apple Silicon and MLX.")
        print("Install with: pip install mlx mlx-lm")
        sys.exit(1)

    # Import after MLX check
    from shannon_control.mlx import MLXTrainingEngine, MLXTrainingConfig

    # Generate job ID if not provided
    job_id = args.job_id
    if not job_id:
        from datetime import datetime
        job_id = f"mlx_scu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create configuration
    config = MLXTrainingConfig(
        base_model=args.model,
        train_data=args.train_data,
        val_data=args.val_data,
        block_size=args.block_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr=args.lr,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=args.lora_layers,
        train_full_params=args.full_params,
        param_scope="trainable" if args.full_params else "lora",
        target_s=args.target_s,
        kp=args.kp,
        ki=args.ki,
        deadband=args.deadband,
        lambda_init=args.lambda_init,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        control_frequency=args.control_freq,
        param_bpt_frequency=args.param_bpt_frequency,
        grad_checkpoint=not args.no_grad_checkpoint,
        adapter_out=args.adapter_out,
        log_dir=args.log_dir,
        log_csv=args.log_csv,
        report_steps=args.report_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        enable_power_monitoring=not args.no_power_monitoring,
    )

    # Print configuration
    print("=" * 60)
    print("MLX-SCU Training Configuration")
    print("=" * 60)
    print(f"Job ID:        {job_id}")
    print(f"Model:         {config.base_model}")
    print(f"Train Data:    {config.train_data}")
    print(f"Val Data:      {config.val_data or 'None'}")
    print(f"Steps:         {config.steps}")
    print(f"Batch Size:    {config.batch_size} (effective: {config.effective_batch_size})")
    print(f"Learning Rate: {config.lr}")
    print(f"Train Mode:    {'full' if config.train_full_params else 'lora'}")
    if not config.train_full_params:
        print(f"LoRA Rank:     {config.lora_r}")
    print(f"Target S:      {config.target_s}")
    print(f"Output:        {config.adapter_out}")
    print("=" * 60)

    # Run training
    engine = MLXTrainingEngine(config, job_id=job_id)

    try:
        adapter_path = engine.run()
        print(f"\nTraining complete! Adapter saved to: {adapter_path}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
