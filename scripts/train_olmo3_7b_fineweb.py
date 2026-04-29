#!/usr/bin/env python3
"""
OLMo 3 7B FineWeb-Edu SCU Training Script

Trains mlx-community/Olmo-3-7B-Instruct-4bit on FineWeb-Edu using
Shannon Control Unit with PI control on Apple Silicon MLX.

Usage:
    # Default training run (5000 steps)
    python scripts/train_olmo3_7b_fineweb.py

    # Custom output directory
    python scripts/train_olmo3_7b_fineweb.py --adapter-out adapters/my_olmo3_run

    # Override SCU parameters
    python scripts/train_olmo3_7b_fineweb.py --target-s 0.02 --steps 3000

    # Quick test run (100 steps)
    python scripts/train_olmo3_7b_fineweb.py --steps 100 --report-steps 10
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_prerequisites():
    """Check that MLX and training data are available."""
    errors = []

    # Check MLX
    try:
        import mlx.core
        import mlx_lm
    except ImportError:
        errors.append(
            "MLX is not available. This script requires Apple Silicon.\n"
            "Install with: pip install mlx mlx-lm"
        )

    # Check training data
    train_data_path = project_root / "data" / "fineweb_edu_1gb.jsonl"
    if not train_data_path.exists():
        errors.append(
            f"Training data not found at: {train_data_path}\n"
            "Download with: python scripts/load_fineweb_edu.py --size 1000 --output data/fineweb_edu_1gb.jsonl"
        )

    if errors:
        print("=" * 60)
        print("PREREQUISITES CHECK FAILED")
        print("=" * 60)
        for err in errors:
            print(f"\n{err}")
        print("=" * 60)
        sys.exit(1)

    return True


def create_parser():
    """Create argument parser with all configurable options."""
    parser = argparse.ArgumentParser(
        description="Train OLMo 3 7B on FineWeb-Edu with SCU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run (5000 steps, ~4-6 hours)
  python scripts/train_olmo3_7b_fineweb.py

  # Quick test (100 steps)
  python scripts/train_olmo3_7b_fineweb.py --steps 100

  # Custom S-ratio target
  python scripts/train_olmo3_7b_fineweb.py --target-s 0.02

  # Custom output
  python scripts/train_olmo3_7b_fineweb.py --adapter-out adapters/olmo3_v2
        """
    )

    # Model Configuration
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        default="mlx-community/Olmo-3-7B-Instruct-4bit",
        help="MLX model ID (default: mlx-community/Olmo-3-7B-Instruct-4bit)"
    )
    model_group.add_argument(
        "--adapter-out",
        default="adapters/olmo3_7b_fineweb_scu",
        help="Output directory for adapter (default: adapters/olmo3_7b_fineweb_scu)"
    )

    # Data Configuration
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--train-data",
        default="data/fineweb_edu_1gb.jsonl",
        help="Path to training data JSONL (default: data/fineweb_edu_1gb.jsonl)"
    )
    data_group.add_argument(
        "--val-data",
        default=None,
        help="Path to validation data (optional)"
    )
    data_group.add_argument(
        "--block-size",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )

    # SCU Control Parameters
    scu_group = parser.add_argument_group("SCU Control")
    scu_group.add_argument(
        "--target-s",
        type=float,
        default=0.03,
        help="Target S-ratio (default: 0.03 = 3%%)"
    )
    scu_group.add_argument(
        "--kp",
        type=float,
        default=0.8,
        help="Proportional gain (default: 0.8)"
    )
    scu_group.add_argument(
        "--ki",
        type=float,
        default=0.15,
        help="Integral gain (default: 0.15)"
    )
    scu_group.add_argument(
        "--deadband",
        type=float,
        default=0.003,
        help="Error deadband (default: 0.003)"
    )
    scu_group.add_argument(
        "--lambda-init",
        type=float,
        default=1.0,
        help="Initial lambda (default: 1.0)"
    )
    scu_group.add_argument(
        "--lambda-min",
        type=float,
        default=0.0001,
        help="Minimum lambda (default: 0.0001)"
    )
    scu_group.add_argument(
        "--lambda-max",
        type=float,
        default=2.0,
        help="Maximum lambda (default: 2.0)"
    )
    scu_group.add_argument(
        "--prior-sigma",
        type=float,
        default=0.01,
        help="Prior sigma for ParamBPT (default: 0.01)"
    )
    scu_group.add_argument(
        "--control-freq",
        type=int,
        default=50,
        help="Control update frequency (default: every 50 steps)"
    )

    # Training Parameters
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Total training steps (default: 5000)"
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per step (default: 1)"
    )
    train_group.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="Gradient accumulation steps (default: 16)"
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    train_group.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Warmup steps (default: 500)"
    )
    train_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # LoRA Parameters
    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank (default: 32)"
    )
    lora_group.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha (default: 64)"
    )
    lora_group.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)"
    )
    lora_group.add_argument(
        "--lora-layers",
        type=int,
        default=32,
        help="Number of layers for LoRA (default: 32)"
    )

    # Logging Parameters
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--log-dir",
        default="logs/olmo3_7b_fineweb_scu",
        help="Log directory (default: logs/olmo3_7b_fineweb_scu)"
    )
    log_group.add_argument(
        "--report-steps",
        type=int,
        default=50,
        help="Report metrics every N steps (default: 50)"
    )
    log_group.add_argument(
        "--save-steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (default: 1000)"
    )
    log_group.add_argument(
        "--job-id",
        default=None,
        help="Custom job ID (auto-generated if not provided)"
    )

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Check prerequisites (MLX, training data)
    check_prerequisites()

    # Import after checks to avoid import errors on non-Mac systems
    from shannon_control.mlx import MLXTrainingEngine, MLXTrainingConfig

    # Generate job ID
    job_id = args.job_id or f"olmo3_fineweb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Resolve paths
    train_data_path = str(project_root / args.train_data)
    adapter_out_path = str(project_root / args.adapter_out)
    log_dir_path = str(project_root / args.log_dir)

    # Create MLX training configuration
    config = MLXTrainingConfig(
        # Model
        base_model=args.model,

        # Data
        train_data=train_data_path,
        val_data=args.val_data,
        block_size=args.block_size,

        # Training
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr=args.lr,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,

        # LoRA
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=args.lora_layers,
        lora_target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],

        # SCU Control
        target_s=args.target_s,
        kp=args.kp,
        ki=args.ki,
        deadband=args.deadband,
        lambda_init=args.lambda_init,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        prior_sigma=args.prior_sigma,
        control_frequency=args.control_freq,

        # Output
        adapter_out=adapter_out_path,
        log_dir=log_dir_path,
        log_csv=f"{log_dir_path}/metrics.csv",
        report_steps=args.report_steps,
        eval_steps=0,  # No validation by default
        save_steps=args.save_steps,

        # MLX settings
        grad_checkpoint=True,
        use_quantized_model=True,
        enable_power_monitoring=True,
    )

    # Print configuration summary
    print("=" * 70)
    print("OLMo 3 7B FineWeb-Edu SCU Training")
    print("=" * 70)
    print(f"Job ID:           {job_id}")
    print(f"Model:            {config.base_model}")
    print(f"Train Data:       {config.train_data}")
    print(f"Steps:            {config.steps}")
    print(f"Target S:         {config.target_s} ({config.target_s*100:.1f}%)")
    print(f"Batch Size:       {config.batch_size} (effective: {config.effective_batch_size})")
    print(f"Learning Rate:    {config.lr}")
    print(f"LoRA:             r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Output:           {config.adapter_out}")
    print(f"Log CSV:          {config.log_csv}")
    print("=" * 70)
    print()

    # Create and run training engine
    engine = MLXTrainingEngine(config, job_id=job_id)

    try:
        adapter_path = engine.run()

        # Get final metrics summary
        summary = engine.scu_callback.get_metrics_summary()

        # Create comprehensive metadata for HuggingFace release
        metadata = {
            "base_model": args.model,
            "architecture": "OLMo3-MLX-SCU",
            "dataset": "FineWeb-Edu",
            "backend": "MLX",

            # SCU results
            "target_s": args.target_s,
            "final_s_ratio": summary.get("final_s_ratio", 0),
            "mean_s_ratio": summary.get("mean_s_ratio", 0),
            "final_lambda": summary.get("final_lambda", 0),
            "lambda_range": summary.get("lambda_range", [0, 0]),

            # Training results
            "final_loss": summary.get("final_loss", 0),
            "mean_loss": summary.get("mean_loss", 0),
            "total_steps": summary.get("total_steps", 0),

            # Full configuration
            "scu_config": {
                "kp": args.kp,
                "ki": args.ki,
                "deadband": args.deadband,
                "lambda_init": args.lambda_init,
                "lambda_min": args.lambda_min,
                "lambda_max": args.lambda_max,
                "prior_sigma": args.prior_sigma,
                "control_frequency": args.control_freq,
            },
            "lora_config": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "layers": args.lora_layers,
                "target_modules": config.lora_target_modules,
            },
            "training_config": {
                "steps": args.steps,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.grad_accum,
                "effective_batch_size": config.effective_batch_size,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "block_size": args.block_size,
                "seed": args.seed,
            },

            "job_id": job_id,
            "adapter_path": str(adapter_path),
            "log_csv": config.log_csv,
            "timestamp": datetime.now().isoformat(),
            "success": True,
        }

        # Save metadata
        metadata_path = Path(adapter_out_path) / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Print results
        print()
        print("=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Final S-ratio:    {metadata['final_s_ratio']:.4f} (target: {args.target_s})")
        print(f"Final Lambda:     {metadata['final_lambda']:.4f}")
        print(f"Final Loss:       {metadata['final_loss']:.4f}")
        print(f"Adapter:          {adapter_path}")
        print(f"Metadata:         {metadata_path}")
        print(f"Metrics CSV:      {config.log_csv}")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Review metrics in the CSV log")
        print("  2. Run: python scripts/prepare_hf_release.py --adapter-path", adapter_out_path)
        print("  3. Upload to HuggingFace Hub")
        print()

        return 0

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training interrupted by user")
        return 1

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
