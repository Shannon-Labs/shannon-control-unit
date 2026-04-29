#!/usr/bin/env python3
"""
DGX Spark + Shannon Control Unit Training Script

Fine-tunes Qwen 3.5 27B (or other models) on NVIDIA DGX Spark using
SCU PI control for adaptive regularization and self-terminating training.

Hardware target:
    NVIDIA DGX Spark — Grace Blackwell GB10, 128GB unified memory,
    1 PFLOP FP4, 273 GB/s bandwidth

Default model:
    Qwen/Qwen3.5-27B — Dense 27B, Apache 2.0, Feb 2026

Usage:
    # Default: Qwen 3.5 27B with full-precision LoRA
    python scripts/train_dgx_spark.py

    # QLoRA mode (4-bit base, bigger LoRA rank, longer context)
    python scripts/train_dgx_spark.py --qlora

    # Custom model
    python scripts/train_dgx_spark.py --model meta-llama/Llama-4-8B --target-s 0.03

    # Quick test (100 steps)
    python scripts/train_dgx_spark.py --steps 100 --report-steps 10
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_prerequisites():
    """Check that CUDA and training data are available."""
    errors = []

    # Check CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append(
                "CUDA is not available. This script requires an NVIDIA GPU.\n"
                "On DGX Spark, ensure NVIDIA drivers and CUDA toolkit are installed."
            )
        else:
            device = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"GPU: {device}")
            print(f"GPU Memory: {mem:.0f} GB")
    except ImportError:
        errors.append("PyTorch is not installed. Install with: pip install torch")

    # Check required packages
    for pkg, install in [
        ("transformers", "transformers>=4.48.0"),
        ("peft", "peft>=0.14.0"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            errors.append(f"{pkg} not found. Install with: pip install {install}")

    if errors:
        print("=" * 60)
        print("PREREQUISITES CHECK FAILED")
        print("=" * 60)
        for err in errors:
            print(f"\n{err}")
        print("\nInstall all requirements with: pip install -r requirements-cuda.txt")
        print("=" * 60)
        sys.exit(1)


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Train on DGX Spark with Shannon Control Unit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Qwen 3.5 27B, full LoRA, bf16
  python scripts/train_dgx_spark.py

  # QLoRA mode (4-bit quantized base, r=64, seq_len=4096)
  python scripts/train_dgx_spark.py --qlora

  # Llama 4 8B
  python scripts/train_dgx_spark.py --model meta-llama/Llama-4-8B --target-s 0.03

  # Quick test
  python scripts/train_dgx_spark.py --steps 100 --report-steps 10

  # Custom data
  python scripts/train_dgx_spark.py --train-data data/my_data.jsonl
        """,
    )

    # Model
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model", default="Qwen/Qwen3.5-27B",
        help="HuggingFace model ID (default: Qwen/Qwen3.5-27B)",
    )
    model_group.add_argument(
        "--adapter-out", default="adapters/qwen35_27b_dgx_scu",
        help="Output directory for adapter",
    )

    # Data
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--train-data", default="data/fineweb_edu_1gb.jsonl",
        help="Training data (.jsonl with 'text' field)",
    )
    data_group.add_argument(
        "--val-data", default=None,
        help="Validation data (optional)",
    )
    data_group.add_argument(
        "--block-size", type=int, default=2048,
        help="Maximum sequence length (default: 2048)",
    )

    # SCU Control
    scu_group = parser.add_argument_group("SCU Control")
    scu_group.add_argument(
        "--target-s", type=float, default=0.04,
        help="Target S-ratio (default: 0.04 = 4%% for xlarge models)",
    )
    scu_group.add_argument("--kp", type=float, default=0.9, help="Proportional gain")
    scu_group.add_argument("--ki", type=float, default=0.18, help="Integral gain")
    scu_group.add_argument("--deadband", type=float, default=0.003, help="Error deadband")
    scu_group.add_argument("--lambda-init", type=float, default=1.0, help="Initial lambda")
    scu_group.add_argument("--lambda-min", type=float, default=0.0001, help="Minimum lambda")
    scu_group.add_argument("--lambda-max", type=float, default=2.0, help="Maximum lambda")
    scu_group.add_argument("--prior-sigma", type=float, default=0.01, help="Prior sigma")
    scu_group.add_argument("--control-freq", type=int, default=50, help="Control update frequency")

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--steps", type=int, default=2000, help="Total training steps")
    train_group.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    train_group.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    train_group.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    train_group.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio")
    train_group.add_argument("--seed", type=int, default=42, help="Random seed")

    # LoRA
    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    lora_group.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    lora_group.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")

    # DGX Spark / CUDA
    cuda_group = parser.add_argument_group("CUDA / DGX Spark")
    cuda_group.add_argument(
        "--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
        help="Training precision (default: bf16 for Blackwell)",
    )
    cuda_group.add_argument("--qlora", action="store_true", help="Use 4-bit QLoRA mode")
    cuda_group.add_argument("--no-flash-attn", action="store_true", help="Disable Flash Attention 2")
    cuda_group.add_argument("--no-grad-checkpoint", action="store_true", help="Disable gradient checkpointing")
    cuda_group.add_argument("--compile", action="store_true", help="Use torch.compile()")

    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log-dir", default="logs/dgx_spark_scu", help="Log directory")
    log_group.add_argument("--report-steps", type=int, default=25, help="Report every N steps")
    log_group.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    log_group.add_argument("--job-id", default=None, help="Custom job ID")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    check_prerequisites()

    from shannon_control.cuda import CUDATrainingEngine, CUDATrainingConfig

    # QLoRA overrides
    if args.qlora:
        print("[DGX] QLoRA mode: 4-bit base model, r=64, seq_len=4096")
        if args.lora_r == 32:  # not explicitly set
            args.lora_r = 64
            args.lora_alpha = 128
        if args.block_size == 2048:  # not explicitly set
            args.block_size = 4096
        if args.batch_size == 2:
            args.batch_size = 4
            args.grad_accum = 4

    # Job ID
    job_id = args.job_id or f"dgx_{Path(args.model).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Resolve paths
    train_data = str(project_root / args.train_data)
    val_data = str(project_root / args.val_data) if args.val_data else None
    adapter_out = str(project_root / args.adapter_out)
    log_dir = str(project_root / args.log_dir)

    # Build config
    config = CUDATrainingConfig(
        base_model=args.model,
        train_data=train_data,
        val_data=val_data,
        block_size=args.block_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr=args.lr,
        steps=args.steps,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_s=args.target_s,
        kp=args.kp,
        ki=args.ki,
        deadband=args.deadband,
        lambda_init=args.lambda_init,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        prior_sigma=args.prior_sigma,
        control_frequency=args.control_freq,
        dtype=args.dtype,
        use_flash_attention=not args.no_flash_attn,
        gradient_checkpointing=not args.no_grad_checkpoint,
        use_4bit=args.qlora,
        compile_model=args.compile,
        adapter_out=adapter_out,
        log_dir=log_dir,
        report_steps=args.report_steps,
        save_steps=args.save_steps,
    )

    # Print banner
    mode = "QLoRA (4-bit)" if args.qlora else "Full LoRA (bf16)"
    print("=" * 70)
    print("DGX Spark + Shannon Control Unit Training")
    print("=" * 70)
    print(f"Job ID:           {job_id}")
    print(f"Model:            {config.base_model}")
    print(f"Mode:             {mode}")
    print(f"Train Data:       {config.train_data}")
    print(f"Steps:            {config.steps}")
    print(f"Target S:         {config.target_s} ({config.target_s * 100:.1f}%)")
    print(f"Batch Size:       {config.batch_size} (effective: {config.effective_batch_size})")
    print(f"Learning Rate:    {config.lr}")
    print(f"LoRA:             r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Precision:        {config.dtype}")
    print(f"Flash Attention:  {config.use_flash_attention}")
    print(f"Grad Checkpoint:  {config.gradient_checkpointing}")
    print(f"Output:           {config.adapter_out}")
    print("=" * 70)
    print()

    # Run training
    engine = CUDATrainingEngine(config, job_id=job_id)

    try:
        adapter_path = engine.run()

        summary = engine.scu_callback.get_metrics_summary()

        # Save comprehensive metadata
        metadata = {
            "base_model": args.model,
            "architecture": f"{Path(args.model).name}-SCU",
            "hardware": "NVIDIA DGX Spark (Grace Blackwell GB10)",
            "dataset": Path(args.train_data).stem,
            "backend": "pytorch-cuda",
            "mode": mode,
            "target_s": args.target_s,
            "final_s_ratio": summary.get("final_s_ratio", 0),
            "final_lambda": summary.get("final_lambda", 0),
            "lambda_range": summary.get("lambda_range", [0, 0]),
            "final_loss": summary.get("final_loss", 0),
            "total_steps": summary.get("total_steps", 0),
            "saturated": summary.get("saturated", False),
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
                "target_modules": config.lora_target_modules,
            },
            "training_config": {
                "steps": args.steps,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.grad_accum,
                "effective_batch_size": config.effective_batch_size,
                "lr": args.lr,
                "block_size": args.block_size,
                "dtype": args.dtype,
                "flash_attention": config.use_flash_attention,
                "gradient_checkpointing": config.gradient_checkpointing,
                "seed": args.seed,
            },
            "job_id": job_id,
            "adapter_path": str(adapter_path),
            "timestamp": datetime.now().isoformat(),
        }

        metadata_path = Path(adapter_out) / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Results
        print()
        print("=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Final S-ratio:    {metadata['final_s_ratio']:.4f} (target: {args.target_s})")
        print(f"Final Lambda:     {metadata['final_lambda']:.4f}")
        print(f"Final Loss:       {metadata['final_loss']:.4f}")
        print(f"MDL Saturated:    {metadata['saturated']}")
        print(f"Adapter:          {adapter_path}")
        print(f"Metadata:         {metadata_path}")
        print(f"Metrics CSV:      {config.log_csv}")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Review metrics CSV and SCU metrics JSON")
        print("  2. Run: python scripts/eval_quality.py")
        print(f"  3. Run: python scripts/prepare_hf_release.py --adapter-path {adapter_out}")
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
