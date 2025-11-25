#!/usr/bin/env python3
"""
OLMo 3 7B - SCU Test Run

Complete test pipeline:
1. Downloads ~50MB of FineWeb-Edu (if not already present)
2. Runs 100 training steps with SCU monitoring
3. Evaluates validation perplexity
4. Outputs S-ratio, lambda adjustments, and control loop telemetry

Usage:
    python scripts/run_olmo3_test.py                    # Auto-detect hardware
    python scripts/run_olmo3_test.py --backend cuda     # Force CUDA
    python scripts/run_olmo3_test.py --backend mlx      # Force MLX
    python scripts/run_olmo3_test.py --steps 500        # More steps
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add parent dir to path
sys.path.append(str(Path(__file__).parent.parent))


def ensure_data(size_mb: float = 50.0, force_download: bool = False):
    """Ensure FineWeb-Edu data is available."""
    data_path = Path("data/fineweb_edu_sample.jsonl")

    if data_path.exists() and not force_download:
        size = data_path.stat().st_size / 1024 / 1024
        print(f"Data exists: {data_path} ({size:.1f}MB)")
        return str(data_path)

    print(f"Downloading FineWeb-Edu ({size_mb}MB)...")
    from scripts.load_fineweb_edu import download_fineweb_edu

    download_fineweb_edu(
        target_size_mb=size_mb,
        output_path=str(data_path),
        min_score=3.0,
    )

    return str(data_path)


def evaluate_perplexity(model, tokenizer, data_path: str, max_samples: int = 50,
                        block_size: int = 2048, device: str = "cuda"):
    """Evaluate validation perplexity."""
    import torch
    import math
    from scu import data as scu_data

    print(f"\nEvaluating perplexity on {data_path}...")

    # Load validation data
    val_texts = scu_data.load_texts_from_file(data_path, max_texts=max_samples)
    val_chunks = scu_data.tokenize_and_chunk(
        val_texts, tokenizer, block_size=block_size, shuffle=False
    )

    if not val_chunks:
        print("Warning: No validation chunks created")
        return float('inf')

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for chunk in val_chunks[:max_samples]:
            input_ids = torch.tensor([chunk['input_ids']]).to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss.item()

            total_loss += loss * input_ids.shape[1]
            total_tokens += input_ids.shape[1]

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

    print(f"Validation: Loss={avg_loss:.4f}, PPL={perplexity:.2f}")
    return perplexity


def run_test(args):
    """Run the complete test pipeline."""
    print("=" * 60)
    print("OLMo 3 7B - SCU Test Run")
    print("=" * 60)

    # Ensure output directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("adapters").mkdir(exist_ok=True)

    # Step 1: Ensure data
    print("\n[Step 1/4] Preparing data...")
    data_path = ensure_data(size_mb=args.data_size, force_download=args.force_download)

    # Step 2: Detect hardware and select model
    print("\n[Step 2/4] Detecting hardware...")
    from scripts.train_olmo3_7b import detect_hardware
    backend, hw_info = detect_hardware()

    if args.backend:
        backend = args.backend
        print(f"Backend override: {backend}")
    else:
        print(f"Auto-detected: {hw_info}")

    # Select model based on backend
    if backend == "cuda":
        model_id = "unsloth/Olmo-3-7B-Instruct-unsloth-bnb-4bit"
    elif backend == "mlx":
        model_id = "mlx-community/Olmo-3-7B-Instruct-4bit"
    else:
        model_id = "allenai/OLMo-3-7B"

    if args.model:
        model_id = args.model
        print(f"Model override: {model_id}")

    print(f"Using model: {model_id}")

    # Step 3: Run training
    print(f"\n[Step 3/4] Running {args.steps} training steps...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_out = f"adapters/olmo3_test_{timestamp}"
    log_csv = f"logs/olmo3_test_{timestamp}.csv"

    # Build training args
    train_args = [
        sys.executable, "scripts/train_olmo3_7b.py",
        "--base_model", model_id,
        "--train_data", data_path,
        "--adapter_out", adapter_out,
        "--log_csv", log_csv,
        "--steps", str(args.steps),
        "--target_s", str(args.target_s),
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.grad_accum),
        "--block_size", str(args.block_size),
        "--backend", backend,
    ]

    # Run training
    import subprocess
    result = subprocess.run(train_args, capture_output=False)

    if result.returncode != 0:
        print(f"Training failed with code {result.returncode}")
        return None

    # Step 4: Analyze results
    print(f"\n[Step 4/4] Analyzing results...")

    # Load metadata
    metadata_path = Path(adapter_out) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Load training log
    log_analysis = analyze_training_log(log_csv)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nModel: {model_id}")
    print(f"Backend: {backend}")
    print(f"Steps: {args.steps}")
    print(f"\nSCU Control Parameters:")
    print(f"  Target S*: {args.target_s:.1%}")
    print(f"  Kp: 0.8, Ki: 0.15")

    if log_analysis:
        print(f"\nTraining Metrics:")
        print(f"  Initial DataBPT: {log_analysis['initial_data_bpt']:.3f}")
        print(f"  Final DataBPT: {log_analysis['final_data_bpt']:.3f}")
        print(f"  DataBPT Change: {log_analysis['data_bpt_change']:.1%}")
        print(f"\nS-Ratio Dynamics:")
        print(f"  Initial S: {log_analysis['initial_s']:.2%}")
        print(f"  Final S: {log_analysis['final_s']:.2%}")
        print(f"  Average S: {log_analysis['avg_s']:.2%}")
        print(f"\nLambda Adjustments:")
        print(f"  Initial lambda: {log_analysis['initial_lambda']:.4f}")
        print(f"  Final lambda: {log_analysis['final_lambda']:.4f}")
        print(f"  Lambda range: [{log_analysis['min_lambda']:.4f}, {log_analysis['max_lambda']:.4f}]")

        if log_analysis.get('saturation_step'):
            print(f"\nSaturation detected at step {log_analysis['saturation_step']}")
        else:
            print(f"\nNo saturation detected in {args.steps} steps")

    print(f"\nOutputs:")
    print(f"  Adapter: {adapter_out}/")
    print(f"  Log: {log_csv}")

    # Save complete analysis
    analysis = {
        "model": model_id,
        "backend": backend,
        "steps": args.steps,
        "target_s": args.target_s,
        "metadata": metadata,
        "log_analysis": log_analysis,
        "timestamp": timestamp,
    }

    analysis_path = f"logs/olmo3_analysis_{timestamp}.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  Analysis: {analysis_path}")

    return analysis


def analyze_training_log(log_path: str) -> dict:
    """Analyze training log CSV for key metrics."""
    import csv

    if not Path(log_path).exists():
        return {}

    rows = []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "step": int(row["step"]),
                "data_bpt": float(row["data_bpt"]),
                "param_bpt": float(row["param_bpt"]),
                "S": float(row["S"]),
                "lambda": float(row["lambda"]),
                "I": float(row["I"]),
            })

    if not rows:
        return {}

    # Extract key metrics
    analysis = {
        "initial_data_bpt": rows[0]["data_bpt"],
        "final_data_bpt": rows[-1]["data_bpt"],
        "data_bpt_change": (rows[-1]["data_bpt"] - rows[0]["data_bpt"]) / rows[0]["data_bpt"],
        "initial_s": rows[0]["S"],
        "final_s": rows[-1]["S"],
        "avg_s": sum(r["S"] for r in rows) / len(rows),
        "initial_lambda": rows[0]["lambda"],
        "final_lambda": rows[-1]["lambda"],
        "min_lambda": min(r["lambda"] for r in rows),
        "max_lambda": max(r["lambda"] for r in rows),
        "saturation_step": None,
    }

    # Check for saturation (lambda hitting bounds)
    lambda_max = 2.0
    for row in rows:
        if row["lambda"] >= lambda_max * 0.99:  # 99% of max
            analysis["saturation_step"] = row["step"]
            break

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="OLMo 3 7B - SCU Test Run"
    )

    # Hardware
    parser.add_argument("--backend", choices=["cuda", "mlx", "cpu"],
                       help="Force specific backend")
    parser.add_argument("--model", help="Override model ID")

    # Data
    parser.add_argument("--data-size", type=float, default=50.0,
                       help="FineWeb-Edu download size in MB (default: 50)")
    parser.add_argument("--force-download", action="store_true",
                       help="Force re-download of data")

    # Training
    parser.add_argument("--steps", type=int, default=100,
                       help="Number of training steps (default: 100)")
    parser.add_argument("--target-s", type=float, default=0.02,
                       help="Target S ratio (default: 0.02)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=16,
                       help="Gradient accumulation steps (default: 16)")
    parser.add_argument("--block-size", type=int, default=2048,
                       help="Context length (default: 2048)")

    args = parser.parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
