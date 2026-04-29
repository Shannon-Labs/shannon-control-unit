#!/usr/bin/env python3
"""
Analyze trained LoRA adapter parameters to understand what SCU training changed.
Compares adapter weights against initialization to see the learned deltas.
"""
import argparse
import numpy as np
from pathlib import Path
import json

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    USE_MLX = True
except ImportError:
    USE_MLX = False
    from safetensors import safe_open

def analyze_adapter(adapter_path: str, prior_sigma: float = 0.01, tokens_per_epoch: int = None):
    """
    Analyze LoRA adapter weights to understand training impact.

    Args:
        adapter_path: Path to adapter directory with safetensors
        prior_sigma: Prior variance used in SCU (default: 0.01)
        tokens_per_epoch: Normalization constant from training (if None, tries to infer)
    """
    adapter_dir = Path(adapter_path)
    adapter_file = adapter_dir / "adapters.safetensors"

    if not adapter_file.exists():
        print(f"ERROR: {adapter_file} not found")
        return

    print("="*70)
    print("SCU LoRA Adapter Analysis")
    print("="*70)
    print(f"Adapter: {adapter_path}")
    print(f"Prior σ: {prior_sigma}")
    print()

    # Load adapter weights
    if USE_MLX:
        # Use MLX to load bfloat16 weights and convert to float32
        weights_mx = mx.load(str(adapter_file))
        weights = {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(weights_mx)}
    else:
        # Fallback to safetensors
        weights = {}
        with safe_open(adapter_file, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

    # Filter to ONLY LoRA weights (matching SCU's extract_lora_params_mlx logic)
    lora_weights = {k: v for k, v in weights.items() if 'lora' in k.lower()}
    lora_a_weights = {k: v for k, v in lora_weights.items() if 'lora_a' in k}
    lora_b_weights = {k: v for k, v in lora_weights.items() if 'lora_b' in k}
    non_lora_weights = {k: v for k, v in weights.items() if 'lora' not in k.lower()}

    print(f"Total weights in file: {len(weights):,}")
    print(f"  LoRA weights: {len(lora_weights):,} (LoRA A: {len(lora_a_weights)}, LoRA B: {len(lora_b_weights)})")
    print(f"  Non-LoRA (normalization, etc.): {len(non_lora_weights):,}")
    print()

    # Calculate statistics ONLY for LoRA weights (matching SCU)
    all_params = np.concatenate([w.flatten() for w in lora_weights.values()])
    total_params = len(all_params)

    print("="*70)
    print("Parameter Statistics")
    print("="*70)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Mean: {all_params.mean():.6f}")
    print(f"Std:  {all_params.std():.6f}")
    print(f"Min:  {all_params.min():.6f}")
    print(f"Max:  {all_params.max():.6f}")
    print(f"L2 norm: {np.linalg.norm(all_params):.2f}")
    print()

    # Calculate ParamBPT (should match SCU logs)
    sum_w_squared = np.sum(all_params ** 2)

    # Try to infer tokens_per_epoch from metadata if not provided
    if tokens_per_epoch is None:
        metadata_file = adapter_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                tokens_per_epoch = metadata.get("tokens_per_epoch", 100000000)
        else:
            # Use default estimate based on 1GB dataset (~250M chars ≈ 62M tokens)
            tokens_per_epoch = 62000000
            print(f"[NOTE] tokens_per_epoch not found, using estimate: {tokens_per_epoch:,}")

    # Correct formula: ParamBPT = Σ(w²) / (2σ² × N × ln(2))
    # where N = tokens_per_epoch (not number of parameters!)
    param_bpt = sum_w_squared / (2 * prior_sigma**2 * tokens_per_epoch * np.log(2))

    print("="*70)
    print("SCU Metrics (Information-Theoretic)")
    print("="*70)
    print(f"Σ(w²): {sum_w_squared:.2f}")
    print(f"Tokens per epoch (N): {tokens_per_epoch:,}")
    print(f"Prior σ: {prior_sigma}")
    print()
    print(f"ParamBPT: {param_bpt:.6f} bits/token")
    print(f"  = {sum_w_squared:.2f} / (2 × {prior_sigma}² × {tokens_per_epoch:,} × ln(2))")
    print()
    print("Interpretation:")
    print(f"  These {total_params:,} LoRA parameters use {param_bpt:.6f} bits")
    print(f"  per token of the {tokens_per_epoch:,}-token training dataset")
    print()
    print("NOTE: Compare this ParamBPT to the value in training logs")
    print("      to verify calculation consistency")
    print()

    # Analyze layer-wise changes (LoRA only)
    print("="*70)
    print("Top 10 LoRA Layers by Parameter Magnitude (L2 norm)")
    print("="*70)

    layer_stats = []
    for name, weight in lora_weights.items():
        l2_norm = np.linalg.norm(weight)
        mean = weight.mean()
        std = weight.std()
        layer_stats.append({
            'name': name,
            'l2_norm': l2_norm,
            'mean': mean,
            'std': std,
            'shape': weight.shape,
            'params': weight.size
        })

    layer_stats.sort(key=lambda x: x['l2_norm'], reverse=True)

    for i, stat in enumerate(layer_stats[:10], 1):
        print(f"{i:2d}. {stat['name']}")
        print(f"    L2: {stat['l2_norm']:.4f} | Mean: {stat['mean']:.6f} | "
              f"Std: {stat['std']:.6f} | Shape: {stat['shape']}")
    print()

    # Analyze LoRA A vs B differences
    print("="*70)
    print("LoRA A vs B Analysis")
    print("="*70)

    lora_a_params = np.concatenate([w.flatten() for w in lora_a_weights.values()])
    lora_b_params = np.concatenate([w.flatten() for w in lora_b_weights.values()])

    print("LoRA A (down-projection):")
    print(f"  Params: {len(lora_a_params):,}")
    print(f"  Mean: {lora_a_params.mean():.6f}")
    print(f"  Std:  {lora_a_params.std():.6f}")
    print(f"  L2:   {np.linalg.norm(lora_a_params):.2f}")
    print()

    print("LoRA B (up-projection):")
    print(f"  Params: {len(lora_b_params):,}")
    print(f"  Mean: {lora_b_params.mean():.6f}")
    print(f"  Std:  {lora_b_params.std():.6f}")
    print(f"  L2:   {np.linalg.norm(lora_b_params):.2f}")
    print()

    # Distribution analysis
    print("="*70)
    print("Parameter Distribution")
    print("="*70)

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_values = np.percentile(all_params, percentiles)

    for pct, val in zip(percentiles, pct_values):
        print(f"  {pct:2d}th percentile: {val:.6f}")
    print()

    # Sparsity analysis
    threshold = 1e-6
    near_zero = np.abs(all_params) < threshold
    sparsity = near_zero.sum() / total_params * 100

    print(f"Sparsity (|w| < {threshold}): {sparsity:.2f}%")
    print(f"Non-zero params: {(~near_zero).sum():,} ({100-sparsity:.2f}%)")
    print()

    # Save detailed stats
    stats_file = adapter_dir / "parameter_analysis.json"
    analysis = {
        "total_params": int(total_params),
        "lora_a_params": int(len(lora_a_params)),
        "lora_b_params": int(len(lora_b_params)),
        "statistics": {
            "mean": float(all_params.mean()),
            "std": float(all_params.std()),
            "min": float(all_params.min()),
            "max": float(all_params.max()),
            "l2_norm": float(np.linalg.norm(all_params))
        },
        "scu_metrics": {
            "sum_w_squared": float(sum_w_squared),
            "param_bpt": float(param_bpt),
            "prior_sigma": prior_sigma,
            "tokens_per_epoch": int(tokens_per_epoch)
        },
        "sparsity": {
            "threshold": threshold,
            "percentage": float(sparsity),
            "non_zero_count": int((~near_zero).sum())
        },
        "percentiles": {
            str(pct): float(val)
            for pct, val in zip(percentiles, pct_values)
        }
    }

    with open(stats_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"Detailed analysis saved to: {stats_file}")
    print()

    return analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze LoRA adapter parameters")
    parser.add_argument(
        "--adapter-path",
        default="adapters/olmo3_7b_fineweb_scu_full",
        help="Path to adapter directory"
    )
    parser.add_argument(
        "--prior-sigma",
        type=float,
        default=0.01,
        help="Prior sigma used in SCU training"
    )
    parser.add_argument(
        "--tokens-per-epoch",
        type=int,
        default=None,
        help="Tokens per epoch normalization constant (auto-detected if not provided)"
    )

    args = parser.parse_args()
    analyze_adapter(args.adapter_path, args.prior_sigma, args.tokens_per_epoch)
