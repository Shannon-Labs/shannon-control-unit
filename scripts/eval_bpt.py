#!/usr/bin/env python3
"""Evaluate base model vs SCU adapter on BPT, perplexity, and control diagnostics."""

import os
import sys
import argparse
import math
import json
import csv
import random
import statistics as stats
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Add parent dir to path
sys.path.append(str(Path(__file__).parent.parent))
from scu import data


def bpt_for_texts(model, tokenizer, texts, max_len=512, device=None):
    """Calculate BPT for each text.
    
    Returns list of BPT values (one per text).
    """
    model.eval()
    bpts = []
    
    for text in texts:
        # Tokenize
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=False
        )
        
        # Move to device
        enc = {k: v.to(device or model.device) for k, v in enc.items()}
        
        # Labels are same as inputs
        labels = enc["input_ids"].clone()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**enc, labels=labels)
            # Convert from nats to bits
            bpt = outputs.loss.item() / math.log(2)
            bpts.append(bpt)
    
    return bpts


def summarize_bpt(model, tokenizer, texts, max_len=512, device=None):
    bpts = bpt_for_texts(model, tokenizer, texts, max_len=max_len, device=device)
    mean_bpt = stats.mean(bpts)
    ppl = 2 ** mean_bpt
    return bpts, mean_bpt, ppl


def bootstrap_ci(delta_list, iters=10000, seed=42):
    """Bootstrap confidence interval for mean difference.
    
    Returns (lower_95, mean, upper_95)
    """
    random.seed(seed)
    means = []
    n = len(delta_list)
    
    for _ in range(iters):
        # Resample with replacement
        sample = [delta_list[random.randrange(n)] for _ in range(n)]
        means.append(stats.mean(sample))
    
    means.sort()
    lower = means[int(0.025 * iters)]
    upper = means[int(0.975 * iters)]
    mean_val = stats.mean(delta_list)
    
    return lower, mean_val, upper


def safe_ratio(numerator, denominator):
    """Return numerator/denominator, or None when invalid."""
    if denominator in (None, 0):
        return None
    return numerator / denominator


def load_metadata(adapter_path):
    if not adapter_path:
        return {}
    metadata_path = Path(adapter_path) / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def load_control_log(control_log_path):
    if not control_log_path:
        return []
    rows = []
    with open(control_log_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                rows.append({
                    "step": int(float(row.get("step", 0))),
                    "s_ratio": float(row.get("S", 0.0)),
                    "lambda": float(row.get("lambda", 0.0)),
                })
            except (ValueError, TypeError):
                continue
    return rows


def convergence_step(steps, s_values, target_s, epsilon):
    if not steps or target_s is None:
        return None
    last_violation = None
    for idx, s_val in enumerate(s_values):
        if abs(s_val - target_s) > epsilon:
            last_violation = idx
    if last_violation is None:
        return steps[0]
    if last_violation >= len(steps) - 1:
        return None
    return steps[last_violation + 1]


def compute_control_metrics(control_log_path, target_s, epsilon=0.002):
    rows = load_control_log(control_log_path)
    if not rows or target_s is None:
        return {}

    steps = [row["step"] for row in rows]
    s_values = [row["s_ratio"] for row in rows]
    lambdas = [row["lambda"] for row in rows]

    errors = [s_val - target_s for s_val in s_values]
    e_rms = math.sqrt(sum(err * err for err in errors) / len(errors))
    steady_state_error = abs(errors[-1])

    overshoot_pct = None
    if target_s > 0:
        overshoot_pct = max(0.0, max(s_values) - target_s) / target_s * 100.0

    lambda_mean = stats.mean(lambdas) if lambdas else None
    lambda_std = stats.pstdev(lambdas) if len(lambdas) > 1 else 0.0
    osc_pct = safe_ratio(lambda_std, lambda_mean)
    if osc_pct is not None:
        osc_pct *= 100.0

    return {
        "target_s": target_s,
        "tracking_error_rms": e_rms,
        "steady_state_error": steady_state_error,
        "overshoot_percent": overshoot_pct,
        "oscillation_percent": osc_pct,
        "lambda_mean": lambda_mean,
        "lambda_std": lambda_std,
        "convergence_step": convergence_step(steps, s_values, target_s, epsilon),
        "epsilon": epsilon,
        "num_steps": len(steps),
    }


def main(args):
    # Suppress tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    metadata = load_metadata(args.adapter_path)
    target_s = args.target_s if args.target_s is not None else metadata.get("target_s")
    steps = args.steps if args.steps is not None else metadata.get("steps")
    tokens_per_epoch = metadata.get("tokens_per_epoch")
    control_log = args.control_log or metadata.get("log_csv")

    # Setup device and dtype
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        use_4bit = not args.no_4bit
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        use_4bit = False
    else:
        device = "cpu"
        dtype = torch.float32
        use_4bit = False
        print("WARNING: Using CPU - evaluation will be slow")
    
    # Quantization config
    quantization_config = None
    if use_4bit and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    
    # Load base model
    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load validation texts
    print(f"Loading validation texts from {args.texts}")
    val_texts = data.load_texts_from_file(args.texts, max_texts=args.max_texts)
    print(f"Loaded {len(val_texts)} texts")

    train_texts = None
    if args.train_texts:
        print(f"Loading training texts from {args.train_texts}")
        train_texts = data.load_texts_from_file(args.train_texts, max_texts=args.max_train_texts)
        print(f"Loaded {len(train_texts)} training texts")

    domain_texts = None
    if args.domain_texts:
        print(f"Loading domain texts from {args.domain_texts}")
        domain_texts = data.load_texts_from_file(args.domain_texts, max_texts=args.max_domain_texts)
        print(f"Loaded {len(domain_texts)} domain texts")
    
    # Evaluate base model
    print("\nEvaluating base model...")
    base_bpts, base_mean_bpt, base_perplexity = summarize_bpt(
        base_model,
        tokenizer,
        val_texts,
        max_len=args.max_length,
        device=device
    )

    base_train_bpt = None
    if train_texts:
        _, base_train_bpt, _ = summarize_bpt(
            base_model,
            tokenizer,
            train_texts,
            max_len=args.max_length,
            device=device
        )

    base_domain_bpt = None
    base_domain_score = args.domain_score_base
    if domain_texts:
        _, base_domain_bpt, _ = summarize_bpt(
            base_model,
            tokenizer,
            domain_texts,
            max_len=args.max_length,
            device=device
        )
        if base_domain_score is None:
            base_domain_score = safe_ratio(1.0, base_domain_bpt)
    
    # Load adapter model if provided
    if args.adapter_path:
        print(f"\nLoading SCU adapter from {args.adapter_path}")
        scu_model = PeftModel.from_pretrained(base_model, args.adapter_path)
        scu_model.eval()
        
        # Evaluate SCU model
        print("Evaluating SCU model...")
        scu_bpts, scu_mean_bpt, scu_perplexity = summarize_bpt(
            scu_model,
            tokenizer,
            val_texts,
            max_len=args.max_length,
            device=device
        )

        scu_train_bpt = None
        if train_texts:
            _, scu_train_bpt, _ = summarize_bpt(
                scu_model,
                tokenizer,
                train_texts,
                max_len=args.max_length,
                device=device
            )

        scu_domain_bpt = None
        scu_domain_score = args.domain_score_scu
        if domain_texts:
            _, scu_domain_bpt, _ = summarize_bpt(
                scu_model,
                tokenizer,
                domain_texts,
                max_len=args.max_length,
                device=device
            )
            if scu_domain_score is None:
                scu_domain_score = safe_ratio(1.0, scu_domain_bpt)
        
        # Calculate differences
        delta_bpts = [b - s for b, s in zip(base_bpts, scu_bpts)]
        delta_mean = stats.mean(delta_bpts)
        generalization_gain = safe_ratio(base_mean_bpt, scu_mean_bpt)

        retention_ratio = None
        if base_domain_score is not None and scu_domain_score is not None:
            retention_ratio = safe_ratio(scu_domain_score, base_domain_score)

        pag = None
        if generalization_gain is not None and retention_ratio is not None:
            pag = math.sqrt(generalization_gain * retention_ratio)

        control_metrics = compute_control_metrics(control_log, target_s)

        efficiency_per_step = safe_ratio(delta_mean, steps)
        efficiency_per_flop = safe_ratio(delta_mean, args.total_flops)
        
        # Bootstrap CI
        if args.bootstrap:
            print("\nCalculating bootstrap confidence interval...")
            ci_lower, ci_mean, ci_upper = bootstrap_ci(delta_bpts, iters=args.bootstrap_iters)
        else:
            ci_lower = ci_mean = ci_upper = delta_mean
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Base Model:    {base_mean_bpt:.3f} BPT (ppl {base_perplexity:.2f})")
        print(f"SCU Model:     {scu_mean_bpt:.3f} BPT (ppl {scu_perplexity:.2f})")
        print(f"Improvement:   {delta_mean:.3f} BPT ({100*delta_mean/base_mean_bpt:.1f}%)")
        print(f"Perplexity:    -{100*(1 - scu_perplexity/base_perplexity):.1f}%")

        if retention_ratio is not None:
            print(f"Retention P:   {retention_ratio:.3f}")
        if pag is not None:
            print(f"PAG:           {pag:.3f}")
        if efficiency_per_step is not None:
            print(f"dBPT/step:     {efficiency_per_step:.6f}")
        if efficiency_per_flop is not None:
            print(f"dBPT/FLOP:     {efficiency_per_flop:.6e}")
        
        if args.bootstrap:
            print(f"\nBootstrap 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
            if ci_lower > 0:
                print("✓ CI excludes zero - improvement is statistically significant")
            else:
                print("✗ CI includes zero - improvement not statistically significant")
        
        # Save results if requested
        if args.output:
            results = {
                "summary_version": "v2",
                'base_model': args.base_model,
                'adapter_path': args.adapter_path,
                'num_texts': len(val_texts),
                'base_bpt': base_mean_bpt,
                'scu_bpt': scu_mean_bpt,
                'delta_bpt': delta_mean,
                'delta_bpt_percent': 100 * delta_mean / base_mean_bpt,
                'base_perplexity': base_perplexity,
                'scu_perplexity': scu_perplexity,
                'perplexity_reduction': 100 * (1 - scu_perplexity/base_perplexity),
                'ci_lower': ci_lower,
                'ci_mean': ci_mean,
                'ci_upper': ci_upper,
                'individual_base_bpts': base_bpts,
                'individual_scu_bpts': scu_bpts,
                "generalization": {
                    "bpt_base": base_mean_bpt,
                    "bpt_finetuned": scu_mean_bpt,
                    "ppl_base": base_perplexity,
                    "ppl_finetuned": scu_perplexity,
                    "delta_bpt": delta_mean,
                    "delta_bpt_percent": 100 * delta_mean / base_mean_bpt,
                    "generalization_gain": generalization_gain,
                },
                "domain_retention": {
                    "domain_texts": args.domain_texts,
                    "num_texts": len(domain_texts) if domain_texts else None,
                    "bpt_base": base_domain_bpt,
                    "bpt_finetuned": scu_domain_bpt,
                    "domain_score_base": base_domain_score,
                    "domain_score_finetuned": scu_domain_score,
                    "retention_ratio": retention_ratio,
                },
                "generalization_gap": {
                    "train_bpt_base": base_train_bpt,
                    "train_bpt_finetuned": scu_train_bpt,
                    "val_bpt_base": base_mean_bpt,
                    "val_bpt_finetuned": scu_mean_bpt,
                    "gap_base": None if base_train_bpt is None else base_train_bpt - base_mean_bpt,
                    "gap_finetuned": None if scu_train_bpt is None else scu_train_bpt - scu_mean_bpt,
                },
                "efficiency": {
                    "steps": steps,
                    "total_flops": args.total_flops,
                    "delta_bpt_per_step": efficiency_per_step,
                    "delta_bpt_per_flop": efficiency_per_flop,
                },
                "pag": {
                    "pag": pag,
                    "generalization_gain": generalization_gain,
                    "retention_ratio": retention_ratio,
                },
                "control_diagnostics": control_metrics,
                "control_log": control_log,
                "metadata": metadata,
                "tokens_per_epoch": tokens_per_epoch,
            }
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to {args.output}")
    
    else:
        # Base model only
        print("\n" + "="*60)
        print("BASE MODEL RESULTS")
        print("="*60)
        print(f"BPT:        {base_mean_bpt:.3f}")
        print(f"Perplexity: {base_perplexity:.2f}")
        print(f"Texts:      {len(val_texts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BPT/PPL, PAG, and control diagnostics")
    
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-1B",
                       help="Base model name")
    parser.add_argument("--adapter_path", default=None,
                       help="Path to SCU adapter (optional)")
    parser.add_argument("--texts", default="data/val.txt",
                       help="Validation texts file")
    parser.add_argument("--max_texts", type=int, default=None,
                       help="Maximum texts to evaluate")
    parser.add_argument("--train_texts", default=None,
                       help="Training texts file for generalization gap")
    parser.add_argument("--max_train_texts", type=int, default=None,
                       help="Maximum training texts to evaluate")
    parser.add_argument("--domain_texts", default=None,
                       help="Domain benchmark texts file for retention")
    parser.add_argument("--max_domain_texts", type=int, default=None,
                       help="Maximum domain texts to evaluate")
    parser.add_argument("--domain_score_base", type=float, default=None,
                       help="External domain score for base model (higher is better)")
    parser.add_argument("--domain_score_scu", type=float, default=None,
                       help="External domain score for SCU model (higher is better)")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--no_4bit", action="store_true",
                       help="Disable 4-bit quantization")
    parser.add_argument("--bootstrap", action="store_true",
                       help="Calculate bootstrap CI")
    parser.add_argument("--bootstrap_iters", type=int, default=10000,
                       help="Bootstrap iterations")
    parser.add_argument("--control_log", default=None,
                       help="CSV log file with S and lambda for control diagnostics")
    parser.add_argument("--target_s", type=float, default=None,
                       help="Target S ratio for control diagnostics (defaults to adapter metadata)")
    parser.add_argument("--steps", type=int, default=None,
                       help="Total training steps for compute-adjusted metrics")
    parser.add_argument("--total_flops", type=float, default=None,
                       help="Total FLOPs for compute-adjusted metrics")
    parser.add_argument("--output", default=None,
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    main(args)
