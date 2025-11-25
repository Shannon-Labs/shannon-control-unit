#!/usr/bin/env python3
"""
OLMo 3 7B - Evaluation Script

Evaluates a trained adapter on validation data, computing:
- Perplexity (PPL)
- Bits per token (BPT)
- DataBPT / ParamBPT / S-ratio

Usage:
    python scripts/eval_olmo3.py --adapter adapters/olmo3_7b_scu
    python scripts/eval_olmo3.py --adapter adapters/olmo3_7b_scu --val-data data/val.txt
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from typing import Optional

# Add parent dir to path
sys.path.append(str(Path(__file__).parent.parent))


def load_model_with_adapter(
    adapter_path: str,
    base_model: Optional[str] = None,
    use_4bit: bool = True,
    device: str = "auto",
):
    """Load model with LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    # Load metadata if available
    metadata_path = Path(adapter_path) / "metadata.json"
    if metadata_path.exists() and not base_model:
        with open(metadata_path) as f:
            metadata = json.load(f)
        base_model = metadata.get("base_model", "allenai/OLMo-3-7B")
        print(f"Base model from metadata: {base_model}")

    if not base_model:
        base_model = "allenai/OLMo-3-7B"

    print(f"Loading base model: {base_model}")

    # Setup quantization
    quantization_config = None
    if use_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    # Determine device map
    if device == "auto":
        device_map = "auto" if torch.cuda.is_available() else None
    else:
        device_map = device

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True
    )

    # Load tokenizer from adapter (has any special tokens)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load adapter
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def evaluate_perplexity(
    model,
    tokenizer,
    data_path: str,
    block_size: int = 2048,
    max_samples: int = 100,
    stride: Optional[int] = None,
) -> dict:
    """
    Evaluate perplexity on validation data.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        data_path: Path to validation data (txt or jsonl)
        block_size: Context window size
        max_samples: Maximum number of chunks to evaluate
        stride: Stride for sliding window (default: block_size)

    Returns:
        Dictionary with evaluation metrics
    """
    import torch
    from scu import data as scu_data

    if stride is None:
        stride = block_size

    print(f"\nEvaluating on: {data_path}")
    print(f"Block size: {block_size}, Stride: {stride}, Max samples: {max_samples}")

    # Load and tokenize data
    texts = scu_data.load_texts_from_file(data_path, max_texts=None)
    chunks = scu_data.tokenize_and_chunk(
        texts, tokenizer, block_size=block_size, shuffle=False
    )

    if not chunks:
        print("Error: No chunks created from data")
        return {"error": "No data chunks"}

    # Limit samples
    eval_chunks = chunks[:max_samples]
    print(f"Evaluating on {len(eval_chunks)} chunks")

    # Get device
    device = next(model.parameters()).device

    # Evaluate
    total_loss = 0.0
    total_tokens = 0
    losses = []

    model.eval()
    with torch.no_grad():
        for i, chunk in enumerate(eval_chunks):
            input_ids = torch.tensor([chunk['input_ids']]).to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss.item()

            num_tokens = input_ids.shape[1] - 1  # Labels shifted by 1
            total_loss += loss * num_tokens
            total_tokens += num_tokens
            losses.append(loss)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(eval_chunks)}")

    # Calculate metrics
    avg_loss_nats = total_loss / total_tokens if total_tokens > 0 else float('inf')
    avg_loss_bits = avg_loss_nats / math.log(2)  # Convert to bits (DataBPT)
    perplexity = math.exp(avg_loss_nats) if avg_loss_nats < 100 else float('inf')

    results = {
        "perplexity": perplexity,
        "avg_loss_nats": avg_loss_nats,
        "data_bpt": avg_loss_bits,
        "total_tokens": total_tokens,
        "num_chunks": len(eval_chunks),
        "loss_std": (sum((l - avg_loss_nats)**2 for l in losses) / len(losses))**0.5,
    }

    return results


def evaluate_scu_metrics(
    model,
    tokenizer,
    train_data_path: str,
    val_data_path: str,
    block_size: int = 2048,
    prior_sigma: float = 0.01,
) -> dict:
    """
    Evaluate SCU-specific metrics (DataBPT, ParamBPT, S-ratio).
    """
    from scu import control, data as scu_data

    print("\nCalculating SCU metrics...")

    # Load training data stats
    train_texts = scu_data.load_texts_from_file(train_data_path, max_texts=1000)
    train_chunks = scu_data.tokenize_and_chunk(
        train_texts, tokenizer, block_size=block_size, shuffle=False
    )
    tokens_per_epoch = len(train_chunks) * block_size

    # Calculate ParamBPT
    param_bpt = control.calculate_param_bpt(
        model, sigma=prior_sigma, tokens_per_epoch=tokens_per_epoch
    )

    # Evaluate validation loss
    val_results = evaluate_perplexity(
        model, tokenizer, val_data_path,
        block_size=block_size, max_samples=50
    )

    data_bpt = val_results["data_bpt"]

    # Calculate S-ratio
    s_ratio = control.calculate_s_ratio(data_bpt, param_bpt)

    return {
        "data_bpt": data_bpt,
        "param_bpt": param_bpt,
        "s_ratio": s_ratio,
        "tokens_per_epoch": tokens_per_epoch,
        "perplexity": val_results["perplexity"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OLMo 3 adapter"
    )

    parser.add_argument("--adapter", required=True,
                       help="Path to adapter directory")
    parser.add_argument("--base-model",
                       help="Override base model (default: from metadata)")
    parser.add_argument("--val-data", default="data/val.txt",
                       help="Validation data file")
    parser.add_argument("--train-data", default="data/fineweb_edu_sample.jsonl",
                       help="Training data file (for ParamBPT normalization)")
    parser.add_argument("--block-size", type=int, default=2048,
                       help="Context length")
    parser.add_argument("--max-samples", type=int, default=100,
                       help="Max validation samples")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization")
    parser.add_argument("--output", help="Output JSON file for results")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_with_adapter(
        args.adapter,
        base_model=args.base_model,
        use_4bit=not args.no_4bit,
    )

    # Evaluate perplexity
    print("\n" + "=" * 60)
    print("PERPLEXITY EVALUATION")
    print("=" * 60)

    ppl_results = evaluate_perplexity(
        model, tokenizer, args.val_data,
        block_size=args.block_size,
        max_samples=args.max_samples,
    )

    print(f"\nResults:")
    print(f"  Perplexity: {ppl_results['perplexity']:.2f}")
    print(f"  DataBPT: {ppl_results['data_bpt']:.4f}")
    print(f"  Avg Loss (nats): {ppl_results['avg_loss_nats']:.4f}")
    print(f"  Total tokens: {ppl_results['total_tokens']}")

    # Evaluate SCU metrics if training data available
    if Path(args.train_data).exists():
        print("\n" + "=" * 60)
        print("SCU METRICS")
        print("=" * 60)

        scu_results = evaluate_scu_metrics(
            model, tokenizer,
            args.train_data, args.val_data,
            block_size=args.block_size,
        )

        print(f"\nSCU Metrics:")
        print(f"  DataBPT: {scu_results['data_bpt']:.4f}")
        print(f"  ParamBPT: {scu_results['param_bpt']:.6f}")
        print(f"  S-ratio: {scu_results['s_ratio']:.2%}")

    # Load adapter metadata
    metadata_path = Path(args.adapter) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        print("\n" + "=" * 60)
        print("TRAINING METADATA")
        print("=" * 60)
        print(f"  Base model: {metadata.get('base_model', 'N/A')}")
        print(f"  Target S: {metadata.get('target_s', 'N/A')}")
        print(f"  Final S: {metadata.get('final_s', 'N/A')}")
        print(f"  Final lambda: {metadata.get('final_lambda', 'N/A')}")
        print(f"  Steps: {metadata.get('steps', 'N/A')}")
    else:
        metadata = {}

    # Combine results
    all_results = {
        "adapter": args.adapter,
        "perplexity": ppl_results,
        "metadata": metadata,
    }

    if Path(args.train_data).exists():
        all_results["scu_metrics"] = scu_results

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return all_results


if __name__ == "__main__":
    main()
