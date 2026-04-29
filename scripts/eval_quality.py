#!/usr/bin/env python3
import argparse
import dataclasses
import json
import time
import types
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.tuner.utils import load_adapters

DEFAULT_PROMPTS = [
    # Logic / Reasoning
    "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?",

    # Math
    "Solve for x: 2x + 5 = 15. Show your work.",

    # Creative Writing
    "Write a haiku about a computer chip dreaming of electric sheep.",

    # Coding (SQL)
    "Write a SQL query to find the top 5 customers who spent the most money from a table named 'orders' with columns 'customer_id' and 'amount'.",

    # Instruction Following (Constraint)
    "Write a single sentence describing the sun without using the letter 'e'.",

    # General Knowledge / Hallucination Check
    "Who was the first person to walk on Mars?",
]


def patch_model_for_adapters(model):
    if not hasattr(model.args, "num_layers") and hasattr(model.args, "num_hidden_layers"):
        try:
            model.args.num_layers = model.args.num_hidden_layers
        except Exception:
            pass

    if not hasattr(model, "config"):
        config_dict = model.args.__dict__ if hasattr(model.args, "__dict__") else {}
        if not config_dict:
            try:
                config_dict = dataclasses.asdict(model.args)
            except Exception:
                config_dict = {}
        if config_dict:
            config_dict["num_layers"] = model.args.num_hidden_layers
            model.config = types.SimpleNamespace(**config_dict)


def run_eval(name, model, tokenizer, prompts, max_tokens):
    print(f"\n{'='*20} {name} {'='*20}")
    results = []
    for prompt_text in prompts:
        print(f"\n[Prompt]: {prompt_text}")
        start = time.time()

        messages = [{"role": "user", "content": prompt_text}]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = prompt_text
        else:
            prompt = prompt_text

        response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        latency_s = time.time() - start
        response_text = response.strip()
        print(f"[Response ({latency_s:.2f}s)]:\n{response_text}")

        results.append({
            "prompt": prompt_text,
            "response": response_text,
            "latency_s": latency_s,
        })
    return results


def average_latency(samples):
    if not samples:
        return None
    return sum(sample["latency_s"] for sample in samples) / len(samples)


def load_prompts(prompt_file):
    if not prompt_file:
        return DEFAULT_PROMPTS
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    if path.suffix == ".json":
        with open(path, "r") as f:
            payload = json.load(f)
        return payload.get("prompts", [])
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def main():
    parser = argparse.ArgumentParser(description="Qualitative evaluation for MLX models")
    parser.add_argument("--base_model", default="mlx-community/Olmo-3-7B-Instruct-4bit",
                        help="Base model to load")
    parser.add_argument("--adapter_path", default="adapters/olmo3_7b_fineweb_scu_full",
                        help="SCU adapter path")
    parser.add_argument("--baseline_adapter_path", default="adapters/olmo3_7b_fineweb_baseline",
                        help="Baseline adapter path (optional)")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum generation tokens")
    parser.add_argument("--prompt_file", default=None,
                        help="Optional prompt file (.txt or .json with 'prompts')")
    parser.add_argument("--output", default=None,
                        help="Output JSON file for qualitative results")
    parser.add_argument("--summary_json", default=None,
                        help="Existing summary JSON to update with qualitative results")
    args = parser.parse_args()

    prompts = load_prompts(args.prompt_file)

    print(f"Loading Base Model: {args.base_model}")
    model_base, tokenizer_base = load(args.base_model)
    base_results = run_eval("Base Model", model_base, tokenizer_base, prompts, args.max_tokens)

    scu_results = None
    if args.adapter_path:
        print(f"\n\nLoading SCU Adapter: {args.adapter_path}")
        model_scu, tokenizer_scu = load(args.base_model)
        patch_model_for_adapters(model_scu)
        try:
            model_scu = load_adapters(model_scu, args.adapter_path)
            scu_results = run_eval("SCU Adapter", model_scu, tokenizer_scu, prompts, args.max_tokens)
        except Exception as exc:
            print(f"CRITICAL ERROR: Could not load adapters: {exc}")

    baseline_results = None
    if args.baseline_adapter_path and Path(args.baseline_adapter_path).exists():
        print(f"\n\nLoading Baseline Adapter: {args.baseline_adapter_path}")
        model_baseline, tokenizer_baseline = load(args.base_model)
        patch_model_for_adapters(model_baseline)
        try:
            model_baseline = load_adapters(model_baseline, args.baseline_adapter_path)
            baseline_results = run_eval("Baseline Adapter", model_baseline, tokenizer_baseline, prompts, args.max_tokens)
        except Exception as exc:
            print(f"Error loading baseline: {exc}")
    else:
        if args.baseline_adapter_path:
            print("Baseline adapter not found (training might still be in progress).")

    qualitative_summary = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "baseline_adapter_path": args.baseline_adapter_path,
        "max_tokens": args.max_tokens,
        "prompts": prompts,
        "results": {
            "base": base_results,
            "scu_adapter": scu_results,
            "baseline_adapter": baseline_results,
        },
        "latency_summary": {
            "base_avg_s": average_latency(base_results),
            "scu_avg_s": average_latency(scu_results) if scu_results else None,
            "baseline_avg_s": average_latency(baseline_results) if baseline_results else None,
        },
    }

    if args.summary_json:
        summary_path = Path(args.summary_json)
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary_payload = json.load(f)
        else:
            summary_payload = {}
        summary_payload["qualitative_eval"] = qualitative_summary
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"\nSummary updated at {args.summary_json}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(qualitative_summary, f, indent=2)
        print(f"\nQualitative results saved to {args.output}")


if __name__ == "__main__":
    main()
