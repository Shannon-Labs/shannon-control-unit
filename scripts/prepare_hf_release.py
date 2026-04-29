#!/usr/bin/env python3
"""
HuggingFace Release Preparation Script

Prepares a trained SCU adapter for HuggingFace Hub release by generating:
- adapter_config.json (PEFT-compatible)
- README.md (model card with training details)
- Validates adapter structure

Usage:
    # Prepare adapter for HF release
    python scripts/prepare_hf_release.py --adapter-path adapters/olmo3_7b_fineweb_scu

    # Preview only (don't write files)
    python scripts/prepare_hf_release.py --adapter-path adapters/olmo3_7b_fineweb_scu --dry-run

    # Upload to HuggingFace (requires huggingface-cli login)
    python scripts/prepare_hf_release.py --adapter-path adapters/olmo3_7b_fineweb_scu --upload --repo-id hunterbown/olmo3-fineweb-scu
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_metadata(adapter_path: Path) -> dict:
    """Load metadata.json from adapter directory."""
    metadata_file = adapter_path / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata.json not found in {adapter_path}")

    with open(metadata_file) as f:
        return json.load(f)


def generate_adapter_config(metadata: dict) -> dict:
    """Generate PEFT-compatible adapter_config.json."""
    lora_config = metadata.get("lora_config", {})

    return {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": metadata.get("base_model", ""),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": lora_config.get("alpha", 64),
        "lora_dropout": lora_config.get("dropout", 0.05),
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_config.get("r", 32),
        "rank_pattern": {},
        "revision": None,
        "target_modules": lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False
    }


def generate_model_card(metadata: dict, repo_id: str = None) -> str:
    """Generate HuggingFace model card README.md."""
    base_model = metadata.get("base_model", "Unknown")
    dataset = metadata.get("dataset", "FineWeb-Edu")
    target_s = metadata.get("target_s", 0.03)
    final_s = metadata.get("final_s_ratio", 0)
    final_lambda = metadata.get("final_lambda", 0)
    final_loss = metadata.get("final_loss", 0)

    scu_config = metadata.get("scu_config", {})
    lora_config = metadata.get("lora_config", {})
    training_config = metadata.get("training_config", {})

    # Extract model name for title
    model_name = base_model.split("/")[-1].replace("-4bit", "").replace("-Instruct", "")

    card = f"""---
base_model: {base_model}
library_name: peft
license: apache-2.0
tags:
- generated_from_trainer
- shannon-control-unit
- scu
- mlx
- olmo
- fineweb-edu
- lora
datasets:
- HuggingFaceFW/fineweb-edu
model-index:
- name: {model_name}-SCU-FineWeb
  results: []
---

# {model_name} SCU (FineWeb-Edu Fine-tuned)

This LoRA adapter was trained using the **Shannon Control Unit (SCU)** method on Apple Silicon MLX.

## Model Details

| Property | Value |
|----------|-------|
| Base Model | [{base_model}](https://huggingface.co/{base_model}) |
| Dataset | FineWeb-Edu (~1GB educational web text) |
| Training Method | SCU (PI Control) |
| Backend | MLX (Apple Silicon) |

## Training Results

| Metric | Value |
|--------|-------|
| Target S-ratio | {target_s:.1%} |
| Final S-ratio | {final_s:.4f} ({final_s*100:.2f}%) |
| Final Lambda | {final_lambda:.4f} |
| Final Loss | {final_loss:.4f} |
| Total Steps | {training_config.get('steps', 'N/A')} |

## SCU Control Parameters

The Shannon Control Unit maintains optimal regularization through PI (Proportional-Integral) control:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Target S | {target_s} | Target information ratio |
| Kp | {scu_config.get('kp', 0.8)} | Proportional gain |
| Ki | {scu_config.get('ki', 0.15)} | Integral gain |
| Deadband | {scu_config.get('deadband', 0.003)} | Error threshold |
| Lambda Range | [{scu_config.get('lambda_min', 0.0001)}, {scu_config.get('lambda_max', 2.0)}] | Regularization bounds |

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | {lora_config.get('r', 32)} |
| Alpha | {lora_config.get('alpha', 64)} |
| Dropout | {lora_config.get('dropout', 0.05)} |
| Target Modules | {', '.join(lora_config.get('target_modules', []))} |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Steps | {training_config.get('steps', 'N/A')} |
| Batch Size | {training_config.get('batch_size', 1)} |
| Effective Batch | {training_config.get('effective_batch_size', 16)} |
| Learning Rate | {training_config.get('lr', 2e-5)} |
| Warmup Steps | {training_config.get('warmup_steps', 500)} |
| Block Size | {training_config.get('block_size', 2048)} |

## Usage

### With MLX (Apple Silicon)

```python
from mlx_lm import load, generate

# Load base model
model, tokenizer = load("{base_model}")

# Load adapter (download or local path)
model.load_adapter("path/to/adapter")

# Generate
prompt = "Explain quantum computing:"
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```

### With Transformers + PEFT (PyTorch)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model (use non-quantized version)
base_model = "allenai/OLMo-3-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "path/to/adapter")

# Generate
inputs = tokenizer("Explain quantum computing:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

## What is SCU?

The **Shannon Control Unit (SCU)** is an information-theoretic approach to LLM fine-tuning that:

1. **Measures Information Ratio (S)**: Computes the ratio of parameter complexity (ParamBPT) to total information (DataBPT + ParamBPT)
2. **Applies PI Control**: Uses a feedback controller to dynamically adjust regularization strength (lambda)
3. **Prevents Overfitting**: Maintains optimal complexity by targeting a specific S-ratio

The control law: `lambda(t+1) = lambda(t) * exp(Kp * error + Ki * integral)`

## License

This adapter is released under the Apache 2.0 license.

## Citation

If you use this adapter, please cite the Shannon Control Unit project:

```bibtex
@software{{shannon_control_unit,
  title = {{Shannon Control Unit: Information-Theoretic LLM Fine-tuning}},
  author = {{Shannon Labs}},
  year = {{2025}},
  url = {{https://github.com/Shannon-Labs/shannon-control-unit}}
}}
```

## Links

- [Shannon Control Unit Repository](https://github.com/Shannon-Labs/shannon-control-unit)
- [Base Model]({f"https://huggingface.co/{base_model}"})
- [FineWeb-Edu Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

---

*Trained on {metadata.get('timestamp', datetime.now().isoformat())[:10]}*
"""
    return card


def validate_adapter(adapter_path: Path) -> list:
    """Validate adapter structure and return list of issues."""
    issues = []
    required_files = ["adapters.safetensors", "metadata.json"]
    optional_files = ["tokenizer.json", "tokenizer_config.json", "scu_metrics.json"]

    for f in required_files:
        if not (adapter_path / f).exists():
            issues.append(f"Missing required file: {f}")

    missing_optional = [f for f in optional_files if not (adapter_path / f).exists()]
    if missing_optional:
        issues.append(f"Missing optional files: {', '.join(missing_optional)}")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SCU adapter for HuggingFace release",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="Path to adapter directory"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID (e.g., hunterbown/olmo3-fineweb-scu)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace Hub after preparation"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make HuggingFace repo private"
    )

    args = parser.parse_args()

    # Resolve adapter path
    adapter_path = args.adapter_path
    if not adapter_path.is_absolute():
        adapter_path = project_root / adapter_path

    if not adapter_path.exists():
        print(f"[ERROR] Adapter path does not exist: {adapter_path}")
        sys.exit(1)

    print("=" * 60)
    print("HuggingFace Release Preparation")
    print("=" * 60)
    print(f"Adapter: {adapter_path}")
    print()

    # Validate adapter
    print("[1/4] Validating adapter structure...")
    issues = validate_adapter(adapter_path)
    if issues:
        for issue in issues:
            print(f"  WARNING: {issue}")
    else:
        print("  OK - All required files present")

    # Load metadata
    print("[2/4] Loading metadata...")
    try:
        metadata = load_metadata(adapter_path)
        print(f"  Base model: {metadata.get('base_model', 'Unknown')}")
        print(f"  Final S-ratio: {metadata.get('final_s_ratio', 'N/A')}")
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)

    # Generate adapter_config.json
    print("[3/4] Generating adapter_config.json...")
    adapter_config = generate_adapter_config(metadata)
    adapter_config_path = adapter_path / "adapter_config.json"

    if args.dry_run:
        print("  [DRY RUN] Would write adapter_config.json:")
        print(f"    r={adapter_config['r']}, alpha={adapter_config['lora_alpha']}")
    else:
        with open(adapter_config_path, "w") as f:
            json.dump(adapter_config, f, indent=2)
        print(f"  Wrote: {adapter_config_path}")

    # Generate README.md
    print("[4/4] Generating README.md model card...")
    readme_content = generate_model_card(metadata, args.repo_id)
    readme_path = adapter_path / "README.md"

    if args.dry_run:
        print("  [DRY RUN] Would write README.md")
        print("  Preview (first 500 chars):")
        print("  " + "-" * 40)
        print(readme_content[:500])
        print("  " + "-" * 40)
    else:
        with open(readme_path, "w") as f:
            f.write(readme_content)
        print(f"  Wrote: {readme_path}")

    print()
    print("=" * 60)
    print("PREPARATION COMPLETE!")
    print("=" * 60)

    # List final files
    print("\nAdapter contents:")
    for f in sorted(adapter_path.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {f.name:<30} {size_str:>10}")

    # Upload to HuggingFace if requested
    if args.upload:
        if not args.repo_id:
            print("\n[ERROR] --repo-id required for upload")
            sys.exit(1)

        print(f"\nUploading to HuggingFace: {args.repo_id}")
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.upload_folder(
                folder_path=str(adapter_path),
                repo_id=args.repo_id,
                repo_type="model",
                create_pr=False,
                private=args.private
            )
            print(f"  Uploaded to: https://huggingface.co/{args.repo_id}")
        except ImportError:
            print("  [ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
            sys.exit(1)
        except Exception as e:
            print(f"  [ERROR] Upload failed: {e}")
            sys.exit(1)
    else:
        print("\nNext steps:")
        print(f"  1. Review files in {adapter_path}")
        print(f"  2. Upload: python scripts/prepare_hf_release.py --adapter-path {args.adapter_path} --upload --repo-id YOUR_REPO_ID")
        print("  3. Or manually: huggingface-cli upload YOUR_REPO_ID", adapter_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
