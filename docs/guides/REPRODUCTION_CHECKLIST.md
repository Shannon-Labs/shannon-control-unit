# SCU Reproduction Checklist

Exact steps to reproduce published SCU results.

---

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/Shannon-Labs/shannon-control-unit.git
cd shannon-control-unit
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Unix/macOS
# or: .venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "from shannon_control import SCUController; print('SCU OK')"
python -c "import mlx; print('MLX OK')"  # Apple Silicon only
python -c "import torch; print('PyTorch OK')"  # CUDA only
```

---

## Exact Versions for Reproduction

Tested with these versions (as of December 2024):

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11+ | Required |
| torch | 2.1.0+ | For CUDA training |
| transformers | 4.36.0+ | HuggingFace models |
| peft | 0.7.0+ | LoRA implementation |
| mlx | 0.4.0+ | Apple Silicon only |
| mlx-lm | 0.4.0+ | MLX model loading |
| bitsandbytes | 0.42.0+ | 4-bit quantization |

### Check Your Versions

```bash
pip list | grep -E "torch|transformers|peft|mlx|bitsandbytes"
```

---

## Reproducing Llama-3.2-1B Results

### Expected Results

| Metric | Baseline | SCU | Improvement |
|--------|----------|-----|-------------|
| BPT | 3.920 | 3.676 | -6.2% |

### Steps

```bash
# 1. Download or prepare data
# Use FineWeb-Edu subset (500MB)

# 2. Run baseline (standard LoRA, no SCU)
python scripts/train_llama.py \
    --model meta-llama/Llama-3.2-1B \
    --adapter-out adapters/llama_1b_baseline \
    --steps 500 \
    --lambda-min 0.0 \
    --lambda-max 0.0

# 3. Run SCU
python scripts/train_llama.py \
    --model meta-llama/Llama-3.2-1B \
    --adapter-out adapters/llama_1b_scu \
    --steps 500 \
    --target-s 0.01 \
    --lambda-init 1.0 \
    --lambda-min 0.0001 \
    --lambda-max 2.0

# 4. Compare results
python scripts/eval_quality.py \
    --baseline adapters/llama_1b_baseline \
    --scu adapters/llama_1b_scu
```

### Verification Criteria

- [ ] Baseline BPT ≈ 3.9 (±0.1)
- [ ] SCU BPT ≈ 3.7 (±0.1)
- [ ] SCU lambda varied during training (not stuck)
- [ ] Improvement ~6% (±2%)

---

## Reproducing Llama-3.2-3B Results

### Expected Results

| Metric | Baseline | SCU | Improvement |
|--------|----------|-----|-------------|
| BPT | 1.830 | 1.635 | -10.6% |

### Steps

```bash
# 1. Run baseline
python scripts/train_llama.py \
    --model meta-llama/Llama-3.2-3B \
    --adapter-out adapters/llama_3b_baseline \
    --steps 500 \
    --lambda-min 0.0 \
    --lambda-max 0.0

# 2. Run SCU
python scripts/train_llama.py \
    --model meta-llama/Llama-3.2-3B \
    --adapter-out adapters/llama_3b_scu \
    --steps 500 \
    --target-s 0.03 \
    --lambda-init 1.0 \
    --lambda-min 0.0001 \
    --lambda-max 2.0

# 3. Compare
python scripts/eval_quality.py \
    --baseline adapters/llama_3b_baseline \
    --scu adapters/llama_3b_scu
```

### Verification Criteria

- [ ] Baseline BPT ≈ 1.83 (±0.05)
- [ ] SCU BPT ≈ 1.64 (±0.05)
- [ ] Improvement ~10% (±2%)

### Reference Data

Check `results/3b_validation_results.json` for detailed metrics.

---

## Reproducing OLMo 3 7B Results

### Expected Behavior

This experiment validates the **automatic stopping criterion**:

| Step | Loss | S-ratio | Lambda | Status |
|------|------|---------|--------|--------|
| 100 | 2.588 | 2.43% | 0.995 | Decreasing |
| 500 | 2.459 | 2.65% | 0.967 | Approaching |
| 1000 | 2.412 | 2.81% | 0.922 | Near target |
| **1500** | **2.408** | **2.93%** | **0.870** | **Equilibrium** |
| 2000 | 2.441 | 2.99% | 0.870 | No change |
| 2800 | 2.435 | 3.14% | 0.870 | No change |

**Key observation:** Lambda stabilizes at 0.870 around step 1500 and does not change through step 2800.

### MLX Training (Apple Silicon)

```bash
python scripts/train_olmo3_7b_fineweb.py \
    --adapter-out adapters/olmo3_7b_scu_repro \
    --steps 2000 \
    --target-s 0.03 \
    --lambda-init 1.0 \
    --lambda-min 0.0001 \
    --lambda-max 2.0 \
    --val-data data/fineweb_edu_1gb_val.jsonl
```

### CUDA Training

```bash
python scripts/train_olmo_cuda.py \
    --model allenai/OLMo-7B \
    --adapter-out adapters/olmo3_7b_scu_cuda \
    --steps 2000 \
    --target-s 0.03
```

### Verification Criteria

- [ ] Lambda starts near 1.0 and decreases
- [ ] Lambda stabilizes (Δλ < 0.01) around step 1500 ±200
- [ ] S-ratio approaches 3% (within 0.5%)
- [ ] Loss at equilibrium < loss at later steps
- [ ] `metadata.json` shows `lambda_range` ≠ [0.0, 0.0]

---

## Reproducing VibeThinker 1.5B Results

### Expected Results

| Model | Method | PPL | Notes |
|-------|--------|-----|-------|
| Base | None | 967 | Math-specialized |
| Baseline | Standard LoRA | 70.27 | Recovered general |
| SCU V3 | Fixed Prior | 70.39 | Comparable + stopped |
| SCU V4 | Dynamic Prior | 108.84 | Overfit (lesson) |

### Steps

```bash
# 1. Run baseline (no SCU)
python scripts/train_vibethinker.py \
    --model WeiboAI/VibeThinker-1.5B \
    --adapter-out adapters/vibethinker_baseline \
    --steps 500 \
    --lambda-min 0.0 \
    --lambda-max 0.0

# 2. Run SCU V3 (fixed prior)
python scripts/train_vibethinker.py \
    --model WeiboAI/VibeThinker-1.5B \
    --adapter-out adapters/vibethinker_v3 \
    --steps 500 \
    --target-s 0.01 \
    --prior-sigma 0.01

# 3. Evaluate
python scripts/eval_ppl.py --adapter adapters/vibethinker_v3
```

### Verification Criteria

- [ ] Baseline PPL ≈ 70 (±2)
- [ ] SCU V3 PPL ≈ 70 (±2)
- [ ] SCU lambda saturated at max (2.0) around step 386
- [ ] Lambda saturation = safety signal (don't continue)

---

## Common Reproduction Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Lambda always 0 | Wrong arguments | Use `--lambda-min 0.0001` not `0.0` |
| Different BPT values | Different data | Use exact FineWeb-Edu subset |
| Much higher loss | Wrong model | Verify model ID matches exactly |
| OOM during training | Batch too large | Reduce batch, increase accum |
| S-ratio always 0 | Missing tokens_per_epoch | Add to config |
| Training fails on Mac | MLX not installed | `pip install mlx mlx-lm` |
| Training fails on CUDA | Wrong torch version | `pip install torch==2.1.0` |

---

## Validation Data Locations

After running experiments, check these files:

| File | Content |
|------|---------|
| `adapters/*/metadata.json` | Final lambda, S-ratio, loss |
| `adapters/*/scu_metrics.json` | Per-step metrics |
| `adapters/*/adapter_config.json` | LoRA configuration |
| `results/3b_validation_results.json` | Llama 3B reference |
| `ablations/*.csv` | Grid search results |

---

## Quick Validation Script

```python
import json

def validate_scu_run(adapter_path):
    """Check if SCU training ran correctly."""
    with open(f"{adapter_path}/metadata.json") as f:
        meta = json.load(f)

    checks = []

    # Check 1: Lambda bounds not [0, 0]
    lambda_range = meta.get("lambda_range", [0, 0])
    checks.append(("Lambda bounds valid", lambda_range != [0.0, 0.0]))

    # Check 2: Final lambda non-zero
    final_lambda = meta.get("final_lambda", 0)
    checks.append(("Final lambda > 0", final_lambda > 0))

    # Check 3: S-ratio computed
    final_s = meta.get("final_s_ratio", 0)
    checks.append(("S-ratio computed", final_s > 0))

    # Check 4: Training completed
    success = meta.get("success", False)
    checks.append(("Training succeeded", success))

    print("SCU Validation Results:")
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")

    return all(passed for _, passed in checks)

# Usage:
# validate_scu_run("adapters/my_adapter")
```

---

## Getting Help

If reproduction fails:

1. **Check versions** - Match exact package versions above
2. **Check data** - Use same dataset and preprocessing
3. **Check args** - Verify lambda bounds are NOT [0, 0]
4. **Check logs** - Look for lambda and S-ratio values
5. **Open issue** - https://github.com/Shannon-Labs/shannon-control-unit/issues

**Contact:** hunter@shannonlabs.dev
