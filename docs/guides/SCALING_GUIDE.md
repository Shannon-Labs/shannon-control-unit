# SCU Scaling Guide: Training on New Models

This guide walks through training SCU on new model architectures and sizes.

---

## Step 1: Determine Model Size Category

| Model Size | Category | Examples |
|------------|----------|----------|
| < 100M | micro | DistilGPT2, TinyBERT |
| 100M - 500M | tiny | GPT-2 Small, BERT-base |
| 500M - 1B | small | GPT-2 Medium, Llama-1B |
| 1B - 3B | medium | Llama-3.2-1B, Phi-2 |
| 3B - 7B | large | Llama-3.2-3B, OLMo-7B |
| 7B+ | xlarge | Llama-8B, Mistral-7B+ |

---

## Step 2: Select Target S*

The target S-ratio (S*) is the information ratio you want the controller to maintain.

### By Model Size

| Category | Recommended S* | Rationale |
|----------|---------------|-----------|
| micro | 0.5% | Very small models need tight regularization |
| tiny | 1.0% | Balance for small parameter counts |
| small | 1.5% | Standard for sub-1B models |
| medium | 2.0% | Validated on VibeThinker 1.5B |
| large | 3.0% | Validated on Llama-3.2-3B, OLMo-7B |
| xlarge | 4.0% | Larger models tolerate more complexity |

### Adjustment Factors

**Dataset size adjustments:**
- Large dataset (>100M tokens): Use standard S*
- Small dataset (<10M tokens): Reduce S* by 0.5%
- Very small dataset (<1M tokens): Reduce S* by 1.0%

**Task-specific adjustments:**
- General text: Standard S*
- Code: Increase S* by 0.5% (more structure needed)
- Math: Increase S* by 0.5%
- Domain-specific: Reduce S* by 0.5% (prevent overfitting)

---

## Step 3: Configure LoRA Parameters

### Rank Selection by Model Size

| Model Size | LoRA Rank (r) | LoRA Alpha (α) | Trainable Params |
|------------|---------------|----------------|------------------|
| < 1B | 8 | 16 | ~5-10M |
| 1B - 3B | 16 | 32 | ~20-40M |
| 3B - 7B | 16-32 | 32-64 | ~40-80M |
| 7B+ | 32 | 64 | ~80-120M |

### Target Modules

For transformer architectures, target all attention and MLP layers:
```yaml
lora_targets:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

---

## Step 4: Set PI Controller Parameters

### Default Values (Recommended)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Kp | 0.8 | Proportional gain (responsiveness) |
| Ki | 0.15 | Integral gain (steady-state correction) |
| deadband | 0.003 | No-update zone (±0.3% around S*) |
| lambda_init | 1.0 | Starting regularization strength |
| lambda_min | 0.0001 | Lower bound for λ |
| lambda_max | 2.0 | Upper bound for λ |

### When to Adjust

**If lambda oscillates wildly:** Reduce Kp to 0.5-0.6
**If convergence is slow:** Increase Kp to 0.9-1.0
**If S-ratio overshoots target:** Increase Ki to 0.2
**If lambda saturates immediately:** Widen lambda bounds or reduce S*

---

## Step 5: Hardware Configuration

### Memory Estimation

| Model Size | 4-bit Quantized | LoRA r=16 | Training Overhead | Total |
|------------|-----------------|-----------|-------------------|-------|
| 1B | ~0.5 GB | ~20 MB | ~1 GB | ~2 GB |
| 3B | ~1.5 GB | ~40 MB | ~2 GB | ~4 GB |
| 7B | ~3.5 GB | ~80 MB | ~4 GB | ~8 GB |
| 13B | ~6.5 GB | ~160 MB | ~8 GB | ~16 GB |

### Batch Size Guidelines

| Available VRAM | Recommended Batch Size | Gradient Accumulation |
|----------------|------------------------|----------------------|
| 8 GB | 1 | 16 |
| 16 GB | 2 | 8 |
| 24 GB | 4 | 4 |
| 40 GB+ | 8 | 2 |

---

## Step 6: Create Configuration

### YAML Template

```yaml
# configs/my_model_scu.yaml
model:
  base_model: "your-model-id"
  architecture: "transformer"

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  targets:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

scu:
  target_s: 0.03
  kp: 0.8
  ki: 0.15
  deadband: 0.003
  lambda_init: 1.0
  lambda_min: 0.0001
  lambda_max: 2.0
  prior_sigma: 0.01

training:
  learning_rate: 2e-5
  batch_size: 1
  gradient_accumulation: 16
  block_size: 2048
  steps: 1500
  warmup_steps: 150
  weight_decay: 0.0  # CRITICAL: Must be 0!

data:
  train_data: "path/to/train.jsonl"
  val_data: "path/to/val.jsonl"
  tokens_per_epoch: 98304000  # Required for ParamBPT calculation
```

---

## Step 7: Run Training

### MLX (Apple Silicon)

```bash
python scripts/train_olmo3_7b_fineweb.py \
    --adapter-out adapters/my_adapter \
    --steps 1500 \
    --target-s 0.03 \
    --lambda-init 1.0 \
    --lambda-min 0.0001 \
    --lambda-max 2.0 \
    --val-data data/val.jsonl
```

### CUDA

```bash
python scripts/train_scu.py \
    --config configs/my_model_scu.yaml \
    --output adapters/my_adapter
```

---

## Step 8: Monitor and Validate

### What to Watch During Training

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| Lambda | Varies within bounds | Stuck at 0 or bounds |
| S-ratio | Approaches target | Stays far from target |
| Loss | Decreasing, then plateaus | Increasing after initial drop |
| Integral | Non-zero when S ≠ S* | Always 0 |

### Stopping Signals

**Stop when ALL of these are true:**
1. Lambda stable (Δλ < 0.001 over 100+ steps)
2. S-ratio within 5% of target
3. Loss plateaued or slightly increasing

### Post-Training Validation

```bash
python scripts/eval_quality.py --adapter adapters/my_adapter
```

Check `adapters/my_adapter/metadata.json`:
- `final_lambda` should be non-zero
- `lambda_range` should NOT be [0.0, 0.0]
- `final_s_ratio` should be near target

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Lambda always 0.0 | Lambda bounds [0, 0] | Set proper bounds |
| S-ratio stuck at 0% | tokens_per_epoch missing | Add to config |
| Lambda oscillates | Kp too high | Reduce to 0.5-0.6 |
| Lambda saturates immediately | S* too aggressive | Increase target_s |
| Loss increases after equilibrium | Overtraining | Stop at equilibrium |
| OOM errors | Batch too large | Reduce batch, increase accum |

---

## Adding to Smart Config

To add your model to the auto-config system, update `scu_api/service/smart_config.py`:

```python
CONFIG_SCALES = {
    # ... existing entries ...
    "my_model": {"s_target": 0.03, "kp": 0.8, "ki": 0.15},
}
```

---

## Quick Reference: Validated Configurations

| Model | S* | LoRA r | Steps | Result |
|-------|-----|--------|-------|--------|
| Llama-3.2-1B | 1.0% | 16 | 500 | -6.2% BPT |
| Llama-3.2-3B | 3.0% | 16 | 500 | -10.6% BPT |
| VibeThinker-1.5B | 1.0% | 16 | 500 | PPL 70.39 |
| OLMo-3-7B | 3.0% | 16 | 1500 | Lambda equilibrium @ step 1500 |

---

**Questions?** Contact hunter@shannonlabs.dev
