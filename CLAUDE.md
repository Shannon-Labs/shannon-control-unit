# CLAUDE.md - Shannon Control Unit

## What is SCU?

Shannon Control Unit (SCU) is a control-theoretic framework for **adaptive regularization** during LLM fine-tuning. It uses a PI (Proportional-Integral) controller to maintain an optimal **Information Ratio (S)** in real-time. The key discovery is that when the PI controller's lambda stabilizes at equilibrium, the model has reached **MDL saturation** - it has learned all meaningful patterns from the data. This transforms SCU from "adaptive regularization" into **"self-terminating training"** that knows when to stop.

---

## Key Concepts

### Information Ratio (S)
```
S = ParamBPT / (DataBPT + ParamBPT)
```
- **DataBPT**: Cross-entropy loss in bits-per-token (how well model fits data)
- **ParamBPT**: Parameter regularization in bits-per-token (model complexity)
- **S**: Ratio of complexity to total information (target: 2-3% for LoRA)

### PI Control Law
```
error = S_measured - S_target
lambda = lambda * exp(Kp * error + Ki * integral)
```
The controller adjusts regularization strength (λ) based on how far S is from target S*.

### MDL Saturation Signal
When λ stops changing (stabilizes at equilibrium) and S is near target → **stop training**.

---

## Critical Files

| File | Purpose |
|------|---------|
| `shannon_control/control.py` | Core PI controller: `update_lambda()`, `calculate_s_ratio()` |
| `shannon_control/mlx/callback.py` | MLX training callback with SCU integration |
| `scu_api/service/smart_config.py` | Auto-config with `CONFIG_SCALES` table |
| `configs/default.yaml` | Default hyperparameters template |
| `scripts/train_olmo3_7b_fineweb.py` | OLMo 7B training script |
| `scripts/eval_quality.py` | Model evaluation script |

---

## Common Tasks

### 1. Training a New Model

```bash
python scripts/train_olmo3_7b_fineweb.py \
    --adapter-out adapters/my_adapter \
    --steps 1500 \
    --target-s 0.03 \
    --lambda-init 1.0 \
    --lambda-min 0.0001 \
    --lambda-max 2.0
```

Key flags:
- `--target-s 0.03`: Target S-ratio (3% recommended for 7B)
- `--lambda-init 1.0`: Starting regularization strength
- `--lambda-min/max`: Bounds for PI controller

### 2. Choosing Target S* by Model Size

| Model Size | Recommended S* | LoRA Rank |
|------------|---------------|-----------|
| 1B | 1-2% | r=8-16 |
| 3B | 2-3% | r=16 |
| 7B | 3% | r=16-32 |

### 3. Monitoring Training

Watch console output for:
```
[SCU] Step 1500: S=0.0293 (target=0.0300), lambda=0.870, loss=2.408
```

**Stop signals:**
- Lambda stable (Δλ < 0.001 over 100+ steps)
- S-ratio near target (within 5%)
- Loss plateaued or increasing

### 4. Evaluating Results

```bash
python scripts/eval_quality.py
```

Check `adapters/*/metadata.json` for:
- `final_lambda`: Should be non-zero, within bounds
- `final_s_ratio`: Should be near target
- `lambda_range`: Should NOT be [0.0, 0.0]

---

## Pitfalls to Avoid

### CRITICAL: weight_decay MUST be 0
SCU provides its own regularization via λ. Using weight_decay doubles regularization and breaks the control loop.

### CRITICAL: tokens_per_epoch Required
ParamBPT calculation needs to know total tokens. Without this, S-ratio is wrong.

### Lambda bounds of [0, 0] = Controller Disabled
If you see `lambda_range: [0.0, 0.0]` in metadata, the PI controller was not active. Check command-line args.

### prior_sigma Affects S-ratio Scale
Default σ=0.01 is calibrated for typical LoRA weights. Changing this changes the S-ratio magnitude.

---

## Quick Reference: CONFIG_SCALES Table

```python
CONFIG_SCALES = {
    "micro":  {"s_target": 0.005, "kp": 0.6, "ki": 0.12},  # <100M params
    "tiny":   {"s_target": 0.01,  "kp": 0.7, "ki": 0.14},  # 100M-500M
    "small":  {"s_target": 0.015, "kp": 0.8, "ki": 0.15},  # 500M-1B
    "medium": {"s_target": 0.02,  "kp": 0.8, "ki": 0.15},  # 1B-3B
    "large":  {"s_target": 0.03,  "kp": 0.8, "ki": 0.15},  # 3B-7B
    "xlarge": {"s_target": 0.04,  "kp": 0.9, "ki": 0.18},  # 7B+
}
```

---

## Repository Structure

```
shannon-control-unit/
├── shannon_control/           # Core control implementation
│   ├── control.py            # PI controller logic
│   ├── mlx/                  # MLX (Apple Silicon) backend
│   └── core/                 # Alternative controller variants
├── scu_api/                  # Training API + CLI
│   └── service/smart_config.py  # Auto-configuration
├── scripts/                  # Training and evaluation scripts
├── configs/                  # YAML configuration templates
├── adapters/                 # Trained adapter outputs
├── docs/                     # Technical documentation
│   ├── technical/           # Theory, math proofs
│   └── guides/              # Practical how-to guides
└── examples/                # Usage examples
```

---

## Key Results

### OLMo 3 7B (Dec 2024)
- **Training**: FineWeb-Edu 98M tokens, LoRA r=16
- **Discovery**: Lambda stabilized at 0.870 @ step 1500
- **Evidence**: Loss at step 1500 (2.408) < loss at step 2800 (2.435)
- **Conclusion**: PI equilibrium = optimal stopping point

### Llama 3.2 (Earlier)
- 1B: 6.2% BPT improvement over baseline
- 3B: 10.6% BPT improvement over baseline

---

## When Helping with SCU

1. **Check training logs** for lambda and S-ratio values
2. **Verify lambda is moving** - if stuck at 0 or bounds, something is wrong
3. **Look for equilibrium** - lambda stable + S near target = done
4. **Don't add weight_decay** - SCU is the regularization
5. **Use CONFIG_SCALES** for starting hyperparameters

---

**Author:** Hunter Bown
**Contact:** hunter@shannonlabs.dev
**Repo:** https://github.com/Shannon-Labs/shannon-control-unit
