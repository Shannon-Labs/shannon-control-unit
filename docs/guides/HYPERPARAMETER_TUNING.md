# SCU Hyperparameter Tuning Guide

Decision trees and guidelines for selecting SCU hyperparameters.

---

## Quick Reference Table

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **target_s** | 0.03 | 0.005 - 0.05 | Target S-ratio (3% for 3B-7B) |
| **Kp** | 0.8 | 0.5 - 1.2 | Proportional gain |
| **Ki** | 0.15 | 0.1 - 0.25 | Integral gain |
| **deadband** | 0.003 | 0.001 - 0.01 | No-update zone around S* |
| **lambda_init** | 1.0 | 0.1 - 2.0 | Starting regularization |
| **lambda_min** | 0.0001 | 0.00001 - 0.01 | Lower bound |
| **lambda_max** | 2.0 | 1.0 - 5.0 | Upper bound |
| **prior_sigma** | 0.01 | 0.001 - 0.1 | Prior std for ParamBPT |

---

## Decision Tree: Target S*

```
START: What is your model size?

├─► < 1B parameters
│   └─► Use S* = 1.0% - 1.5%
│
├─► 1B - 3B parameters
│   └─► Use S* = 2.0%
│
├─► 3B - 7B parameters
│   └─► Use S* = 3.0%
│
└─► > 7B parameters
    └─► Use S* = 4.0%

THEN: Adjust for dataset size

├─► Dataset < 10M tokens
│   └─► Reduce S* by 0.5%
│
├─► Dataset 10M - 100M tokens
│   └─► Keep S* as-is
│
└─► Dataset > 100M tokens
    └─► Keep S* as-is (or increase by 0.5% for very large)

THEN: Adjust for task

├─► General text / conversation
│   └─► Keep S* as-is
│
├─► Code or math
│   └─► Increase S* by 0.5%
│
└─► Domain-specific / specialized
    └─► Reduce S* by 0.5%
```

---

## Decision Tree: Controller Gains (Kp, Ki)

```
SYMPTOM: Lambda oscillating between high and low values

├─► Oscillation amplitude > 50% of range
│   └─► Problem: Kp too high
│       └─► Solution: Reduce Kp to 0.5 - 0.6
│
└─► Oscillation amplitude 20-50%
    └─► Problem: Slight overshoot
        └─► Solution: Reduce Kp to 0.7

---

SYMPTOM: S-ratio converges very slowly (>1000 steps to reach target)

├─► Lambda barely changes
│   └─► Problem: Kp too low
│       └─► Solution: Increase Kp to 0.9 - 1.0
│
└─► Lambda changes but S-ratio doesn't
    └─► Problem: Plant gain issue
        └─► Check: Is weight_decay = 0?

---

SYMPTOM: S-ratio overshoots target, then oscillates around it

├─► Overshoot > 50% of target
│   └─► Problem: Integral windup
│       └─► Solution: Increase Ki to 0.2 - 0.25
│
└─► Steady-state error (S-ratio settles away from target)
    └─► Problem: Ki too low
        └─► Solution: Increase Ki to 0.18 - 0.20
```

---

## Empirical Grid Search Results

From ablation studies on Llama-3.2-3B:

### Kp vs Ki Grid

| Kp \ Ki | 0.10 | 0.15 | 0.20 | 0.25 |
|---------|------|------|------|------|
| **0.5** | Slow | Slow | OK | OK |
| **0.7** | OK | Good | Good | Overshoot |
| **0.8** | Good | **Best** | Good | Overshoot |
| **0.9** | Good | Good | Overshoot | Oscillate |
| **1.0** | OK | Overshoot | Oscillate | Oscillate |

**Best combination:** Kp = 0.8, Ki = 0.15 (default)

### S* vs Final Performance

| S* | Final Loss | Lambda Behavior | Notes |
|----|------------|-----------------|-------|
| 1% | 2.52 | Saturated early | Underfit |
| 2% | 2.44 | Stable at 0.92 | Good |
| **3%** | **2.41** | Stable at 0.87 | **Best** |
| 4% | 2.43 | Stable at 0.78 | Slight overfit |
| 5% | 2.48 | Low bound | Overfit |

---

## Lambda Bounds Selection

```
Default: lambda_min = 0.0001, lambda_max = 2.0

When to widen:
├─► Lambda hits max early and stays there
│   └─► Increase lambda_max to 3.0 - 5.0
│
└─► Lambda hits min and S-ratio is still high
    └─► Decrease lambda_min to 0.00001

When to narrow:
├─► Lambda never exceeds 1.0
│   └─► Reduce lambda_max to 1.5
│
└─► Lambda never drops below 0.5
    └─► Increase lambda_min to 0.1
```

---

## Prior Sigma Selection

The prior sigma (σ) affects how ParamBPT is calculated:

```
ParamBPT = Σ(w²) / (2σ² × N × ln(2))
```

| Sigma | Effect on S-ratio | When to Use |
|-------|-------------------|-------------|
| 0.001 | Very high S-ratio | Very regularized training |
| 0.005 | High S-ratio | Strong regularization |
| **0.01** | **Standard** | **Default for most cases** |
| 0.05 | Low S-ratio | Less regularization |
| 0.1 | Very low S-ratio | Minimal regularization |

**Rule:** If your S-ratio is always near 0%, increase sigma. If always near 100%, decrease sigma.

---

## LoRA Configuration by Model Size

### Rank Selection

```
Model Size → Rank

├─► < 500M params
│   └─► r = 4 - 8
│
├─► 500M - 1B
│   └─► r = 8 - 16
│
├─► 1B - 3B
│   └─► r = 16
│
├─► 3B - 7B
│   └─► r = 16 - 32
│
└─► > 7B
    └─► r = 32 - 64
```

### Alpha Selection

**Rule:** α = 2 × r (default scaling)

| Rank | Alpha | Scale |
|------|-------|-------|
| 8 | 16 | 2.0 |
| 16 | 32 | 2.0 |
| 32 | 64 | 2.0 |

---

## Batch Size and Gradient Accumulation

### Memory-Constrained Selection

```
Available VRAM → Configuration

├─► 8 GB
│   └─► batch_size = 1, grad_accum = 16
│
├─► 16 GB
│   └─► batch_size = 2, grad_accum = 8
│
├─► 24 GB
│   └─► batch_size = 4, grad_accum = 4
│
└─► 40 GB+
    └─► batch_size = 8, grad_accum = 2
```

**Effective batch size = batch_size × grad_accum**

For most training, aim for effective batch size of 16.

---

## Learning Rate by Model Size

| Model Size | Learning Rate | Notes |
|------------|--------------|-------|
| < 1B | 5e-5 | More aggressive |
| 1B - 3B | 2e-5 | Standard |
| 3B - 7B | 2e-5 | Standard |
| 7B+ | 1e-5 | Conservative |

---

## Block Size Selection

| Model Context | Block Size | Notes |
|---------------|------------|-------|
| 2K tokens | 1024 - 2048 | Fit full context |
| 4K tokens | 2048 | Standard |
| 8K+ tokens | 2048 - 4096 | Memory constrained |

---

## Complete Configuration Example

For OLMo-3-7B on FineWeb-Edu:

```yaml
# Based on validated training run (Dec 2024)
model:
  base_model: mlx-community/Olmo-3-7B-Instruct-4bit

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  targets: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

scu:
  target_s: 0.03      # 3% for 7B model
  kp: 0.8             # Default proportional
  ki: 0.15            # Default integral
  deadband: 0.003     # ±0.3% tolerance
  lambda_init: 1.0    # Start at 1.0
  lambda_min: 0.0001  # Lower bound
  lambda_max: 2.0     # Upper bound
  prior_sigma: 0.01   # Standard prior

training:
  learning_rate: 2e-5
  batch_size: 1
  gradient_accumulation: 16
  block_size: 2048
  steps: 1500
  warmup_steps: 150
  weight_decay: 0.0   # CRITICAL!

# Expected results:
# - Lambda stabilizes at ~0.87 around step 1500
# - S-ratio converges to ~2.93%
# - Loss minimum at step 1500 (~2.408)
```

---

## Tuning Workflow

1. **Start with defaults** - Use CONFIG_SCALES for your model size
2. **Run 500 steps** - Observe lambda and S-ratio behavior
3. **If oscillating** - Reduce Kp
4. **If too slow** - Increase Kp
5. **If steady-state error** - Adjust Ki
6. **If S-ratio scale wrong** - Adjust prior_sigma
7. **Full run** - Once stable, run to equilibrium

---

**Questions?** Contact hunter@shannonlabs.dev
