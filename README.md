# Shannon Control Unit (SCU)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-VibeThinker_SCU-yellow)](https://huggingface.co/hunterbown/VibeThinker-1.5B-SCU)
[![License](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)

**SCU is a drop-in controller that automatically tunes regularization during LLM fine-tuning — and tells you when to stop.**

It watches an information-theoretic ratio in real time, adjusts regularization via a PI (Proportional-Integral) feedback loop, and stops training the moment the model has learned everything useful from your data (MDL saturation). No manual early-stopping, no guessing at step counts.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Training Your Own Model](#training-your-own-model)
- [Choosing Hyperparameters](#choosing-hyperparameters)
- [Reading the Training Logs](#reading-the-training-logs)
- [Results](#results)
- [Common Pitfalls](#common-pitfalls)
- [Training API & CLI](#training-api--cli)
- [Project Structure](#project-structure)
- [Related Work](#related-work)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

---

## How It Works

SCU maintains a target **Information Ratio (S)** during training. Think of S as "what fraction of the model's total information budget is spent on parameter complexity vs. fitting the data."

```
S = ParamBPT / (DataBPT + ParamBPT)
```

| Term | What it measures |
|------|-----------------|
| **DataBPT** | Cross-entropy loss in bits-per-token (how well the model fits the data) |
| **ParamBPT** | Parameter complexity in bits-per-token (how complex the model has become) |
| **S** | Ratio of complexity to total information (typical target: 2-3% for LoRA) |

A PI controller watches S and adjusts the regularization strength (lambda) to keep it near a target S*:

```
error = S_measured - S_target
lambda = lambda * exp(Kp * error + Ki * integral)
```

- S too high? The controller increases regularization to push it down.
- S too low? The controller eases off to let the model learn.
- Lambda stops changing? **Training is done.** The model has reached MDL saturation.

```
Data --> Tokenize/Batch --> Model
                             |
                      Loss --> DataBPT, ParamBPT --> S = ParamBPT / (DataBPT + ParamBPT)
                             |                           |
                             +--- Feedback (e = S - S*) -+
                                        |
                                 PI Controller (Kp, Ki, deadband, clamp)
                                        |
                                   lambda(t+1) update
                                        |
                                 Applied at grad-accum boundaries
```

---

## Quick Start

### Use a Pre-trained SCU Adapter (Inference)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_id = "meta-llama/Llama-3.2-3B"
base = AutoModelForCausalLM.from_pretrained(
    base_id, device_map="auto", torch_dtype=torch.float16
)

# Load SCU-trained adapter
model = PeftModel.from_pretrained(
    base, "hunterbown/shannon-control-unit", subfolder="3b-scu"
)
tokenizer = AutoTokenizer.from_pretrained(base_id)

# Generate
inputs = tokenizer("The key insight is", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Installation

```bash
git clone https://github.com/Shannon-Labs/shannon-control-unit.git
cd shannon-control-unit
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+. On Apple Silicon, MLX is supported natively. On Linux/Windows, CUDA is recommended.

For the API and CLI tools:

```bash
pip install -e .[dev,server]
```

---

## Training Your Own Model

### MLX (Apple Silicon)

```bash
python scripts/train_olmo3_7b_fineweb.py \
    --adapter-out adapters/my_adapter \
    --steps 1500 \
    --target-s 0.03 \
    --lambda-init 1.0 \
    --lambda-min 0.0001 \
    --lambda-max 2.0
```

### Using the SCU Controller in Your Own Training Loop

```python
from shannon_control.control import (
    update_lambda, calculate_param_bpt,
    calculate_data_bpt, calculate_s_ratio
)

# Initialize
lmbda = 1.0       # Starting regularization strength
I = 0.0            # Integral term
S_hat = None       # EMA of S (auto-initialized)
S_target = 0.03    # Target S-ratio (3% for 7B models)

# Inside your training loop, at each gradient accumulation boundary:
data_bpt = calculate_data_bpt(loss.item())
param_bpt = calculate_param_bpt(model, tokens_per_epoch=98_000_000)
s_ratio = calculate_s_ratio(data_bpt, param_bpt)

lmbda, I, S_hat = update_lambda(
    lmbda, s_ratio, S_target, I,
    Kp=0.8, Ki=0.15,
    lmin=0.0001, lmax=2.0,
    S_hat=S_hat
)

# Apply lambda as your regularization weight
reg_loss = lmbda * param_bpt
total_loss = loss + reg_loss
```

### Automatic Stopping

```python
def should_stop_training(lambda_history, s_ratio, target_s, window=100):
    """Stop when the PI controller reaches stable equilibrium."""
    lambda_stable = abs(lambda_history[-1] - lambda_history[-window]) < 0.001
    s_near_target = abs(s_ratio - target_s) / target_s < 0.05  # 5% tolerance
    return lambda_stable and s_near_target
```

When `should_stop_training()` returns `True`, the model has learned everything meaningful from the data. Continuing past this point adds noise without improving performance.

---

## Choosing Hyperparameters

### Target S* by Model Size

| Model Size | Recommended S* | LoRA Rank | Notes |
|------------|---------------|-----------|-------|
| < 100M | 0.5% | r=8 | Micro models, very tight regularization |
| 100M-500M | 1% | r=8-16 | Small models |
| 500M-1B | 1.5% | r=8-16 | |
| 1B-3B | 2% | r=16 | |
| 3B-7B | 3% | r=16-32 | Best validated range |
| 7B+ | 4% | r=32 | |

### Controller Gains (Auto-Config)

SCU ships with a pre-tuned table. You usually don't need to change these:

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

Use `scu auto-config --model-id <your-model>` to get recommended settings automatically.

### Key Training Flags

| Flag | Default | What it does |
|------|---------|-------------|
| `--target-s` | 0.03 | Target S-ratio for the PI controller |
| `--lambda-init` | 1.0 | Starting regularization strength |
| `--lambda-min` | 0.0001 | Lower bound for lambda (prevents zero regularization) |
| `--lambda-max` | 2.0 | Upper bound for lambda |
| `--steps` | 1500 | Max training steps (SCU may stop earlier) |
| `--adapter-out` | `adapters/` | Where to save the trained adapter |

---

## Reading the Training Logs

During training, SCU prints a status line at each logging step:

```
[SCU] Step 1500: S=0.0293 (target=0.0300), lambda=0.870, loss=2.408
```

### What to look for

| Signal | What it means |
|--------|--------------|
| Lambda decreasing steadily | Model is learning, S approaching target |
| Lambda stable (no change for 100+ steps) | **MDL saturation reached — stop training** |
| S-ratio within 5% of target | Controller is doing its job |
| S-ratio far from target, lambda at bounds | Check your hyperparameters |
| Lambda stuck at 0 or at min/max | Controller may be disabled — check args |

### After Training

Check `adapters/*/metadata.json` for:
- `final_lambda` — should be non-zero, within your bounds
- `final_s_ratio` — should be near your target
- `lambda_range` — should NOT be `[0.0, 0.0]` (that means the controller was inactive)

---

## Results

### OLMo 3 7B (Dec 2025)

Trained a LoRA adapter for OLMo 3 7B Instruct (4-bit MLX) on FineWeb-Edu (98M tokens).

| Step | Loss | S-ratio | Lambda | Status |
|------|------|---------|--------|--------|
| 100 | 2.588 | 2.43% | 0.995 | Lambda decreasing |
| 500 | 2.459 | 2.65% | 0.967 | Approaching target |
| 1000 | 2.412 | 2.81% | 0.922 | Near equilibrium |
| **1500** | **2.408** | **2.93%** | **0.870** | **Lambda stable** |
| 2000 | 2.441 | 2.99% | 0.870 | No change |
| 2800 | 2.435 | 3.14% | 0.870 | No change |

Lambda stabilized at 0.870 around step 1500 and did not change through step 2800. Loss at step 1500 (2.408) was *lower* than loss at step 2800 (2.435) — training beyond the saturation point added noise.

### VibeThinker 1.5B

| Variant | Method | Validation PPL |
|---------|--------|---------------|
| Baseline | Standard fine-tuning (no regularization) | 70.27 |
| SCU V3 | Fixed prior, PI control | **70.39** (matched baseline) |
| SCU V4 | Dynamic prior (loosened) | 108.84 (overfit) |

SCU V3 matched the baseline while also signaling when to stop. When V4 loosened the regularization, it overfit — confirming that V3's lambda saturation was a meaningful signal, not a limitation.

### Llama 3.2

| Model | Baseline BPT | SCU BPT | Improvement |
|-------|-------------|---------|-------------|
| Llama-3.2-1B | 3.920 | 3.676 | **-6.2%** |
| Llama-3.2-3B | 1.830 | 1.635 | **-10.6%** |

---

## Common Pitfalls

### weight_decay must be 0

SCU provides its own regularization via lambda. Setting `weight_decay > 0` doubles the regularization and breaks the control loop. Always use `weight_decay=0` in your optimizer.

### tokens_per_epoch is required

The ParamBPT calculation normalizes by total tokens. Without this value, the S-ratio will be wrong and the controller will misbehave.

### Lambda bounds of [0, 0] means the controller is disabled

If you see `lambda_range: [0.0, 0.0]` in your metadata, the PI controller was not active. Check your `--lambda-min` and `--lambda-max` arguments.

### prior_sigma affects S-ratio scale

The default `sigma=0.01` is calibrated for typical LoRA weights. Changing this changes the magnitude of ParamBPT and therefore S. Only change it if you understand the implications.

---

## Training API & CLI

SCU includes a FastAPI server and CLI for managing training jobs:

```bash
# Launch the API server
python -m scu_api.server

# Check system health
scu health

# Get auto-configured settings for your model
scu auto-config --model-id gpt2 --train-data data/train.txt

# Submit a training job
scu train --base-model sshleifer/tiny-gpt2 \
          --train-data data/train.txt \
          --steps 5 --wait

# Check job status
scu status <job-id>
scu jobs

# Download trained adapter
scu download <job-id> --output adapters/
```

The job queue persists to SQLite (`jobs.db`). For programmatic use, `SCUClient` is exported at the package root.

---

## Project Structure

```
shannon-control-unit/
├── shannon_control/           # Core controller
│   ├── control.py            # PI controller: update_lambda(), calculate_s_ratio()
│   ├── mlx/                  # Apple Silicon (MLX) backend
│   │   └── callback.py       # MLX training callback with SCU
│   └── core/                 # Controller variants
├── scu_api/                  # Training API & CLI
│   └── service/
│       └── smart_config.py   # Auto-configuration (CONFIG_SCALES table)
├── scripts/
│   ├── train_olmo3_7b_fineweb.py  # OLMo 7B training script
│   └── eval_quality.py            # Model evaluation
├── configs/                  # YAML config templates
│   ├── default.yaml
│   └── olmo3_7b.yaml
├── docs/
│   ├── technical/           # Theory, proofs, math
│   └── guides/              # CUDA guide, deployment, hyperparameter tuning
├── examples/                # Usage examples
├── adapters/                # Trained adapter outputs
└── experiments/             # Training logs and results
```

---

## Related Work

**EntroPIC** (arXiv:2511.15248) independently applies PI control to stabilize policy entropy in reinforcement learning. SCU regulates the information ratio in supervised fine-tuning. Both validate the value of feedback control for neural training dynamics.

---

## Limitations

- Validated up to 7B parameters with LoRA fine-tuning only (not full-parameter training yet)
- No direct measurement of specialized capability preservation (e.g., math benchmarks after general fine-tuning)
- Optimal S* currently requires empirical selection (auto-config helps but isn't proven universal)
- Not yet compared against DoRA, QLoRA, or other modern PEFT methods with careful tuning

---

## Citation

```bibtex
@misc{bown2025scu,
  author = {Bown, Hunter},
  title = {Shannon Control Unit: Information-Theoretic Regularization via PI Control},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Shannon-Labs/shannon-control-unit}}
}
```

---

## License

Dual-licensed:

- **Research & Open Source:** [AGPL-3.0](LICENSE) — free for academic and open-source use
- **Commercial:** Proprietary licenses available for closed-source applications — contact [hunter@shannonlabs.dev](mailto:hunter@shannonlabs.dev)

---

**Author:** Hunter Bown
**Website:** [shannonlabs.dev](https://shannonlabs.dev)
**Contact:** hunter@shannonlabs.dev
