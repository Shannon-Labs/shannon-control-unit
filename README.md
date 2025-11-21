# Shannon Control Unit (SCU): Information-Theoretic Regularization via PI Control

[![Patent Pending](https://img.shields.io/badge/Patent-Pending-orange.svg)](https://shannonlabs.dev)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/hunterbown/shannon-control-unit)
[![License](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)

**Abstract**

Shannon Control Unit (SCU) applies closed-loop control to large-scale language model training. Treating regularization strength (λ) as an actuator and the Minimum Description Length (MDL) information ratio (S) as the controlled variable, SCU uses a proportional-integral (PI) controller to maintain a target (S*) throughout optimization. This feedback stabilizes model complexity without manual hyperparameter sweeps. On Llama 3.2 (1B, 3B) fine-tuning, SCU improves bits-per-token by 6-12% over tuned fixed-λ baselines while preserving training stability.

---

## 1. Problem Statement

Conventional regularization (weight decay, dropout) is scheduled open-loop. The effective tendency to overfit varies over the course of training, so static or hand-tuned schedules either under-penalize (memorization) or over-penalize (underfitting). A feedback mechanism that measures the model’s instantaneous information balance and adjusts $\lambda$ accordingly is required.

## 2. Methodology

SCU couples information theory with PI control. We monitor the MDL-derived information ratio:

```
S(t) = ParamBPT(t) / (DataBPT(t) + ParamBPT(t))
```

where **DataBPT** is the bits-per-token of the loss and **ParamBPT** is the bits-per-token of the parameter update. The control objective is `S(t) → S*`. Let `e(t) = S(t) - S*`. With plant gain `∂S/∂λ < 0`, the PI law updates the regularization strength as:

```
λ(t+1) = λ(t) * exp(-(K_p × e(t) + K_i × Σ e(τ)))
```

optionally with deadband and integral clamping for anti-windup. Updates are applied at gradient-accumulation boundaries to maintain stability.

### Control System Highlights
- **Closed-loop, not scheduled:** PI controller keeps S near the target S*; λ adapts continually instead of following a fixed decay or manual grid.
- **Stability primitives:** Deadband to avoid chatter, integral clamp/leak for anti-windup, and λ min/max guards to prevent runaway behavior.
- **Observation + actuation:** Telemetry exposes S/DataBPT/ParamBPT; λ updates gate at accumulation boundaries to avoid noise amplification.
- **Hardware-aware path:** CUDA 4-bit + Unsloth fast path when requested; MPS and CPU fall back safely with sane defaults.
- **Bootstrapping:** Auto-config seeds controller gains/targets from model and dataset scale so feedback starts stable on first step.

### Control Loop (text diagram)

```
Data → Tokenize/Batch → Model (plant)
          │                 │
          │          Loss → DataBPT, ParamBPT → S = ParamBPT / (DataBPT + ParamBPT)
          │                 │                           │
          └─────────────── Feedback (e = S - S*) ───────┘
                                 │
                          PI Controller (Kp, Ki, deadband, clamp)
                                 │
                            λ(t+1) actuator (regularization weight)
                                 │
                          Applied at grad-accum boundaries
```


## 3. Results

We validated SCU by fine-tuning Llama 3.2 models on a subset of WikiText-103. The results show significant improvements in compression efficiency (Bits Per Token) and Perplexity compared to an optimally tuned cross-entropy baseline.

| Model | Metric | Baseline (Cross-Entropy) | SCU (PI Control) | Improvement |
|-------|--------|--------------------------|------------------|-------------|
| **Llama-3.2-1B** | BPT | 3.920 | **3.676** | **-6.2%** |
| | Perplexity | 15.14 | **12.78** | **-15.6%** |
| **Llama-3.2-3B** | BPT | 1.830 | **1.635** | **-10.6%** |
| | Perplexity | 3.56 | **3.11** | **-12.6%** |

*Note: Validation performed on Llama 3.2 LoRA adapters. Baseline represents the best-performing fixed-λ configuration found via grid search.*

### Key Finding: Scaling Behavior and Data Requirements

Our VibeThinker 1.5B experiments demonstrate SCU's principled response to varying dataset scales. The system exhibits clear quantitative relationships between dataset size, parameter complexity, and the information ratio S.

#### Observed Scaling Behavior

**Configuration 1: Limited Data**
- Model: 1.5B parameters (18M trainable LoRA parameters)
- Dataset: 2MB text corpus (530k tokens)
- Measured metrics:
  - ParamBPT: 14.2 bits/token
  - DataBPT: 8.0 bits/token
  - S ratio: 64% (target: 1%)
  - Controller response: λ saturated at maximum (2.0)

**Configuration 2: Extended Data**
- Dataset: HuggingFaceFW/finewiki, 100MB text corpus (26M tokens)
- Measured metrics:
  - ParamBPT: 0.287 bits/token (50× reduction from Configuration 1)
  - S ratio: 4.1%
  - Controller response: λ in active regulation region

**Configuration 3: Extended Data with Normalization**
- Same data as Configuration 2
- Applied tokens-per-epoch normalization (100M tokens)
- Results: S ratio 1.5% at target range

#### Interpretation

The observed scaling follows the expected mathematical relationship:
```
ParamBPT ∝ 1 / tokens_per_epoch
```

With insufficient data (Configuration 1), ParamBPT dominates the information budget, resulting in elevated S ratios. SCU responds by maximizing regularization strength (λ saturation) to constrain model complexity.

With adequate data scaling (Configuration 2), ParamBPT decreases proportionally, bringing S into a regulable region. The controller operates within its designed range.

Configuration 3 demonstrates that parameter complexity normalization (via `tokens_per_epoch_override`) enables fine-tuning on datasets where the natural S ratio would exceed target, providing a practical mechanism for domain-specific adaptation.

#### Scaling Guidelines

For target S=1% with prior σ=0.01:
- Required: ~10 tokens per trainable parameter
- Implications:
  - 18M LoRA parameters: ~180M tokens (~720MB text) for natural convergence to 1% S
  - 100M full parameters: ~1B tokens (~4GB text) for natural convergence to 1% S

| Configuration | Dataset | Tokens | Measured S | Normalized S | Lambda State |
|---------------|---------|--------|------------|--------------|--------------|
| 1 | 2MB | 530k | 64% | - | Saturated |
| 2 | 100MB | 26M | 4.1% | - | Active regulation |
| 3 | 100MB | 26M | 4.1% | 1.5% | Active regulation |

These results validate SCU's ability to quantitatively assess model-data scaling relationships and respond appropriately through adaptive regularization.

## 4. Related & Concurrent Work

The application of control theory to LLM training is an emerging and promising field.

### 4.1 Independent Convergence: EntroPIC
Recent independent work, **EntroPIC** (arXiv:2511.15248), applies PI control to stabilize policy entropy in reinforcement learning. While EntroPIC regulates policy entropy in RL, SCU regulates the information ratio in supervised learning. Both validate the necessity of feedback control for neural training dynamics.

## 5. Future Directions

Our ongoing research focuses on:

- **Scaling Laws for S***: Deriving the optimal target S* from first principles based on model size (N) and dataset size (D), removing the need for a target setpoint entirely.
- **Full-Parameter Training**: Extending validation beyond LoRA to full model pretraining.
- **Unified Control**: Investigating if regulating Information Ratio implicitly stabilizes entropy (potentially unifying SCU and EntroPIC).

## 6. Usage

### Installation
```bash
git clone https://github.com/Shannon-Labs/shannon-control-unit.git
cd shannon-control-unit
pip install -r requirements.txt
```

### Quick Start (Inference)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load Base Model
base_id = "meta-llama/Llama-3.2-3B"
base = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto", torch_dtype=torch.float16)

# Load SCU Adapter
model = PeftModel.from_pretrained(base, "hunterbown/shannon-control-unit", subfolder="3b-scu")
```

For reproduction scripts and training details, see [`examples/`](./examples/) and [`scripts/`](./scripts/).

### Training API + CLI (MVP)
```bash
# Install with server + CLI extras
pip install -e .[dev,server]

# Launch API server (FastAPI + SQLite-backed job queue)
python -m scu_api.server

# Health + auto-config
scu health
scu auto-config --model-id gpt2 --train-data data/train.txt

# Submit training via CLI (wait and download artifacts)
scu train --base-model sshleifer/tiny-gpt2 --train-data data/train.txt --steps 5 --wait
scu status <job-id>
scu jobs
scu download <job-id> --output adapters/
```

Notes:
- The job queue persists to SQLite (`jobs.db`) and exposes `/jobs`, `/jobs/{id}`, and `/jobs/{id}/adapter`.
- CLI auto-config merges suggested settings with your overrides; defaults map to `TrainingConfig`.
- `SCUClient` is exported at package root for SDK usage if you prefer Python over the CLI.

## 7. Citation

If you use SCU in your research, please cite:

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

## 8. License

This repository is dual-licensed:

*   **Research & Open Source:** [AGPL-3.0](LICENSE). Free for academic and open-source use.
*   **Commercial:** Proprietary licenses available for closed-source applications. Contact `hunter@shannonlabs.dev`.

**Intellectual Property:** The SCU methodology is subject to a U.S. Provisional Patent (Filed September 2025).

**Technical Paper:** See [SCU_Technical_Report_v1.pdf](./SCU_Technical_Report_v1.pdf) for detailed methodology and evaluation.

---

## Appendix: Control Math (tl;dr)

- **S-ratio (controlled variable):**
  
  \(S = \frac{\text{ParamBPT}}{\text{DataBPT} + \text{ParamBPT}}\); reduces to the information ratio between parameter update cost and data fit.
- **Error:** \(e(t) = S(t) - S^*\) with plant gain \(\partial S / \partial \lambda < 0\).
- **PI update with deadband and clamp:**
  
  \(e_d = 0\) if \(|e| < \delta\) (deadband), else \(e_d = e\).
  
  \(I_{t+1} = \text{clip}(I_t + e_d, I_{\min}, I_{\max})\).
  
  \(\lambda_{t+1} = \text{clip}\big( \lambda_t \cdot \exp(-(K_p e_d + K_i I_{t+1})), \lambda_{\min}, \lambda_{\max} \big)\).
- **Actuation point:** apply λ update at gradient-accumulation boundaries to avoid per-microbatch noise.
- **Stability aids:** deadband to prevent chatter; integral clamp/leak to avoid windup; λ bounds to avoid runaway; controller gains set by auto-config from model/dataset scale.

---

**Author:** Hunter Bown  
**Contact:** hunter@shannonlabs.dev  
**Website:** https://shannonlabs.dev
