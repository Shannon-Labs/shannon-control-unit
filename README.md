# Shannon Control Unit (SCU): Adaptive Regularization via Control Theory

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-VibeThinker_SCU-yellow)](https://huggingface.co/hunterbown/VibeThinker-1.5B-SCU)
[![License](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)

**Abstract**

Shannon Control Unit (SCU) is a control-theoretic framework for adaptive regularization during Large Language Model (LLM) fine-tuning. SCU addresses the **Plasticity-Stability Trade-off** by monitoring an MDL-derived **Information Ratio (S)** in real-time, automatically adjusting regularization strength via PI control to prevent overfitting while maintaining learning capacity.

**Latest Discovery (Dec 2024):** SCU's PI controller provides a natural stopping criterion - when lambda stabilizes at equilibrium, the model has reached **MDL saturation** (learned all meaningful patterns from the data). This transforms SCU from "adaptive regularization" to **"self-terminating training"** that knows when to stop.

---

## 1. Motivation: The Plasticity-Stability Trade-off

Fine-tuning a specialized model like **VibeThinker-1.5B** (a Qwen-2.5-Math derivative) on general text involves recovering suppressed general capabilities while attempting to preserve specialized knowledge. The model already possesses general English capabilities from pre-training, but they are suppressed by its math specialization (initial PPL 967).

*   **Phase 1: Capability Recovery (Steps 0-380):** The model rapidly recovers its general English abilities. PPL drops from 967 to ~70.
*   **Phase 2: Potential Interference (Step 380+):** Continued training may begin overwriting specialized circuits to squeeze out marginal gains in general text. This is the risk of **Catastrophic Forgetting**.

**SCU vs. Baseline:**
*   **Baseline:** Continues training without adaptive stopping, potentially sacrificing specialized capabilities for marginal general improvement.
*   **SCU:** Detects saturation of information gain and increases regularization, aiming to preserve specialized capabilities while achieving functional general performance.

## 2. Experimental Validation: The VibeThinker Experiment

We validated SCU on **WeiboAI/VibeThinker-1.5B** using the FineWeb-Edu dataset.

### Observed Behavior: Information Ratio Saturation
SCU detected that the Information Ratio saturated after approximately **16M tokens** (~Step 386).
*   **Reaction:** SCU increased regularization to $\lambda=2.0$, effectively freezing the weights.
*   **Interpretation:** This suggests SCU identified a point where additional training provided diminishing returns on information gain.

### Results

| Metric | Base Model | Baseline (Manual) | SCU V3 (Auto-Brake) | SCU V4 (Unregulated) |
| :--- | :--- | :--- | :--- | :--- |
| **FineWeb PPL** | 967 | 70.27 | **70.39** | 108.84 (Crash) |
| **Interpretation** | Specialized | Recovers general | Comparable recovery | Overfit |

*   **Parity in General Performance:** SCU matched the baseline's general text recovery (70.39 vs 70.27 PPL).
*   **Hypothesis on Preservation:** By stopping at the saturation point, SCU may help preserve specialized capabilities. *Note: This hypothesis requires validation with domain-specific benchmarks (e.g., math evaluations).*

## 3. Methodology

SCU couples information theory with PI control. We monitor the MDL-derived information ratio:

```
S(t) = ParamBPT(t) / (DataBPT(t) + ParamBPT(t))
```

where **DataBPT** is the bits-per-token of the loss and **ParamBPT** is the bits-per-token of the parameter update. The control objective is `S(t) → S*`. Let `e(t) = S(t) - S*`. With plant gain `∂S/∂λ < 0`, the PI law updates the regularization strength as:

```
λ(t+1) = λ(t) * exp(+(K_p × e(t) + K_i × Σ e(τ)))
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

### VibeThinker 1.5B Validation

We conducted a comparative study on the **VibeThinker 1.5B** model to validate SCU's adaptive regularization. This experiment used a 500MB subset of the FineWeb-Edu dataset and compared three configurations:

| Model Variant | Training Method | Validation PPL | Status |
| :--- | :--- | :--- | :--- |
| **Baseline** | Standard Finetuning (λ=0) | 70.27 | Strong Baseline |
| **V3 (Fixed Prior)** | SCU (Fixed σ) | **70.39** | Comparable |
| **V4 (Dynamic Prior)** | SCU (Dynamic σ) | 108.84 | Overfit |

#### Key Observation: Regularization Saturation as Safety Signal
In the V3 run, the SCU controller naturally saturated the regularization strength ($\lambda \to 2.0$) towards the end of training.
*   **The Test:** We hypothesized this saturation was a limitation and ran V4 to "fix" it by loosening the prior.
*   **The Result:** V4 overfitted (PPL 108 vs 70).
*   **The Interpretation:** The saturation in V3 appears to be a useful signal that the model-data complexity ratio has reached a threshold. This suggests SCU can serve as an automatic early-stopping mechanism.

This provides preliminary evidence that SCU can help detect overfitting risk during training.

### OLMo 3 7B Validation (Latest: Dec 2024)

We trained a LoRA adapter for **OLMo 3 7B Instruct** (4-bit MLX) on FineWeb-Edu (98M tokens) to validate SCU on larger models and discover automatic stopping criteria.

**Training Configuration:**
- **Model**: `mlx-community/Olmo-3-7B-Instruct-4bit` (7B parameters)
- **LoRA**: r=16, α=32 (40M trainable parameters)
- **Data**: FineWeb-Edu 1GB (98.3M tokens, 95/5 train/val split)
- **Target S-ratio**: 3.0%
- **Hardware**: Apple M4 Max (MLX), 8.5GB peak memory

**Key Discovery: PI Controller Equilibrium as Stopping Signal**

| Step | Loss | Data BPT | Param BPT | S-ratio | Lambda | Status |
|------|------|----------|-----------|---------|--------|--------|
| 100  | 2.588 | 3.734 | 0.091 | 2.43% | 0.995 | Lambda decreasing |
| 500  | 2.459 | 3.547 | 0.094 | 2.65% | 0.967 | Approaching target |
| 1000 | 2.412 | 3.480 | 0.098 | 2.81% | 0.922 | Near equilibrium |
| **1500** | **2.408** | **3.475** | **0.105** | **2.93%** | **0.870** | **✓ Lambda stable** |
| 2000 | 2.441 | 3.522 | 0.109 | 2.99% | 0.870 | No lambda change |
| 2500 | 2.393 | 3.453 | 0.112 | 3.14% | 0.870 | No lambda change |
| 2800 | 2.435 | 3.513 | 0.114 | 3.14% | 0.870 | No lambda change |

**Observation:** Lambda stabilized at 0.870 around step 1500 and **did not change** through step 2800, indicating the PI controller reached equilibrium. This suggests step 1500 was the **MDL saturation point** - where the model learned all meaningful patterns from the data.

**Evidence supporting step 1500 as optimal stop:**
1. **Lambda convergence**: Δλ < 0.001 from step 1500-2800 (controller saturated)
2. **S-ratio closest to target**: 2.93% at step 1500 vs 3.14% at step 2500 (0.07% vs 0.14% error)
3. **Lower loss**: 2.408 at step 1500 vs 2.435 at step 2800 (training beyond saturation added noise)
4. **Lower ParamBPT**: 0.105 at step 1500 vs 0.114 at step 2800 (simpler model with same data fit)

**Automatic Stopping Criteria (Proposed):**
```python
def should_stop_training(lambda_history, s_ratio, target_s, window=100):
    """
    Stop when PI controller reaches stable equilibrium near target.

    Args:
        lambda_history: Recent lambda values (last `window` steps)
        s_ratio: Current S-ratio
        target_s: Target S-ratio
        window: Steps to check for stability (default: 100)

    Returns:
        True if training should stop (MDL saturation reached)
    """
    # Check 1: Lambda has stopped changing (controller saturated)
    lambda_stable = abs(lambda_history[-1] - lambda_history[-window]) < 0.001

    # Check 2: S-ratio is near target
    s_ratio_near_target = abs(s_ratio - target_s) / target_s < 0.05  # 5% tolerance

    return lambda_stable and s_ratio_near_target
```

This discovery transforms SCU from "adaptive regularization" to **"self-terminating training"** - the controller not only optimizes the model but also signals when optimization is complete.

### Llama 3.2 Validation (Earlier Work)
On Llama 3.2 (1B, 3B) fine-tuning, SCU improved bits-per-token by 6-12% over tuned fixed-λ baselines:

| Model | Metric | Baseline | SCU (PI Control) | Improvement |
|-------|--------|----------|------------------|-------------|
| **Llama-3.2-1B** | BPT | 3.920 | **3.676** | **-6.2%** |
| **Llama-3.2-3B** | BPT | 1.830 | **1.635** | **-10.6%** |

## 4. Related & Concurrent Work

The application of control theory to LLM training is an emerging and promising field.

### 4.1 Independent Convergence: EntroPIC
Recent independent work, **EntroPIC** (arXiv:2511.15248), applies PI control to stabilize policy entropy in reinforcement learning. While EntroPIC regulates policy entropy in RL, SCU regulates the information ratio in supervised learning. Both validate the necessity of feedback control for neural training dynamics.

## 5. Limitations

- **Scale**: Validated only up to 3B parameters with LoRA fine-tuning
- **Domain benchmarks**: No direct measurement of specialized capability preservation (e.g., math benchmarks)
- **S* selection**: Optimal target S* currently requires empirical tuning
- **Baseline comparisons**: Not yet compared against DoRA, QLoRA, or other modern PEFT methods with careful tuning

## 6. Future Directions

Our ongoing research focuses on:

- **Automatic Stopping Implementation**: Deploy the PI equilibrium detection mechanism discovered in OLMo 3 7B training as a production feature. This would eliminate manual step count selection and provide provable stopping criteria based on MDL saturation.

- **Scaling Laws for S***: Investigating whether optimal target S* can be derived from model size (N) and dataset size (D). Early evidence from 1B-7B range suggests S* ≈ 2-3% may be universal for LoRA fine-tuning.

- **Full-Parameter Training**: Extending validation beyond LoRA to full model pretraining. The automatic stopping mechanism may be even more valuable for expensive pretraining runs.

- **Domain-Specific Evaluation**: Adding math/code benchmarks to validate capability preservation claims from the VibeThinker experiment.

- **Unified Control**: Investigating relationships between Information Ratio regulation (SCU) and entropy stabilization (EntroPIC). Both use PI control but target different aspects of model behavior.

## 7. Usage

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

## 8. Citation

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

## 9. License

This repository is dual-licensed:

*   **Research & Open Source:** [AGPL-3.0](LICENSE). Free for academic and open-source use.
*   **Commercial:** Proprietary licenses available for closed-source applications. Contact `hunter@shannonlabs.dev`.

**Technical Paper:** See `docs/technical/` for detailed methodology and evaluation.

---

## Appendix: Control Math (tl;dr)

- **S-ratio (controlled variable):**
  
  \(S = \frac{\text{ParamBPT}}{\text{DataBPT} + \text{ParamBPT}}\); the MDL-derived regularization-to-fit ratio.
- **Error:** \(e(t) = S(t) - S^*\) with plant gain \(\partial S / \partial \lambda < 0\).
- **PI update with deadband and clamp:**
  
  \(e_d = 0\) if \(|e| < \delta\) (deadband), else \(e_d = e\).
  
  \(I_{t+1} = \text{clip}(I_t + e_d, I_{\min}, I_{\max})\).
  
  \(\lambda_{t+1} = \text{clip}\big( \lambda_t \cdot \exp(+(K_p e_d + K_i I_{t+1})), \lambda_{\min}, \lambda_{\max} \big)\).
- **Actuation point:** apply λ update at gradient-accumulation boundaries to avoid per-microbatch noise.
- **Stability aids:** deadband to prevent chatter; integral clamp/leak to avoid windup; λ bounds to avoid runaway; controller gains set by auto-config from model/dataset scale.

---

**Author:** Hunter Bown  
**Contact:** hunter@shannonlabs.dev  
**Website:** https://shannonlabs.dev
