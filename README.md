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
λ_(t+1) = λ_t × exp(-(K_p × e(t) + K_i × Σ e(τ)))
```

optionally with deadband and integral clamping for anti-windup. Updates are applied at gradient-accumulation boundaries to maintain stability.

## 3. Results

We validated SCU by fine-tuning Llama 3.2 models on a subset of WikiText-103. The results show significant improvements in compression efficiency (Bits Per Token) and Perplexity compared to an optimally tuned cross-entropy baseline.

| Model | Metric | Baseline (Cross-Entropy) | SCU (PI Control) | Improvement |
|-------|--------|--------------------------|------------------|-------------|
| **Llama-3.2-1B** | BPT | 3.920 | **3.676** | **-6.2%** |
| | Perplexity | 15.14 | **12.78** | **-15.6%** |
| **Llama-3.2-3B** | BPT | 1.830 | **1.635** | **-10.6%** |
| | Perplexity | 3.56 | **3.11** | **-12.6%** |

*Note: Validation performed on Llama 3.2 LoRA adapters. Baseline represents the best-performing fixed-λ configuration found via grid search.*

## 4. Related & Concurrent Work

The application of control theory to LLM training is an emerging and promising field.

### 4.1 Independent Convergence: EntroPIC
Recent independent work, **EntroPIC** (arXiv:2511.15248), applies PI control to stabilize policy entropy in reinforcement learning. This convergence indicates that control-theoretic feedback is effective for stabilizing training dynamics. SCU targets the MDL information ratio during supervised pretraining/fine-tuning, whereas EntroPIC targets policy entropy in RL; the objectives are complementary and suggest a broader control lens on neural training.

## 5. Future Directions

Our ongoing research focuses on:

- **Scaling Laws for S***: Deriving the optimal target S* from first principles based on model size (N) and dataset size (D), removing the need for a target setpoint entirely.
- **Full-Parameter Training**: Extending validation beyond LoRA to full model pretraining.
- **Unified Control**: Investigating if regulating Information Ratio implicitly stabilizes entropy (unifying SCU and EntroPIC findings).

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
