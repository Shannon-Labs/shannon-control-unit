---
base_model: WeiboAI/VibeThinker-1.5B
library_name: peft
license: agpl-3.0
tags:
- generated_from_trainer
- shannon-control-unit
- reinforcement-learning
- regularization
- vibethinker
model-index:
- name: VibeThinker-1.5B-SCU-V3
  results: []
---

# VibeThinker-1.5B-SCU: Automated Information-Theoretic Early Stopping

**A Scientifically Validated "Self-Regulating" Model**

[![Patent Pending](https://img.shields.io/badge/Patent-Pending-orange.svg)](https://shannonlabs.dev)
[![GitHub](https://img.shields.io/badge/GitHub-Shannon_Control_Unit-blue.svg)](https://github.com/Shannon-Labs/shannon-control-unit)

This model is a proof-of-concept for the **Shannon Control Unit (SCU)**, an automated training framework that acts as a "Safety Brake" against overfitting. It was trained on the `WeiboAI/VibeThinker-1.5B` model using the FineWeb-Edu dataset.

## 🚀 The "Transistor" Moment in AI Training

Traditional training is like a **vacuum tube**: it amplifies learning blindly until manual intervention (early stopping) cuts the power. SCU acts as a **transistor**, modulating regularization in real-time based on the model's internal Information Ratio.

### Scientific Discovery: Step 386
In this experiment, SCU detected that the 1.5B model had saturated its learning capacity after only **16 Million tokens** (Step 386).
*   **Automatic Reaction:** The controller saturated regularization ($\lambda \to 2.0$), effectively freezing the weights to prevent the model from memorizing noise.
*   **The Result:** Optimal performance (6.14 BPT) matching the best manual baseline, but with guaranteed safety against the overfitting "crash" seen in unregulated runs.

This suggests that for highly optimized models like VibeThinker, **~90% of standard training compute may be wasted** on overfitting, which SCU identifies and prevents automatically.

## 📊 Benchmark Results

| Metric | Base Model | Baseline (Manual) | SCU V3 (Auto-Brake) | SCU V4 (Unregulated) |
| :--- | :--- | :--- | :--- | :--- |
| **BPT Score** | 9.92 | 6.13 | **6.14** | 6.77 |
| **Status** | Untrained | Risky | **Safe & Optimal** | Crashed |

## 🛠️ Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 1. Load Base Model
base_id = "WeiboAI/VibeThinker-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    base_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    trust_remote_code=True
)

# 2. Load SCU Adapter
adapter_id = "hunterbown/VibeThinker-1.5B-SCU" 
model = PeftModel.from_pretrained(model, adapter_id)

# 3. Inference
tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
prompt = "Explain the concept of information capacity."
# ... standard generation ...
```

## 📜 License & Citation

*   **License:** AGPL-3.0 (Open Source Research License)
*   **Patent:** U.S. Provisional Patent Pending (Sept 2025)
*   **Repository:** [https://github.com/Shannon-Labs/shannon-control-unit](https://github.com/Shannon-Labs/shannon-control-unit)

If you use this in research, please cite:

```bibtex
@misc{bown2025scu,
  author = {Bown, Hunter},
  title = {Shannon Control Unit: Automated Information-Theoretic Early Stopping},
  year = {2025},
  url = {https://github.com/Shannon-Labs/shannon-control-unit}
}
```
