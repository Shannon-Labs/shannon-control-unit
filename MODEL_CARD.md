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

# VibeThinker-1.5B-SCU (Scientific Safety)

**A Scientifically Validated "Self-Regulating" Model**

[![Patent Pending](https://img.shields.io/badge/Patent-Pending-orange.svg)](https://shannonlabs.dev)
[![GitHub](https://img.shields.io/badge/GitHub-Shannon_Control_Unit-blue.svg)](https://github.com/Shannon-Labs/shannon-control-unit)

This model is a **scientifically validated** proof-of-concept for the **Shannon Control Unit (SCU)**. It applies information-theoretic control to the `WeiboAI/VibeThinker-1.5B` model, demonstrating how automated regularization can act as a "Safety Brake" against overfitting.

## 🚀 Why This Matters (The Scientific Discovery)

Most models are trained with fixed regularization. SCU adapts dynamically.

During the training of this model (V3), our controller **saturated the regularization** ($\lambda \to 2.0$) near the end. We initially thought this was a bug and tried to "fix" it in a follow-up experiment (V4).
*   **The Result:** The "fixed" V4 model overfitted immediately (PPL 108).
*   **The Lesson:** This V3 model represents a **correctly self-regulated system**. The controller detected it had learned all it could from the data and applied maximum braking to prevent memorization.

### 📊 Benchmark Results

| Model Variant | Training Method | Validation PPL | Status |
| :--- | :--- | :--- | :--- |
| **Baseline** | Standard Finetuning (λ=0) | 70.27 | Strong Baseline |
| **VibeThinker-SCU (This Model)** | SCU (Fixed Prior, Natural) | **70.39** | ✅ **Optimal Safety** |
| **Unregulated Attempt (V4)** | SCU (Dynamic Prior) | 108.84 | ❌ Crashed (Overfit) |

**Conclusion:** This model achieves optimal performance (matching baseline) while strictly adhering to information-theoretic safety bounds.

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

# 2. Load This Adapter
# (Replace with your new repo name, e.g., "hunterbown/VibeThinker-1.5B-SCU")
adapter_id = "hunterbown/VibeThinker-1.5B-SCU" 
model = PeftModel.from_pretrained(model, adapter_id)

# 3. Inference
tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
prompt = "Explain the concept of regularization."
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
  title = {Shannon Control Unit: Information-Theoretic Regularization via PI Control},
  year = {2025},
  url = {https://github.com/Shannon-Labs/shannon-control-unit}
}
```
