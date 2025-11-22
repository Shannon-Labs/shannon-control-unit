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

# VibeThinker-1.5B-SCU (Scientifically Validated)

This is the **scientifically validated** release of the Shannon Control Unit (SCU) applied to the `WeiboAI/VibeThinker-1.5B` model.

**Shannon Control Unit (SCU)** is a research project that applies control-theoretic principles to Large Language Model (LLM) training. Like cruise control maintains vehicle speed regardless of hills, SCU maintains optimal regularization regardless of data complexity.

## 🔬 Scientific Validation Results

We conducted a rigorous comparative study on the VibeThinker 1.5B model to validate SCU's safety mechanisms.

| Model Variant | Training Method | Final Train DataBPT | Validation PPL | Validation BPT | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Model** | None (Zero-shot) | N/A | 967.22 | 9.92 | Reference |
| **Baseline** | Standard Finetuning (λ=0) | 3.49 | **70.27** | **6.13** | Strong Baseline |
| **V3 (Scientific)** | SCU (Fixed Prior, Natural) | 3.49 | **70.39** | **6.14** | ✅ **Optimal** |
| **V4 (Adaptive)** | SCU (Dynamic Prior) | **3.02** | 108.84 | 6.77 | ❌ Overfit |

### The "Safety Brake" Discovery
In this V3 run, the SCU controller saturated the regularization strength ($\lambda \to 2.0$) towards the end of training. 
*   We initially hypothesized this was a limitation and tried to "fix" it in V4 by loosening the prior. 
*   The result was immediate overfitting (PPL 108 vs 70).
*   **Conclusion:** The saturation was a **correct safety signal**. The SCU detected that the model had fully exploited the 500MB dataset and applied maximum braking to prevent memorization.

## 🛠️ Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 1. Load Base Model (WeiboAI/VibeThinker-1.5B)
base_id = "WeiboAI/VibeThinker-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    base_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)

# 2. Load SCU Adapter
adapter_id = "hunterbown/shannon-control-unit" # (or path to local adapter)
# Note: When released, use the specific revision or subfolder if applicable
model = PeftModel.from_pretrained(model, adapter_id, subfolder="adapters/vibethinker_1.5b_v3")

# 3. Inference
prompt = "Explain the concept of regularization."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.6)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
```

## 📜 License & Patent

*   **License:** AGPL-3.0 (Open Source)
*   **Patent:** U.S. Provisional Patent Pending (Sept 2025)
*   **Commercial Use:** Contact `hunter@shannonlabs.dev`

**Repository:** [https://github.com/Shannon-Labs/shannon-control-unit](https://github.com/Shannon-Labs/shannon-control-unit)
