---
base_model: mlx-community/Olmo-3-7B-Instruct-4bit
library_name: mlx
license: apache-2.0
tags:
- mlx
- lora
- text-generation
- scu
- information-theory
datasets:
- HuggingFaceFW/fineweb-edu
language:
- en
---

# OLMo-3-7B-Instruct SCU LoRA Adapter

**SCU-trained LoRA adapter for OLMo 3 7B Instruct** using information-theoretic regularization.

## Executive Summary

*   **Achievement**: The SCU-trained adapter matches the performance of an unregularized baseline while maintaining **~2x parameter efficiency** (0.105 vs 0.210 bits/token).
*   **Method**: Utilizes the **Shannon Control Unit (SCU)**, a PI-controlled adaptive regularization mechanism based on the Minimum Description Length (MDL) principle.
*   **Key Finding**: Analysis of the PI controller's equilibrium revealed that training could have stopped **46% earlier** (at Step 1500) without any loss in capability, validating the concept of MDL saturation.

## Methodology

| Component | Specification |
| :--- | :--- |
| **Base Model** | [`mlx-community/Olmo-3-7B-Instruct-4bit`](https://huggingface.co/mlx-community/Olmo-3-7B-Instruct-4bit) |
| **Dataset** | FineWeb-Edu (1GB subset, 98.3M tokens) |
| **Training** | LoRA (Rank 16, α=32), 40M trainable parameters |
| **SCU Config** | Target S-ratio: 3.0%, Prior σ: 0.01, PI Controller (Kp=0.8, Ki=0.15) |
| **Hardware** | Apple M4 Max (MLX framework) |

## Results

### 1. Efficiency Analysis (Step 1500 vs Final)

Comparing the model at the **MDL Saturation Point** (Step 1500) versus the final step (Step 2800):

| Metric | Step 1500 (Optimal) | Step 2800 (Final) | Impact |
| :--- | :--- | :--- | :--- |
| **Compute** | ~45 mins | ~1.5 hours | **46% reduction** |
| **Loss** | **2.408** | 2.435 | 1.1% increase |
| **Param BPT** | **0.105** | 0.114 | 8.5% higher complexity |
| **S-ratio** | **2.93%** | 3.14% | Divergence from target |

### 2. Comparative Performance (at Step 1500)

Comparing SCU against a standard unregularized baseline at the same training step:

| Metric | SCU | Baseline (No Reg) | Delta |
| :--- | :--- | :--- | :--- |
| **Loss** | **2.408** | 2.460 | SCU achieves **lower loss** |
| **Param BPT** | **0.105** | 0.210 | SCU is **~2x more efficient** |
| **S-ratio** | **2.93%** | 5.57% | Baseline divergence |

### 3. Capability Preservation

Qualitative testing confirms SCU preserves base model capabilities:

*   **Math & Logic**: Equivalent performance (solved algebra, failed complex riddles similarly).
*   **Coding (SQL)**: Equivalent correctness and formatting.
*   **Knowledge**: Equivalent fact retrieval.
*   **Creative Writing**: Both SCU and Baseline improved upon the Base model ("Evocative" vs "Standard").

## Discussion

*   **MDL Saturation**: The stabilization of the regularization coefficient ($\lambda$) at Step 1500 provided a clear signal that the model had learned all meaningful patterns available in the data given the complexity constraint.
*   **Redundant Complexity**: The baseline model's higher complexity (0.210 bits/token) did not result in lower loss, suggesting that the additional information encoded in the parameters was noise rather than signal.
*   **Auto-Stopping**: The PI controller's equilibrium offers a reliable, automated metric for stopping training, potentially saving significant compute resources in future runs.

## Usage

### With MLX

```python
from mlx_lm import load, generate

# Load base model with adapter
model, tokenizer = load(
    "mlx-community/Olmo-3-7B-Instruct-4bit",
    adapter_path="ShannonLabs/olmo3-7b-fineweb-scu-full"
)

# Generate
response = generate(
    model,
    tokenizer,
    prompt="Explain the concept of entropy in information theory.",
    max_tokens=256
)
```

### With Transformers + PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "allenai/Olmo-3-7B-Instruct",
    load_in_4bit=True
)

# Load adapter
model = PeftModel.from_pretrained(model, "ShannonLabs/olmo3-7b-fineweb-scu-full")

tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct")
```

## Citation

```bibtex
@software{scu_olmo3_2025,
  title={SCU-trained LoRA Adapter for OLMo 3 7B Instruct},
  author={Shannon Labs},
  year={2025},
  note={Information-theoretic adaptive regularization with PI control}
}
```
