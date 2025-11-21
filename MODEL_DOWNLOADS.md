# SCU Adapter Models

This directory contains configuration files for Shannon Control Unit (SCU) trained LoRA adapters.

## 🤗 HuggingFace Models

**All model weights are hosted on HuggingFace Hub:**

- **1B SCU**: [`hunterbown/shannon-control-unit`](https://huggingface.co/hunterbown/shannon-control-unit) (subfolder: `1b-scu`)
- **3B SCU**: [`hunterbown/shannon-control-unit`](https://huggingface.co/hunterbown/shannon-control-unit) (subfolder: `3b-scu`)
- **3B Fixed**: [`hunterbown/shannon-control-unit`](https://huggingface.co/hunterbown/shannon-control-unit) (subfolder: `3b-fixed`)

## Quick Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B", 
    device_map="auto",
    torch_dtype=torch.float16
)

# Load SCU adapter
model = PeftModel.from_pretrained(
    base, 
    "hunterbown/shannon-control-unit", 
    subfolder="3b-scu"
)
```

## Local Files

Each adapter directory contains:
- `adapter_config.json`: LoRA configuration
- `metrics.json`: Training metrics
- `README.md`: Model documentation

**Note**: Model weights (`.safetensors`) are hosted on HuggingFace and excluded from this repository.