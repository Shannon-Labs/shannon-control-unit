# Examples

Practical examples and tutorials for using Shannon Control Unit.

## Quick Start

### 1. Loading Pre-trained Models

The simplest way to use SCU is to load our pre-trained adapters:

```python
# examples/load_pretrained.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model and tokenizer
base_id = "meta-llama/Llama-3.2-1B"
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(base_id)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base_model.config.pad_token_id = tokenizer.pad_token_id

# Load SCU adapter
model = PeftModel.from_pretrained(base_model, "hunterbown/shannon-control-unit")

# Generate text
inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### 2. Basic Training with SCU

Train your own model with adaptive regularization:

```python
# examples/train_basic.py
from scu.control import SCUController
from transformers import AutoModelForCausalLM, TrainingArguments
import torch

# Initialize model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Initialize SCU controller
# For 1B models: target_s ~ 0.01
# For 3B models: target_s ~ 0.03
controller = SCUController(
    target_s=0.01,
    kp=0.1,
    ki=0.01,
    lambda_init=1.0
)

# Training loop (simplified)
def training_step(batch, model, controller):
    # Forward pass
    outputs = model(**batch)
    data_loss = outputs.loss
    
    # Calculate BPT metrics
    data_bpt = data_loss.item() / math.log(2)
    param_bpt = calculate_param_bpt(model)  # L2 norm / num_params
    
    # Get adaptive lambda
    lambda_t = controller.step(data_bpt, param_bpt)
    
    # Total loss with regularization
    param_loss = sum(p.pow(2).sum() for p in model.parameters())
    total_loss = data_loss + lambda_t * param_loss
    
    return total_loss

# See scripts/train_scu.py for complete implementation
```

### 3. Evaluation

Evaluate model performance:

```python
# examples/evaluate.py
from scripts.eval_bpt import evaluate_bpt
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-model-path")

results = evaluate_bpt(
    model=model,
    data_path="data/val.txt",
    block_size=1024
)

print(f"BPT: {results['bpt']:.3f}")
print(f"Perplexity: {results['perplexity']:.2f}")
```

## Complete Examples

For complete, runnable examples, see:

- [`scripts/train_scu.py`](../scripts/train_scu.py) - Full training pipeline
- [`scripts/eval_bpt.py`](../scripts/eval_bpt.py) - Complete evaluation
- [`scripts/test_quickstart.py`](../scripts/test_quickstart.py) - Quick validation test
- [`notebooks/SCU_Demo.ipynb`](../notebooks/SCU_Demo.ipynb) - Interactive Jupyter notebook

## Notebooks

### Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hmbown/shannon-control-unit/blob/main/notebooks/SCU_Demo.ipynb)

The Colab notebook provides an interactive walkthrough of:
- Loading pre-trained SCU models
- Comparing baseline vs SCU performance
- Visualizing control dynamics
- Running inference

## Tips & Best Practices

### Choosing Target S*

Based on our validation:
- **1B models**: `target_s ≈ 0.01` (1%)
- **3B models**: `target_s ≈ 0.03` (3%)
- **Larger models**: Scaling laws under investigation

### Tuning Controller Gains

- Start with `kp=0.1`, `ki=0.01`
- Increase `kp` for faster response (may oscillate)
- Increase `ki` for better steady-state tracking
- Use lambda bounds to prevent instability

### Monitoring Training

Key metrics to track:
- Current S ratio vs target
- Lambda trajectory (should be bounded)
- Data BPT and Param BPT separately
- Validation loss trends

## Next Steps

- Read the [API documentation](../docs/README.md)
- Explore [validation results](../results/)
- Review [training scripts](../scripts/)
- Check [technical documentation](../docs/technical/) (coming soon)
