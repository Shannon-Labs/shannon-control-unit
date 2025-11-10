# Shannon Control Unit - Documentation

Comprehensive documentation for the Shannon Control Unit (SCU) adaptive regularization system.

## Quick Links

- **[Getting Started](../README.md)** - Installation and quick start
- **[API Reference](#api-reference)** - Core API documentation
- **[Examples](../examples/)** - Code examples and tutorials
- **[Technical Documentation](./technical/)** - Deep technical details (coming soon)

## Overview

Shannon Control Unit (SCU) provides cruise-control-style adaptive regularization for LLM training. Like cruise control maintains speed regardless of road conditions, SCU maintains optimal information ratio regardless of data complexity.

## API Reference

### Core Classes

#### `SCUController`

Located in [`scu/control.py`](../scu/control.py)

The main controller class implementing PI-based adaptive regularization.

**Key Parameters:**
- `target_s`: Target information ratio S* (e.g., 0.01 for 1B models, 0.03 for 3B)
- `kp`: Proportional gain (default: 0.1)
- `ki`: Integral gain (default: 0.01)
- `lambda_init`: Initial regularization strength (default: 1.0)
- `lambda_min`: Minimum lambda bound (default: 0.1)
- `lambda_max`: Maximum lambda bound (default: 10.0)

**Key Methods:**
- `step(data_bpt, param_bpt)`: Update lambda based on current BPT values
- `get_lambda()`: Get current regularization strength
- `get_s()`: Get current information ratio

**Example:**
```python
from scu.control import SCUController

controller = SCUController(target_s=0.01, kp=0.1, ki=0.01)
lambda_t = controller.step(data_bpt=3.2, param_bpt=0.05)
```

### Data Utilities

Located in [`scu/data.py`](../scu/data.py)

Functions for loading and processing training data.

## Training with SCU

### Basic Training Loop

```python
from scu.control import SCUController
from transformers import AutoModelForCausalLM, Trainer

# Initialize controller
controller = SCUController(target_s=0.01)

# In training loop
for batch in dataloader:
    # Calculate BPT metrics
    data_bpt = calculate_data_bpt(batch, model)
    param_bpt = calculate_param_bpt(model)
    
    # Update lambda
    lambda_t = controller.step(data_bpt, param_bpt)
    
    # Apply regularization with current lambda
    loss = data_loss + lambda_t * param_loss
```

See [`examples/`](../examples/) for complete training scripts.

## Model Weights

Pre-trained SCU adapters are available on Hugging Face:

- **Llama-3.2-1B + SCU**: `hunterbown/shannon-control-unit`
- **Llama-3.2-3B + SCU**: `hunterbown/shannon-control-unit` (subfolder: `3b-scu`)

### Loading Pre-trained Adapters

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model = PeftModel.from_pretrained(base_model, "hunterbown/shannon-control-unit")
```

## Validation Results

| Model | Metric | Baseline | SCU | Improvement |
|-------|--------|----------|-----|-------------|
| Llama-3.2-1B | BPT | 3.920 | 3.676 | **-6.2%** |
| Llama-3.2-1B | Perplexity | 15.14 | 12.78 | **-15.6%** |
| Llama-3.2-3B | BPT | 1.830 | 1.635 | **-10.6%** |
| Llama-3.2-3B | Perplexity | 3.56 | 3.11 | **-12.6%** |

Full validation results: [`results/3b_validation_results.json`](../results/3b_validation_results.json)

## Configuration

Default configuration is in [`configs/default.yaml`](../configs/default.yaml)

## Scripts

Common scripts for training and evaluation:

- [`scripts/train_scu.py`](../scripts/train_scu.py) - Main training script
- [`scripts/eval_bpt.py`](../scripts/eval_bpt.py) - BPT evaluation
- [`scripts/run_ablation.py`](../scripts/run_ablation.py) - Ablation studies

## Citation

```bibtex
@software{bown2025shannon,
  author = {Bown, Hunter},
  title = {Shannon Control Unit: Adaptive Regularization via PI Control},
  year = {2025},
  url = {https://github.com/Hmbown/shannon-control-unit}
}
```

## License

- **Code**: AGPL-3.0 (commercial licenses available)
- **Model Weights**: Meta Llama 3.2 Community License

For commercial licensing: hunter@shannonlabs.dev
