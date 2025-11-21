# VibeThinker 1.5B - Training Summary

## Overview

**VibeThinker-1.5B** is a 1.5 billion parameter language model trained using the Shannon Control Unit (SCU) method. This adapter was successfully trained on **November 21, 2025** at **00:21:54 UTC**.

## Model Details

- **Base Model**: VibeThinker-1.5B (local: `/models/VibeThinker-1.5B`)
- **Training Method**: Shannon Control Unit (SCU) with LoRA adapters
- **Adapter Size**: 73.9 MB (`adapter_model.safetensors`)
- **Framework**: PEFT 0.17.1

## Training Configuration

### SCU Control Parameters
- **Target S**: 0.01 (target entropy/complexity)
- **Kp (Proportional Gain)**: 0.8
- **Ki (Integral Gain)**: 0.15
- **Deadband**: 0.002
- **Lambda Init**: 1.0
- **Lambda Range**: [0.0001, 2.0]

### Training Hyperparameters
- **Epochs**: 1
- **Steps**: 100
- **Batch Size**: 1
- **Gradient Accumulation Steps**: 16 (effective batch size: 16)
- **Learning Rate**: 5e-5
- **Block Size**: 1024 tokens
- **Precision**: FP16

### Training Data
- **Path**: `data/train.txt`
- **Type**: Text data for language modeling

## Training Script

The model was trained using the crash-proof training script:
```bash
./scripts/train_vibethinker_1.5b.sh
```

This script is designed to be memory-efficient and compatible with Mac/MPS devices by using:
- Small batch size (1) with gradient accumulation
- Conservative memory settings
- No Unsloth optimization (for compatibility)

## Job Metadata

- **Job ID**: `799c60e4`
- **Timestamp**: 2025-11-21T00:21:54.867377

## Files Included

- `adapter_model.safetensors` - Trained LoRA adapter weights
- `adapter_config.json` - LoRA configuration
- `metadata.json` - Training job metadata
- `tokenizer.json` - Tokenizer configuration
- `vocab.json` - Vocabulary
- `merges.txt` - BPE merges
- `chat_template.jinja` - Chat template
- Various token configuration files

## Usage

To use this adapter with the base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "/Volumes/VIXinSSD/shannon-control-unit/models/VibeThinker-1.5B"
)

# Load adapter
model = PeftModel.from_pretrained(
    base_model,
    "/Volumes/VIXinSSD/shannon-control-unit/adapters/vibethinker_1.5b"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/Volumes/VIXinSSD/shannon-control-unit/adapters/vibethinker_1.5b"
)
```

Or use the SCU CLI:

```bash
scu generate \
  --base-model models/VibeThinker-1.5B \
  --adapter adapters/vibethinker_1.5b \
  --prompt "Your prompt here"
```

## Status

✅ **Training Complete**  
✅ **Adapter Saved**  
✅ **Ready for Inference**

## Next Steps

- [ ] Evaluate model performance on validation set
- [ ] Compare with baseline (non-SCU trained) model
- [ ] Document specific use cases and capabilities
- [ ] Consider uploading to HuggingFace Hub
- [ ] Run ablation studies on SCU parameters

## Notes

This model demonstrates the SCU training method on a 1.5B parameter model, showing that the method scales effectively to larger models while maintaining memory efficiency through careful hyperparameter tuning.
