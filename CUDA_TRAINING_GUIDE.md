# CUDA-Optimized SCU Training Guide

## Overview

The `train_granite_cuda.py` script provides CUDA-optimized training for IBM Granite-4.0-H-1B with Shannon Control Unit (SCU) integration. This script is designed for NVIDIA GPUs and includes advanced optimizations for maximum performance.

## Features

### 🚀 Performance Optimizations
- **Mixed Precision Training**: FP16/bfloat16 support with automatic detection
- **Multi-GPU Support**: Distributed training across multiple GPUs
- **torch.compile()**: Graph optimization for faster execution
- **Memory Efficient Attention**: Flash attention and memory-efficient implementations
- **cuDNN Benchmarking**: Automatic optimization for your specific hardware

### 🧠 SCU Integration
- **Real-time Control**: Adaptive regularization every N steps
- **CUDA Memory Tracking**: Detailed memory usage monitoring
- **EMA Smoothing**: Exponential moving average for stable control
- **Anti-windup Protection**: Prevents integral term saturation

### 💾 Memory Management
- **4-bit Quantization**: Optional QLoRA for extreme memory efficiency
- **Gradient Checkpointing**: Trade compute for memory
- **Automatic Cache Management**: Intelligent CUDA cache clearing
- **Memory Fraction Control**: Prevents OOM errors

## Requirements

### Hardware
- **NVIDIA GPU** with 8GB+ VRAM (16GB+ recommended)
- **CUDA 11.8+** or **CUDA 12.0+**
- **Compute Capability 7.0+** (Volta, Turing, Ampere, Ada Lovelace)

### Software
```bash
# Core dependencies
pip install torch>=2.0.0 transformers>=4.36.0 peft>=0.7.0

# CUDA optimizations
pip install bitsandbytes>=0.41.0  # For 4-bit quantization
pip install flash-attn>=2.0.0     # For flash attention (optional)

# Additional CUDA tools
pip install nvidia-ml-py3>=7.352.0  # For GPU monitoring
```

## Quick Start

### 1. Check Requirements
```bash
python scripts/train_granite_cuda.py --check-requirements
```

### 2. Basic Training (Single GPU)
```bash
python scripts/train_granite_cuda.py \
    --fp16 \
    --batch-size 8 \
    --max-steps 1000 \
    --output-dir ./granite_cuda_output
```

### 3. Advanced Training (Multi-GPU)
```bash
# Multi-GPU with bfloat16
cuda_VISIBLE_DEVICES=0,1 python scripts/train_granite_cuda.py \
    --bf16 \
    --multi-gpu \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --torch-compile \
    --max-steps 2000
```

### 4. Memory-Efficient Training
```bash
# For GPUs with limited memory
python scripts/train_granite_cuda.py \
    --fp16 \
    --4bit-quantization \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --lora-r 8 \
    --max-steps 1000
```

## Command Line Arguments

### Core Training Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 8 | Batch size per GPU |
| `--gradient-accumulation-steps` | 4 | Gradient accumulation steps |
| `--learning-rate` | 1e-4 | Learning rate |
| `--max-steps` | 1000 | Maximum training steps |
| `--block-size` | 1024 | Sequence length for training |
| `--lora-r` | 16 | LoRA rank |

### Precision & Optimization
| Argument | Description |
|----------|-------------|
| `--fp16` | Enable FP16 mixed precision |
| `--bf16` | Enable bfloat16 mixed precision |
| `--torch-compile` | Enable torch.compile optimization |
| `--4bit-quantization` | Enable 4-bit quantization (QLoRA) |

### Multi-GPU & Distribution
| Argument | Description |
|----------|-------------|
| `--multi-gpu` | Enable multi-GPU training |

### Control & Monitoring
| Argument | Description |
|----------|-------------|
| `--test-run` | Run minimal steps for testing |
| `--check-requirements` | Check CUDA requirements |
| `--resume-from-checkpoint` | Resume from checkpoint |

## Hardware-Specific Recommendations

### RTX 4090 (24GB)
```bash
python scripts/train_granite_cuda.py \
    --bf16 \
    --batch-size 16 \
    --gradient-accumulation-steps 2 \
    --torch-compile \
    --max-steps 2000
```

### A100 (40GB)
```bash
python scripts/train_granite_cuda.py \
    --bf16 \
    --batch-size 32 \
    --torch-compile \
    --max-steps 5000
```

### RTX 3080 (10GB)
```bash
python scripts/train_granite_cuda.py \
    --fp16 \
    --4bit-quantization \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --lora-r 8 \
    --max-steps 1000
```

## Performance Benchmarks

### Expected Training Speed (tokens/second)
| GPU | Precision | Batch Size | Speed |
|-----|-----------|------------|--------|
| RTX 4090 | BF16 | 16 | ~15,000 |
| RTX 3080 | FP16 | 8 | ~8,000 |
| A100 | BF16 | 32 | ~25,000 |
| T4 | FP16 | 4 | ~3,000 |

### Memory Usage
| Configuration | VRAM Usage |
|---------------|------------|
| FP16, batch=8 | ~6-8GB |
| BF16, batch=16 | ~12-16GB |
| 4-bit, batch=4 | ~4-6GB |

## Monitoring & Troubleshooting

### Real-time Monitoring
The script provides detailed logging including:
- **S-ratio tracking**: Current vs target information balance
- **Lambda updates**: Regularization strength adjustments
- **CUDA memory**: Real-time memory usage
- **Training speed**: Tokens per second

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size and enable gradient accumulation
--batch-size 2 --gradient-accumulation-steps 16

# Enable 4-bit quantization
--4bit-quantization

# Reduce sequence length
--block-size 512
```

#### 2. Slow Training
```bash
# Enable torch.compile
--torch-compile

# Use optimal dtype for your GPU
--bf16  # For RTX 30xx+ and A100
--fp16  # For older GPUs

# Increase batch size if memory allows
--batch-size 16
```

#### 3. Multi-GPU Issues
```bash
# Set CUDA_VISIBLE_DEVICES explicitly
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_granite_cuda.py ...

# Use smaller per-GPU batch size
--batch-size 4
```

## Output Structure

```
granite_cuda_output/
├── scu_config.json              # Training configuration
├── scu_metrics.json             # SCU control metrics
├── scu_history.json             # S-ratio and lambda history
├── training_config.json         # Full training config
├── pytorch_model.bin           # Model weights
├── adapter_config.json         # LoRA configuration
├── tokenizer.json              # Tokenizer files
└── logs/
    └── cuda_optimized_training.log
```

## Advanced Usage

### Custom SCU Parameters
Edit the configuration file to adjust:
- Target S-ratio
- PI controller gains
- Control frequency
- Lambda bounds

### Resume Training
```bash
python scripts/train_granite_cuda.py \
    --resume-from-checkpoint ./granite_cuda_output/checkpoint-500 \
    --max-steps 2000
```

### Distributed Training
```bash
# Using torch.distributed
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/train_granite_cuda.py \
    --multi-gpu \
    --batch-size 4
```

## Comparison with Other Scripts

| Feature | CUDA Script | CPU Script | Memory-Efficient |
|---------|-------------|------------|------------------|
| **Speed** | 🚀 Fastest | 🐌 Slowest | ⚡ Moderate |
| **Memory** | 💾 High VRAM | 💽 Low RAM | 🧠 Optimized |
| **Precision** | FP16/BF16 | FP32 | FP32 |
| **Multi-GPU** | ✅ Yes | ❌ No | ❌ No |
| **Quantization** | ✅ 4-bit | ❌ No | ❌ No |
| **Compile** | ✅ torch.compile | ❌ No | ❌ No |

## Best Practices

1. **Start Conservative**: Begin with smaller batch sizes and scale up
2. **Monitor Memory**: Watch CUDA memory usage in logs
3. **Use Appropriate Precision**: BF16 for Ampere+, FP16 for older GPUs
4. **Enable torch.compile**: Significant speedup with minimal overhead
5. **Gradient Accumulation**: Effective way to increase batch size
6. **Regular Checkpoints**: Use `--save-steps` for frequent saves

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs in `logs/cuda_optimized_training.log`
3. Use `--test-run` to validate setup
4. Check CUDA requirements with `--check-requirements`

## Next Steps

After training, you can:
1. **Evaluate BPT**: Use `scripts/eval_bpt.py`
2. **Visualize Results**: Check `viz/generate_ablation_plots.py`
3. **Deploy Model**: Use `tools/push_to_hf.py`
4. **Compare Controllers**: Use `scripts/compare_controllers.py`