# Quick Start: Granite SCU Training with CUDA

This guide provides quick commands for training IBM Granite-4.0-H-1B with SCU control on NVIDIA GPUs.

## Installation

```bash
# Clone the repository
git clone https://github.com/Hmbown/shannon-control-unit.git
cd shannon-control-unit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For CUDA optimizations
pip install bitsandbytes>=0.41.0  # For 4-bit quantization
pip install flash-attn>=2.0.0     # Optional: for flash attention
```

## Quick Training Commands

### 1. Check CUDA Setup
```bash
python scripts/train_granite_cuda.py --check-cuda
```

### 2. Quick Test Run (5 minutes)
```bash
python scripts/train_granite_cuda.py \
    --test-run \
    --max-steps 100 \
    --batch-size 4 \
    --fp16
```

### 3. Full Training Run
```bash
python scripts/train_granite_cuda.py \
    --batch-size 8 \
    --max-steps 1000 \
    --fp16 \
    --torch-compile \
    --gradient-checkpointing
```

### 4. Multi-GPU Training
```bash
python scripts/train_granite_cuda.py \
    --multi-gpu \
    --batch-size 16 \
    --max-steps 2000 \
    --bf16 \
    --torch-compile
```

### 5. Memory-Efficient Training (8GB GPU)
```bash
python scripts/train_granite_cuda.py \
    --batch-size 4 \
    --fp16 \
    --gradient-checkpointing \
    --use-4bit \
    --max-steps 500
```

### 6. Resume Training from Checkpoint
```bash
python scripts/train_granite_cuda.py \
    --resume-from-checkpoint granite_cuda_scu_output/checkpoint-500 \
    --batch-size 8 \
    --max-steps 1000
```

## Training Script Options

### Hardware Optimization
- `--fp16`: Use FP16 mixed precision (for Volta/Turing/Ampere GPUs)
- `--bf16`: Use bfloat16 mixed precision (for Ampere+ GPUs)
- `--multi-gpu`: Enable multi-GPU training
- `--torch-compile`: Enable torch.compile() optimization (2-3x speedup)
- `--gradient-checkpointing`: Save memory at cost of compute

### Memory Management
- `--batch-size`: Batch size per GPU (default: 8)
- `--use-4bit`: Enable 4-bit quantization (QLoRA)
- `--memory-fraction`: Limit GPU memory usage (0.0-1.0)

### SCU Control
- `--target-s`: Target S-ratio (default: 0.01)
- `--kp`: Proportional gain (default: 0.8)
- `--ki`: Integral gain (default: 0.15)
- `--control-frequency`: SCU update frequency (default: 10 steps)

### Training Duration
- `--max-steps`: Maximum training steps
- `--num-epochs`: Number of epochs (alternative to max-steps)
- `--early-stopping`: Enable early stopping

### Logging & Output
- `--logging-dir`: Directory for logs (default: logs/)
- `--output-dir`: Directory for model outputs (default: granite_cuda_scu_output/)
- `--save-frequency`: Save checkpoint every N steps
- `--eval-frequency`: Evaluate every N steps

## Other Training Scripts

### CPU-Only Training
```bash
python scripts/train_granite_cpu_only.py
```

### Simple SCU Training
```bash
python scripts/train_granite_simple.py
```

### Fixed Lambda Ablation
```bash
python scripts/train_granite_fixed.py
```

## Repository Management

### Check Disk Usage
```bash
python scripts/cleanup_outputs.py --list
```

### Interactive Cleanup
```bash
python scripts/cleanup_outputs.py --interactive
```

### Clean Everything
```bash
python scripts/cleanup_outputs.py --all --force
```

### Clean Specific Types
```bash
# Clean only model outputs
python scripts/cleanup_outputs.py --outputs --force

# Clean only cache
python scripts/cleanup_outputs.py --caches --force

# Clean only logs
python scripts/cleanup_outputs.py --logs --force
```

## Monitoring Training

### Real-time Metrics
The CUDA training script displays:
- **Loss**: Cross-entropy loss
- **S-ratio**: Current information ratio
- **Lambda**: Adaptive regularization strength
- **GPU Memory**: Current and peak memory usage
- **Throughput**: Tokens/second
- **ETA**: Estimated time to completion

### Log Files
Training logs are saved to `logs/cuda_optimized_training.log` with:
- Detailed SCU control metrics
- Memory usage statistics
- Training progress
- Performance benchmarks

### TensorBoard (Optional)
```bash
# Install tensorboard
pip install tensorboard

# Launch during training
tensorboard --logdir logs/
```

## Expected Performance

### Single GPU (RTX 4090, 24GB)
- **Batch Size**: 16
- **Speed**: ~150 tokens/sec
- **Memory Usage**: ~18GB
- **Training Time**: ~2 hours for 1000 steps

### Single GPU (RTX 3090, 24GB)
- **Batch Size**: 16
- **Speed**: ~120 tokens/sec
- **Memory Usage**: ~18GB
- **Training Time**: ~2.5 hours for 1000 steps

### Single GPU (T4, 16GB)
- **Batch Size**: 8
- **Speed**: ~40 tokens/sec
- **Memory Usage**: ~14GB
- **Training Time**: ~6 hours for 1000 steps

### Multi-GPU (2x RTX 4090)
- **Batch Size**: 32 (16 per GPU)
- **Speed**: ~280 tokens/sec
- **Training Time**: ~1.2 hours for 1000 steps

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python scripts/train_granite_cuda.py --batch-size 4

# Enable gradient checkpointing
python scripts/train_granite_cuda.py --gradient-checkpointing

# Use 4-bit quantization
python scripts/train_granite_cuda.py --use-4bit

# Limit memory fraction
python scripts/train_granite_cuda.py --memory-fraction 0.8
```

### Slow Training
```bash
# Enable torch.compile()
python scripts/train_granite_cuda.py --torch-compile

# Use bfloat16 if supported
python scripts/train_granite_cuda.py --bf16

# Enable cuDNN benchmarking
export TORCH_CUDNN_BENCHMARK=1
python scripts/train_granite_cuda.py
```

### Multi-GPU Issues
```bash
# Specify GPUs explicitly
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_granite_cuda.py --multi-gpu

# Single GPU mode if multi-GPU fails
python scripts/train_granite_cuda.py --batch-size 8
```

## Model Output

After training, you'll find:
- **Model**: `granite_cuda_scu_output/adapter_model.bin`
- **Config**: `granite_cuda_scu_output/adapter_config.json`
- **Logs**: `logs/cuda_optimized_training.log`
- **Metrics**: `granite_cuda_scu_output/scu_metrics.json`
- **Checkpoints**: `granite_cuda_scu_output/checkpoint-*/`

## Next Steps

1. **Evaluate Model**: Use `scripts/eval_bpt.py` to evaluate BPT performance
2. **Push to Hub**: Use `tools/push_to_hf.py` to upload to HuggingFace
3. **Compare**: Run ablations with different SCU parameters
4. **Scale**: Try larger models or longer training runs

## Support

- **Documentation**: See `CUDA_TRAINING_GUIDE.md` for detailed information
- **Issues**: Report problems on GitHub Issues
- **Examples**: Check `examples/` directory for sample configurations
