# SCU + Granite Deployment Guide

## Quick Start Checklist

### ✅ Prerequisites
- [ ] Apple Silicon Mac with 16GB+ RAM (36GB recommended)
- [ ] Python 3.11+ with virtual environment
- [ ] 10GB+ free disk space
- [ ] Internet connection for initial model download

### ✅ Environment Setup
```bash
# 1. Clone and setup
git clone https://github.com/hunterbown/shannon-control-unit.git
cd shannon-control-unit
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test installation
python test_quick_scu.py
```

### ✅ Training Options

#### Option 1: Quick Test (5 minutes)
```bash
python scripts/train_granite_1b_scu.py \
    --test-run \
    --memory-safe \
    --output-dir ./quick_test
```

#### Option 2: Full Training (3-4 hours)
```bash
python scripts/train_granite_1b_scu.py \
    --memory-safe \
    --output-dir ./full_training
```

#### Option 3: Custom Configuration
```bash
# Create custom config
cp configs/granite_1b_scu_config.py my_config.py
# Edit my_config.py with your settings

python scripts/train_granite_1b_scu.py \
    --config-override my_config.json \
    --output-dir ./custom_training
```

## 📋 Pre-Deployment Checklist

### System Requirements
- [ ] **Memory Check**: Run `python -c "import psutil; print(f'{psutil.virtual_memory().total/1024**3:.1f}GB available')"`
- [ ] **Storage Check**: Ensure 10GB+ free space
- [ ] **Python Version**: Confirm Python 3.11+ with `python --version`

### Model Validation
- [ ] **Quick Test**: Run `python test_quick_scu.py` successfully
- [ ] **SCU Logic**: Verify control system responds correctly
- [ ] **Memory Test**: Confirm no memory explosions during test

### Configuration Review
- [ ] **Target S-ratio**: Default 2% appropriate for your use case
- [ ] **Control frequency**: Every 5 steps (adjust if needed)
- [ ] **Memory settings**: Conservative values for your hardware
- [ ] **Dataset**: WikiText-2 included (replace if needed)

## 🚀 Deployment Options

### Option A: Research Use
```bash
# Academic/research deployment
python scripts/train_granite_1b_scu.py \
    --output-dir ./research_model \
    --memory-safe \
    --max-steps 1000  # Adjust as needed
```

### Option B: Production Fine-tuning
```bash
# Production deployment with custom data
python scripts/train_granite_1b_scu.py \
    --output-dir ./production_model \
    --memory-safe \
    --train-file ./your_data.txt \
    --validation-file ./your_validation.txt \
    --max-steps 2000  # Longer training
```

### Option C: HuggingFace Upload
```bash
# After training, upload to HuggingFace
# 1. Create model card from README_HF_GRANITE_SCU.md
# 2. Use HuggingFace CLI to upload
huggingface-cli upload your-username/your-model-name ./output_dir/
```

## 📊 Monitoring and Validation

### During Training
Watch for these key indicators:
- **Memory usage**: Should stay <32GB (warnings at 30GB)
- **S-ratio tracking**: Should converge to ~2% target
- **Control actions**: Should be smooth, not oscillating
- **Loss progression**: Should decrease steadily

### Post-Training Validation
```bash
# Evaluate BPT performance
python scripts/eval_bpt.py \
    --model-path ./output_dir \
    --dataset wikitext-2

# Generate analysis plots
python viz/generate_scu_analysis.py \
    --metrics-path ./output_dir/scu_metrics.json
```

## 🔧 Troubleshooting

### Memory Issues
```bash
# If memory usage is too high:
# 1. Reduce batch size to 1
# 2. Decrease sequence length to 256
# 3. Increase gradient accumulation to 64
# 4. Enable --memory-safe flag
```

### Training Instability
```bash
# If training is unstable:
# 1. Reduce PI gains (Kp=0.4, Ki=0.05)
# 2. Increase deadband to 0.005
# 3. Decrease control frequency to every 10 steps
# 4. Tighten lambda bounds to [0.01, 1.0]
```

### Slow Training
```bash
# If training is too slow:
# 1. Reduce max_steps to 500
# 2. Increase logging_steps to 50
# 3. Reduce dataset size to 25K examples
# 4. Disable validation if not needed
```

## 📁 File Structure

```
granite_1b_scu_output/
├── adapter_model.bin          # LoRA adapter weights
├── adapter_config.json        # LoRA configuration
├── scu_metrics.json           # Training metrics
├── scu_history.json           # Control history
├── scu_config.json            # Configuration used
├── training_args.bin          # Training arguments
└── logs/
    ├── memory_efficient_training.log
    └── tensorboard/
```

## 🎯 Expected Results

### Training Metrics
- **Final loss**: ~3.2 ± 0.1
- **S-ratio average**: 2.1% ± 0.2%
- **Memory peak**: <30GB
- **Training time**: 2-4 hours
- **Control actions**: 200 per 1000 steps

### Model Performance
- **Information balance**: Maintains target S-ratio
- **Training stability**: No explosions or crashes
- **Memory safety**: Stays within hardware limits
- **Convergence**: Smooth loss reduction

## 🔐 Security Considerations

### Model Safety
- Only load from trusted HuggingFace repositories
- Verify model cards and licensing
- Test in isolated environment first
- Monitor for unexpected behaviors

### Data Privacy
- Training data stays local
- No external API calls during training
- Configurable logging levels
- Secure model storage

## 📞 Support and Community

### Getting Help
1. **Check logs**: Review training logs in `output_dir/logs/`
2. **Test configuration**: Use `--test-run` to validate setup
3. **Memory monitoring**: Watch system resources during training
4. **Community support**: Open issues on GitHub

### Contributing
- Report bugs and improvements
- Share hardware compatibility results
- Contribute optimizations and features
- Help with documentation and examples

## 📚 Additional Resources

### Documentation
- [SCU Paper](paper_huggingface_scu_granite.md) - Full technical details
- [Model Card](README_HF_GRANITE_SCU.md) - HuggingFace model documentation
- [AGENTS.md](AGENTS.md) - Project overview and coding standards

### Related Work
- [IBM Granite Models](https://huggingface.co/ibm-granite)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Information Theory in ML](https://arxiv.org/abs/2207.07413)

---

**Ready to deploy? Start with the quick test and scale up as needed!** 🚀