# Shannon Control Unit (SCU) - AI Coding Agent Reference

## Project Overview

**Shannon Control Unit (SCU)** is a pioneering research project that applies control-theoretic principles to Large Language Model (LLM) training. Like cruise control maintains vehicle speed regardless of hills, SCU maintains optimal regularization regardless of data complexity.

**Core Innovation**: Automatically adjusts regularization strength (λ) during training to maintain a target information ratio (S*), eliminating manual hyperparameter tuning.

**Key Metrics**:
- **S-ratio**: S = ParamBPT / (DataBPT + ParamBPT)
- **Target**: Maintain S* throughout training via PI control
- **Validated**: 6.2% BPT improvement (1B), 10.6% (3B)

**Status**: Patent pending, dual-licensed (AGPL-3.0 + commercial), seeking 7B+ validation partners.

## Core Concepts: Model-Data Scaling

SCU provides principled detection and response to model-data scaling imbalances. The system quantitatively measures when parameter complexity exceeds dataset justification through the S-ratio and responds via adaptive regularization.

### Quantifying Model-Data Balance

The information ratio S reveals the relative contribution of parameter complexity versus data fit:

```
S = ParamBPT / (DataBPT + ParamBPT)
```

**ParamBPT** (parameter bits per token) represents the description length of model updates, scaling inversely with dataset size:

```
ParamBPT ∝ 1 / tokens_per_epoch
```

**Interpretation thresholds**:
- **S ≈ 1%**: Optimal balance (target)
- **S > 10%**: Model is over-parameterized for dataset
- **S > 50%**: Severe data starvation; regularization saturates

### SCU Response to Scaling Mismatch

When S exceeds target:
1. Controller increases λ (regularization strength)
2. ParamBPT decreases proportionally
3. S approaches target through feedback

When data is insufficient, λ may saturate at maximum (default: 2.0), indicating the system cannot achieve target S with available data.

### Scaling Guidelines

For prior σ=0.01 and target S=1%:

```
tokens_per_parameter ≈ 1 / (2 × σ² × S × ln(2))
tokens_per_parameter ≈ 140
```

**Example requirements**:
- 18M LoRA parameters: ~2.5B tokens (~10GB text)
- 100M full parameters: ~14B tokens (~56GB text)

### Resolution Strategies

When λ saturates (S too high):

1. **Increase Dataset**: Most theoretically rigorous
   - Follows Minimum Description Length principles
   - Maintains mathematical consistency
   - Example: Download additional data via `scripts/download_finewiki.py`

2. **Parameter Count Normalization**: Practical alternative
   - Use `--tokens_per_epoch_override` to adjust normalization
   - Effective for domain-specific fine-tuning
   - Document override usage clearly

3. **Prior Adjustment**: Modify regularization scale
   - Increase σ (e.g., from 0.01 to 0.1) → 100× ParamBPT reduction
   - Loosens weight prior assumptions
   - Changes theoretical interpretation

4. **Architecture Modification**: Reduce parameters
   - Lower LoRA rank (e.g., r=16 → r=8)
   - Reduces ParamBPT quadratically
   - May impact representational capacity

**Documentation**: Always report whether results use natural token counts or normalization overrides.

### VibeThinker 1.5B Example

**Configuration A**: Insufficient data
- Model: 1.5B parameters, 18M LoRA parameters
- Dataset: 2MB text, 530k tokens
- Result: ParamBPT=14.2, S=64%, λ saturated at 2.0
- Diagnosis: Severe model-data imbalance (≈0.03 tokens/parameter)

**Configuration B**: Extended data
- Dataset: HuggingFaceFW/finewiki, 100MB, 26M tokens
- Result: ParamBPT=0.287, S=4.1%, λ in active regulation
- Normalization: With 100M token override, S=1.5%
- Conclusion: Adequate scaling for this adaptation task

This demonstrates SCU's quantitative assessment capability and the importance of appropriate model-data scaling.

## Technology Stack

**Core Dependencies**:
```python
transformers>=4.36.0  # Model architectures
torch>=2.0.0         # Deep learning framework
peft>=0.7.0          # LoRA adapters
accelerate>=0.25.0   # Distributed training
bitsandbytes>=0.41.0 # 4-bit quantization
datasets>=2.14.0     # Data loading
```

**Hardware Support**:
- **CUDA**: FP16 + 4-bit quantization
- **Apple Silicon**: MPS with MLX framework support
- **CPU**: Fallback (slow, for testing only)

**No Build System**: Direct Python execution, requirements.txt dependency management.

## Project Architecture

### Directory Structure
```
scu/                    # Core SCU v1.0 implementation
├── control.py         # PI controller for adaptive lambda
├── data.py           # Data loading utilities
└── metrics.py        # Training metrics calculation

scu2/                   # Advanced SCU v2.0
├── core/              # Multi-scale entropy controllers
├── production/        # Production training scripts
└── experiments/       # Research configurations

scripts/                # Training workflows
├── train_scu.py      # Main training pipeline
├── eval_bpt.py       # BPT evaluation
└── run_ablation.py   # Comparative studies

tools/                  # Deployment utilities
├── hf_sync.py        # HuggingFace integration
└── visual_audit.js   # Quality assurance

configs/               # Configuration management
└── default.yaml      # Training parameters

results/               # Validation artifacts
1b-scu/, 3b-scu/      # Model adapter weights
```

### Key Components

**Core Control System** (`scu/control.py`):
- **PI Controller**: Proportional-Integral control for λ adjustment
- **Control Law**: λ ← λ × exp(+(Kp×error + Ki×∫error))
- **Anti-windup**: Integral term clamping and leakage
- **Deadband**: ±0.2pp tolerance to prevent oscillation

**Training Pipeline** (`scripts/train_scu.py`):
- **Multi-platform**: CUDA, MPS, CPU support
- **LoRA Integration**: Memory-efficient fine-tuning
- **Mixed Precision**: FP16 training with gradient accumulation
- **Real-time Monitoring**: S-ratio tracking and λ adjustment

## Development Workflow

### Quick Start Commands
```bash
# Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Validate installation
python scripts/test_quickstart.py

# Run training (1B model)
python scripts/train_scu.py --config configs/default.yaml

# Evaluate BPT performance
python scripts/eval_bpt.py --model-path adapters/scu_adapter

# Generate ablation plots
python viz/generate_ablation_plots.py
```

### Configuration Management
- **YAML-based**: `configs/default.yaml` for training parameters
- **Key Settings**: target_s=0.01 (1%), kp=0.8, ki=0.15
- **LoRA Config**: r=16, alpha=32, dropout=0.05
- **Hardware Auto-detection**: CUDA → MPS → CPU fallback

### Testing Strategy
```bash
# Unit tests
pytest tests/test_control.py

# Integration test
python scripts/test_quickstart.py

# Validation run
python scripts/eval_bpt.py --quick-test

# Visual QA
node tools/visual_audit.js
```

## Code Style Guidelines

**Python Standards**:
- **PEP 8**: Follow standard Python conventions
- **Type Hints**: Use for all public functions
- **Docstrings**: Google-style documentation
- **Functions**: Keep focused, single responsibility
- **Error Handling**: Graceful degradation with logging

**Research Code Principles**:
- **Reproducibility**: Fixed seeds, deterministic operations
- **Modularity**: Clear separation of concerns
- **Validation**: Statistical significance testing
- **Documentation**: Comprehensive inline comments

**Example Function Pattern**:
```python
def update_lambda(lmbda: float, S_meas: float, S_target: float, I: float,
                  *, Kp: float = 0.8, Ki: float = 0.15) -> Tuple[float, float, float]:
    """Update regularization strength using PI control.
    
    Args:
        lmbda: Current regularization strength
        S_meas: Measured S-ratio
        S_target: Target S-ratio
        I: Integral error accumulator
        Kp: Proportional gain (default: 0.8)
        Ki: Integral gain (default: 0.15)
        
    Returns:
        Tuple of (new_lambda, new_I, error)
        
    Note:
        Plant gain is negative: dS/dλ < 0
    """
```

## Testing Instructions

### Validation Requirements
- **Multi-seed**: Minimum 3 seeds for statistical validity
- **Baseline Comparison**: Fixed λ with hyperparameter sweep
- **Statistical Testing**: 95% confidence intervals
- **Reproducibility**: All hyperparameters documented

### Performance Metrics
- **Primary**: Bits Per Token (BPT) reduction
- **Secondary**: Perplexity improvement
- **Control**: S-ratio tracking accuracy
- **Efficiency**: Training time overhead < 2%

### Test Data
- **Dataset**: WikiText-103 subset (~512k tokens)
- **Rationale**: Resource-constrained, challenging for regularization
- **Split**: 90% train, 10% validation
- **Format**: Plain text, tokenized on-the-fly

## Security Considerations

**Model Loading**:
- **Trusted Sources**: Only load from verified HuggingFace repos
- **Local Validation**: Check model card and licenses
- **Sandboxing**: No arbitrary code execution in model loading

**Data Handling**:
- **Local Processing**: No external API calls during training
- **Privacy**: Training data stays local
- **Sanitization**: Input validation for all file paths

**Deployment**:
- **License Compliance**: Respect Meta Llama 3.2 Community License
- **IP Protection**: Patent pending technology
- **Commercial Use**: Dual licensing enforced

## Common Development Patterns

### Adding New Controllers
1. Inherit from base controller class
2. Implement `update()` method with control logic
3. Add configuration parameters to YAML
4. Write unit tests with mock data
5. Validate against baseline methods

### Integrating New Models
1. Update model loading in `train_scu.py`
2. Verify LoRA compatibility
3. Test quantization support
4. Benchmark training performance
5. Document model-specific considerations

### Running Experiments
1. Create configuration file in `configs/`
2. Set appropriate hyperparameters
3. Use deterministic seeds
4. Log all settings
5. Save results to `results/`

## Debugging Guidelines

**Common Issues**:
- **CUDA OOM**: Reduce batch_size, enable gradient checkpointing
- **MPS Issues**: Disable 4-bit quantization, use FP32
- **Control Oscillation**: Adjust deadband, reduce Ki gain
- **Slow Convergence**: Check learning rate, verify data loading

**Debugging Tools**:
- **Logging**: Detailed CSV logs in `logs/`
- **Visualization**: Matplotlib plots in `viz/`
- **Metrics**: Real-time S-ratio and λ tracking
- **Validation**: Automated BPT evaluation

## External Integration

**HuggingFace Hub**:
- **Model Upload**: `tools/push_to_hf.py`
- **Adapter Loading**: PEFT integration
- **Model Cards**: Auto-generated documentation

**Apple Silicon**:
- **MLX Framework**: Optimized for M1/M2/M3
- **Unified Memory**: Efficient 4-bit training
- **Metal Performance**: Hardware-accelerated inference

## Research Ethics

**Scientific Rigor**:
- **Honest Reporting**: Include negative results
- **Statistical Validity**: Proper significance testing
- **Reproducibility**: Complete experimental details
- **Peer Review**: Open to external validation

**IP and Licensing**:
- **Patent Pending**: Respect provisional filing
- **Dual License**: AGPL-3.0 for open source
- **Commercial Terms**: Contact for proprietary use
- **Attribution**: Maintain copyright notices

---

**Contact**: hunter@shannonlabs.dev  
**Website**: https://shannonlabs.dev  
**Models**: https://huggingface.co/hunterbown/shannon-control-unit