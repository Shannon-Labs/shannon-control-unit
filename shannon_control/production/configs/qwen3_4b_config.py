"""
Production T-SCU Configuration for Qwen3:4B Training

Optimized configuration for training Qwen3:4B with thermodynamic control.
Designed for HuggingFace distribution with production-ready settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Qwen3ProductionConfig:
    """Production configuration for T-SCU Qwen3:4B training"""

    # Model configuration
    model_name: str = "Qwen/Qwen2.5-3B"  # Use 3B as base for 4B target
    tokenizer_name: str = "Qwen/Qwen2.5-3B"
    model_max_length: int = 2048

    # Training configuration
    batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Effective batch size = 8
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    max_steps: int = 5000
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 25

    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"

    # T-SCU configuration
    enable_thermodynamic_control: bool = True
    power_budget_watts: float = 25.0  # Conservative for Apple Silicon
    target_efficiency_bits_per_joule: float = 1e-5
    max_temperature_celsius: float = 80.0
    control_frequency: int = 50  # Apply control every N steps

    # Data configuration
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    preprocessing_num_workers: int = 4
    block_size: int = 1024
    overwrite_cache: bool = False

    # Optimization configuration
    fp16: bool = True
    bf16: bool = False
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 4
    gradient_checkpointing: bool = True
    use_fast_tokenizer: bool = True

    # Evaluation configuration
    evaluation_strategy: str = "steps"
    prediction_loss_only: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True

    # Checkpointing configuration
    save_strategy: str = "steps"
    save_total_limit: int = 3
    save_safetensors: bool = True
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None

    # Hardware configuration
    device_map: str = "auto"
    torch_dtype: str = "float16"
    trust_remote_code: bool = True

    # Logging configuration
    output_dir: str = "./scu2_production_output"
    logging_dir: str = "./scu2_production_output/logs"
    logging_first_step: bool = True
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    # Production flags
    run_validation: bool = True
    run_training: bool = True
    local_rank: int = -1
    deepspeed: Optional[str] = None

    # T-SCU specific settings
    thermodynamic_logging: bool = True
    power_monitor_type: str = "apple_silicon"  # "apple_silicon", "nvidia", "generic"
    emergency_shutdown_temp: float = 90.0
    adaptive_batch_sizing: bool = True
    energy_optimization_level: str = "balanced"  # "conservative", "balanced", "aggressive"


@dataclass
class HuggingFaceModelConfig:
    """HuggingFace model card and metadata configuration"""

    # Model metadata
    model_name: str = "T-SCU-Qwen3-4B"
    model_type: str = "qwen2"
    architecture: str = "Qwen2ForCausalLM"

    # Training metadata
    base_model: str = "Qwen/Qwen2.5-3B"
    training_method: str = "Thermodynamic SCU (T-SCU)"
    training_data: str = "WikiText-103"
    training_steps: int = 5000

    # Performance metrics
    bits_per_token_baseline: Optional[float] = None
    bits_per_token_scu: Optional[float] = None
    improvement_percentage: Optional[float] = None
    energy_efficiency_improvement: Optional[float] = None

    # Technical specifications
    parameter_count: int = 4000000000  # 4B parameters
    context_length: int = 2048
    vocab_size: int = 152064  # Qwen vocab size

    # Hardware requirements
    min_memory_gb: int = 12
    recommended_memory_gb: int = 16
    apple_silicon_optimized: bool = True

    # Tags for HuggingFace
    tags: List[str] = field(default_factory=lambda: [
        "thermodynamic-control",
        "energy-efficient",
        "qwen2",
        "lora",
        "apple-silicon",
        "t-scu",
        "information-theory",
        "control-theory"
    ])

    # License information
    license: str = "apache-2.0"
    library_name: str = "transformers"
    tasks: List[str] = field(default_factory=lambda: ["text-generation"])

    # Model card content
    model_card_content: str = """
# Thermodynamic SCU Qwen3-4B

## Model Description
Qwen3-4B fine-tuned with Thermodynamic Shannon Control Unit (T-SCU), an advanced energy-aware training system that optimizes the trade-off between information entropy and physical thermodynamic entropy during neural network training.

## Training Method
- **Base Model**: Qwen/Qwen2.5-3B
- **Training Method**: T-SCU with multi-scale entropy control
- **Regularization**: Adaptive λ control targeting optimal information ratio
- **Energy Optimization**: Real-time power monitoring and efficiency control
- **Hardware**: Apple Silicon optimized

## Performance Metrics
- **Bits per Token Improvement**: {bits_per_token_improvement:.2f}%
- **Energy Efficiency**: {energy_efficiency_improvement:.2f}x better than baseline
- **Thermal Efficiency**: Optimized for Apple Silicon hardware

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

# Load T-SCU adapter
model = PeftModel.from_pretrained(base_model, "your-username/T-SCU-Qwen3-4B")
```

## Hardware Requirements
- **Minimum**: 12GB RAM (Apple Silicon recommended)
- **Recommended**: 16GB+ RAM (Apple Silicon M1 Pro/Max or newer)
- **Power Consumption**: 15-25W during training

## Technical Details
- **Architecture**: Qwen2 with LoRA adapters
- **Context Length**: 2048 tokens
- **Training Data**: WikiText-103
- **Training Steps**: 5000 steps
- **Energy Monitoring**: Real-time power and thermal control

## Citation
If you use this model, please cite:
```
@model{T-SCU-Qwen3-4B,
  title={{Thermodynamic SCU Qwen3-4B}},
  author={{Your Name}},
  year={{2024}},
  url={{https://huggingface.co/your-username/T-SCU-Qwen3-4B}}
}
```
"""