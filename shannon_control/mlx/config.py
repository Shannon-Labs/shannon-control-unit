"""
MLX Training Configuration for Shannon Control Unit

Provides a comprehensive configuration dataclass for native MLX training
with SCU control integration.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class MLXTrainingConfig:
    """
    Configuration for MLX-native SCU training.

    This configuration covers:
    - Model selection and LoRA parameters
    - SCU PI controller settings
    - Training hyperparameters
    - Apple Silicon optimizations
    - Logging and output paths

    Example:
        config = MLXTrainingConfig(
            base_model="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
            train_data="data/train.jsonl",
            target_s=0.01,
            steps=1000
        )
    """

    # ===== Model Configuration =====
    base_model: str = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
    """HuggingFace model ID or path. Use mlx-community models for best performance."""

    # ===== Data Configuration =====
    train_data: str = "data/train.jsonl"
    """Path to training data (.jsonl with 'text' field or .txt)"""

    val_data: Optional[str] = None
    """Path to validation data (optional)"""

    block_size: int = 1024
    """Maximum sequence length for training"""

    # ===== Training Hyperparameters =====
    batch_size: int = 2
    """Batch size per step. MLX can handle larger batches than PyTorch MPS."""

    gradient_accumulation_steps: int = 4
    """Gradient accumulation steps. Effective batch = batch_size * grad_accum"""

    lr: float = 2e-4
    """Learning rate for AdamW optimizer"""

    steps: int = 1000
    """Total training steps"""

    warmup_steps: int = 100
    """Number of warmup steps for learning rate scheduler"""

    seed: int = 42
    """Random seed for reproducibility"""

    # ===== LoRA Configuration =====
    lora_r: int = 16
    """LoRA rank (dimension of low-rank matrices)"""

    lora_alpha: int = 32
    """LoRA alpha (scaling factor). Typically 2x lora_r."""

    lora_dropout: float = 0.05
    """Dropout rate for LoRA layers"""

    lora_layers: int = 16
    """Number of model layers to apply LoRA to"""

    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    """Target modules for LoRA adaptation"""

    train_full_params: bool = False
    """Train full model parameters instead of LoRA (experimental)."""

    # ===== SCU Control Parameters =====
    target_s: float = 0.01
    """Target S ratio (information ratio). 0.01 = 1% for small models."""

    kp: float = 0.8
    """Proportional gain for PI controller"""

    ki: float = 0.15
    """Integral gain for PI controller"""

    deadband: float = 0.002
    """Error threshold below which no control update occurs"""

    lambda_init: float = 1.0
    """Initial regularization strength"""

    lambda_min: float = 1e-4
    """Minimum lambda bound"""

    lambda_max: float = 2.0
    """Maximum lambda bound"""

    prior_sigma: float = 0.01
    """Prior standard deviation for parameter BPT calculation"""

    control_frequency: int = 50
    """Apply SCU control every N steps"""

    param_scope: str = "lora"
    """Parameter scope for ParamBPT: "lora", "trainable", or "all"."""

    param_bpt_frequency: Optional[int] = None
    """ParamBPT update interval (defaults to control frequency for non-LoRA)."""

    # ===== MLX-Specific Configuration =====
    grad_checkpoint: bool = True
    """Enable gradient checkpointing for memory efficiency"""

    use_quantized_model: bool = True
    """Whether base model is quantized (4-bit). Affects memory usage."""

    # ===== Output Configuration =====
    adapter_out: str = "adapters/mlx_scu_adapter"
    """Directory to save trained LoRA adapter"""

    log_dir: Optional[str] = "logs/mlx_scu"
    """Directory for logs and metrics"""

    log_csv: Optional[str] = None
    """Path for CSV metrics log (optional)"""

    report_steps: int = 10
    """Log metrics every N steps"""

    eval_steps: int = 100
    """Run validation every N steps (if val_data provided)"""

    save_steps: int = 500
    """Save checkpoint every N steps"""

    # ===== Power Monitoring =====
    enable_power_monitoring: bool = True
    """Enable Apple Silicon power and thermal monitoring"""

    def __post_init__(self):
        """Validate and expand paths after initialization."""
        self.adapter_out = str(Path(self.adapter_out).expanduser())
        self.train_data = str(Path(self.train_data).expanduser())

        if self.val_data:
            self.val_data = str(Path(self.val_data).expanduser())
        if self.log_dir:
            self.log_dir = str(Path(self.log_dir).expanduser())
        if self.log_csv:
            self.log_csv = str(Path(self.log_csv).expanduser())

        if self.train_full_params and self.param_scope == "lora":
            self.param_scope = "trainable"

    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "base_model": self.base_model,
            "train_data": self.train_data,
            "val_data": self.val_data,
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "lr": self.lr,
            "steps": self.steps,
            "warmup_steps": self.warmup_steps,
            "seed": self.seed,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_layers": self.lora_layers,
            "lora_target_modules": self.lora_target_modules,
            "train_full_params": self.train_full_params,
            "target_s": self.target_s,
            "kp": self.kp,
            "ki": self.ki,
            "deadband": self.deadband,
            "lambda_init": self.lambda_init,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "prior_sigma": self.prior_sigma,
            "control_frequency": self.control_frequency,
            "param_scope": self.param_scope,
            "param_bpt_frequency": self.param_bpt_frequency,
            "grad_checkpoint": self.grad_checkpoint,
            "adapter_out": self.adapter_out,
            "log_dir": self.log_dir,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MLXTrainingConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Pre-configured profiles for common use cases
DEEPSEEK_R1_1_5B_CONFIG = MLXTrainingConfig(
    base_model="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
    target_s=0.01,
    lora_r=16,
    lora_alpha=32,
    batch_size=2,
    steps=1000,
)

LLAMA_3_2_1B_CONFIG = MLXTrainingConfig(
    base_model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    target_s=0.01,
    lora_r=16,
    lora_alpha=32,
    batch_size=4,
    steps=500,
)

LLAMA_3_2_3B_CONFIG = MLXTrainingConfig(
    base_model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    target_s=0.03,  # Higher target for larger model
    lora_r=16,
    lora_alpha=32,
    batch_size=1,
    gradient_accumulation_steps=8,
    steps=1000,
)
