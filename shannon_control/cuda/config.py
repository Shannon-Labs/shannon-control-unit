"""
CUDA Training Configuration for Shannon Control Unit

Configuration dataclass for PyTorch/CUDA training with SCU control,
optimized for NVIDIA DGX Spark hardware.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class CUDATrainingConfig:
    """
    Configuration for CUDA-native SCU training.

    Covers model selection, LoRA parameters, SCU PI controller settings,
    training hyperparameters, and DGX Spark optimizations.

    Example:
        config = CUDATrainingConfig(
            base_model="Qwen/Qwen3.5-27B",
            train_data="data/fineweb_edu_1gb.jsonl",
            target_s=0.04,
            steps=2000,
        )
    """

    # ===== Model Configuration =====
    base_model: str = "Qwen/Qwen3.5-27B"
    """HuggingFace model ID or local path."""

    # ===== Data Configuration =====
    train_data: str = "data/fineweb_edu_1gb.jsonl"
    """Path to training data (.jsonl with 'text' field)."""

    val_data: Optional[str] = None
    """Path to validation data (optional)."""

    block_size: int = 2048
    """Maximum sequence length for training."""

    # ===== Training Hyperparameters =====
    batch_size: int = 2
    """Per-device batch size."""

    gradient_accumulation_steps: int = 8
    """Gradient accumulation steps. Effective batch = batch_size * grad_accum."""

    lr: float = 2e-5
    """Learning rate for AdamW optimizer."""

    steps: int = 2000
    """Total training steps."""

    warmup_ratio: float = 0.05
    """Fraction of steps used for warmup."""

    seed: int = 42
    """Random seed for reproducibility."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    # ===== LoRA Configuration =====
    lora_r: int = 32
    """LoRA rank."""

    lora_alpha: int = 64
    """LoRA alpha (scaling factor). Typically 2x lora_r."""

    lora_dropout: float = 0.05
    """Dropout rate for LoRA layers."""

    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    """Target modules for LoRA adaptation."""

    # ===== SCU Control Parameters =====
    target_s: float = 0.04
    """Target S ratio. 0.04 = 4% for xlarge (7B+) models."""

    kp: float = 0.9
    """Proportional gain for PI controller."""

    ki: float = 0.18
    """Integral gain for PI controller."""

    deadband: float = 0.003
    """Error threshold below which no control update occurs."""

    lambda_init: float = 1.0
    """Initial regularization strength."""

    lambda_min: float = 1e-4
    """Minimum lambda bound."""

    lambda_max: float = 2.0
    """Maximum lambda bound."""

    prior_sigma: float = 0.01
    """Prior standard deviation for parameter BPT calculation."""

    control_frequency: int = 50
    """Apply SCU control every N steps."""

    # ===== CUDA / DGX Spark Configuration =====
    dtype: str = "bf16"
    """Training precision: 'bf16', 'fp16', or 'fp32'. bf16 recommended for Blackwell."""

    use_flash_attention: bool = True
    """Enable Flash Attention 2 (requires flash-attn package)."""

    gradient_checkpointing: bool = True
    """Enable gradient checkpointing for memory efficiency."""

    use_4bit: bool = False
    """Load base model in 4-bit (QLoRA). False = full precision LoRA."""

    use_8bit: bool = False
    """Load base model in 8-bit."""

    device_map: str = "auto"
    """Device map for model loading."""

    dataloader_num_workers: int = 4
    """Number of dataloader workers."""

    dataloader_pin_memory: bool = True
    """Pin memory for faster data transfer."""

    tf32: bool = True
    """Enable TF32 for matmuls (Blackwell native)."""

    compile_model: bool = False
    """Use torch.compile() for potential speedups."""

    # ===== Output Configuration =====
    adapter_out: str = "adapters/qwen35_27b_scu"
    """Directory to save trained LoRA adapter."""

    log_dir: Optional[str] = "logs/dgx_spark_scu"
    """Directory for logs and metrics."""

    log_csv: Optional[str] = None
    """Path for CSV metrics log (auto-generated if None)."""

    report_steps: int = 25
    """Log metrics every N steps."""

    eval_steps: int = 200
    """Run validation every N steps (if val_data provided)."""

    save_steps: int = 500
    """Save checkpoint every N steps."""

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

        if self.log_csv is None and self.log_dir:
            self.log_csv = str(Path(self.log_dir) / "metrics.csv")

    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def warmup_steps(self) -> int:
        """Compute warmup steps from ratio."""
        return int(self.steps * self.warmup_ratio)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict) -> "CUDATrainingConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Pre-configured profiles

QWEN35_27B_DGX_SPARK = CUDATrainingConfig(
    base_model="Qwen/Qwen3.5-27B",
    target_s=0.04,
    kp=0.9,
    ki=0.18,
    lora_r=32,
    lora_alpha=64,
    batch_size=2,
    gradient_accumulation_steps=8,
    block_size=2048,
    steps=2000,
    dtype="bf16",
    use_4bit=False,
)

QWEN35_27B_DGX_SPARK_QLORA = CUDATrainingConfig(
    base_model="Qwen/Qwen3.5-27B",
    target_s=0.04,
    kp=0.9,
    ki=0.18,
    lora_r=64,
    lora_alpha=128,
    batch_size=4,
    gradient_accumulation_steps=4,
    block_size=4096,
    steps=2000,
    dtype="bf16",
    use_4bit=True,
)

LLAMA4_8B_DGX_SPARK = CUDATrainingConfig(
    base_model="meta-llama/Llama-4-8B",
    target_s=0.03,
    kp=0.8,
    ki=0.15,
    lora_r=32,
    lora_alpha=64,
    batch_size=4,
    gradient_accumulation_steps=4,
    block_size=2048,
    steps=2000,
    dtype="bf16",
)
