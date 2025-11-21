from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, validator


class TrainingConfig(BaseModel):
    """Minimal training configuration for the SCU API."""

    base_model: str = Field("meta-llama/Llama-3.2-1B", description="HF model id")
    train_data: str = Field("data/train.txt", description="Path to training data")
    adapter_out: str = Field("adapters/scu_adapter", description="Where to save adapter")

    # Control parameters
    target_s: float = Field(0.01, ge=0.0, le=1.0)
    kp: float = Field(0.8)
    ki: float = Field(0.15)
    deadband: float = Field(0.002)
    lambda_init: float = Field(1.0)
    lambda_min: float = Field(1e-4)
    lambda_max: float = Field(2.0)

    # Training parameters
    prior_sigma: float = Field(0.01)
    epochs: int = Field(1, ge=1)
    steps: Optional[int] = Field(None, ge=1)
    batch_size: int = Field(1, ge=1)
    lr: float = Field(5e-5, gt=0)
    block_size: int = Field(1024, ge=64)
    gradient_accumulation_steps: int = Field(4, ge=1)
    fp16: bool = Field(True)
    seed: int = Field(42)
    max_texts: Optional[int] = Field(None, ge=1)
    use_unsloth: bool = Field(False, description="Use Unsloth FastLanguageModel loader")

    log_csv: Optional[str] = Field(None, description="Optional CSV metrics log path")

    class Config:
        validate_assignment = True

    @validator("adapter_out")
    def _expand_adapter_out(cls, v: str) -> str:
        return str(Path(v).expanduser())

    @property
    def target_steps(self) -> int:
        if self.steps:
            return self.steps
        # Assume ~1 step per chunk per epoch if steps not specified; caller sets later.
        return 0
