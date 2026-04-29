"""
MLX Native Training Module for Shannon Control Unit

This package provides native Apple MLX framework support for SCU training,
enabling production-ready fine-tuning on Apple Silicon.

Key Components:
- SCUTrainingCallback: mlx-lm compatible callback with PI control
- MLXTrainingConfig: Configuration for MLX-native training
- MLXTrainingEngine: Complete training pipeline for MLX

Target Model: DeepSeek-R1-Distill-Qwen-1.5B (with <think> reasoning tags)

Usage:
    from shannon_control.mlx import MLXTrainingEngine, MLXTrainingConfig

    config = MLXTrainingConfig(
        base_model="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
        train_data="data/train.jsonl",
        target_s=0.01
    )
    engine = MLXTrainingEngine(config)
    engine.run()
"""

__version__ = "1.0.0"

from .callback import SCUTrainingCallback, SCUState
from .config import MLXTrainingConfig
from .engine import MLXTrainingEngine

__all__ = [
    "SCUTrainingCallback",
    "SCUState",
    "MLXTrainingConfig",
    "MLXTrainingEngine",
]
