"""
CUDA Training Backend for Shannon Control Unit

Provides PyTorch + CUDA training pipeline with SCU PI control integration,
optimized for NVIDIA DGX Spark (Grace Blackwell, 128GB unified memory).

Supports: Qwen 3.5, Llama 4, DeepSeek, and other HuggingFace models.
"""

from .config import CUDATrainingConfig
from .callback import SCUCUDACallback
from .engine import CUDATrainingEngine

__all__ = [
    "CUDATrainingConfig",
    "SCUCUDACallback",
    "CUDATrainingEngine",
]
