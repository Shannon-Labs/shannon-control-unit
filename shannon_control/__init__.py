"""
Thermodynamic Shannon Control Unit 2.0 (T-SCU)

Revolutionary energy-aware training system that bridges Shannon information entropy
with physical thermodynamic entropy for optimal computational efficiency.

Core Innovation: Controls both information entropy AND physical energy dissipation
during neural network training, pushing toward Landauer limit optimization.

Supports PyTorch/CUDA (DGX Spark, NVIDIA GPUs) and native Apple MLX backends.
"""

__version__ = "2.0.0"
__author__ = "Hunter Bown, Shannon Labs"

from .core.simplified_controller import SimplifiedSCU, TrainingState, ControlAction
from .core.multiscale_entropy import MultiScaleEntropyAnalyzer

# Core control functions (framework-agnostic)
from .control import (
    update_lambda,
    calculate_data_bpt,
    calculate_s_ratio,
    calculate_param_bpt,
    calculate_param_bpt_from_stats,
)

# MLX adapters (available even if MLX not installed)
from .mlx_adapters import (
    is_mlx_available,
    get_mlx_device_info,
)

__all__ = [
    # Core controllers
    "SimplifiedSCU",
    "TrainingState",
    "ControlAction",
    "MultiScaleEntropyAnalyzer",
    # Control functions
    "update_lambda",
    "calculate_data_bpt",
    "calculate_s_ratio",
    "calculate_param_bpt",
    "calculate_param_bpt_from_stats",
    # MLX utilities
    "is_mlx_available",
    "get_mlx_device_info",
    # CUDA utilities
    "is_cuda_available",
    "get_cuda_module",
]


def is_cuda_available() -> bool:
    """Check if CUDA is available for training.

    Returns:
        True if PyTorch with CUDA support is installed.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_cuda_module():
    """
    Lazily import the CUDA module to avoid import errors on non-CUDA systems.

    Returns:
        The shannon_control.cuda module if CUDA is available, None otherwise.

    Example:
        cuda_module = get_cuda_module()
        if cuda_module:
            engine = cuda_module.CUDATrainingEngine(config)
    """
    if is_cuda_available():
        from . import cuda
        return cuda
    return None


def get_mlx_module():
    """
    Lazily import the MLX module to avoid import errors on non-Apple systems.

    Returns:
        The shannon_control.mlx module if MLX is available, None otherwise.

    Example:
        mlx_module = get_mlx_module()
        if mlx_module:
            engine = mlx_module.MLXTrainingEngine(config)
    """
    if is_mlx_available():
        from . import mlx
        return mlx
    return None