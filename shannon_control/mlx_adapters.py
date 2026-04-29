"""
MLX Adapters for Shannon Control Unit

Provides MLX-specific utilities for extracting LoRA parameters and computing
SCU metrics. These adapters bridge the gap between MLX models and the
framework-agnostic SCU control functions.

Usage:
    from shannon_control.mlx_adapters import calculate_param_bpt_mlx

    param_bpt = calculate_param_bpt_mlx(model, tokens_per_epoch=100000)
"""

from typing import Dict, Tuple, Iterable

# Import control functions (framework-agnostic)
from .control import calculate_param_bpt_from_stats


def _iter_mlx_params(model, scope: str) -> Iterable[Tuple[str, "mx.array"]]:
    """Yield (name, param) pairs for the requested scope."""
    try:
        from mlx.utils import tree_flatten
    except ImportError as exc:
        raise ImportError(
            "MLX is required for MLX adapters. "
            "Install with: pip install mlx mlx-lm"
        ) from exc

    if scope not in {"lora", "trainable", "all"}:
        raise ValueError(f"Unsupported param scope: {scope}")

    if scope == "all":
        params = model.parameters() if hasattr(model, "parameters") else model.trainable_parameters()
    else:
        params = model.trainable_parameters() if hasattr(model, "trainable_parameters") else model.parameters()

    flat_params = dict(tree_flatten(params))
    for name, param in flat_params.items():
        if scope == "lora" and "lora" not in name.lower():
            continue
        yield name, param


def _to_float(value) -> float:
    """Best-effort conversion of MLX scalars to float."""
    try:
        return float(value)
    except TypeError:
        return float(value.item())


def extract_lora_params_mlx(model) -> Dict[str, "np.ndarray"]:
    """Extract LoRA parameters from an MLX model as NumPy arrays.

    Traverses the MLX model's trainable parameters and extracts any
    parameters with 'lora' in their name, converting them to NumPy
    arrays for framework-agnostic processing.

    Args:
        model: MLX model with LoRA layers applied

    Returns:
        Dict mapping parameter names to NumPy arrays

    Example:
        >>> lora_params = extract_lora_params_mlx(model)
        >>> for name, arr in lora_params.items():
        ...     print(f"{name}: shape={arr.shape}")
    """
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        import numpy as np
    except ImportError:
        raise ImportError(
            "MLX is required for MLX adapters. "
            "Install with: pip install mlx mlx-lm"
        )

    lora_params = {}

    # Get trainable parameters from model
    if hasattr(model, 'trainable_parameters'):
        params = model.trainable_parameters()
    elif hasattr(model, 'parameters'):
        params = model.parameters()
    else:
        raise ValueError("Model does not have parameters() or trainable_parameters() method")

    # Flatten the nested parameter dict
    flat_params = dict(tree_flatten(params))

    for name, param in flat_params.items():
        if "lora" in name.lower():
            # Force evaluation of lazy MLX array
            mx.eval(param)
            # Convert to NumPy for processing
            lora_params[name] = np.array(param)

    return lora_params


def get_param_stats_mlx(model, scope: str = "lora") -> Tuple[float, int]:
    """Get parameter statistics from an MLX model.

    Computes the sum of squared weights and total parameter count for
    the requested scope without materializing full arrays on CPU.

    Args:
        model: MLX model
        scope: "lora", "trainable", or "all"

    Returns:
        Tuple of (sum_of_squares, param_count)
    """
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise ImportError(
            "MLX is required for MLX adapters. "
            "Install with: pip install mlx mlx-lm"
        ) from exc

    param_sum = 0.0
    param_count = 0

    for _, param in _iter_mlx_params(model, scope):
        sumsq = mx.sum(mx.square(param))
        mx.eval(sumsq)
        param_sum += _to_float(sumsq)
        param_count += param.size

    return param_sum, param_count


def get_lora_param_stats_mlx(model) -> Tuple[float, int]:
    """Get LoRA parameter statistics from an MLX model.

    Computes the sum of squared weights and total parameter count
    for all LoRA parameters in the model.

    Args:
        model: MLX model with LoRA layers applied

    Returns:
        Tuple of (sum_of_squares, param_count)

    Example:
        >>> param_sum, param_count = get_lora_param_stats_mlx(model)
        >>> print(f"LoRA params: {param_count}, sum_sq: {param_sum:.4f}")
    """
    return get_param_stats_mlx(model, scope="lora")


def calculate_param_bpt_mlx(
    model,
    tokens_per_epoch: int,
    sigma: float = 0.01,
    param_scope: str = "lora",
) -> float:
    """Calculate Parameter BPT for an MLX model.

    This is the MLX equivalent of calculate_param_bpt() from control.py.
    It extracts parameters from the requested scope and computes the
    parameter bits-per-token using the framework-agnostic formula.

    ParamBPT = (1 / (N * ln(2))) * Σ(w² / (2σ²))

    Args:
        model: MLX model
        tokens_per_epoch: Fixed normalization constant (N)
        sigma: Prior standard deviation (default: 0.01)
        param_scope: "lora", "trainable", or "all"

    Returns:
        Parameter bits per token

    Example:
        >>> from mlx_lm import load
        >>> model, tokenizer = load("mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit")
        >>> param_bpt = calculate_param_bpt_mlx(model, tokens_per_epoch=100000)
    """
    param_sum, param_count = get_param_stats_mlx(model, scope=param_scope)

    if param_count == 0:
        return 1e-6

    return calculate_param_bpt_from_stats(param_sum, tokens_per_epoch, sigma)


def is_mlx_available() -> bool:
    """Check if MLX is available on this system.

    Returns:
        True if MLX and mlx-lm are installed and importable
    """
    try:
        import mlx.core
        import mlx_lm
        return True
    except ImportError:
        return False


def get_mlx_device_info() -> Dict[str, str]:
    """Get information about the MLX device (Apple Silicon).

    Returns:
        Dict with device information including chip model and memory
    """
    import platform
    import subprocess

    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip_model": "Unknown",
        "is_apple_silicon": False,
    }

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        info["is_apple_silicon"] = True
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                timeout=5
            )
            info["chip_model"] = result.stdout.strip()
        except Exception:
            pass

    return info
