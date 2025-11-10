"""Shannon Control Unit - Adaptive regularization via control theory."""

from .control import (
    update_lambda,
    calculate_param_bpt,
    calculate_data_bpt,
    calculate_s_ratio,
    ema
)

from .data import (
    tokenize_and_chunk,
    load_texts_from_file,
    create_data_iterator,
    prepare_dataset
)

from .mpc_controller import (
    ControllerBase,
    MPCController,
    PIController
)

from .metrics import (
    MetricBase,
    S_Ratio,
    AttentionEntropy,
    GradientOrthogonality
)

__version__ = "1.0.0"
__all__ = [
    "update_lambda",
    "calculate_param_bpt", 
    "calculate_data_bpt",
    "calculate_s_ratio",
    "ema",
    "tokenize_and_chunk",
    "load_texts_from_file",
    "create_data_iterator",
    "prepare_dataset",
    "ControllerBase",
    "MPCController",
    "PIController",
    "MetricBase",
    "S_Ratio",
    "AttentionEntropy",
    "GradientOrthogonality"
]