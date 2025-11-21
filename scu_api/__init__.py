"""SCU Training API package (minimal runnable version).

This module exposes the FastAPI app at `scu_api.server.app` and a thin client
(`TrainingEngine`) that can be used programmatically without HTTP.
"""

from scu_api.config import TrainingConfig  # noqa: F401
from scu_api.training_engine import TrainingEngine  # noqa: F401

__all__ = ["TrainingConfig", "TrainingEngine"]
