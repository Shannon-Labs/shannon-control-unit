"""SCU Training API package (minimal runnable version).

This module exposes the FastAPI app at `scu_api.server.app` and a thin client
(`TrainingEngine`) that can be used programmatically without HTTP.
"""

from scu_api.config import TrainingConfig  # noqa: F401
from scu_api.client.sync_client import SCUClient  # noqa: F401

try:  # Avoid forcing heavy deps (torch/peft) on lightweight clients
    from scu_api.training_engine import TrainingEngine  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    TrainingEngine = None  # type: ignore

__all__ = ["TrainingConfig", "TrainingEngine", "SCUClient"]
