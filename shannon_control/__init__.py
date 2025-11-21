"""
Thermodynamic Shannon Control Unit 2.0 (T-SCU)

Revolutionary energy-aware training system that bridges Shannon information entropy
with physical thermodynamic entropy for optimal computational efficiency.

Core Innovation: Controls both information entropy AND physical energy dissipation
during neural network training, pushing toward Landauer limit optimization.
"""

__version__ = "2.0.0"
__author__ = "Hunter Bown, Shannon Labs"

from .core.simplified_controller import SimplifiedSCU, TrainingState, ControlAction
from .core.multiscale_entropy import MultiScaleEntropyAnalyzer

__all__ = [
    "SimplifiedSCU",
    "TrainingState",
    "ControlAction",
    "MultiScaleEntropyAnalyzer"
]