"""Compatibility shim for legacy `scu` imports.

This project primarily lives under the `shannon_control` package, but several
scripts and examples still import `scu.*`. To keep those entrypoints working
without modifying every caller, we re-export the public modules from
`shannon_control` here.
"""

from shannon_control.control import *  # noqa: F401,F403
from shannon_control import control  # noqa: F401
from shannon_control import data  # noqa: F401
from shannon_control import metrics  # noqa: F401
from shannon_control import mpc_controller  # noqa: F401

__all__ = []
__all__ += control.__all__ if hasattr(control, "__all__") else []
__all__ += metrics.__all__ if hasattr(metrics, "__all__") else []
