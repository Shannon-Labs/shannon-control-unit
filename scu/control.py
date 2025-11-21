"""Thin wrapper that forwards to `shannon_control.control`.

Keeping this module allows legacy imports like `from scu import control` and
`from scu.control import update_lambda` to continue working while the canonical
implementation lives under `shannon_control`.
"""

from shannon_control.control import *  # noqa: F401,F403
