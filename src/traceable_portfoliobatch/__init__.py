"""Reproducible utilities for the traceable batch decision layer."""

import platform
import sys


def _force_nonblocking_windows_platform_fallback() -> None:
    """Avoid the CPython 3.12 WMI probe, which can block indefinitely.

    The standard-library ``platform`` module already falls back to
    ``sys.getwindowsversion`` and processor environment variables when WMI
    raises ``OSError``.  For repository commands on Windows, select that
    built-in fallback before NumPy/SciPy request ``platform.machine()``.
    """
    if sys.platform != "win32" or not hasattr(platform, "_wmi_query"):
        return

    def _wmi_disabled(*_args: object) -> object:
        raise OSError("WMI platform probe disabled for deterministic CLI startup")

    platform._wmi_query = _wmi_disabled  # type: ignore[attr-defined]


_force_nonblocking_windows_platform_fallback()

__version__ = "0.45a3.dev0"
