from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import traceable_portfoliobatch  # noqa: E402,F401
import joblib  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import PIL  # noqa: E402
import pytest  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from threadpoolctl import threadpool_info  # noqa: E402

from check_canonical_cpu_dispatch import (  # noqa: E402
    build_report as build_cpu_dispatch_report,
    evaluate_normalization,
)


def _capture_config(func) -> str:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        func()
    return stream.getvalue()


def _git(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *command], cwd=ROOT, text=True, capture_output=True, check=False
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def build_diagnostics() -> dict[str, Any]:
    env_names = [
        "PYTHONHASHSEED",
        "SOURCE_DATE_EPOCH",
        "NPY_DISABLE_CPU_FEATURES",
        "OPENBLAS_CORETYPE",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MPLCONFIGDIR",
    ]
    try:
        dispatch = build_cpu_dispatch_report()
        dispatch_errors = evaluate_normalization(dispatch)
        dispatch["normalization_status"] = "PASS" if not dispatch_errors else "FAIL"
        dispatch["normalization_errors"] = dispatch_errors
    except Exception as exc:
        dispatch = {
            "normalization_status": "FAIL",
            "normalization_errors": [
                f"runtime inspection failed: {type(exc).__name__}: {exc}"
            ],
        }
    return {
        "schema_version": "2.0",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "libc": list(platform.libc_ver()),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__,
            "pillow": PIL.__version__,
            "pytest": pytest.__version__,
        },
        "numerical_runtime": {
            "threadpools": threadpool_info(),
            "numpy_config": _capture_config(np.show_config),
            "scipy_config": _capture_config(scipy.show_config),
        },
        "physical_host_capabilities": {
            "physical_cpu_flags": dispatch.get("physical_cpu_flags", []),
            "physical_avx2": dispatch.get("physical_avx2"),
            "physical_avx512": dispatch.get("physical_avx512"),
            "physical_cpu_class": dispatch.get("physical_cpu_class"),
            "lscpu": dispatch.get("lscpu"),
        },
        "effective_numerical_dispatch": {
            key: value
            for key, value in dispatch.items()
            if key
            not in {
                "physical_cpu_flags",
                "physical_avx2",
                "physical_avx512",
                "physical_cpu_class",
                "lscpu",
            }
        },
        "environment": {name: os.environ.get(name) for name in env_names},
        "git": {
            "head": _git(["rev-parse", "HEAD"]),
            "branch": _git(["branch", "--show-current"]),
            "status_porcelain": _git(["status", "--porcelain"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Paper13 runtime diagnostics")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = build_diagnostics()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
