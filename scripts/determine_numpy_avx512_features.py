from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Iterable


EXPECTED_NUMPY_VERSION = "2.3.5"


def select_avx512_dispatch_features(features: Iterable[str]) -> list[str]:
    """Return a deterministic list of recognized AVX512 dispatch targets."""
    return sorted(
        {str(feature) for feature in features if str(feature).upper().startswith("AVX512")}
    )


def discover() -> dict[str, object]:
    # This is the one intentionally unnormalized NumPy process. It only discovers
    # the pinned wheel's recognized dispatch names and never imports Paper13 code.
    import numpy as np
    from numpy._core import _multiarray_umath as umath

    baseline = [str(value) for value in getattr(umath, "__cpu_baseline__", ())]
    dispatch = [str(value) for value in getattr(umath, "__cpu_dispatch__", ())]
    effective = {
        str(key): bool(value)
        for key, value in dict(getattr(umath, "__cpu_features__", {})).items()
    }
    disabled = select_avx512_dispatch_features(dispatch)
    return {
        "schema_version": "1.0",
        "purpose": "discover recognized NumPy AVX512 dispatch names before canonical execution",
        "numpy_version": np.__version__,
        "numpy_cpu_baseline": baseline,
        "numpy_cpu_dispatch_recognized": dispatch,
        "numpy_cpu_features_effective_in_discovery_process": effective,
        "recognized_avx512_dispatch_features": disabled,
        "NPY_DISABLE_CPU_FEATURES": ",".join(disabled),
        "preexisting_NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover the pinned NumPy wheel's recognized AVX512 dispatch names"
    )
    parser.add_argument("--github-env", type=Path)
    parser.add_argument("--shell-env", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--require-unset", action="store_true")
    args = parser.parse_args()

    payload = discover()
    errors: list[str] = []
    if payload["numpy_version"] != EXPECTED_NUMPY_VERSION:
        errors.append(
            f"NumPy version {payload['numpy_version']!r} != pinned {EXPECTED_NUMPY_VERSION!r}"
        )
    if args.require_unset and payload["preexisting_NPY_DISABLE_CPU_FEATURES"]:
        errors.append("NPY_DISABLE_CPU_FEATURES was already set in the discovery process")
    disabled = str(payload["NPY_DISABLE_CPU_FEATURES"])
    if not disabled:
        errors.append("the pinned NumPy runtime reported no recognized AVX512 dispatch names")
    elif any(not re.fullmatch(r"[A-Z0-9_]+", token) for token in disabled.split(",")):
        errors.append("the discovered AVX512 list contains an invalid environment token")
    payload["status"] = "PASS" if not errors else "FAIL"
    payload["errors"] = errors

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")

    if not errors:
        if args.github_env:
            with args.github_env.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(f"NPY_DISABLE_CPU_FEATURES={disabled}\n")
        if args.shell_env:
            args.shell_env.parent.mkdir(parents=True, exist_ok=True)
            args.shell_env.write_text(
                f"export NPY_DISABLE_CPU_FEATURES='{disabled}'\n", encoding="utf-8"
            )

    print(text, end="")
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
