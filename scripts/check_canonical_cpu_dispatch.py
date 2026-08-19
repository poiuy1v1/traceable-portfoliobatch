from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any


THREAD_ENV_NAMES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
EXPECTED_NUMPY_VERSION = "2.3.5"
EXPECTED_OPENBLAS_CORE = "Haswell"


def _physical_cpu_flags() -> list[str]:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() in {"flags", "features"}:
                return sorted({flag.lower() for flag in value.split()})
    return []


def _lscpu() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["lscpu", "--json"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return {"status": "UNAVAILABLE", "exit_code": None, "stdout": "", "stderr": ""}
    parsed: Any = None
    if completed.returncode == 0:
        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError:
            parsed = None
    return {
        "status": "PASS" if completed.returncode == 0 and parsed is not None else "FAIL",
        "exit_code": completed.returncode,
        "parsed": parsed,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _physical_class(flags: list[str]) -> str:
    if any(flag.startswith("avx512") for flag in flags):
        return "AVX512_CAPABLE"
    if "avx2" in flags:
        return "AVX2_NO_AVX512"
    return "OTHER_X64"


def _component(filepath: str) -> str:
    lowered = filepath.replace("\\", "/").lower()
    if "numpy.libs/" in lowered:
        return "numpy"
    if "scipy.libs/" in lowered:
        return "scipy"
    return "other"


def build_report() -> dict[str, Any]:
    physical_flags = _physical_cpu_flags()

    import numpy as np
    import scipy
    from numpy._core import _multiarray_umath as umath
    from scipy.linalg import blas
    from threadpoolctl import threadpool_info

    # Load both independently bundled BLAS runtimes using tiny infrastructure-only
    # probes. No Paper13 module or scientific artifact is touched by this gate.
    matrix = np.ones((2, 2), dtype=np.float64)
    np.matmul(matrix, matrix)
    blas.dgemm(alpha=1.0, a=matrix, b=matrix)

    baseline = [str(value) for value in getattr(umath, "__cpu_baseline__", ())]
    dispatch = [str(value) for value in getattr(umath, "__cpu_dispatch__", ())]
    features = {
        str(key): bool(value)
        for key, value in dict(getattr(umath, "__cpu_features__", {})).items()
    }
    recognized_avx512 = sorted(name for name in dispatch if name.upper().startswith("AVX512"))

    pools: list[dict[str, Any]] = []
    for item in threadpool_info():
        if str(item.get("internal_api", "")).lower() != "openblas":
            continue
        filepath = str(item.get("filepath") or "")
        pools.append(
            {
                "component": _component(filepath),
                "user_api": item.get("user_api"),
                "internal_api": item.get("internal_api"),
                "prefix": item.get("prefix"),
                "filepath": filepath,
                "version": item.get("version"),
                "threading_layer": item.get("threading_layer"),
                "architecture": item.get("architecture"),
                "num_threads": item.get("num_threads"),
            }
        )

    return {
        "schema_version": "1.0",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "physical_cpu_flags": physical_flags,
        "physical_avx2": "avx2" in physical_flags,
        "physical_avx512": any(flag.startswith("avx512") for flag in physical_flags),
        "physical_cpu_class": _physical_class(physical_flags),
        "lscpu": _lscpu(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "numpy_cpu_baseline": baseline,
        "numpy_cpu_dispatch": dispatch,
        "numpy_cpu_features_effective": features,
        "numpy_recognized_avx512_dispatch": recognized_avx512,
        "numpy_avx512_effective": any(features.get(name, False) for name in recognized_avx512),
        "numpy_avx2_effective": bool(features.get("AVX2", False)),
        "numpy_fma3_effective": bool(features.get("FMA3", False)),
        "NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES"),
        "openblas_version": sorted({str(pool["version"]) for pool in pools if pool["version"]}),
        "openblas_architecture": sorted(
            {str(pool["architecture"]) for pool in pools if pool["architecture"]}
        ),
        "OPENBLAS_CORETYPE": os.environ.get("OPENBLAS_CORETYPE"),
        "openblas_threads": sorted(
            {int(pool["num_threads"]) for pool in pools if pool["num_threads"] is not None}
        ),
        "openblas_libraries": pools,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def evaluate_normalization(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("numpy_version") != EXPECTED_NUMPY_VERSION:
        errors.append(
            f"NumPy version {report.get('numpy_version')!r} != pinned {EXPECTED_NUMPY_VERSION!r}"
        )

    flags = {str(value).lower() for value in report.get("physical_cpu_flags", [])}
    if str(report.get("machine") or platform.machine()).lower() in {"x86_64", "amd64"} and "avx2" not in flags:
        errors.append("physical x86-64 host does not report AVX2 required by the Haswell target")

    recognized = [str(value) for value in report.get("numpy_recognized_avx512_dispatch", [])]
    if not recognized:
        errors.append("NumPy reported no recognized AVX512 dispatch targets")
    requested_text = report.get("NPY_DISABLE_CPU_FEATURES")
    requested_tokens = [] if requested_text is None else str(requested_text).split(",")
    if any(not token or not re.fullmatch(r"[A-Z0-9_]+", token) for token in requested_tokens):
        errors.append("NPY_DISABLE_CPU_FEATURES contains an empty or invalid token")
    if len(requested_tokens) != len(set(requested_tokens)):
        errors.append("NPY_DISABLE_CPU_FEATURES contains duplicate tokens")
    if set(requested_tokens) != set(recognized):
        errors.append(
            "NPY_DISABLE_CPU_FEATURES does not exactly match recognized AVX512 dispatch targets"
        )

    effective = {
        str(key): bool(value)
        for key, value in dict(report.get("numpy_cpu_features_effective", {})).items()
    }
    missing_effective = sorted(set(recognized) - set(effective))
    if missing_effective:
        errors.append(f"effective NumPy feature map is missing: {missing_effective}")
    active_avx512 = sorted(name for name in recognized if effective.get(name, False))
    if active_avx512:
        errors.append(f"AVX512 dispatch remains effective: {active_avx512}")
    if not effective.get("AVX2", False):
        errors.append("AVX2 is not effective under the Haswell target")
    if not effective.get("FMA3", False):
        errors.append("FMA3 is not effective under the Haswell target")

    if str(report.get("OPENBLAS_CORETYPE") or "").casefold() != EXPECTED_OPENBLAS_CORE.casefold():
        errors.append("OPENBLAS_CORETYPE is not Haswell")
    pools = list(report.get("openblas_libraries", []))
    if not pools:
        errors.append("threadpoolctl found no OpenBLAS runtime")
    components = {str(pool.get("component")) for pool in pools}
    for required in ("numpy", "scipy"):
        if required not in components:
            errors.append(f"threadpoolctl did not identify the {required} OpenBLAS runtime")
    for pool in pools:
        architecture = str(pool.get("architecture") or "")
        if not architecture:
            errors.append(f"OpenBLAS architecture is unavailable for {pool.get('filepath')}")
        elif architecture.casefold() != EXPECTED_OPENBLAS_CORE.casefold():
            errors.append(
                f"OpenBLAS architecture {architecture!r} is not Haswell for {pool.get('filepath')}"
            )
        if pool.get("num_threads") != 1:
            errors.append(
                f"OpenBLAS thread count {pool.get('num_threads')!r} is not 1 for {pool.get('filepath')}"
            )
        if not pool.get("version"):
            errors.append(f"OpenBLAS version is unavailable for {pool.get('filepath')}")

    for name in THREAD_ENV_NAMES:
        if str(report.get(name) or "") != "1":
            errors.append(f"{name} is not 1")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify canonical numerical CPU dispatch")
    parser.add_argument("--json-output", "--output", dest="json_output", type=Path)
    args = parser.parse_args()

    try:
        report = build_report()
        errors = evaluate_normalization(report)
    except Exception as exc:
        report = {
            "schema_version": "1.0",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
        errors = [f"runtime inspection failed: {type(exc).__name__}: {exc}"]
    report["normalization_status"] = "PASS" if not errors else "FAIL"
    report["normalization_errors"] = errors
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
