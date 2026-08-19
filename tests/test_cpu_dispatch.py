from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_report() -> dict:
    avx512 = [
        "AVX512F",
        "AVX512CD",
        "AVX512_KNL",
        "AVX512_KNM",
        "AVX512_SKX",
        "AVX512_CLX",
        "AVX512_CNL",
        "AVX512_ICL",
        "AVX512_SPR",
    ]
    features = {name: False for name in avx512}
    features.update({"AVX2": True, "FMA3": True})
    pools = [
        {
            "component": component,
            "filepath": f"/site-packages/{component}.libs/libscipy_openblas.so",
            "architecture": "Haswell",
            "num_threads": 1,
            "version": "0.3.30",
        }
        for component in ("numpy", "scipy")
    ]
    return {
        "numpy_version": "2.3.5",
        "physical_cpu_flags": ["avx2", "fma"],
        "numpy_recognized_avx512_dispatch": avx512,
        "numpy_cpu_features_effective": features,
        "NPY_DISABLE_CPU_FEATURES": ",".join(sorted(avx512)),
        "OPENBLAS_CORETYPE": "Haswell",
        "openblas_libraries": pools,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def test_discovery_selects_only_recognized_avx512_dispatch_names():
    discovery = _load_script("determine_numpy_avx512_features.py")
    assert discovery.select_avx512_dispatch_features(
        ["SSE3", "AVX2", "AVX512_SKX", "AVX512F", "FMA3"]
    ) == ["AVX512F", "AVX512_SKX"]


def test_pure_cpu_dispatch_gate_accepts_normalized_snapshot_and_rejects_failures():
    gate = _load_script("check_canonical_cpu_dispatch.py")
    baseline = _normalized_report()
    assert gate.evaluate_normalization(baseline) == []

    cases = []
    missing_numpy_env = copy.deepcopy(baseline)
    missing_numpy_env["NPY_DISABLE_CPU_FEATURES"] = None
    cases.append(missing_numpy_env)
    wrong_openblas = copy.deepcopy(baseline)
    wrong_openblas["openblas_libraries"][1]["architecture"] = "SkylakeX"
    cases.append(wrong_openblas)
    avx512_active = copy.deepcopy(baseline)
    avx512_active["numpy_cpu_features_effective"]["AVX512F"] = True
    cases.append(avx512_active)
    avx2_disabled = copy.deepcopy(baseline)
    avx2_disabled["numpy_cpu_features_effective"]["AVX2"] = False
    cases.append(avx2_disabled)
    threaded = copy.deepcopy(baseline)
    threaded["openblas_libraries"][0]["num_threads"] = 2
    cases.append(threaded)
    for report in cases:
        assert gate.evaluate_normalization(report)


def test_live_gate_fails_without_normalization_and_does_not_touch_scientific_files(tmp_path: Path):
    sentinels = [
        ROOT / "data/split_manifest.csv",
        ROOT / "models/activation_logistic_train_split.json",
        ROOT / "traces/action_traces/portfolio_batch_test_primary_cost_contract_trace.json",
    ]
    before = {path: _sha256(path) for path in sentinels}
    environment = os.environ.copy()
    environment.pop("NPY_DISABLE_CPU_FEATURES", None)
    environment.pop("OPENBLAS_CORETYPE", None)
    output = tmp_path / "cpu_dispatch_gate.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_canonical_cpu_dispatch.py"),
            "--json-output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert json.loads(output.read_text(encoding="utf-8"))["normalization_status"] == "FAIL"
    assert {path: _sha256(path) for path in sentinels} == before


def test_cpu_dispatch_contract_preserves_tolerances_and_keeps_selection_out_of_gate():
    contract = json.loads(
        (ROOT / "results/reproduction/regression_contract.json").read_text(encoding="utf-8")
    )
    canonical = contract["comparison_classes"]["numerical_tolerance"]["canonical"]
    assert canonical["default_abs"] == 1e-12
    assert canonical["default_rel"] == 1e-12
    assert contract["comparison_classes"]["canonical_byte_identity"]["enforced_by"] == (
        "scripts/check_clean_reproduction.py --mode canonical"
    )
    cpu = contract["canonical_platform"]["cpu_dispatch_contract"]
    assert cpu["openblas_coretype"] == "Haswell"
    assert cpu["runtime_gate_must_precede_scientific_execution"] is True
    gate_source = (ROOT / "scripts/check_canonical_cpu_dispatch.py").read_text(encoding="utf-8")
    assert "selected_C" not in gate_source
    assert "0.001" not in gate_source


def test_canonical_workflow_and_docker_normalize_before_scientific_python():
    workflow = (ROOT / ".github/workflows/reproduce.yml").read_text(encoding="utf-8")
    discovery = workflow.index("determine_numpy_avx512_features.py")
    gate = workflow.index("check_canonical_cpu_dispatch.py")
    diagnostics = workflow.index("collect_runtime_diagnostics.py")
    clean_copy = workflow.index("check_clean_reproduction.py --mode canonical")
    reproduction = workflow.index("run_full_reproduction.py")
    assert discovery < gate < diagnostics < clean_copy < reproduction
    assert "OPENBLAS_CORETYPE: Haswell" in workflow
    assert '--github-env "$GITHUB_ENV"' in workflow

    dockerfile = (ROOT / "Dockerfile.canonical").read_text(encoding="utf-8")
    assert "OPENBLAS_CORETYPE=Haswell" in dockerfile
    assert dockerfile.index("determine_numpy_avx512_features.py") < dockerfile.index(
        "check_canonical_cpu_dispatch.py"
    ) < dockerfile.index("check_clean_reproduction.py")

    environment_lock = (ROOT / "environment-lock.yml").read_text(encoding="utf-8")
    assert "OPENBLAS_CORETYPE: Haswell" in environment_lock
    assert "dynamically_disable_recognized_avx512_targets" in environment_lock
