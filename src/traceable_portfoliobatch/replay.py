from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import write_json
from .modeling import preprocess_from_artifact
from .policies import PolicyConfig, metrics, select_batch
from .tracing import records, sha256


def _nested_differences(
    expected: Any,
    observed: Any,
    *,
    path: str = "root",
    absolute_tolerance: float = 1e-12,
) -> tuple[list[str], float]:
    """Compare archived trace payloads exactly, with bounded float round-trip noise."""
    errors: list[str] = []
    max_numeric_difference = 0.0

    def walk(left: Any, right: Any, current: str) -> None:
        nonlocal max_numeric_difference
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                errors.append(
                    f"{current}: key mismatch {sorted(set(left).symmetric_difference(right))}"
                )
                return
            for key in sorted(left):
                walk(left[key], right[key], f"{current}.{key}")
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                errors.append(f"{current}: list length {len(left)} != {len(right)}")
                return
            for index, (left_value, right_value) in enumerate(zip(left, right)):
                walk(left_value, right_value, f"{current}[{index}]")
            return
        if isinstance(left, bool) or isinstance(right, bool):
            if type(left) is not type(right) or left != right:
                errors.append(f"{current}: {left!r} != {right!r}")
            return
        if isinstance(left, Real) and isinstance(right, Real):
            left_float = float(left)
            right_float = float(right)
            if not (math.isfinite(left_float) and math.isfinite(right_float)):
                if left_float != right_float:
                    errors.append(f"{current}: non-finite {left!r} != {right!r}")
                return
            difference = abs(left_float - right_float)
            max_numeric_difference = max(max_numeric_difference, difference)
            if difference > absolute_tolerance:
                errors.append(
                    f"{current}: {left!r} != {right!r}; abs_diff={difference:.6g}"
                )
            return
        if left != right:
            errors.append(f"{current}: {left!r} != {right!r}")

    walk(expected, observed, path)
    return errors, max_numeric_difference


def _config_lock_differences(trace: dict[str, Any], locked: dict[str, Any]) -> list[str]:
    config = trace["config"]
    policy = locked["policy"]
    errors: list[str] = []

    def require_equal(name: str, observed: Any, expected: Any) -> None:
        differences, _ = _nested_differences(
            expected, observed, path=f"config.{name}", absolute_tolerance=1e-12
        )
        errors.extend(differences)

    batch_size = int(config["batch_size"])
    require_equal("batch_size_is_locked", batch_size in locked["batch_sizes"], True)
    for field in ["alpha", "beta", "gamma", "delta", "diversity_distance_scale"]:
        require_equal(field, config[field], policy[field])
    require_equal("cost_scale", config["cost_scale"], locked["cost_scale_calibration_median"])
    require_equal(
        "cost_budget",
        config["cost_budget"],
        float(policy["cost_budget_per_item"]) * batch_size,
    )
    if locked.get("primary_policy_mode") == "cost_only":
        require_equal("primary_policy_mode", policy.get("primary_policy_mode"), "cost_only")
        require_equal("primary_risk_budget", locked.get("primary_risk_budget"), None)
        require_equal("risk_budget", config["risk_budget"], None)
    else:
        require_equal(
            "risk_budget",
            config["risk_budget"],
            float(policy["risk_budget_per_item"]) * batch_size,
        )
    expected_seed = (
        locked["random_baseline_seed_policy"]["primary_trace_seed"]
        if trace["method"] == "random_baseline"
        else None
    )
    require_equal("random_seed", config["random_seed"], expected_seed)
    return errors


def replay_trace(root: Path, trace_path: Path) -> dict:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    manifest_path = root / trace["candidate_manifest"]
    feature_path = root / trace.get(
        "feature_table", "data/derived/full_public_stability_features_relocked.csv"
    )
    artifact_path = root / trace["model_artifact"]
    code_path = root / trace["selector_code"]
    environment_path = root / trace["environment_lock"]
    source_manifest_path = root / trace["canonical_source_manifest"]
    locked_parameters_path = root / trace["locked_parameters_record"]
    checks = {
        "manifest_hash_match": sha256(manifest_path) == trace["candidate_manifest_sha256"],
        "source_manifest_hash_match": (
            sha256(source_manifest_path) == trace["canonical_source_manifest_sha256"]
        ),
        "feature_table_hash_match": sha256(feature_path) == trace.get("feature_table_sha256", sha256(feature_path)),
        "model_hash_match": sha256(artifact_path) == trace["model_artifact_sha256"],
        "selector_code_hash_match": sha256(code_path) == trace["selector_code_sha256"],
        "environment_lock_hash_match": sha256(environment_path) == trace["environment_lock_sha256"],
        "locked_parameters_hash_match": (
            sha256(locked_parameters_path) == trace["locked_parameters_record_sha256"]
        ),
        "locked_parameters_payload_match": (
            json.loads(locked_parameters_path.read_text(encoding="utf-8"))
            == trace["locked_parameters"]
        ),
    }
    # Preserve intentional empty provenance strings.  The trace is written from
    # the in-memory scored table where these fields are empty strings, and the
    # replay must not reinterpret them as NaN.
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    manifest_split = (
        "final_evaluation_split"
        if "final_evaluation_split" in manifest
        else "model_split"
    )
    manifest = manifest[
        manifest[manifest_split] == trace["evaluation_split"]
    ].reset_index(drop=True)
    features = pd.read_csv(feature_path)
    feature_split = (
        "final_evaluation_split"
        if "final_evaluation_split" in features
        else "model_split"
    )
    feature_pool = features[
        features[feature_split] == trace["evaluation_split"]
    ].copy()
    feature_pool = feature_pool.set_index("candidate_id").loc[manifest["candidate_id"]].reset_index()
    z = preprocess_from_artifact(feature_pool, artifact_path)
    cfg = PolicyConfig(**trace["config"])
    selected, alternatives, steps = select_batch(manifest, z, trace["method"], cfg)
    replay_metrics = metrics(selected)
    expected_ids = [r["candidate_id"] for r in trace["selected_candidates"]]
    replay_ids = selected["candidate_id"].tolist()
    checks["selected_ids_match"] = expected_ids == replay_ids
    metric_keys = sorted(replay_metrics)
    metric_diffs = {
        key: float(replay_metrics[key]) - float(trace["metrics"][key])
        for key in metric_keys
    }
    checks["metrics_match"] = all(abs(v) <= 1e-12 for v in metric_diffs.values())
    expected_alt = [r["candidate_id"] for r in trace["logged_alternative_candidates"]]
    checks["alternative_prefix_match"] = expected_alt == alternatives["candidate_id"].tolist()
    expected_step_ids = [
        s.get("selected_candidate", {}).get("candidate_id")
        for s in trace["stepwise_decisions"]
        if s.get("status") == "selected"
    ]
    replay_step_ids = [
        s.get("selected_candidate", {}).get("candidate_id")
        for s in steps
        if s.get("status") == "selected"
    ]
    checks["stepwise_selected_ids_match"] = expected_step_ids == replay_step_ids
    selected_differences, selected_max_diff = _nested_differences(
        trace["selected_candidates"],
        records(selected),
        path="selected_candidates",
    )
    alternative_differences, alternative_max_diff = _nested_differences(
        trace["logged_alternative_candidates"],
        records(alternatives),
        path="logged_alternative_candidates",
    )
    step_differences, step_max_diff = _nested_differences(
        trace["stepwise_decisions"],
        steps,
        path="stepwise_decisions",
    )
    checks["selected_candidate_records_match"] = not selected_differences
    checks["alternative_candidate_records_match"] = not alternative_differences
    checks["full_stepwise_decisions_match"] = not step_differences

    selection_event_count = sum(step.get("status") == "selected" for step in steps)
    stop_event_count = sum(
        str(step.get("status", "")).startswith("stopped") for step in steps
    )
    checks["event_counts_match"] = (
        trace["stepwise_decision_count"] == len(steps)
        and trace["selection_event_count"] == selection_event_count
        and trace["stop_event_count"] == stop_event_count
        and trace["full_batch"] == (selection_event_count == int(cfg.batch_size))
        and trace["full_batch_status"]
        == (
            "full_batch"
            if selection_event_count == int(cfg.batch_size)
            else "stopped_before_requested_batch"
        )
    )
    config_lock_differences = _config_lock_differences(trace, trace["locked_parameters"])
    checks["config_matches_locked_parameters"] = not config_lock_differences
    passed = all(checks.values())
    return {
        "trace": trace_path.relative_to(root).as_posix(),
        "passed": passed,
        "checks": checks,
        "metric_differences": metric_diffs,
        "full_record_differences": {
            "selected_candidates": selected_differences[:100],
            "logged_alternative_candidates": alternative_differences[:100],
            "stepwise_decisions": step_differences[:100],
            "config_lock": config_lock_differences[:100],
        },
        "maximum_full_record_numeric_difference": max(
            selected_max_diff, alternative_max_diff, step_max_diff
        ),
        "replayed_selected_candidates": replay_ids,
    }


def replay_all(root: Path, trace: Path | None = None) -> dict:
    trace_dir = root / "traces" / "action_traces"
    paths = [trace] if trace else sorted(trace_dir.glob("*.json"))
    out_dir = root / "traces" / "replay_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()
    results = []
    for path in paths:
        result = replay_trace(root, path)
        write_json(out_dir / f"{path.stem}_replay.json", result)
        results.append(result)
    summary = {
        "status": "PASS" if results and all(r["passed"] for r in results) else "FAIL",
        "trace_count": len(results),
        "passed_count": sum(bool(r["passed"]) for r in results),
        "failed": [r["trace"] for r in results if not r["passed"]],
        "semantics": "exact archived-implementation replay, not independent implementation validation",
    }
    write_json(root / "results" / "reproduction" / "replay_summary.json", summary)
    return summary
