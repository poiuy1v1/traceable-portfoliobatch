from __future__ import annotations

import hashlib
from pathlib import Path
import platform
from typing import Any

import pandas as pd

from .io_utils import write_json


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def write_trace(
    path: Path,
    *,
    run_id: str,
    method: str,
    config: dict,
    manifest_rel: str,
    manifest_sha256: str,
    source_manifest_rel: str,
    source_manifest_sha256: str,
    feature_table_rel: str,
    feature_table_sha256: str,
    model_artifact_rel: str,
    model_sha256: str,
    selected: pd.DataFrame,
    alternatives: pd.DataFrame,
    steps: list[dict[str, Any]],
    metrics: dict,
    code_rel: str,
    code_sha256: str,
    environment_rel: str,
    environment_sha256: str,
    evaluation_split: str,
    locked_parameters_rel: str,
    locked_parameters_sha256: str,
    locked_parameters: dict,
    policy_layer: str = "policy-agnostic trace wrapper",
) -> None:
    selection_event_count = sum(step.get("status") == "selected" for step in steps)
    stop_event_count = sum(
        str(step.get("status", "")).startswith("stopped") for step in steps
    )
    payload = {
        "schema_version": "3.0",
        "run_id": run_id,
        "method": method,
        "selector_identity": {"run_id": run_id, "method": method},
        "evaluation_split": evaluation_split,
        "trace_layer": policy_layer,
        "config": config,
        "candidate_manifest": manifest_rel,
        "candidate_manifest_sha256": manifest_sha256,
        "canonical_source_manifest": source_manifest_rel,
        "canonical_source_manifest_sha256": source_manifest_sha256,
        "feature_table": feature_table_rel,
        "feature_table_sha256": feature_table_sha256,
        "model_artifact": model_artifact_rel,
        "model_artifact_sha256": model_sha256,
        "selector_code": code_rel,
        "selector_code_sha256": code_sha256,
        "environment_lock": environment_rel,
        "environment_lock_sha256": environment_sha256,
        "locked_parameters_record": locked_parameters_rel,
        "locked_parameters_record_sha256": locked_parameters_sha256,
        "locked_parameters": locked_parameters,
        "runtime": {
            "python": platform.python_version(),
            "platform_policy": ("canonical Linux byte identity; cross-platform exact discrete "
                                "identities plus field-specific numerical and visual tolerances"),
        },
        "selected_candidates": records(selected),
        "logged_alternative_candidates": records(alternatives),
        "stepwise_decisions": steps,
        "stepwise_decision_count": len(steps),
        "selection_event_count": selection_event_count,
        "stop_event_count": stop_event_count,
        "full_batch": selection_event_count == int(config["batch_size"]),
        "full_batch_status": (
            "full_batch"
            if selection_event_count == int(config["batch_size"])
            else "stopped_before_requested_batch"
        ),
        "metrics": metrics,
        "replay_command": f"python scripts/run_audit_replay.py --trace traces/action_traces/{path.name}",
        "replay_semantics": (
            "exact deterministic re-execution with the archived implementation; "
            "not an independent software implementation"
        ),
    }
    write_json(path, payload)
