from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageChops, ImageStat

from .io_utils import sha256_file, write_json

GENERATED_GLOBS = (
    "data/derived/**/*.csv",
    "data/processed/full_public_stability_manifest_scored.csv",
    "data/processed/full_public_thermal_model_scores.csv",
    "data/split_manifest.csv",
    "models/*.json",
    "results/reproduction/**/*.json",
    "results/reproduction/**/*.csv",
    "figures/*.png",
    "figures/*.pdf",
    "figures/*.svg",
    "source_data/*.csv",
    "traces/action_traces/*.json",
    "traces/replay_checks/*.json",
    "traces/trace_archive_manifest.csv",
)

MANIFEST_REL = "results/reproduction/generated_artifact_manifest.json"
REFERENCE_REL = "results/reproduction/regression_reference.json"
CONTRACT_REL = "results/reproduction/regression_contract.json"
RUNTIME_REL = "results/reproduction/runtime_diagnostics.json"


def _artifact_class(relative_path: str) -> str:
    """Assign an artifact to the regression class used by v45A2R1.

    Canonical byte identity is a Linux-container property, not a cross-platform
    promise. PDF and SVG files carry renderer metadata and are checked by
    content/visual rules instead. PNG files are visual assets because font
    rasterization may legitimately differ across operating systems.
    """
    suffix = Path(relative_path).suffix.lower()
    if suffix in {".pdf", ".svg", ".png"}:
        return "visual_equivalence"
    return "canonical_byte_identity"


def generated_artifact_paths(root: Path) -> list[Path]:
    paths: set[Path] = set()
    for pattern in GENERATED_GLOBS:
        paths.update(p for p in root.glob(pattern) if p.is_file())
    excluded = {root / MANIFEST_REL, root / REFERENCE_REL, root / CONTRACT_REL, root / RUNTIME_REL}
    return sorted(
        (p for p in paths if p not in excluded),
        key=lambda p: p.relative_to(root).as_posix(),
    )


def build_generated_artifact_manifest(root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in generated_artifact_paths(root):
        rel = path.relative_to(root).as_posix()
        files[rel] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "regression_class": _artifact_class(rel),
        }
    return {
        "schema_version": "3.0",
        "comparison_policy": {
            "canonical_byte_identity": (
                "required only in the pinned canonical Linux environment for files "
                "labelled canonical_byte_identity"
            ),
            "cross_platform_exact": (
                "candidate identities, split sizes, selected C, selected counts, hits, "
                "trace event counts and schemas must match exactly"
            ),
            "cross_platform_numeric": (
                f"field-specific absolute tolerances defined in {CONTRACT_REL}"
            ),
            "visual_equivalence": (
                "figure dimensions and bounded rendered-pixel differences; PDF/SVG byte "
                "identity is not a platform-neutral scientific requirement"
            ),
            "self_coverage": (
                f"{MANIFEST_REL}, {REFERENCE_REL}, {CONTRACT_REL} and {RUNTIME_REL} are "
                "excluded to avoid self-referential hashing and are covered by package-level manifests"
            ),
        },
        "file_count": len(files),
        "files": files,
    }


def write_generated_artifact_manifest(root: Path) -> dict[str, Any]:
    payload = build_generated_artifact_manifest(root)
    write_json(root / MANIFEST_REL, payload)
    return payload


def _trace_snapshot(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    exact: dict[str, Any] = {}
    numeric: dict[str, Any] = {}
    primary_paths = sorted(
        (root / "traces/action_traces").glob("*_test_primary_cost_contract_trace.json")
    )
    paths = primary_paths or sorted((root / "traces/action_traces").glob("*.json"))
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = payload["method"]
        config = payload.get("config", {})
        exact[method] = {
            "selected_candidate_ids": [r["candidate_id"] for r in payload["selected_candidates"]],
            "selected_count": int(payload["metrics"]["selected_count"]),
            "hits": int(payload["metrics"]["hits"]),
            "stepwise_decision_count": int(payload["stepwise_decision_count"]),
            "selection_event_count": int(payload["selection_event_count"]),
            "stop_event_count": int(payload["stop_event_count"]),
            "full_batch": bool(payload["full_batch"]),
            "full_batch_status": payload["full_batch_status"],
            "policy_contract": config.get(
                "policy_contract", config.get("primary_policy_mode", "cost_only")
            ),
            "risk_budget_per_item": config.get(
                "risk_budget_per_item", config.get("risk_budget")
            ),
        }
        numeric[method] = {
            key: value
            for key, value in payload["metrics"].items()
            if key not in {"selected_count", "hits"}
        }
    return exact, numeric


def _frame_records(path: Path, *, key_columns: list[str] | None = None) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    if key_columns:
        frame = frame.sort_values(key_columns).reset_index(drop=True)
    return frame.to_dict(orient="records")


def build_semantic_snapshot(root: Path) -> dict[str, Any]:
    activation = json.loads((root / "results/reproduction/activation_model_metrics.json").read_text())
    locked = json.loads(
        (root / "results/reproduction/tuning/v49_locked_parameters_before_test.json").read_text()
    )
    source_group = json.loads((root / "results/reproduction/source_group_audit.json").read_text())
    thermal = json.loads((root / "results/reproduction/thermal_regression_metrics.json").read_text())
    replay = json.loads((root / "results/reproduction/replay_summary.json").read_text())
    pca = json.loads((root / "results/reproduction/descriptor_pca_variance.json").read_text())
    dummy = json.loads(
        (root / "results/reproduction/activation_dummy_baselines.json").read_text()
    )
    bootstrap = json.loads(
        (root / "results/reproduction/activation_test_source_group_bootstrap_summary.json").read_text()
    )
    ablation_summary = json.loads(
        (root / "results/reproduction/diagnostics/source_overlap_C_ablation_summary.json").read_text()
    )
    trace_exact, trace_numeric = _trace_snapshot(root)

    random_summary = _frame_records(
        root / "results/reproduction/random_baseline_primary_cost_only_summary.csv",
        key_columns=["batch_size"],
    )
    pool_summary = _frame_records(
        root / "results/reproduction/test_pool_resampling_primary_cost_only_summary.csv",
        key_columns=["batch_size", "method"],
    )
    policy_primary = _frame_records(
        root / "results/reproduction/test_policy_comparison_primary_cost_only.csv",
        key_columns=["method"],
    )
    ablation = _frame_records(
        root / "results/reproduction/diagnostics/source_overlap_C_ablation.csv",
        key_columns=["C", "evaluation_universe"],
    )
    risk_audit = _frame_records(
        root / "results/reproduction/risk_feasibility_audit.csv",
        key_columns=["pool", "requested_batch_size"],
    )
    legacy_stress = _frame_records(
        root / "results/reproduction/diagnostics/test_policy_risk_stress_legacy_0p35.csv",
        key_columns=["batch_size", "method"],
    )

    random_full = pd.read_csv(
        root / "results/reproduction/random_baseline_primary_cost_only_raw.csv"
    )
    pool = pd.read_csv(
        root / "results/reproduction/test_pool_resampling_primary_cost_only_raw.csv"
    )

    exact_policy = []
    numeric_policy = []
    for row in policy_primary:
        exact_policy.append({
            "method": row["method"],
            "selected_candidate_ids": json.loads(row["selected_candidate_ids"]),
            "selected_count": int(row["selected_count"]),
            "hits": int(row["hits"]),
            "batch_size": int(row["batch_size"]),
            "evaluation_split": row["evaluation_split"],
            "full_batch": bool(row["full_batch"]),
            "full_batch_status": row["full_batch_status"],
            "selection_event_count": int(row["selection_event_count"]),
            "stop_event_count": int(row["stop_event_count"]),
            "policy_contract": row["policy_contract"],
            "risk_contract": row["risk_contract"],
            "risk_role": row["risk_role"],
            "risk_budget_per_item": None,
        })
        numeric_policy.append({
            k: v for k, v in row.items()
            if k not in {
                "method", "selected_candidate_ids", "selected_count", "hits",
                "batch_size", "evaluation_split", "full_batch", "full_batch_status",
                "selection_event_count", "stop_event_count", "policy_contract",
                "risk_contract", "risk_role", "risk_budget_per_item",
            }
        } | {"method": row["method"]})

    exact_ablation = [
        {
            "C": float(row["C"]),
            "evaluation_universe": row["evaluation_universe"],
            "n_rows": int(row["n_rows"]),
            "n_source_doi_groups": int(row["n_source_doi_groups"]),
            "model_role": row["model_role"],
            "diagnostic_only": bool(row["diagnostic_only"]),
        }
        for row in ablation
    ]
    numeric_ablation = [
        {
            "C": float(row["C"]),
            "evaluation_universe": row["evaluation_universe"],
            "roc_auc": row["roc_auc"],
            "accuracy_at_0_5": row["accuracy_at_0_5"],
            "brier_score": row["brier_score"],
        }
        for row in ablation
    ]
    exact_risk = [
        {
            "pool": row["pool"],
            "requested_batch_size": int(row["requested_batch_size"]),
            "pool_size": int(row["pool_size"]),
            "pool_source_group_count": int(row["pool_source_group_count"]),
            "full_batch_risk_feasible": bool(row["full_batch_risk_feasible"]),
            "max_feasible_cardinality_under_legacy_cap": int(
                row["max_feasible_cardinality_under_legacy_cap"]
            ),
            "full_batch_cost_feasible": bool(row["full_batch_cost_feasible"]),
            "max_feasible_cardinality_under_cost_cap": int(
                row["max_feasible_cardinality_under_cost_cap"]
            ),
            "risk_score_formula": row["risk_score_formula"],
            "risk_score_interpretation": row["risk_score_interpretation"],
            "legacy_risk_contract_role": row["legacy_risk_contract_role"],
            "primary_selected_risk": bool(row["primary_selected_risk"]),
        }
        for row in risk_audit
    ]
    numeric_risk = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "pool",
                "requested_batch_size",
                "pool_size",
                "pool_source_group_count",
                "full_batch_risk_feasible",
                "max_feasible_cardinality_under_legacy_cap",
                "full_batch_cost_feasible",
                "risk_score_formula",
                "risk_score_interpretation",
                "legacy_risk_contract_role",
                "primary_selected_risk",
                "five_lowest_risk_values",
            }
        }
        | {"pool": row["pool"], "requested_batch_size": int(row["requested_batch_size"])}
        | {
            "five_lowest_risk_values": (
                json.loads(row["five_lowest_risk_values"])
                if isinstance(row["five_lowest_risk_values"], str)
                and row["five_lowest_risk_values"].strip().startswith("[")
                else []
            )
        }
        for row in risk_audit
    ]
    exact_stress = []
    numeric_stress = []
    for row in legacy_stress:
        exact_stress.append({
            "method": row["method"],
            "batch_size": int(row["batch_size"]),
            "selected_candidate_ids": json.loads(row["selected_candidate_ids"]),
            "selected_count": int(row["selected_count"]),
            "hits": int(row["hits"]),
            "evaluation_split": row["evaluation_split"],
            "full_batch": bool(row["full_batch"]),
            "full_batch_status": row["full_batch_status"],
            "selection_event_count": int(row["selection_event_count"]),
            "stop_event_count": int(row["stop_event_count"]),
            "policy_contract": row["policy_contract"],
            "contract_role": row["contract_role"],
            "risk_interpretation": row["risk_interpretation"],
            "risk_budget_per_item": float(row["risk_budget_per_item"]),
        })
        numeric_stress.append({
            key: value
            for key, value in row.items()
            if key
            not in {
                "method",
                "batch_size",
                "selected_candidate_ids",
                "selected_count",
                "hits",
                "evaluation_split",
                "full_batch",
                "full_batch_status",
                "selection_event_count",
                "stop_event_count",
                "policy_contract",
                "contract_role",
                "risk_interpretation",
                "risk_budget_per_item",
            }
        } | {"method": row["method"], "batch_size": int(row["batch_size"])} )
    dummy_rows = sorted(dummy["baselines"], key=lambda row: row["strategy"])

    return {
        "schema_version": "4.0",
        "exact": {
            "activation": {
                "selected_C": activation["selected_C"],
                "n_train": int(activation["n_train"]),
                "n_calibration": int(activation["n_calibration"]),
                "n_test": int(activation["n_test"]),
                "solver": activation["solver"],
                "cv_splitter": activation["cv_splitter"],
                "cv_fold_group_overlap_count_max": int(
                    activation["cv_fold_group_overlap_count_max"]
                ),
                "n_excluded_train_source_overlap": int(
                    activation["n_excluded_train_source_overlap"]
                ),
                "all_activation_fits_converged": bool(activation["all_activation_fits_converged"]),
            },
            "locked_policy_grid_values": {
                "primary_policy_mode": locked["primary_policy_mode"],
                "primary_risk_budget": locked["primary_risk_budget"],
                "alpha": locked["policy"]["alpha"],
                "beta": locked["policy"]["beta"],
                "gamma": locked["policy"]["gamma"],
                "delta": locked["policy"]["delta"],
                "risk_budget_per_item": locked["policy"]["risk_budget_per_item"],
                "diversity_distance_scale": locked["policy"]["diversity_distance_scale"],
                "legacy_risk_stress_budget_per_item": locked[
                    "legacy_risk_stress_contract"
                ]["risk_budget_per_item"],
                "legacy_risk_stress_role": locked["legacy_risk_stress_contract"]["role"],
            },
            "source_group": source_group,
            "replay": {
                "status": replay["status"],
                "trace_count": int(replay["trace_count"]),
                "passed_count": int(replay["passed_count"]),
            },
            "traces": trace_exact,
            "policy_comparison": exact_policy,
            "source_overlap_C_ablation": exact_ablation,
            "dummy_baselines": [
                {
                    "strategy": row["strategy"],
                    "fit_split": row.get("fit_split", dummy.get("fit_split")),
                    "evaluation_split": row.get(
                        "evaluation_split", dummy.get("evaluation_split")
                    ),
                    "training_n": int(row["training_n"]),
                    "training_majority_class": int(row["training_majority_class"]),
                    "test_n": int(row["test_n"]),
                }
                for row in dummy_rows
            ],
            "test_source_group_bootstrap": {
                "requested_resamples": int(bootstrap["requested_resamples"]),
                "valid_auc_resamples": int(bootstrap["valid_auc_resamples"]),
                "invalid_auc_resamples": int(bootstrap["invalid_auc_resamples"]),
                "seed": int(bootstrap["seed"]),
                "bootstrap_group_unit": bootstrap["bootstrap_group_unit"],
                "source_group_count": int(bootstrap["source_group_count"]),
                "confidence_interval": bootstrap["confidence_interval"],
                "used_for_tuning": bool(bootstrap["used_for_tuning"]),
            },
            "risk_feasibility": exact_risk,
            "legacy_risk_stress": exact_stress,
            "random_design": {
                "full_pool_runs_per_batch_size": int(random_full.groupby("batch_size").size().min()),
                "pool_count": int(pool["pool_seed"].nunique()),
                "random_seeds_per_pool": int(
                    pool[pool["method"].eq("random_baseline")]
                    .groupby(["pool_seed", "batch_size"])["random_seed"]
                    .nunique()
                    .min()
                ),
            },
            "thermal": {
                "selected_alpha": thermal["selected_alpha"],
                "n_train": int(thermal["n_train"]),
                "n_validation": int(thermal["n_validation"]),
            },
            "descriptor_contrast_count": int(
                len(pd.read_csv(root / "source_data/figure4_descriptor_contrasts.csv"))
            ),
        },
        "numeric": {
            "activation": {
                "calibration_roc_auc": activation["calibration_roc_auc"],
                "calibration_brier_score": activation["calibration_brier_score"],
                "test_roc_auc": activation["test_roc_auc"],
                "test_accuracy_at_0_5": activation["test_accuracy_at_0_5"],
                "test_brier_score": activation["test_brier_score"],
                "test_positive_fraction": activation["test_positive_fraction"],
                "test_majority_class_fraction_descriptive": activation[
                    "test_majority_class_fraction_descriptive"
                ],
                "dummy_prior_accuracy": activation["dummy_prior_accuracy"],
                "dummy_prior_roc_auc": activation["dummy_prior_roc_auc"],
                "dummy_prior_brier": activation["dummy_prior_brier"],
                "dummy_most_frequent_accuracy": activation[
                    "dummy_most_frequent_accuracy"
                ],
                "test_source_group_bootstrap_auc_ci_low": activation[
                    "test_source_group_bootstrap_auc_ci_low"
                ],
                "test_source_group_bootstrap_auc_ci_high": activation[
                    "test_source_group_bootstrap_auc_ci_high"
                ],
                "test_source_group_bootstrap_auc_standard_error": activation[
                    "test_source_group_bootstrap_auc_standard_error"
                ],
                "artifact_regeneration_max_abs_diff": activation["artifact_regeneration_max_abs_diff"],
            },
            "locked_policy": {
                "cost_scale_calibration_median": locked["cost_scale_calibration_median"],
                "cost_budget_per_item": locked["policy"]["cost_budget_per_item"],
            },
            "traces": trace_numeric,
            "policy_comparison": numeric_policy,
            "source_overlap_C_ablation": numeric_ablation,
            "source_overlap_C_ablation_deltas": ablation_summary,
            "dummy_baselines": [
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "strategy",
                        "fit_split",
                        "evaluation_split",
                        "training_n",
                        "training_majority_class",
                        "test_n",
                    }
                }
                | {"strategy": row["strategy"]}
                for row in dummy_rows
            ],
            "test_source_group_bootstrap": {
                key: value
                for key, value in bootstrap.items()
                if key
                not in {
                    "requested_resamples",
                    "valid_auc_resamples",
                    "invalid_auc_resamples",
                    "seed",
                    "bootstrap_group_unit",
                    "source_group_count",
                    "confidence_interval",
                    "used_for_tuning",
                }
            },
            "risk_feasibility": numeric_risk,
            "legacy_risk_stress": numeric_stress,
            "random_summary": random_summary,
            "pool_summary": pool_summary,
            "thermal": {
                "mae_C": thermal["mae_C"],
                "rmse_C": thermal["rmse_C"],
                "r2": thermal["r2"],
                "training_mean_baseline_rmse_C": thermal["training_mean_baseline_rmse_C"],
            },
            "pca": pca,
        },
        "diagnostics": {
            "final_fit_n_iter": activation["final_fit_n_iter"],
            "maximum_cv_n_iter": activation["maximum_cv_n_iter"],
            "max_iter": activation["max_iter"],
            "tol": activation["tol"],
            "note": (
                "iteration counts and machine-precision reload differences are environment diagnostics, "
                "not cross-platform scientific claims"
            ),
        },
    }


def _tolerance_for(path: str, contract: dict[str, Any], mode: str) -> tuple[float, float]:
    numeric = contract["comparison_classes"]["numerical_tolerance"]
    profile = numeric["canonical" if mode == "canonical" else "portability"]
    abs_tol = float(profile["default_abs"])
    rel_tol = float(profile["default_rel"])
    for rule in profile.get("overrides", []):
        if fnmatch.fnmatch(path, rule["pattern"]):
            abs_tol = float(rule.get("abs", abs_tol))
            rel_tol = float(rule.get("rel", rel_tol))
            break
    return abs_tol, rel_tol


def compare_semantic_snapshots(
    reference: dict[str, Any],
    current: dict[str, Any],
    *,
    contract: dict[str, Any],
    mode: str = "portability",
) -> list[str]:
    errors: list[str] = []

    def exact_walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                errors.append(f"exact.{path}: key mismatch {sorted(set(a) ^ set(b))}")
                return
            for key in sorted(a):
                exact_walk(a[key], b[key], f"{path}.{key}" if path else key)
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                errors.append(f"exact.{path}: list length {len(a)} != {len(b)}")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                exact_walk(x, y, f"{path}[{i}]")
            return
        if a != b:
            errors.append(f"exact.{path}: {a!r} != {b!r}")

    def numeric_walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                errors.append(f"numeric.{path}: key mismatch {sorted(set(a) ^ set(b))}")
                return
            for key in sorted(a):
                numeric_walk(a[key], b[key], f"{path}.{key}" if path else key)
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                errors.append(f"numeric.{path}: list length {len(a)} != {len(b)}")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                numeric_walk(x, y, f"{path}[{i}]")
            return
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            abs_tol, rel_tol = _tolerance_for(f"numeric.{path}", contract, mode)
            diff = abs(float(a) - float(b))
            allowed = max(abs_tol, rel_tol * max(abs(float(a)), abs(float(b))))
            if diff > allowed:
                errors.append(
                    f"numeric.{path}: {a!r} != {b!r}; abs_diff={diff:.6g}, allowed={allowed:.6g}"
                )
            return
        if a != b:
            errors.append(f"numeric.{path}: {a!r} != {b!r}")

    if reference.get("schema_version") != current.get("schema_version"):
        errors.append(
            f"schema_version: {reference.get('schema_version')!r} != {current.get('schema_version')!r}"
        )
        return errors
    exact_walk(reference["exact"], current["exact"], "")
    numeric_walk(reference["numeric"], current["numeric"], "")
    return errors


def compare_png_visuals(reference_path: Path, current_path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    rules = contract["comparison_classes"]["visual_equivalence"]
    with Image.open(reference_path) as ref_img, Image.open(current_path) as cur_img:
        ref = ref_img.convert("RGB")
        cur = cur_img.convert("RGB")
        if ref.size != cur.size:
            return {
                "status": "FAIL",
                "reason": f"dimension mismatch {ref.size} != {cur.size}",
            }
        diff = ImageChops.difference(ref, cur)
        stat = ImageStat.Stat(diff)
        mean_abs = sum(stat.mean) / 3.0
        histogram = diff.convert("L").histogram()
        total = ref.size[0] * ref.size[1]
        unchanged = histogram[0]
        changed_fraction = 1.0 - (unchanged / total)
        status = (
            "PASS"
            if mean_abs <= float(rules["max_mean_absolute_channel_difference"])
            and changed_fraction <= float(rules["max_changed_pixel_fraction"])
            else "FAIL"
        )
        return {
            "status": status,
            "dimensions": list(ref.size),
            "mean_absolute_channel_difference": mean_abs,
            "changed_pixel_fraction": changed_fraction,
            "thresholds": {
                "max_mean_absolute_channel_difference": rules["max_mean_absolute_channel_difference"],
                "max_changed_pixel_fraction": rules["max_changed_pixel_fraction"],
            },
        }
