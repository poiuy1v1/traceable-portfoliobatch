from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceable_portfoliobatch.io_utils import sha256_file  # noqa: E402
from traceable_portfoliobatch.regression import (  # noqa: E402
    MANIFEST_REL,
    build_generated_artifact_manifest,
)


EXPECTED_SPLIT_COUNTS = {
    "train": 1394,
    "calibration": 158,
    "test": 193,
    "excluded_train_source_overlap": 434,
}
EXPECTED_GROUP_COUNTS = {
    "train": 865,
    "calibration": 136,
    "test": 173,
    "excluded_train_source_overlap": 270,
}
PRIMARY_METHODS = {
    "top_score",
    "uncertainty",
    "cost_aware",
    "portfolio_batch",
    "random_baseline",
}
PRIMARY_TRACE_PATHS = {
    method: Path("traces/action_traces")
    / f"{method}_test_primary_cost_contract_trace.json"
    for method in PRIMARY_METHODS
}

REQUIRED_FILES = [
    "data/processed/full_public_stability_source_manifest.csv",
    "data/processed/full_public_stability_manifest_scored.csv",
    "data/derived/full_public_stability_features_relocked.csv",
    "models/activation_logistic_train_split.json",
    "results/reproduction/activation_model_metrics.json",
    "results/reproduction/source_group_audit.json",
    "results/reproduction/diagnostics/source_overlap_C_ablation.csv",
    "results/reproduction/diagnostics/source_overlap_C_ablation_summary.json",
    "results/reproduction/activation_dummy_baselines.json",
    "results/reproduction/activation_test_source_group_bootstrap_raw.csv",
    "results/reproduction/activation_test_source_group_bootstrap_summary.json",
    "results/reproduction/risk_feasibility_audit.csv",
    "results/reproduction/tuning/activation_C_train_internal_SOURCE_GROUP_CV.csv",
    "results/reproduction/tuning/activation_train_group_cv_fold_audit.csv",
    "results/reproduction/tuning/portfolio_policy_cost_only_calibration_selection.csv",
    "results/reproduction/tuning/portfolio_policy_cost_only_group_bootstrap_stability.csv",
    "results/reproduction/tuning/v49_locked_parameters_before_test.json",
    "results/reproduction/test_policy_comparison_primary_cost_only.csv",
    "results/reproduction/random_baseline_primary_cost_only_raw.csv",
    "results/reproduction/random_baseline_primary_cost_only_summary.csv",
    "results/reproduction/test_pool_resampling_primary_cost_only_raw.csv",
    "results/reproduction/test_pool_resampling_primary_cost_only_summary.csv",
    "results/reproduction/diagnostics/test_policy_risk_stress_legacy_0p35.csv",
    "results/reproduction/v49_statistical_summary.json",
    "results/reproduction/replay_summary.json",
    "results/reproduction/regression_contract.json",
    MANIFEST_REL,
    "source_data/source_overlap_C_ablation_source.csv",
    "source_data/activation_dummy_baselines_source.csv",
    "source_data/activation_test_source_group_bootstrap_source.csv",
    "source_data/risk_feasibility_audit_source.csv",
    "source_data/figure3_test_policy_primary_cost_source.csv",
    "source_data/figure3_random_baseline_primary_cost_raw_points.csv",
    "source_data/figure3_random_baseline_primary_cost_summary_source.csv",
    "source_data/figure3_pool_resampling_primary_cost_raw_points.csv",
    "source_data/figure3_pool_resampling_primary_cost_summary_source.csv",
    "source_data/figure_source_data_manifest.csv",
    *[path.as_posix() for path in PRIMARY_TRACE_PATHS.values()],
]


def _read_json(relative: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _read_csv(relative: str | Path) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative)


def _boolish(value: Any) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass"}
    return bool(value)


def _nullish(value: Any) -> bool:
    return value is None or bool(pd.isna(value))


def _normalize_doi(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip().rstrip("/.,;")


def _valid_normalized_doi(value: Any) -> bool:
    text = "" if pd.isna(value) else str(value).strip().lower()
    return re.fullmatch(r"10\.\d{4,9}/\S+", text) is not None


def _ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if _nullish(value):
        return []
    text = str(value).strip()
    if text.startswith("["):
        return [str(item) for item in json.loads(text)]
    return [item for item in text.split() if item]


def _recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(str(key) for key in value)
        for child in value.values():
            keys.update(_recursive_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_recursive_keys(child))
    return keys


def _record(
    checks: dict[str, bool],
    details: dict[str, Any],
    name: str,
    passed: Any,
    detail: Any | None = None,
) -> None:
    checks[name] = bool(passed)
    if detail is not None:
        details[name] = detail


def _declared_hash_matches(payload: dict[str, Any], path_key: str, hash_key: str) -> bool:
    relative = payload.get(path_key)
    expected = payload.get(hash_key)
    return bool(
        relative
        and expected
        and (ROOT / str(relative)).is_file()
        and sha256_file(ROOT / str(relative)) == expected
    )


def _selected_C_from_cv(table: pd.DataFrame) -> float:
    selected = table[table["selected_one_standard_error"].map(_boolish)]
    if len(selected) != 1:
        return float("nan")
    return float(selected.iloc[0]["C"])


def validate_repository_state() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    _record(checks, details, "required_v49_files", not missing, {"missing": missing})
    if missing:
        return _finish(checks, details)

    source = _read_csv("data/processed/full_public_stability_source_manifest.csv")
    relocked = _read_csv("data/derived/full_public_stability_features_relocked.csv")
    scored = _read_csv("data/processed/full_public_stability_manifest_scored.csv")
    metrics = _read_json("results/reproduction/activation_model_metrics.json")
    source_audit = _read_json("results/reproduction/source_group_audit.json")
    model = _read_json("models/activation_logistic_train_split.json")
    lock = _read_json("results/reproduction/tuning/v49_locked_parameters_before_test.json")

    required_source_columns = {
        "candidate_id",
        "source_doi",
        "source_doi_normalized",
        "source_group_id",
        "source_group_basis",
        "original_model_split",
        "final_evaluation_split",
    }
    _record(
        checks,
        details,
        "source_manifest_columns",
        required_source_columns <= set(source.columns),
        {"missing": sorted(required_source_columns - set(source.columns))},
    )
    if not checks["source_manifest_columns"]:
        return _finish(checks, details)

    _record(
        checks,
        details,
        "source_manifest_identity",
        len(source) == 2179
        and source["candidate_id"].is_unique
        and relocked["candidate_id"].is_unique
        and set(source["candidate_id"].astype(str))
        == set(relocked["candidate_id"].astype(str)),
        {
            "source_rows": int(len(source)),
            "relocked_rows": int(len(relocked)),
            "source_candidate_ids_unique": bool(source["candidate_id"].is_unique),
            "relocked_candidate_ids_unique": bool(relocked["candidate_id"].is_unique),
        },
    )

    split_counts = source["final_evaluation_split"].value_counts().to_dict()
    group_counts = {
        split: int(
            source.loc[source["final_evaluation_split"].eq(split), "source_group_id"].nunique()
        )
        for split in EXPECTED_SPLIT_COUNTS
    }
    _record(checks, details, "final_split_counts", split_counts == EXPECTED_SPLIT_COUNTS, split_counts)
    _record(checks, details, "final_group_counts", group_counts == EXPECTED_GROUP_COUNTS, group_counts)

    active_groups = {
        split: set(
            source.loc[source["final_evaluation_split"].eq(split), "source_group_id"].astype(str)
        )
        for split in ("train", "calibration", "test")
    }
    overlap_counts = {
        "train_calibration": len(active_groups["train"] & active_groups["calibration"]),
        "train_test": len(active_groups["train"] & active_groups["test"]),
        "calibration_test": len(active_groups["calibration"] & active_groups["test"]),
    }
    _record(
        checks,
        details,
        "active_source_groups_disjoint",
        set(overlap_counts.values()) == {0},
        overlap_counts,
    )
    active = source[source["final_evaluation_split"].isin(["train", "calibration", "test"])]
    normalized_raw_doi = source["source_doi"].map(_normalize_doi)
    _record(
        checks,
        details,
        "publication_doi_group_complete",
        source["source_doi_normalized"].map(_valid_normalized_doi).all()
        and normalized_raw_doi.eq(source["source_doi_normalized"].astype(str)).all()
        and source["source_group_id"].astype(str).eq(
            source["source_doi_normalized"].astype(str)
        ).all()
        and source["source_group_basis"].eq("normalized_source_doi").all()
        and active["source_group_id"].fillna("").astype(str).str.strip().ne("").all(),
        {
            "all_normalized_doi_syntax_valid": bool(
                source["source_doi_normalized"].map(_valid_normalized_doi).all()
            ),
            "raw_doi_normalization_matches": bool(
                normalized_raw_doi.eq(source["source_doi_normalized"].astype(str)).all()
            ),
            "group_id_equals_normalized_doi": bool(
                source["source_group_id"].astype(str).eq(
                    source["source_doi_normalized"].astype(str)
                ).all()
            ),
        },
    )
    excluded_ids = set(
        source.loc[
            source["final_evaluation_split"].eq("excluded_train_source_overlap"),
            "candidate_id",
        ]
    )
    _record(
        checks,
        details,
        "excluded_rows_not_scored",
        set(scored["final_evaluation_split"]) == {"calibration", "test"}
        and excluded_ids.isdisjoint(set(scored["candidate_id"])),
    )
    _record(
        checks,
        details,
        "source_group_audit_consistent",
        source_audit.get("status") == "PASS_COMPLETE_ACTIVE_SOURCE_GROUP_DISJOINT"
        and source_audit.get("training_rows_without_packaged_source_doi") == 0
        and source_audit.get("training_rows_with_packaged_source_doi") == 1394
        and source_audit.get("train_calibration_source_doi_overlap_count") == 0
        and source_audit.get("train_test_source_doi_overlap_count") == 0
        and source_audit.get("policy_calibration_test_source_doi_overlap_count") == 0,
    )

    group_cv = _read_csv("results/reproduction/tuning/activation_C_train_internal_SOURCE_GROUP_CV.csv")
    fold_audit = _read_csv("results/reproduction/tuning/activation_train_group_cv_fold_audit.csv")
    selected_c = _selected_C_from_cv(group_cv)
    within = group_cv[group_cv["within_one_standard_error"].map(_boolish)]
    model_c = float(model.get("selected_C", model.get("parameters", {}).get("C", float("nan"))))
    _record(
        checks,
        details,
        "activation_group_cv_design",
        len(fold_audit) == 5
        and fold_audit["group_overlap_count"].eq(0).all()
        and set(group_cv["cv_splitter"]) == {"StratifiedGroupKFold"}
        and set(group_cv["random_state"].astype(int)) == {45}
        and set(group_cv["n_splits"].astype(int)) == {5}
        and group_cv["all_folds_converged"].map(_boolish).all(),
    )
    _record(
        checks,
        details,
        "selected_C_training_only_consistency",
        len(within) > 0
        and np.isclose(selected_c, float(within["C"].min()), rtol=0, atol=0)
        and np.isclose(selected_c, float(metrics["selected_C"]), rtol=0, atol=0)
        and np.isclose(selected_c, model_c, rtol=0, atol=0)
        and np.isclose(selected_c, float(lock["activation_C"]), rtol=0, atol=0),
        {"selected_C": selected_c},
    )
    _record(
        checks,
        details,
        "activation_convergence",
        metrics.get("all_activation_fits_converged") is True
        and int(metrics.get("final_fit_n_iter", 10**9)) < int(metrics.get("max_iter", 0))
        and int(metrics.get("maximum_cv_n_iter", 10**9)) < int(metrics.get("max_iter", 0)),
    )

    legacy = lock.get("legacy_risk_stress_contract", {})
    policy = lock.get("policy", {})
    _record(
        checks,
        details,
        "v49_pretest_lock_contract",
        lock.get("primary_policy_mode") == "cost_only"
        and lock.get("primary_risk_budget") is None
        and policy.get("risk_budget_per_item") is None
        and np.isclose(float(legacy.get("risk_budget_per_item", float("nan"))), 0.35)
        and legacy.get("role") == "stress_test_only"
        and lock.get("test_evaluation_may_start_only_after_this_file_is_written") is True
        and lock.get("test_use") == "none during model or policy tuning"
        and "test labels are not used" in str(lock.get("test_use_claim", "")).lower(),
    )
    hash_checks = {
        "source_manifest": _declared_hash_matches(
            lock, "source_manifest", "training_source_manifest_sha256"
        ),
        "split_manifest": _declared_hash_matches(
            lock, "final_split_manifest", "final_split_manifest_sha256"
        ),
        "model_artifact": _declared_hash_matches(
            lock, "activation_model_artifact", "activation_model_artifact_sha256"
        ),
    }
    _record(checks, details, "v49_pretest_lock_hashes", all(hash_checks.values()), hash_checks)

    tuning = _read_csv(
        "results/reproduction/tuning/portfolio_policy_cost_only_calibration_selection.csv"
    )
    tuning_required = {
        "alpha",
        "beta",
        "gamma",
        "delta",
        "diversity_distance_scale",
        "cost_budget_per_item",
        "selected_count",
        "hits",
        "proxy_cost_normalized_yield",
        "total_proxy_cost",
        "full_batch",
        "selected",
        "primary_policy_mode",
    }
    tuning_columns_ok = tuning_required <= set(tuning.columns)
    risk_null = "risk_budget_per_item" not in tuning or tuning["risk_budget_per_item"].map(_nullish).all()
    grid = set(
        zip(
            tuning.get("beta", pd.Series(dtype=float)).astype(float).round(12),
            tuning.get("gamma", pd.Series(dtype=float)).astype(float).round(12),
            tuning.get("delta", pd.Series(dtype=float)).astype(float).round(12),
        )
    )
    expected_grid = {
        (beta, gamma, delta)
        for beta in (0.0, 0.15, 0.30)
        for gamma in (0.10, 0.20, 0.35)
        for delta in (0.0, 0.10, 0.20)
    }
    selected_tuning = tuning[tuning.get("selected", False).map(_boolish)] if tuning_columns_ok else tuning.iloc[0:0]
    ranked = (
        tuning.assign(_full_batch=tuning["full_batch"].map(_boolish))
        .sort_values(
            [
                "_full_batch",
                "hits",
                "proxy_cost_normalized_yield",
                "total_proxy_cost",
                "gamma",
                "beta",
                "delta",
            ],
            ascending=[False, False, False, True, True, True, True],
            kind="mergesort",
        )
        if tuning_columns_ok
        else tuning
    )
    selected_matches_rank = (
        tuning_columns_ok
        and len(selected_tuning) == 1
        and int(selected_tuning.index[0]) == int(ranked.index[0])
    )
    selected_matches_lock = len(selected_tuning) == 1 and all(
        np.isclose(float(selected_tuning.iloc[0][key]), float(policy[key]), rtol=0, atol=1e-12)
        for key in ("alpha", "beta", "gamma", "delta", "diversity_distance_scale", "cost_budget_per_item")
    )
    _record(
        checks,
        details,
        "cost_only_policy_grid_and_ranking",
        tuning_columns_ok
        and len(tuning) == 27
        and grid == expected_grid
        and tuning["alpha"].astype(float).eq(1.0).all()
        and tuning["diversity_distance_scale"].astype(float).eq(2.0).all()
        and tuning["primary_policy_mode"].eq("cost_only").all()
        and risk_null
        and selected_matches_rank
        and selected_matches_lock,
    )
    stability = _read_csv(
        "results/reproduction/tuning/portfolio_policy_cost_only_group_bootstrap_stability.csv"
    )
    _record(
        checks,
        details,
        "cost_only_policy_group_bootstrap",
        len(stability) == 20
        and stability["bootstrap_seed"].astype(int).nunique() == 20
        and stability["primary_policy_mode"].eq("cost_only").all()
        and "full_batch_frequency" in stability
        and np.allclose(
            stability["full_batch_frequency"].astype(float),
            float(stability["full_batch"].map(_boolish).mean()),
            rtol=0,
            atol=1e-12,
        )
        and ("risk_budget_per_item" not in stability or stability["risk_budget_per_item"].map(_nullish).all()),
    )

    ablation = _read_csv("results/reproduction/diagnostics/source_overlap_C_ablation.csv")
    ablation_summary = _read_json(
        "results/reproduction/diagnostics/source_overlap_C_ablation_summary.json"
    )
    ablation_pairs = {
        (float(row.C), str(row.evaluation_universe))
        for row in ablation.itertuples(index=False)
    }
    expected_pairs = {
        (c_value, universe)
        for c_value in (0.001, 3.0)
        for universe in (
            "historical_overlap_inclusive_test",
            "final_publication_source_disjoint_test",
        )
    }
    expected_rows = {
        "historical_overlap_inclusive_test": 436,
        "final_publication_source_disjoint_test": 193,
    }
    summary_keys = _recursive_keys(ablation_summary)
    _record(
        checks,
        details,
        "source_overlap_C_ablation_exact_contract",
        len(ablation) == 4
        and ablation_pairs == expected_pairs
        and all(
            int(row.n_rows) == expected_rows[str(row.evaluation_universe)]
            and int(row.n_source_doi_groups) > 0
            and _boolish(row.diagnostic_only)
            for row in ablation.itertuples(index=False)
        )
        and {
            "split_effect_at_C_3",
            "split_effect_at_C_0p001",
            "C_effect_on_overlap_inclusive_test",
            "C_effect_on_source_disjoint_test",
        }
        <= summary_keys,
    )

    dummy = _read_json("results/reproduction/activation_dummy_baselines.json")
    dummy_source = _read_csv("source_data/activation_dummy_baselines_source.csv")
    dummy_rows = dummy.get("baselines", [])
    dummy_strategies = {str(row.get("strategy")) for row in dummy_rows}
    train = source[source["final_evaluation_split"].eq("train")]
    train_positive_fraction = float(
        (train["outcome_value"].astype(float) > 0).mean()
    )
    train_majority_class = int(train_positive_fraction >= 0.5)
    dummy_lookup = dummy_source.set_index("strategy")
    _record(
        checks,
        details,
        "training_fitted_dummy_baselines",
        dummy_strategies == {"prior", "most_frequent"}
        and set(dummy_source["strategy"]) == {"prior", "most_frequent"}
        and dummy_source["fit_split"].eq("train").all()
        and dummy_source["evaluation_split"].eq(
            "final_publication_source_disjoint_test"
        ).all()
        and dummy_source["training_n"].astype(int).eq(1394).all()
        and dummy_source["test_n"].astype(int).eq(193).all()
        and np.allclose(
            dummy_source["training_positive_fraction"].astype(float),
            train_positive_fraction,
            rtol=0,
            atol=1e-12,
        )
        and dummy_source["training_majority_class"].astype(int).eq(
            train_majority_class
        ).all()
        and np.isclose(
            float(dummy_lookup.loc["prior", "predicted_positive_probability"]),
            train_positive_fraction,
            rtol=0,
            atol=1e-12,
        )
        and np.isclose(
            float(
                dummy_lookup.loc[
                    "most_frequent", "predicted_positive_probability"
                ]
            ),
            float(train_majority_class),
            rtol=0,
            atol=1e-12,
        )
        and np.isfinite(
            dummy_source[["accuracy", "roc_auc", "brier_score"]]
            .astype(float)
            .to_numpy()
        ).all()
        and all(str(row.get("fit_split", dummy.get("fit_split", ""))) == "train" for row in dummy_rows),
    )

    bootstrap_raw = _read_csv("results/reproduction/activation_test_source_group_bootstrap_raw.csv")
    bootstrap = _read_json("results/reproduction/activation_test_source_group_bootstrap_summary.json")
    invalid_auc = bootstrap_raw["invalid_for_auc"].map(_boolish)
    _record(
        checks,
        details,
        "source_group_bootstrap_exact_design",
        bootstrap.get("requested_resamples") == 10000
        and bootstrap.get("seed") == 45
        and bootstrap.get("bootstrap_group_unit") == "normalized_source_doi"
        and int(bootstrap.get("valid_auc_resamples", -1))
        + int(bootstrap.get("invalid_auc_resamples", -1))
        == 10000
        and len(bootstrap_raw) == 10000
        and bootstrap_raw["bootstrap_resample"].nunique() == 10000
        and set(bootstrap_raw["bootstrap_resample"].astype(int))
        == set(range(1, 10001))
        and bootstrap_raw["sampled_group_occurrences"].astype(int).eq(173).all()
        and int(invalid_auc.sum()) == int(bootstrap["invalid_auc_resamples"])
        and bootstrap_raw.loc[invalid_auc, "roc_auc"].isna().all()
        and bootstrap_raw.loc[~invalid_auc, "roc_auc"].notna().all(),
    )

    risk = _read_csv("results/reproduction/risk_feasibility_audit.csv")
    risk_pairs = set(
        zip(risk["pool"].astype(str), risk["requested_batch_size"].astype(int))
    )
    expected_risk_pairs = {
        (pool_name, batch_size)
        for pool_name in ("final_calibration", "final_test")
        for batch_size in (5, 10, 20)
    }
    risk_formula_ok = True
    for row in risk.itertuples(index=False):
        split_name = {
            "final_calibration": "calibration",
            "final_test": "test",
        }[str(row.pool)]
        pool_scores = scored[scored["final_evaluation_split"].eq(split_name)]
        k = int(row.requested_batch_size)
        sorted_risk = np.sort(pool_scores["risk_score"].astype(float).to_numpy())
        sorted_cost = np.sort(pool_scores["proxy_total_cost"].astype(float).to_numpy())
        expected_risk_min = float(sorted_risk[:k].sum())
        expected_cost_min = float(sorted_cost[:k].sum())
        expected_risk_cap = 0.35 * k
        expected_cost_cap = float(lock["policy"]["cost_budget_per_item"]) * k
        expected_risk_cardinality = int(
            np.count_nonzero(np.cumsum(sorted_risk[:k]) <= expected_risk_cap)
        )
        expected_cost_cardinality = int(
            np.count_nonzero(np.cumsum(sorted_cost[:k]) <= expected_cost_cap)
        )
        lowest_risk = (
            json.loads(str(row.five_lowest_risk_values)) if k == 5 else []
        )
        risk_formula_ok = risk_formula_ok and (
            np.isclose(float(row.legacy_risk_budget_per_item), 0.35, rtol=0, atol=1e-12)
            and np.isclose(float(row.legacy_total_risk_cap), expected_risk_cap, rtol=0, atol=1e-12)
            and np.isclose(
                float(row.minimum_possible_total_risk_for_exact_batch),
                expected_risk_min,
                rtol=0,
                atol=1e-12,
            )
            and _boolish(row.full_batch_risk_feasible) == (expected_risk_min <= expected_risk_cap)
            and int(row.max_feasible_cardinality_under_legacy_cap)
            == expected_risk_cardinality
            and np.isclose(
                float(row.minimum_risk_budget_per_item_for_full_batch),
                expected_risk_min / k,
                rtol=0,
                atol=1e-12,
            )
            and np.isclose(float(row.total_cost_cap), expected_cost_cap, rtol=0, atol=1e-12)
            and np.isclose(
                float(row.minimum_possible_total_cost_for_exact_batch),
                expected_cost_min,
                rtol=0,
                atol=1e-12,
            )
            and _boolish(row.full_batch_cost_feasible) == (expected_cost_min <= expected_cost_cap)
            and int(row.max_feasible_cardinality_under_cost_cap)
            == expected_cost_cardinality
            and (
                k != 5
                or np.allclose(
                    lowest_risk,
                    sorted_risk[:5],
                    rtol=0,
                    atol=1e-12,
                )
            )
            and str(row.legacy_risk_contract_role) == "stress_test_only"
            and str(row.risk_score_interpretation)
            == "benchmark-defined uncalibrated proxy"
            and not _boolish(row.primary_selected_risk)
        )
    batch5_cost_feasible = risk[
        risk["requested_batch_size"].astype(int).eq(5)
    ]["full_batch_cost_feasible"].map(_boolish).all()
    _record(
        checks,
        details,
        "risk_and_cost_feasibility_mathematics",
        len(risk) == 6
        and risk_pairs == expected_risk_pairs
        and risk_formula_ok
        and batch5_cost_feasible,
    )

    primary = _read_csv("results/reproduction/test_policy_comparison_primary_cost_only.csv")
    primary_risk_null = primary["risk_budget_per_item"].map(_nullish).all()
    primary_ids = {str(row.method): _ids(row.selected_candidate_ids) for row in primary.itertuples(index=False)}
    _record(
        checks,
        details,
        "primary_cost_only_full_batch_comparison",
        len(primary) == 5
        and set(primary["method"]) == PRIMARY_METHODS
        and primary["batch_size"].astype(int).eq(5).all()
        and primary["evaluation_split"].eq("test").all()
        and primary["policy_contract"].eq("cost_only").all()
        and primary["risk_contract"].eq("none").all()
        and primary["risk_role"].eq("descriptive_only").all()
        and primary_risk_null
        and primary["selected_count"].astype(int).eq(5).all()
        and primary["selection_event_count"].astype(int).eq(5).all()
        and primary["stop_event_count"].astype(int).eq(0).all()
        and primary["full_batch"].map(_boolish).all()
        and all(len(ids) == len(set(ids)) == 5 for ids in primary_ids.values()),
    )

    random_raw = _read_csv("results/reproduction/random_baseline_primary_cost_only_raw.csv")
    random_summary = _read_csv("results/reproduction/random_baseline_primary_cost_only_summary.csv")
    _record(
        checks,
        details,
        "primary_random_design_100",
        set(random_raw["batch_size"].astype(int)) == {5, 10, 20}
        and random_raw.groupby("batch_size").size().eq(100).all()
        and set(random_summary["batch_size"].astype(int)) == {5, 10, 20}
        and random_summary["n_runs"].astype(int).eq(100).all()
        and ("risk_budget_per_item" not in random_raw or random_raw["risk_budget_per_item"].map(_nullish).all()),
    )
    pool_raw = _read_csv("results/reproduction/test_pool_resampling_primary_cost_only_raw.csv")
    random_pool = pool_raw[pool_raw["method"].eq("random_baseline")]
    _record(
        checks,
        details,
        "primary_source_group_pool_design_20_10",
        set(pool_raw["batch_size"].astype(int)) == {5, 10, 20}
        and pool_raw["pool_seed"].astype(int).nunique() == 20
        and random_pool.groupby(["pool_seed", "batch_size"])["random_seed"].nunique().eq(10).all()
        and ("risk_budget_per_item" not in pool_raw or pool_raw["risk_budget_per_item"].map(_nullish).all()),
    )

    stress = _read_csv("results/reproduction/diagnostics/test_policy_risk_stress_legacy_0p35.csv")
    _record(
        checks,
        details,
        "legacy_risk_stress_not_primary",
        stress["policy_contract"].eq("legacy_risk_stress_0p35").all()
        and len(stress) == 15
        and set(stress["batch_size"].astype(int)) == {5, 10, 20}
        and set(stress["method"]) == PRIMARY_METHODS
        and stress["contract_role"].eq("stress_test_only").all()
        and np.allclose(
            stress["risk_budget_per_item"].astype(float),
            0.35,
            rtol=0,
            atol=1e-12,
        )
        and stress["risk_interpretation"].eq(
            "benchmark-defined uncalibrated proxy"
        ).all(),
    )

    figure_manifest = _read_csv("source_data/figure_source_data_manifest.csv")
    figure3_sources = " ".join(
        figure_manifest.loc[
            figure_manifest["figure_file"].astype(str).str.contains("Figure3_policy_comparison"),
            "source",
        ].astype(str)
    )
    required_figure3_sources = {
        "figure3_test_policy_primary_cost_source.csv",
        "figure3_random_baseline_primary_cost_raw_points.csv",
        "figure3_random_baseline_primary_cost_summary_source.csv",
        "figure3_pool_resampling_primary_cost_raw_points.csv",
        "figure3_pool_resampling_primary_cost_summary_source.csv",
    }
    _record(
        checks,
        details,
        "figure3_primary_cost_only_sources",
        all(name in figure3_sources for name in required_figure3_sources)
        and "common_risk" not in figure3_sources,
    )

    trace_checks: dict[str, bool] = {}
    for method, relative in PRIMARY_TRACE_PATHS.items():
        trace = _read_json(relative)
        config = trace.get("config", {})
        trace_risk = config.get("risk_budget_per_item", config.get("risk_budget"))
        trace_ids = [str(item["candidate_id"]) for item in trace.get("selected_candidates", [])]
        trace_checks[method] = bool(
            trace.get("method") == method
            and trace_risk is None
            and config.get("policy_contract", config.get("primary_policy_mode", "cost_only"))
            == "cost_only"
            and trace.get("full_batch") is True
            and int(trace.get("selection_event_count", -1)) == 5
            and int(trace.get("stop_event_count", -1)) == 0
            and trace_ids == primary_ids.get(method)
            and trace.get("locked_parameters_record")
            == "results/reproduction/tuning/v49_locked_parameters_before_test.json"
            and trace.get("locked_parameters_record_sha256")
            == sha256_file(ROOT / "results/reproduction/tuning/v49_locked_parameters_before_test.json")
            and trace.get("locked_parameters") == lock
        )
    _record(checks, details, "primary_trace_cost_only_contract", all(trace_checks.values()), trace_checks)

    replay = _read_json("results/reproduction/replay_summary.json")
    _record(
        checks,
        details,
        "replay_5_of_5",
        replay.get("status") == "PASS"
        and int(replay.get("trace_count", -1)) == 5
        and int(replay.get("passed_count", -1)) == 5,
        replay,
    )

    required_metric_keys = {
        "test_positive_fraction",
        "test_majority_class_fraction_descriptive",
        "dummy_prior_accuracy",
        "dummy_prior_roc_auc",
        "dummy_prior_brier",
        "dummy_most_frequent_accuracy",
        "test_source_group_bootstrap_auc_ci_low",
        "test_source_group_bootstrap_auc_ci_high",
        "test_source_group_bootstrap_auc_standard_error",
        "bootstrap_resamples_requested",
        "bootstrap_valid_auc_resamples",
        "bootstrap_invalid_auc_resamples",
        "bootstrap_group_unit",
    }
    old_majority_ok = "test_majority_class_accuracy" not in metrics or str(
        metrics.get("test_majority_class_accuracy_status", "")
    ).upper() == "DEPRECATED_DESCRIPTIVE_ONLY"
    _record(
        checks,
        details,
        "activation_metrics_v49_schema",
        required_metric_keys <= set(metrics)
        and metrics.get("bootstrap_group_unit") == "normalized_source_doi"
        and metrics.get("bootstrap_resamples_requested") == 10000
        and old_majority_ok,
    )

    contract = _read_json("results/reproduction/regression_contract.json")
    exact_contract = contract.get("v49_exact_discrete_requirements", {})
    cpu = contract.get("canonical_platform", {}).get("cpu_dispatch_contract", {})
    _record(
        checks,
        details,
        "regression_contract_v49",
        contract.get("schema_version") == "4.0"
        and exact_contract.get("primary_policy_contract") == "cost_only"
        and exact_contract.get("primary_risk_budget") is None
        and exact_contract.get("legacy_risk_stress_budget_per_item") == 0.35
        and exact_contract.get("dummy_baseline_strategies") == ["prior", "most_frequent"]
        and exact_contract.get("bootstrap_requested_resamples") == 10000
        and exact_contract.get("bootstrap_seed") == 45
        and cpu.get("openblas_coretype") == "Haswell"
        and cpu.get("runtime_gate_must_precede_scientific_execution") is True,
    )

    stored_manifest = _read_json(MANIFEST_REL)
    _record(
        checks,
        details,
        "generated_artifact_manifest_current",
        stored_manifest == build_generated_artifact_manifest(ROOT),
    )
    return _finish(checks, details)


def _finish(checks: dict[str, bool], details: dict[str, Any]) -> dict[str, Any]:
    errors = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "v49-scientific-gate-1.0",
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "details": details,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Paper13 v49 scientific contract")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_repository_state()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
