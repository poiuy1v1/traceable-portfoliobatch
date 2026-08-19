from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import traceable_portfoliobatch  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from traceable_portfoliobatch.io_utils import sha256_file
from traceable_portfoliobatch.regression import (
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
EXPECTED_UPSTREAM = {
    "commit": "5693968b3e9b9e26eab3bdb1db908ae2877d4bb7",
    "git_blob_sha1": "dc67ae0a5b72863cdc8c6850257dab40ede74b1c",
    "raw_sha256": "fb62d307feac907d63de7bb2bcfd2880abaa60b751c185f78a3556ae20b88165",
}
EXCLUSION_REASON = "source DOI group also present in training"
LOCK_REL = "results/reproduction/tuning/v45A3_locked_parameters_before_test.json"
POLICY_GRID = {
    "beta": (0.0, 0.15, 0.30),
    "gamma": (0.10, 0.20, 0.35),
    "delta": (0.0, 0.10, 0.20),
    "risk_budget_per_item": (0.16, 0.18, 0.20, 0.25, 0.35),
}
POLICY_FIELDS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "diversity_distance_scale",
    "risk_budget_per_item",
    "cost_budget_per_item",
]
POLICY_SORT_FIELDS = [
    "full_batch",
    "hits",
    "proxy_cost_normalized_yield",
    "risk_used",
    "gamma",
    "beta",
    "delta",
    "risk_budget_per_item",
]
POLICY_SORT_ASCENDING = [False, False, False, True, True, True, True, True]


def _read_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_normalized_doi(value: object) -> bool:
    text = "" if pd.isna(value) else str(value).strip().lower()
    return re.fullmatch(r"10\.\d{4,9}/\S+", text) is not None


def _validate_policy_selection_table(
    table: pd.DataFrame,
    locked_policy: dict[str, Any],
    calibration_cost_median: float,
) -> tuple[bool, str]:
    """Independently validate the complete calibration grid and its winner."""
    required = set(POLICY_FIELDS + POLICY_SORT_FIELDS + ["selected"])
    missing = sorted(required - set(table.columns))
    if missing:
        return False, f"missing policy columns={missing}"

    numeric_columns = sorted(required - {"selected"})
    numeric = table.copy()
    try:
        for column in numeric_columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="raise")
    except (TypeError, ValueError) as exc:
        return False, f"nonnumeric policy field: {exc}"
    if (
        numeric[numeric_columns].isna().any().any()
        or not np.isfinite(numeric[numeric_columns].to_numpy(dtype=float)).all()
    ):
        return False, "policy table contains nonfinite or missing numeric values"

    selected_text = table["selected"].astype(str).str.strip().str.lower()
    if not selected_text.isin({"true", "false"}).all():
        return False, "selected must contain only exact boolean values"
    selected_mask = selected_text.eq("true")

    expected_grid = {
        tuple(round(float(value), 12) for value in values)
        for values in product(
            POLICY_GRID["beta"],
            POLICY_GRID["gamma"],
            POLICY_GRID["delta"],
            POLICY_GRID["risk_budget_per_item"],
        )
    }
    observed_grid = [
        tuple(round(float(row[column]), 12) for column in POLICY_GRID)
        for row in numeric.to_dict("records")
    ]
    expected_cost = 1.5 * float(calibration_cost_median)
    grid_ok = (
        len(numeric) == 135
        and len(set(observed_grid)) == 135
        and set(observed_grid) == expected_grid
        and np.allclose(numeric["alpha"], 1.0, atol=0, rtol=0)
        and np.allclose(
            numeric["diversity_distance_scale"], 2.0, atol=0, rtol=0
        )
        and np.allclose(
            numeric["cost_budget_per_item"], expected_cost, atol=1e-12, rtol=0
        )
    )
    if not grid_ok:
        missing_grid = sorted(expected_grid - set(observed_grid))[:5]
        unexpected_grid = sorted(set(observed_grid) - expected_grid)[:5]
        return False, (
            "policy table is not the exact 135-row Cartesian calibration grid; "
            f"rows={len(numeric)}, unique={len(set(observed_grid))}, "
            f"missing_examples={missing_grid}, unexpected_examples={unexpected_grid}, "
            f"alpha_constant={np.allclose(numeric['alpha'], 1.0, atol=0, rtol=0)}, "
            "diversity_constant="
            f"{np.allclose(numeric['diversity_distance_scale'], 2.0, atol=0, rtol=0)}, "
            "cost_constant="
            f"{np.allclose(numeric['cost_budget_per_item'], expected_cost, atol=1e-12, rtol=0)}"
        )

    ranked = (
        numeric.assign(_original_row=np.arange(len(numeric), dtype=int))
        .sort_values(
            POLICY_SORT_FIELDS,
            ascending=POLICY_SORT_ASCENDING,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    if ranked[POLICY_SORT_FIELDS].duplicated().any():
        return False, "policy ranking does not have a unique total ordering"
    if ranked["_original_row"].astype(int).tolist() != list(range(135)):
        return False, "policy CSV order differs from the independently recomputed ranking"
    selected_indices = np.flatnonzero(selected_mask.to_numpy()).tolist()
    if selected_indices != [int(ranked.loc[0, "_original_row"])]:
        return False, "selected row is not the independently recomputed rank-1 policy"

    winner = ranked.iloc[0]
    if any(
        abs(float(winner[field]) - float(locked_policy[field])) > 1e-12
        for field in POLICY_FIELDS
    ):
        return False, "independently recomputed policy winner does not match the lock"
    if abs(float(locked_policy["cost_budget_per_item"]) - expected_cost) > 1e-12:
        return False, "locked policy cost budget does not equal 1.5 times calibration median"
    return True, "complete 135-row grid and deterministic rank-1 winner verified"


def validate_repository_state() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    def require(name: str, condition: Any, detail: str) -> None:
        passed = bool(condition)
        checks[name] = passed
        if not passed:
            errors.append(f"{name}: {detail}")

    required = [
        "data/processed/full_public_stability_source_manifest.csv",
        "data/derived/full_public_stability_features_relocked.csv",
        "data/processed/full_public_stability_manifest_scored.csv",
        "data/provenance/mofsimplify_training_source_provenance.json",
        "models/activation_logistic_train_split.json",
        "results/reproduction/training_source_doi_backfill_audit.json",
        "results/reproduction/source_group_audit.json",
        "results/reproduction/activation_model_metrics.json",
        "results/reproduction/train_held_source_overlap_exclusions.csv",
        "results/reproduction/tuning/activation_C_train_internal_SOURCE_GROUP_CV.csv",
        "results/reproduction/tuning/activation_train_group_cv_fold_audit.csv",
        "results/reproduction/tuning/portfolio_policy_calibration_selection.csv",
        "results/reproduction/tuning/portfolio_policy_group_bootstrap_stability.csv",
        LOCK_REL,
        "results/reproduction/tuning/locked_parameters.json",
        "results/reproduction/test_policy_comparison_common_risk.csv",
        "results/reproduction/replay_summary.json",
        "results/reproduction/random_baseline_full_test_distribution.csv",
        "results/reproduction/test_pool_resampling_raw.csv",
        "results/reproduction/regression_contract.json",
        MANIFEST_REL,
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    require("required_files", not missing, f"missing={missing}")
    if missing:
        return {
            "status": "FAIL",
            "schema_version": "v45A3-scientific-gate-1.0",
            "checks": checks,
            "errors": errors,
        }

    source = pd.read_csv(
        ROOT / "data/processed/full_public_stability_source_manifest.csv",
        low_memory=False,
    )
    relocked = pd.read_csv(
        ROOT / "data/derived/full_public_stability_features_relocked.csv",
        low_memory=False,
    )
    scored = pd.read_csv(
        ROOT / "data/processed/full_public_stability_manifest_scored.csv",
        low_memory=False,
    )
    exclusions = pd.read_csv(
        ROOT / "results/reproduction/train_held_source_overlap_exclusions.csv",
        low_memory=False,
    )
    cv = pd.read_csv(
        ROOT
        / "results/reproduction/tuning/activation_C_train_internal_SOURCE_GROUP_CV.csv"
    )
    folds = pd.read_csv(
        ROOT / "results/reproduction/tuning/activation_train_group_cv_fold_audit.csv"
    )
    policy_selection = pd.read_csv(
        ROOT / "results/reproduction/tuning/portfolio_policy_calibration_selection.csv"
    )
    policy_stability = pd.read_csv(
        ROOT / "results/reproduction/tuning/portfolio_policy_group_bootstrap_stability.csv"
    )
    common_policy = pd.read_csv(
        ROOT / "results/reproduction/test_policy_comparison_common_risk.csv"
    )
    provenance = _read_json("data/provenance/mofsimplify_training_source_provenance.json")
    backfill = _read_json("results/reproduction/training_source_doi_backfill_audit.json")
    source_audit = _read_json("results/reproduction/source_group_audit.json")
    activation = _read_json("results/reproduction/activation_model_metrics.json")
    lock = _read_json(LOCK_REL)
    compatibility_lock = _read_json("results/reproduction/tuning/locked_parameters.json")
    model = _read_json("models/activation_logistic_train_split.json")
    replay = _read_json("results/reproduction/replay_summary.json")
    contract = _read_json("results/reproduction/regression_contract.json")

    required_columns = {
        "candidate_id",
        "model_split",
        "original_model_split",
        "final_evaluation_split",
        "raw_refcode",
        "source_doi",
        "source_doi_normalized",
        "source_group_id",
        "source_group_basis",
        "exclusion_reason",
    }
    require(
        "source_manifest_columns",
        required_columns.issubset(source.columns),
        f"missing={sorted(required_columns - set(source.columns))}",
    )
    require(
        "source_manifest_identity",
        len(source) == 2179
        and source["candidate_id"].is_unique
        and set(source["candidate_id"]) == set(relocked["candidate_id"]),
        "expected 2,179 unique candidates matching the derived table",
    )

    split_counts = {
        key: int(value)
        for key, value in source["final_evaluation_split"].value_counts().to_dict().items()
    }
    require(
        "final_split_counts",
        split_counts == EXPECTED_SPLIT_COUNTS,
        f"observed={split_counts}, expected={EXPECTED_SPLIT_COUNTS}",
    )
    original_split = source["original_model_split"].astype(str)
    original_train_groups = set(
        source.loc[original_split.eq("train"), "source_group_id"].astype(str)
    )
    expected_final_split = original_split.copy()
    expected_excluded = original_split.isin(["calibration", "test"]) & source[
        "source_group_id"
    ].astype(str).isin(original_train_groups)
    expected_final_split.loc[expected_excluded] = "excluded_train_source_overlap"
    require(
        "original_assignment_preserved",
        original_split.isin(["train", "calibration", "test"]).all()
        and original_split.eq(source["model_split"].astype(str)).all()
        and source["final_evaluation_split"].astype(str).eq(expected_final_split).all(),
        (
            "every original training row must remain training and every original held "
            "row must retain its split or carry the exact train-DOI exclusion"
        ),
    )
    require(
        "source_group_basis",
        source["source_group_basis"].eq("normalized_source_doi").all()
        and source["source_group_id"].astype(str).eq(
            source["source_doi_normalized"].astype(str)
        ).all(),
        "every row must use normalized_source_doi as source_group_id",
    )

    active = source[source["final_evaluation_split"].isin(["train", "calibration", "test"])]
    require(
        "active_doi_complete",
        active["source_doi_normalized"].map(_valid_normalized_doi).all(),
        "all active train/calibration/test rows require syntactically valid normalized DOI",
    )
    require(
        "all_manifest_doi_syntax",
        source["source_doi_normalized"].map(_valid_normalized_doi).all(),
        "every canonical manifest row requires syntactically valid normalized DOI",
    )
    groups = {
        split: set(
            source.loc[
                source["final_evaluation_split"].eq(split), "source_group_id"
            ].astype(str)
        )
        for split in EXPECTED_SPLIT_COUNTS
    }
    group_counts = {split: len(values) for split, values in groups.items()}
    require(
        "final_group_counts",
        group_counts == EXPECTED_GROUP_COUNTS,
        f"observed={group_counts}, expected={EXPECTED_GROUP_COUNTS}",
    )
    overlap_counts = {
        "train_calibration": len(groups["train"] & groups["calibration"]),
        "train_test": len(groups["train"] & groups["test"]),
        "calibration_test": len(groups["calibration"] & groups["test"]),
    }
    require(
        "active_source_group_disjoint",
        set(overlap_counts.values()) == {0},
        f"observed={overlap_counts}",
    )

    excluded = source[
        source["final_evaluation_split"].eq("excluded_train_source_overlap")
    ]
    require(
        "conservative_exclusions",
        excluded["original_model_split"].isin(["calibration", "test"]).all()
        and excluded["source_group_id"].astype(str).isin(groups["train"]).all()
        and excluded["exclusion_reason"].eq(EXCLUSION_REASON).all()
        and set(exclusions["candidate_id"]) == set(excluded["candidate_id"])
        and exclusions["exclusion_reason"].eq(EXCLUSION_REASON).all(),
        "all and only held rows sharing a training DOI must carry the exact exclusion reason",
    )
    require(
        "excluded_rows_not_scored",
        set(excluded["candidate_id"]).isdisjoint(set(scored["candidate_id"]))
        and set(scored["final_evaluation_split"]) == {"calibration", "test"}
        and len(scored) == EXPECTED_SPLIT_COUNTS["calibration"] + EXPECTED_SPLIT_COUNTS["test"],
        "scored table must contain only active calibration and test rows",
    )

    require(
        "upstream_provenance",
        provenance.get("upstream_repository") == "hjkgrp/MOFSimplify"
        and provenance.get("upstream_commit") == EXPECTED_UPSTREAM["commit"]
        and provenance.get("upstream_git_blob_sha1") == EXPECTED_UPSTREAM["git_blob_sha1"]
        and provenance.get("raw_file_sha256") == EXPECTED_UPSTREAM["raw_sha256"]
        and provenance.get("row_count") == 1394
        and provenance.get("dataset_doi") == "10.5281/zenodo.5736562"
        and provenance.get("mofsimplify_dataset_version") == "1.1.0",
        "official immutable upstream provenance does not match the authenticated source",
    )
    require(
        "backfill_audit",
        backfill.get("status") == "PASS"
        and backfill["rows"].get("training_rows_with_doi") == 1394
        and backfill["rows"].get("unresolved") == 0
        and backfill["rows"].get("ambiguous_or_conflicting") == 0
        and backfill["rows"].get("held_rows_preserved") == 785
        and backfill.get("active_source_doi_group_overlaps")
        == {"train_calibration": 0, "train_test": 0, "calibration_test": 0},
        "backfill audit must pass with complete training DOI and preserved held rows",
    )
    require(
        "source_group_audit",
        source_audit.get("status") == "PASS_COMPLETE_ACTIVE_SOURCE_GROUP_DISJOINT"
        and source_audit.get("source_group_basis") == "normalized_source_doi"
        and source_audit.get("active_training_doi_coverage") == 1.0
        and source_audit.get("active_calibration_doi_coverage") == 1.0
        and source_audit.get("active_test_doi_coverage") == 1.0
        and source_audit.get("train_calibration_source_doi_overlap_count") == 0
        and source_audit.get("train_test_source_doi_overlap_count") == 0
        and source_audit.get("policy_calibration_test_source_doi_overlap_count") == 0,
        "persisted source-group audit is incomplete or inconsistent",
    )

    selected_rows = cv[cv["selected_one_standard_error"].astype(bool)]
    selected_c = float(selected_rows.iloc[0]["C"]) if len(selected_rows) == 1 else None
    cv_ranked = cv.sort_values(
        ["mean_train_cv_roc_auc", "mean_train_cv_brier", "C"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    best_mean = float(cv_ranked.loc[0, "mean_train_cv_roc_auc"])
    best_sd = float(cv_ranked.loc[0, "sd_train_cv_roc_auc"])
    one_se_threshold = best_mean - best_sd / np.sqrt(5)
    expected_within_one_se = cv["mean_train_cv_roc_auc"].ge(one_se_threshold)
    expected_selected_c = float(cv.loc[expected_within_one_se, "C"].min())
    parsed_iterations: list[list[int]] = []
    try:
        parsed_iterations = [json.loads(value) for value in cv["fold_n_iter_values"]]
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed_iterations = []
    require(
        "group_cv_design",
        len(cv) == 13
        and len(folds) == 5
        and set(folds["fold"].astype(int)) == {1, 2, 3, 4, 5}
        and folds["group_overlap_count"].eq(0).all()
        and folds["train_positive_fraction"].between(0, 1, inclusive="neither").all()
        and folds["validation_positive_fraction"].between(0, 1, inclusive="neither").all()
        and set(cv["cv_splitter"]) == {"StratifiedGroupKFold"}
        and set(cv["random_state"].astype(int)) == {45}
        and set(cv["n_splits"].astype(int)) == {5}
        and set(cv["training_doi_group_count"].astype(int)) == {865},
        "five fixed, class-valid, source-DOI-disjoint StratifiedGroupKFold folds are required",
    )
    require(
        "group_cv_convergence",
        bool(cv["all_folds_converged"].astype(bool).all())
        and bool((cv["max_n_iter"] < cv["max_iter"]).all())
        and len(parsed_iterations) == len(cv)
        and all(len(values) == 5 for values in parsed_iterations)
        and all(max(values) == int(row_max) for values, row_max in zip(parsed_iterations, cv["max_n_iter"])),
        "every C/fold fit must converge before max_iter and record five iteration counts",
    )
    require(
        "one_standard_error_selection",
        len(selected_rows) == 1
        and bool(selected_rows.iloc[0]["within_one_standard_error"])
        and selected_c == expected_selected_c
        and cv["within_one_standard_error"].astype(bool).equals(
            expected_within_one_se.reset_index(drop=True)
        ),
        (
            "exactly the minimum C within the independently recomputed one-standard-error "
            f"threshold must be selected; expected_C={expected_selected_c}, "
            f"threshold={one_se_threshold}"
        ),
    )

    model_c = float(model.get("selected_C", model.get("parameters", {}).get("C", float("nan"))))
    require(
        "selected_C_consistency",
        selected_c is not None
        and selected_c == float(activation["selected_C"])
        and selected_c == float(lock["activation_C"])
        and selected_c == model_c
        and selected_c == float(model["parameters"]["C"]),
        "selected C must match CV, metrics, lock, and fitted model artifact",
    )
    require(
        "activation_convergence",
        activation.get("all_activation_fits_converged") is True
        and int(activation["final_fit_n_iter"]) < int(activation["max_iter"])
        and int(activation["maximum_cv_n_iter"]) < int(activation["max_iter"])
        and activation.get("cv_splitter") == "StratifiedGroupKFold"
        and int(activation.get("cv_fold_group_overlap_count_max", -1)) == 0,
        "activation CV and final fit must converge with zero group overlap",
    )

    lock_path = ROOT / LOCK_REL
    lock_hash = sha256_file(lock_path)
    require(
        "pretest_lock_hash",
        compatibility_lock.get("authoritative_pre_test_lock") == LOCK_REL
        and compatibility_lock.get("authoritative_pre_test_lock_sha256") == lock_hash
        and activation.get("locked_parameters_before_test_sha256") == lock_hash
        and compatibility_lock
        == {
            **lock,
            "authoritative_pre_test_lock": LOCK_REL,
            "authoritative_pre_test_lock_sha256": lock_hash,
        },
        "authoritative pre-test lock hash/payload must propagate unchanged",
    )
    hash_checks = {
        lock["source_manifest"]: lock["training_source_manifest_sha256"],
        lock["final_split_manifest"]: lock["final_split_manifest_sha256"],
        lock["activation_model_artifact"]: lock["activation_model_artifact_sha256"],
        lock["environment_lock"]: lock["environment_lock_sha256"],
        **lock["code_hashes"],
    }
    require(
        "pretest_lock_file_hashes",
        all(sha256_file(ROOT / path) == expected for path, expected in hash_checks.items())
        and lock["activation_feature_list_sha256"]
        == _json_sha256(lock["activation_feature_list"])
        and lock["code_sha256"] == _json_sha256(lock["code_hashes"]),
        "one or more files or structured lists no longer match the pre-test lock",
    )
    require(
        "pretest_lock_design",
        lock.get("test_evaluation_may_start_only_after_this_file_is_written") is True
        and lock.get("test_use") == "none during model or policy tuning"
        and lock.get("test_use_claim")
        == "Within the final locked evaluation pipeline, test labels are not used for model or policy selection."
        and lock.get("training_cv")
        == {
            "splitter": "StratifiedGroupKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 45,
            "group_key": "normalized_source_doi",
            "group_overlap_count_max": 0,
        }
        and lock.get("batch_sizes") == [5, 10, 20]
        and lock.get("final_evaluation_row_counts") == EXPECTED_SPLIT_COUNTS,
        "pre-test lock does not contain the complete fixed v45A3 design",
    )

    calibration_cost_median = float(
        scored.loc[
            scored["final_evaluation_split"].eq("calibration"), "proxy_total_cost"
        ].median()
    )
    policy_grid_ok, policy_grid_detail = _validate_policy_selection_table(
        policy_selection, lock["policy"], calibration_cost_median
    )
    require(
        "policy_selection_complete_grid_and_winner",
        policy_grid_ok,
        policy_grid_detail,
    )
    require(
        "policy_selection_matches_lock",
        policy_grid_ok
        and abs(calibration_cost_median - float(lock["cost_scale_calibration_median"]))
        <= 1e-12,
        "complete independently ranked policy search must match the lock and cost scale",
    )
    require(
        "policy_group_bootstrap_design",
        len(policy_stability) == 20
        and set(policy_stability["bootstrap_seed"].astype(int)) == set(range(20))
        and policy_stability["calibration_group_count"].gt(0).all()
        and policy_stability["equivalent_top_config_count"].gt(0).all()
        and np.allclose(
            policy_stability["cost_budget_per_item"],
            lock["policy"]["cost_budget_per_item"],
            atol=1e-12,
            rtol=0,
        )
        and lock.get("policy_calibration_group_bootstrap")
        == {
            "unit": "normalized source DOI group",
            "bootstrap_resamples": 20,
            "bootstrap_seeds": "0..19",
        },
        "calibration policy stability must use 20 fixed normalized-DOI-group bootstraps",
    )

    random_full = pd.read_csv(
        ROOT / "results/reproduction/random_baseline_full_test_distribution.csv"
    )
    pool = pd.read_csv(ROOT / "results/reproduction/test_pool_resampling_raw.csv")
    full_counts = random_full.groupby("batch_size").size().to_dict()
    random_pool = pool[pool["method"].eq("random_baseline")]
    seeds_per_pool = random_pool.groupby(["pool_seed", "batch_size"])[
        "random_seed"
    ].nunique()
    require(
        "random_design_100_20_10",
        full_counts == {5: 100, 10: 100, 20: 100}
        and all(
            set(group["seed"].astype(int)) == set(range(100))
            for _, group in random_full.groupby("batch_size")
        )
        and set(pool["pool_seed"].astype(int)) == set(range(20))
        and set(pool["batch_size"].astype(int)) == {5, 10, 20}
        and len(seeds_per_pool) == 60
        and seeds_per_pool.eq(10).all()
        and all(
            set(group["random_seed"].astype(int)) == set(range(10))
            for _, group in random_pool.groupby(["pool_seed", "batch_size"])
        )
        and lock["random_baseline_seed_policy"]["primary_trace_seed"] == 44
        and lock["random_baseline_seed_policy"]["full_pool_seeds"]
        == "0..99 per requested batch size"
        and lock["random_baseline_seed_policy"]["full_pool_runs_per_batch_size"] == 100
        and lock["random_baseline_seed_policy"]["batch_sensitivity_seed_offset"] == 4400
        and lock["random_baseline_seed_policy"]["batch_sensitivity_seeds_by_batch_size"]
        == {"5": 4405, "10": 4410, "20": 4420}
        and lock["pool_resampling_design"]["unit"] == "normalized source DOI group"
        and lock["pool_resampling_design"]["fraction"] == 0.5
        and lock["pool_resampling_design"]["pool_resamples"] == 20
        and lock["pool_resampling_design"]["pool_seeds"] == "0..19"
        and lock["pool_resampling_design"][
            "random_seeds_per_resampled_pool_and_batch_size"
        ]
        == 10,
        "random design must remain 100 full-pool runs, 20 group pools, and 10 seeds per pool",
    )

    trace_paths = sorted((ROOT / "traces/action_traces").glob("*.json"))
    trace_ok = len(trace_paths) == 5
    trace_configs_ok = trace_ok
    trace_event_counts_ok = trace_ok
    trace_by_method: dict[str, dict[str, Any]] = {}
    for path in trace_paths:
        trace = json.loads(path.read_text(encoding="utf-8"))
        trace_by_method[trace["method"]] = trace
        trace_ok = trace_ok and (
            trace.get("schema_version") == "3.0"
            and trace.get("canonical_source_manifest_sha256")
            == sha256_file(ROOT / trace["canonical_source_manifest"])
            and trace.get("locked_parameters_record") == LOCK_REL
            and trace.get("locked_parameters_record_sha256") == lock_hash
            and trace.get("locked_parameters") == lock
            and trace.get("evaluation_split") == "test"
        )
        config = trace["config"]
        batch_size = int(config["batch_size"])
        expected_seed = 44 if trace["method"] == "random_baseline" else None
        trace_configs_ok = trace_configs_ok and (
            batch_size == 5
            and all(
                abs(float(config[field]) - float(lock["policy"][field])) <= 1e-12
                for field in [
                    "alpha",
                    "beta",
                    "gamma",
                    "delta",
                    "diversity_distance_scale",
                ]
            )
            and abs(
                float(config["cost_scale"])
                - float(lock["cost_scale_calibration_median"])
            )
            <= 1e-12
            and abs(
                float(config["cost_budget"])
                - float(lock["policy"]["cost_budget_per_item"]) * batch_size
            )
            <= 1e-12
            and abs(
                float(config["risk_budget"])
                - float(lock["policy"]["risk_budget_per_item"]) * batch_size
            )
            <= 1e-12
            and config["random_seed"] == expected_seed
        )
        selection_events = sum(
            step.get("status") == "selected" for step in trace["stepwise_decisions"]
        )
        stop_events = sum(
            str(step.get("status", "")).startswith("stopped")
            for step in trace["stepwise_decisions"]
        )
        trace_event_counts_ok = trace_event_counts_ok and (
            trace.get("stepwise_decision_count") == len(trace["stepwise_decisions"])
            and trace.get("selection_event_count") == selection_events
            and trace.get("stop_event_count") == stop_events
            and trace.get("full_batch") == (selection_events == batch_size)
        )
    require(
        "trace_source_and_lock_binding",
        trace_ok,
        "all five schema-3 test traces must bind the source manifest and authoritative lock",
    )
    require(
        "trace_configs_match_lock",
        trace_configs_ok,
        "every trace config must be derived from the embedded authoritative policy/seed lock",
    )
    require(
        "trace_event_counts",
        trace_event_counts_ok,
        "every trace must explicitly and correctly record selection, stop, and full-batch state",
    )
    common_columns = {
        "selected_candidate_ids",
        "full_batch",
        "full_batch_status",
        "selection_event_count",
        "stop_event_count",
    }
    common_events_ok = common_columns.issubset(common_policy.columns)
    if common_events_ok:
        for row in common_policy.to_dict("records"):
            trace = trace_by_method.get(row["method"], {})
            try:
                ids = json.loads(row["selected_candidate_ids"])
            except (TypeError, json.JSONDecodeError):
                ids = []
            common_events_ok = common_events_ok and (
                ids
                == [item["candidate_id"] for item in trace.get("selected_candidates", [])]
                and int(row["selection_event_count"])
                == int(trace.get("selection_event_count", -1))
                and int(row["stop_event_count"]) == int(trace.get("stop_event_count", -1))
                and bool(row["full_batch"]) == bool(trace.get("full_batch"))
                and row["full_batch_status"] == trace.get("full_batch_status")
            )
    require(
        "primary_policy_reports_ids_and_stop_events",
        common_events_ok,
        "primary common-risk table must report IDs, selection/stop counts, and full-batch status consistent with traces",
    )
    require(
        "replay_5_of_5",
        replay.get("status") == "PASS"
        and replay.get("trace_count") == 5
        and replay.get("passed_count") == 5
        and not replay.get("failed"),
        "exact archived-implementation replay must pass for all five traces",
    )

    exact_contract = contract.get("v45A3_exact_discrete_requirements", {})
    require(
        "regression_contract_v45A3",
        contract.get("schema_version") == "3.0"
        and exact_contract.get("active_training_doi_coverage") == 1.0
        and exact_contract.get("active_calibration_doi_coverage") == 1.0
        and exact_contract.get("active_test_doi_coverage") == 1.0
        and exact_contract.get("train_calibration_source_doi_overlap_count") == 0
        and exact_contract.get("train_test_source_doi_overlap_count") == 0
        and exact_contract.get("calibration_test_source_doi_overlap_count") == 0
        and exact_contract.get("activation_cv_splitter") == "StratifiedGroupKFold"
        and exact_contract.get("activation_cv_fold_group_overlap_count") == 0,
        "schema-3 contract must retain the v45A3 exact scientific invariants",
    )
    require(
        "generated_artifact_manifest_current",
        _read_json(MANIFEST_REL) == build_generated_artifact_manifest(ROOT),
        "generated artifact manifest differs from a fresh reconstruction",
    )

    details = {
        "split_counts": split_counts,
        "group_counts": group_counts,
        "active_source_group_overlap_counts": overlap_counts,
        "selected_C": selected_c,
        "cv_splitter": "StratifiedGroupKFold",
        "cv_fold_group_overlap_count_max": int(folds["group_overlap_count"].max()),
        "pretest_lock_sha256": lock_hash,
        "replay": {
            "status": replay.get("status"),
            "passed_count": replay.get("passed_count"),
            "trace_count": replay.get("trace_count"),
        },
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "schema_version": "v45A3-scientific-gate-1.0",
        "checks": checks,
        "errors": errors,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently validate the v45A3 scientific bootstrap invariants"
    )
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
