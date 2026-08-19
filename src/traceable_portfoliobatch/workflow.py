from __future__ import annotations

import hashlib
import json
import re
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from .io_utils import write_csv, write_json
from .leakage import write_audit
from .modeling import (
    ACTIVATION_MAX_ITER,
    ACTIVATION_META,
    ACTIVATION_SOLVER,
    ACTIVATION_TOL,
    THERMAL_META,
    activation_pipeline,
    descriptor_proxy_cost,
    feature_columns,
    fit_activation_checked,
    normalized_binary_entropy,
    predict_from_artifact,
    save_linear_pipeline_artifact,
    select_activation_C_train_cv,
    select_thermal_alpha_cv,
    thermal_pipeline,
)
from .policies import PolicyConfig, config_dict, metrics, select_batch
from .regression import write_generated_artifact_manifest
from .tracing import sha256, write_trace
from .v49_science import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    LEGACY_RISK_BUDGET_PER_ITEM,
    RISK_SCORE_FORMULA,
    RISK_SCORE_INTERPRETATION,
    cost_feasibility_table,
    evaluate_training_fitted_dummy_classifiers,
    risk_feasibility_table,
    source_group_cluster_bootstrap,
)

METHODS = ["top_score", "uncertainty", "cost_aware", "portfolio_batch", "random_baseline"]
DETERMINISTIC_METHODS = METHODS[:-1]
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
DEFAULT_SOURCE_MANIFEST = "data/processed/full_public_stability_source_manifest.csv"
DERIVED_FEATURE_TABLE = "data/derived/full_public_stability_features_relocked.csv"
V49_LOCK_REL = "results/reproduction/tuning/v49_locked_parameters_before_test.json"
EXCLUDED_SPLIT = "excluded_train_source_overlap"
BATCH_SENSITIVITY_RANDOM_SEED_OFFSET = 4400
POOL_RANDOM_SEED_BASE = 100000


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _norm_doi(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    for prefix in ["https://doi.org/", "http://doi.org/", "doi:"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip().rstrip("/.,;")


def _is_valid_normalized_doi(value: object) -> bool:
    """Return whether a normalized value has the required DOI structure."""
    text = "" if pd.isna(value) else str(value).strip().lower()
    return re.fullmatch(r"10\.\d{4,9}/\S+", text) is not None


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _attach_source_groups(features: pd.DataFrame, source_manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach and validate the canonical DOI-complete final split manifest."""
    if features["candidate_id"].duplicated().any():
        raise ValueError("Feature table contains duplicate candidate_id values")
    if source_manifest["candidate_id"].duplicated().any():
        raise ValueError("Canonical source manifest contains duplicate candidate_id values")
    feature_ids = set(features["candidate_id"].astype(str))
    manifest_ids = set(source_manifest["candidate_id"].astype(str))
    if feature_ids != manifest_ids:
        raise ValueError(
            "Canonical source manifest must cover every feature row exactly: "
            f"missing={len(feature_ids - manifest_ids)}, extra={len(manifest_ids - feature_ids)}"
        )

    source_fields = {
        "source_doi", "source_doi_normalized", "source_url", "source_title",
        "source_group_id", "source_group_basis", "original_model_split",
        "final_evaluation_split", "exclusion_reason",
    }
    clean_features = features.drop(
        columns=[c for c in features.columns if c in source_fields],
        errors="ignore",
    ).copy()
    manifest = source_manifest.copy()
    if "source_doi" not in manifest:
        manifest["source_doi"] = ""
    manifest["source_doi"] = manifest["source_doi"].fillna("").astype(str)
    manifest["source_doi_normalized"] = manifest["source_doi"].map(_norm_doi)
    if manifest["source_doi_normalized"].eq("").any():
        examples = manifest.loc[
            manifest["source_doi_normalized"].eq(""), "candidate_id"
        ].head(10).tolist()
        raise ValueError(
            "Canonical source manifest contains unresolved DOI values; "
            f"examples={examples}"
        )
    invalid_doi = ~manifest["source_doi_normalized"].map(_is_valid_normalized_doi)
    if invalid_doi.any():
        examples = manifest.loc[
            invalid_doi, ["candidate_id", "source_doi_normalized"]
        ].head(10).to_dict("records")
        raise ValueError(
            "Canonical source manifest contains invalid DOI syntax; "
            f"examples={examples}"
        )
    keep = [
        c for c in [
            "candidate_id", "source_doi", "source_doi_normalized", "source_url",
            "source_title", "original_model_split", "final_evaluation_split",
            "exclusion_reason",
        ] if c in manifest.columns
    ]
    out = clean_features.merge(
        manifest[keep], on="candidate_id", how="left", validate="one_to_one"
    )
    out["source_doi_normalized"] = out["source_doi_normalized"].fillna("")
    out["source_group_id"] = out["source_doi_normalized"]
    out["source_group_basis"] = "normalized_source_doi"
    if "original_model_split" not in out:
        out["original_model_split"] = out["model_split"].astype(str)
    else:
        out["original_model_split"] = out["original_model_split"].fillna("").astype(str)
        mismatch = out["original_model_split"].ne(out["model_split"].astype(str))
        if mismatch.any():
            examples = out.loc[mismatch, "candidate_id"].head(10).tolist()
            raise ValueError(
                "Canonical original_model_split differs from the frozen feature table; "
                f"examples={examples}"
            )
    if "final_evaluation_split" not in out:
        out["final_evaluation_split"] = out["original_model_split"]
    else:
        out["final_evaluation_split"] = (
            out["final_evaluation_split"].fillna("").astype(str)
        )
    if "exclusion_reason" not in out:
        out["exclusion_reason"] = ""
    out["exclusion_reason"] = out["exclusion_reason"].fillna("").astype(str)

    allowed = {"train", "calibration", "test", EXCLUDED_SPLIT}
    invalid = ~out["final_evaluation_split"].isin(allowed)
    if invalid.any():
        values = sorted(out.loc[invalid, "final_evaluation_split"].unique().tolist())
        raise ValueError(f"Invalid final_evaluation_split values: {values}")

    train_groups = set(
        out.loc[out["original_model_split"].eq("train"), "source_group_id"]
    )
    held_mask = out["original_model_split"].isin(["calibration", "test"])
    expected_excluded = held_mask & out["source_group_id"].isin(train_groups)
    observed_excluded = out["final_evaluation_split"].eq(EXCLUDED_SPLIT)
    expected_final = out["original_model_split"].copy()
    expected_final.loc[expected_excluded] = EXCLUDED_SPLIT
    changed_membership = out["final_evaluation_split"].ne(expected_final)
    if changed_membership.any():
        examples = out.loc[
            changed_membership,
            ["candidate_id", "original_model_split", "final_evaluation_split"],
        ].head(10).to_dict("records")
        raise ValueError(
            "Final evaluation membership changed original training membership or "
            "failed the exact conservative held-row source-overlap rule; "
            f"examples={examples}"
        )
    bad_reason = observed_excluded & out["exclusion_reason"].ne(
        "source DOI group also present in training"
    )
    if bad_reason.any():
        raise ValueError("Excluded rows must carry the exact source-overlap exclusion reason")
    if ((~observed_excluded) & out["exclusion_reason"].ne("")).any():
        raise ValueError("Active rows must not carry an exclusion reason")

    active_groups = {
        split: set(out.loc[out["final_evaluation_split"].eq(split), "source_group_id"])
        for split in ["train", "calibration", "test"]
    }
    intersections = {
        "train-calibration": active_groups["train"] & active_groups["calibration"],
        "train-test": active_groups["train"] & active_groups["test"],
        "calibration-test": active_groups["calibration"] & active_groups["test"],
    }
    if any(intersections.values()):
        counts = {key: len(value) for key, value in intersections.items()}
        raise ValueError(f"Active source DOI group overlap remains: {counts}")

    # model_split remains as a compatibility alias for generated downstream
    # artifacts; the two explicit fields preserve provenance and final use.
    out["model_split"] = out["final_evaluation_split"]
    return out


def _score_table(split: pd.DataFrame, model, columns: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    probabilities = model.predict_proba(split[columns])[:, 1]
    z = model.named_steps["scaler"].transform(
        model.named_steps["imputer"].transform(split[columns])
    )
    uncertainty = normalized_binary_entropy(probabilities)
    cost = descriptor_proxy_cost(z)
    risk = 0.5 * (1 - probabilities) + 0.5 * uncertainty
    fields = [
        "candidate_id", "model_split", "raw_split", "raw_source_file",
        "raw_source_row", "raw_refcode", "outcome_binary", "source_doi",
        "source_doi_normalized", "source_group_id", "source_group_basis",
        "original_model_split", "final_evaluation_split", "exclusion_reason",
    ]
    base = split[[c for c in fields if c in split]].copy()
    base["predicted_success"] = probabilities
    base["uncertainty"] = uncertainty
    base["proxy_total_cost"] = cost
    base["cost_synthesis"] = cost * 2 / 3
    base["cost_characterization"] = cost / 3
    base["risk_score"] = risk
    base["score_source"] = "train_fit_logistic_C_selected_by_train_internal_CV"
    base["uncertainty_source"] = "normalized_binary_entropy_proxy_not_calibrated"
    base["cost_source"] = "1_plus_mean_absolute_train_standardized_descriptor_value"
    base["risk_source"] = (
        "0.5*(1-predicted_success)+0.5*uncertainty; benchmark-defined "
        "uncalibrated proxy; descriptive in the primary cost-only comparison "
        "and constrained only in the legacy stress test"
    )
    return base.reset_index(drop=True), z


def _fixed_C_source_overlap_ablation(
    features: pd.DataFrame,
    train: pd.DataFrame,
    columns: list[str],
    *,
    selected_C: float,
) -> tuple[pd.DataFrame, dict]:
    """Run the post-lock, fixed-C source-universe diagnostic.

    The returned table is deliberately aggregate-only: rows excluded from the
    final evaluation are never added to the primary scored manifest.
    """
    if float(selected_C) != 0.001:
        raise RuntimeError(
            "The unchanged training-only C selection drifted from 0.001; "
            "the fixed-C v49 diagnostic cannot proceed"
        )
    y_train = (train["outcome_binary"].astype(float) > 0).astype(int)
    universes = {
        "historical_overlap_inclusive_test": features[
            features["original_model_split"].eq("test")
        ].reset_index(drop=True),
        "final_publication_source_disjoint_test": features[
            features["final_evaluation_split"].eq("test")
        ].reset_index(drop=True),
    }
    decomposition = features[
        features["original_model_split"].eq("test")
        & features["final_evaluation_split"].eq(EXCLUDED_SPLIT)
    ].reset_index(drop=True)
    expected_counts = {
        "historical_overlap_inclusive_test": 436,
        "final_publication_source_disjoint_test": 193,
    }
    observed_counts = {name: len(frame) for name, frame in universes.items()}
    if observed_counts != expected_counts:
        raise RuntimeError(
            "Fixed-C diagnostic evaluation universes changed: "
            f"observed={observed_counts}, expected={expected_counts}"
        )

    rows: list[dict] = []
    decomposition_rows: list[dict] = []
    for C in [3.0, 0.001]:
        model = activation_pipeline(C)
        fit_activation_checked(
            model,
            train[columns],
            y_train,
            context=f"v49 post-lock fixed-C diagnostic C={C:g}",
        )
        model_role = (
            "current_source_group_cv_model"
            if C == 0.001
            else "historical_fixed_C_diagnostic_comparator"
        )
        for universe, frame in universes.items():
            y = (frame["outcome_binary"].astype(float) > 0).astype(int).to_numpy()
            if np.unique(y).size != 2:
                raise RuntimeError(
                    f"Fixed-C diagnostic universe {universe} lacks both outcome classes"
                )
            probability = model.predict_proba(frame[columns])[:, 1]
            rows.append({
                "C": C,
                "evaluation_universe": universe,
                "n_rows": int(len(frame)),
                "n_source_doi_groups": int(frame["source_group_id"].nunique()),
                "roc_auc": float(roc_auc_score(y, probability)),
                "accuracy_at_0_5": float(accuracy_score(y, probability >= 0.5)),
                "brier_score": float(brier_score_loss(y, probability)),
                "model_role": model_role,
                "diagnostic_only": True,
                "affects_selected_C": False,
            })
        y_decomposition = (
            decomposition["outcome_binary"].astype(float) > 0
        ).astype(int).to_numpy()
        probability_decomposition = model.predict_proba(decomposition[columns])[:, 1]
        decomposition_rows.append({
            "C": C,
            "evaluation_universe": "historical_test_excluded_for_train_source_overlap",
            "n_rows": int(len(decomposition)),
            "n_source_doi_groups": int(decomposition["source_group_id"].nunique()),
            "roc_auc": float(roc_auc_score(y_decomposition, probability_decomposition)),
            "accuracy_at_0_5": float(
                accuracy_score(y_decomposition, probability_decomposition >= 0.5)
            ),
            "brier_score": float(
                brier_score_loss(y_decomposition, probability_decomposition)
            ),
            "model_role": model_role,
            "diagnostic_only": True,
        })

    table = pd.DataFrame(rows)
    lookup = table.set_index(["C", "evaluation_universe"])["roc_auc"]
    split_C3 = float(
        lookup.loc[(3.0, "final_publication_source_disjoint_test")]
        - lookup.loc[(3.0, "historical_overlap_inclusive_test")]
    )
    split_C0001 = float(
        lookup.loc[(0.001, "final_publication_source_disjoint_test")]
        - lookup.loc[(0.001, "historical_overlap_inclusive_test")]
    )
    C_overlap = float(
        lookup.loc[(0.001, "historical_overlap_inclusive_test")]
        - lookup.loc[(3.0, "historical_overlap_inclusive_test")]
    )
    C_disjoint = float(
        lookup.loc[(0.001, "final_publication_source_disjoint_test")]
        - lookup.loc[(3.0, "final_publication_source_disjoint_test")]
    )
    interpretation_supported = bool(
        split_C3 < 0
        and split_C0001 < 0
        and min(abs(split_C3), abs(split_C0001))
        > max(abs(C_overlap), abs(C_disjoint))
    )
    summary = {
        "schema_version": "v49-source-overlap-C-ablation-1.0",
        "status": "PASS",
        "diagnostic_only": True,
        "affects_selected_C": False,
        "fixed_C_values": [0.001, 3.0],
        "selected_C_before_and_after_diagnostic": float(selected_C),
        "delta_definition": {
            "split_effect": (
                "ROC-AUC(final_publication_source_disjoint_test) - "
                "ROC-AUC(historical_overlap_inclusive_test) at fixed C"
            ),
            "C_effect": "ROC-AUC(C=0.001) - ROC-AUC(C=3.0) at fixed universe",
        },
        "split_effect_at_C_3": split_C3,
        "split_effect_at_C_0p001": split_C0001,
        "C_effect_on_overlap_inclusive_test": C_overlap,
        "C_effect_on_source_disjoint_test": C_disjoint,
        "historical_test_excluded_for_train_source_overlap": decomposition_rows,
        "source_universe_reduction_larger_than_fixed_C_effect": interpretation_supported,
        "claim_boundary": (
            "post-hoc publication-source diagnostic; no chemical-family, structural-"
            "similarity, or causal leakage claim"
        ),
    }
    return table, summary


def _policy_config(
    method: str,
    batch_size: int,
    cost_per_item: float,
    risk_per_item: float | None,
    cost_scale: float,
    tuned: dict,
    seed: int | None = None,
) -> PolicyConfig:
    return PolicyConfig(
        batch_size=int(batch_size),
        cost_budget=float(cost_per_item * batch_size),
        risk_budget=None if risk_per_item is None else float(risk_per_item * batch_size),
        alpha=float(tuned["alpha"]),
        beta=float(tuned["beta"]),
        gamma=float(tuned["gamma"]),
        delta=float(tuned["delta"]),
        cost_scale=float(cost_scale),
        diversity_distance_scale=float(tuned["diversity_distance_scale"]),
        random_seed=seed,
    )


def _tune_cost_only_policy(
    calibration: pd.DataFrame,
    z: np.ndarray,
    cost_scale: float,
    cost_per_item: float,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    for beta, gamma, delta in product(
        [0, 0.15, 0.30], [0.10, 0.20, 0.35], [0, 0.10, 0.20]
    ):
        values = {
            "alpha": 1.0,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "diversity_distance_scale": 2.0,
        }
        config = _policy_config(
            "portfolio_batch", 5, cost_per_item, None, cost_scale, values
        )
        selected, _, _ = select_batch(calibration, z, "portfolio_batch", config, record_details=False)
        result = metrics(selected)
        rows.append({
            **values,
            "cost_budget_per_item": cost_per_item,
            **result,
            "full_batch": int(result["selected_count"] == 5),
            "primary_policy_mode": "cost_only",
        })
    table = pd.DataFrame(rows).sort_values(
        [
            "full_batch", "hits", "proxy_cost_normalized_yield", "total_proxy_cost",
            "gamma", "beta", "delta",
        ],
        ascending=[False, False, False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    table["selected"] = False
    table.loc[0, "selected"] = True
    best = table.iloc[0]
    return {
        "alpha": float(best.alpha),
        "beta": float(best.beta),
        "gamma": float(best.gamma),
        "delta": float(best.delta),
        "diversity_distance_scale": float(best.diversity_distance_scale),
        "risk_budget_per_item": None,
        "cost_budget_per_item": float(best.cost_budget_per_item),
        "primary_policy_mode": "cost_only",
    }, table


def _cost_only_policy_stability(
    calibration: pd.DataFrame,
    z: np.ndarray,
    cost_scale: float,
    cost_per_item: float,
    n: int = 20,
) -> pd.DataFrame:
    groups = calibration["source_group_id"].astype(str).unique()
    rows = []
    group_values = calibration["source_group_id"].astype(str).to_numpy()
    for seed in range(n):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(groups, size=max(2, int(round(0.8 * len(groups)))), replace=True)
        indices = np.concatenate([np.flatnonzero(group_values == group) for group in chosen])
        subset = calibration.iloc[indices].reset_index(drop=True)
        z_subset = z[indices]
        tuned, table = _tune_cost_only_policy(
            subset, z_subset, cost_scale, cost_per_item
        )
        equivalent = (
            (table["full_batch"] == table.loc[0, "full_batch"])
            & (table["hits"] == table.loc[0, "hits"])
            & (
                table["proxy_cost_normalized_yield"]
                >= 0.99 * table.loc[0, "proxy_cost_normalized_yield"]
            )
        )
        rows.append({
            "bootstrap_seed": seed,
            **tuned,
            "calibration_group_count": len(set(chosen)),
            "equivalent_top_config_count": int(equivalent.sum()),
            "full_batch": bool(table.loc[0, "full_batch"]),
        })
    out = pd.DataFrame(rows)
    tuple_fields = ["beta", "gamma", "delta"]
    frequencies = out.groupby(tuple_fields)["bootstrap_seed"].transform("count")
    out["selected_tuple_frequency"] = frequencies / float(n)
    out["full_batch_frequency"] = float(out["full_batch"].astype(bool).mean())
    out["primary_policy_mode"] = "cost_only"
    return out


def _eval(
    pool: pd.DataFrame,
    z: np.ndarray,
    batch_size: int,
    risk: float | None,
    cost: float,
    cost_scale: float,
    tuned: dict,
    random_seed: int = 44,
    methods: list[str] = METHODS,
    detailed: bool = True,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    outputs = {}
    for method in methods:
        config = _policy_config(
            method,
            batch_size,
            cost,
            risk,
            cost_scale,
            tuned,
            random_seed if method == "random_baseline" else None,
        )
        selected, alternatives, steps = select_batch(pool, z, method, config, record_details=detailed)
        selection_event_count = sum(step.get("status") == "selected" for step in steps)
        stop_event_count = sum(
            str(step.get("status", "")).startswith("stopped") for step in steps
        )
        if not detailed:
            selection_event_count = int(len(selected))
            stop_event_count = int(len(selected) < batch_size)
        full_batch = bool(len(selected) == batch_size)
        rows.append({
            "method": method,
            "batch_size": batch_size,
            "risk_budget_per_item": risk,
            "cost_budget_per_item": cost,
            "selected_candidate_ids": json.dumps(
                selected["candidate_id"].astype(str).tolist(), separators=(",", ":")
            ),
            "full_batch": full_batch,
            "full_batch_status": (
                "full_batch" if full_batch else "stopped_before_requested_batch"
            ),
            "selection_event_count": int(selection_event_count),
            "stop_event_count": int(stop_event_count),
            **metrics(selected),
        })
        outputs[method] = (selected, alternatives, steps)
    return pd.DataFrame(rows), outputs


def _group_subsample(frame: pd.DataFrame, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = frame["source_group_id"].astype(str).unique()
    n = max(2, int(round(len(groups) * fraction)))
    selected = rng.choice(groups, size=n, replace=False)
    return np.flatnonzero(frame["source_group_id"].astype(str).isin(selected).to_numpy())


def _jaccard(a, b) -> float:
    left, right = set(a), set(b)
    return 1.0 if not left and not right else len(left & right) / len(left | right)


def _write_figure_source_manifest(root: Path) -> None:
    rows = []

    def add(stem: str, placement: str, sources: str) -> None:
        for extension in ["png", "pdf", "svg"]:
            rows.append({
                "figure_file": f"figures/{stem}.{extension}",
                "placement": placement,
                "source": sources,
                "regeneration_status": "regenerated by run_full_reproduction.py",
            })

    add("Figure1_workflow", "main", V49_LOCK_REL)
    add(
        "Figure2_benchmark_boundary",
        "main",
        "results/reproduction/leakage_audit.json; results/reproduction/source_group_audit.json; results/reproduction/replay_summary.json; results/reproduction/activation_model_metrics.json",
    )
    add(
        "Figure3_policy_comparison",
        "main",
        "source_data/figure3_test_policy_primary_cost_source.csv; source_data/figure3_random_baseline_primary_cost_raw_points.csv; source_data/figure3_random_baseline_primary_cost_summary_source.csv; source_data/figure3_pool_resampling_primary_cost_raw_points.csv; source_data/figure3_pool_resampling_primary_cost_summary_source.csv",
    )
    add(
        "Figure4_descriptor_materials_analysis",
        "main",
        "source_data/figure4_pca_coordinates.csv; results/reproduction/descriptor_pca_variance.json; data/processed/full_public_stability_manifest_scored.csv; traces/action_traces/portfolio_batch_test_primary_cost_contract_trace.json",
    )
    add(
        "Figure5_trace_replay_audit",
        "main",
        "results/reproduction/replay_summary.json; results/reproduction/activation_model_metrics.json; results/reproduction/leakage_audit.json; results/reproduction/source_group_audit.json; traces/action_traces/portfolio_batch_test_primary_cost_contract_trace.json",
    )
    add(
        "Supplementary_Figure1_secondary_thermal_repository_context",
        "SI",
        "results/reproduction/thermal_regression_metrics.json",
    )
    add(
        "Supplementary_Figure2_descriptor_contrasts",
        "SI",
        "source_data/figure4_descriptor_contrasts.csv",
    )
    write_csv(pd.DataFrame(rows), root / "source_data/figure_source_data_manifest.csv")


def run_full_reproduction(
    root: Path | None = None,
    *,
    source_manifest_rel: str | Path | None = None,
) -> dict:
    root = Path(root) if root else _root()
    processed = root / "data/processed"
    derived = root / "data/derived"
    results = root / "results/reproduction"
    tuning = results / "tuning"
    diagnostics = results / "diagnostics"
    models = root / "models"
    source = root / "source_data"
    for directory in [derived, results, tuning, diagnostics, models, source]:
        directory.mkdir(parents=True, exist_ok=True)

    frozen_features_path = processed / "full_public_stability_features.csv"
    source_manifest_path = Path(source_manifest_rel or DEFAULT_SOURCE_MANIFEST)
    if not source_manifest_path.is_absolute():
        source_manifest_path = root / source_manifest_path
    features = pd.read_csv(frozen_features_path, low_memory=False)
    source_manifest = pd.read_csv(source_manifest_path, low_memory=False)
    features = _attach_source_groups(features, source_manifest)
    meta = ACTIVATION_META | {
        "source_doi", "source_doi_normalized", "source_group_id", "source_group_basis",
        "source_url", "source_title", "original_model_split", "final_evaluation_split",
        "exclusion_reason",
    }
    columns = [
        c for c in feature_columns(features, meta)
        if pd.api.types.is_numeric_dtype(features[c])
    ]
    train = features[features["final_evaluation_split"].eq("train")].copy().reset_index(drop=True)
    calibration = features[
        features["final_evaluation_split"].eq("calibration")
    ].copy().reset_index(drop=True)
    test = features[features["final_evaluation_split"].eq("test")].copy().reset_index(drop=True)
    excluded = features[
        features["final_evaluation_split"].eq(EXCLUDED_SPLIT)
    ].copy().reset_index(drop=True)
    for name, frame in [("calibration", calibration), ("test", test)]:
        if frame.empty:
            raise RuntimeError(f"Final {name} split contains zero rows")

    relocked = features.copy().reset_index(drop=True)
    derived_feature_path = root / DERIVED_FEATURE_TABLE
    write_csv(relocked, derived_feature_path)
    write_csv(
        excluded[[
            "candidate_id", "raw_refcode", "source_doi", "original_model_split",
            "final_evaluation_split", "exclusion_reason",
        ]],
        results / "train_held_source_overlap_exclusions.csv",
    )

    selected_C, C_table, fold_audit = select_activation_C_train_cv(train, columns, C_GRID)
    write_csv(C_table, tuning / "activation_C_train_internal_SOURCE_GROUP_CV.csv")
    write_csv(fold_audit, tuning / "activation_train_group_cv_fold_audit.csv")
    model = activation_pipeline(selected_C)
    y_train = (train["outcome_binary"].astype(float) > 0).astype(int)
    final_fit = fit_activation_checked(
        model,
        train[columns],
        y_train,
        context=f"final activation model C={selected_C:g}",
    )

    artifact_path = models / "activation_logistic_train_split.json"
    save_linear_pipeline_artifact(
        model,
        columns,
        artifact_path,
        {
            "selected_C": selected_C,
            "C_grid": C_GRID,
            "selection": (
                "five-fold StratifiedGroupKFold by normalized training source DOI; "
                "fixed random_state=45; one-standard-error rule"
            ),
            "policy_calibration": (
                "preserved v45A2R2 calibration assignment after conservative "
                "train-source-overlap exclusion"
            ),
            "training_source_doi_status": "complete",
            "score_interpretation": "model-derived score; not calibrated",
            "convergence": final_fit,
        },
    )

    # Policy calibration is the last label-bearing stage before the machine-
    # readable test lock is written.
    calibration_scored, z_calibration = _score_table(calibration, model, columns)
    p_calibration = model.predict_proba(calibration[columns])[:, 1]
    y_calibration = (calibration["outcome_binary"].astype(float) > 0).astype(int).to_numpy()
    if np.unique(y_calibration).size != 2:
        raise RuntimeError("Final calibration split does not contain both outcome classes")

    cost_scale = float(calibration_scored["proxy_total_cost"].median())
    cost_per_item = float(1.5 * cost_scale)
    tuned, policy_table = _tune_cost_only_policy(
        calibration_scored, z_calibration, cost_scale, cost_per_item
    )
    write_csv(
        policy_table,
        tuning / "portfolio_policy_cost_only_calibration_selection.csv",
    )
    # Historical filenames remain deterministic compatibility aliases, but the
    # table itself is explicitly the v49 cost-only 27-configuration search.
    write_csv(policy_table, tuning / "portfolio_policy_calibration_selection.csv")
    stability = _cost_only_policy_stability(
        calibration_scored, z_calibration, cost_scale, cost_per_item, 20
    )
    write_csv(
        stability,
        tuning / "portfolio_policy_cost_only_group_bootstrap_stability.csv",
    )
    write_csv(stability, tuning / "portfolio_policy_group_bootstrap_stability.csv")
    source_manifest_display = (
        source_manifest_path.relative_to(root).as_posix()
        if source_manifest_path.is_relative_to(root)
        else str(source_manifest_path)
    )
    code_paths = [
        root / "src/traceable_portfoliobatch/modeling.py",
        root / "src/traceable_portfoliobatch/policies.py",
        root / "src/traceable_portfoliobatch/v49_science.py",
        root / "src/traceable_portfoliobatch/workflow.py",
    ]
    code_hashes = {
        path.relative_to(root).as_posix(): sha256(path) for path in code_paths
    }
    locked = {
        "schema_version": "v49-lock-1.0",
        "activation_C": selected_C,
        "activation_solver": ACTIVATION_SOLVER,
        "activation_max_iter": ACTIVATION_MAX_ITER,
        "activation_tol": ACTIVATION_TOL,
        "activation_convergence": final_fit,
        "activation_feature_list": columns,
        "activation_feature_list_sha256": _json_sha256(columns),
        "activation_model_artifact": "models/activation_logistic_train_split.json",
        "activation_model_artifact_sha256": sha256(artifact_path),
        "primary_policy_mode": "cost_only",
        "primary_risk_budget": None,
        "policy": tuned,
        "selected": {
            field: tuned[field]
            for field in [
                "alpha",
                "beta",
                "gamma",
                "delta",
                "diversity_distance_scale",
            ]
        },
        "cost_budget_per_item": cost_per_item,
        "cost_scale": cost_scale,
        "cost_scale_calibration_median": cost_scale,
        "selection_split": (
            "model C selected inside training by source-DOI-group CV; policy selected "
            "on preserved final calibration only under a cost-only feasibility contract"
        ),
        "test_use": "none during model or policy tuning",
        "test_labels_not_used_for_model_or_policy_selection": True,
        "test_use_claim": (
            "Within the final locked evaluation pipeline, test labels are not used "
            "for model or policy selection."
        ),
        "training_cv": {
            "splitter": "StratifiedGroupKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 45,
            "group_key": "normalized_source_doi",
            "group_overlap_count_max": int(fold_audit["group_overlap_count"].max()),
        },
        "source_manifest": source_manifest_display,
        "training_source_manifest_sha256": sha256(source_manifest_path),
        "final_split_manifest": DERIVED_FEATURE_TABLE,
        "final_split_manifest_sha256": sha256(derived_feature_path),
        "training_source_doi_status": "complete",
        "batch_sizes": [5, 10, 20],
        "fixed_C_diagnostic": {
            "C_values": [0.001, 3.0],
            "role": "post_lock_diagnostic_only",
            "affects_selected_C": False,
        },
        "dummy_classifier_design": {
            "fit_split": "train",
            "evaluation_split": "final_publication_source_disjoint_test",
            "strategies": ["prior", "most_frequent"],
        },
        "test_source_group_bootstrap_design": {
            "unit": "normalized_source_doi",
            "resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "confidence_interval": "95_percentile",
        },
        "legacy_risk_stress_contract": {
            "risk_budget_per_item": LEGACY_RISK_BUDGET_PER_ITEM,
            "role": "stress_test_only",
            "risk_score_formula": RISK_SCORE_FORMULA,
            "risk_score_interpretation": RISK_SCORE_INTERPRETATION,
            "statement": (
                "legacy risk 0.35 is retained for stress-testing only; it is not "
                "the primary selector contract and is not retuned in v49."
            ),
        },
        "random_baseline_seed_policy": {
            "primary_trace_seed": 44,
            "full_pool_seeds": "0..99 per requested batch size",
            "full_pool_runs_per_batch_size": 100,
            "batch_sensitivity_seed_offset": BATCH_SENSITIVITY_RANDOM_SEED_OFFSET,
            "batch_sensitivity_seeds_by_batch_size": {
                str(batch_size): BATCH_SENSITIVITY_RANDOM_SEED_OFFSET + batch_size
                for batch_size in [5, 10, 20]
            },
        },
        "pool_resampling_design": {
            "unit": "normalized source DOI group",
            "fraction": 0.5,
            "pool_resamples": 20,
            "pool_seeds": "0..19",
            "random_seeds_per_resampled_pool_and_batch_size": 10,
            "within_pool_random_seed_indices": "0..9",
            "random_seed_formula": (
                "100000 + pool_seed*1000 + batch_size*20 + within_pool_random_seed"
            ),
            "random_seed_base": POOL_RANDOM_SEED_BASE,
        },
        "policy_calibration_group_bootstrap": {
            "unit": "normalized source DOI group",
            "bootstrap_resamples": 20,
            "bootstrap_seeds": "0..19",
        },
        "code_hashes": code_hashes,
        "code_sha256": _json_sha256(code_hashes),
        "environment_lock": "environment-lock.yml",
        "environment_lock_sha256": sha256(root / "environment-lock.yml"),
        "final_evaluation_row_counts": {
            "train": int(len(train)),
            "calibration": int(len(calibration)),
            "test": int(len(test)),
            EXCLUDED_SPLIT: int(len(excluded)),
        },
        "test_evaluation_may_start_only_after_this_file_is_written": True,
    }
    v49_lock_path = root / V49_LOCK_REL
    write_json(v49_lock_path, locked)
    v49_lock_sha256 = sha256(v49_lock_path)
    write_json(
        tuning / "locked_parameters.json",
        {
            **locked,
            "authoritative_pre_test_lock": V49_LOCK_REL,
            "authoritative_pre_test_lock_sha256": v49_lock_sha256,
        },
    )

    # No test scoring, test-label validation, or test-label metric is evaluated
    # above this boundary.
    y_test = (test["outcome_binary"].astype(float) > 0).astype(int).to_numpy()
    if np.unique(y_test).size != 2:
        raise RuntimeError("Final test split does not contain both outcome classes")
    test_scored, z_test = _score_table(test, model, columns)
    p_test = model.predict_proba(test[columns])[:, 1]
    scored = pd.concat([calibration_scored, test_scored], ignore_index=True)
    scored_path = processed / "full_public_stability_manifest_scored.csv"
    write_csv(scored, scored_path)
    regenerated = predict_from_artifact(test, artifact_path)

    # Task 1 is explicitly post-lock.  The aggregate diagnostic never changes
    # the selected model, the primary scored manifest, or policy tuning.
    ablation_table, ablation_summary = _fixed_C_source_overlap_ablation(
        features,
        train,
        columns,
        selected_C=selected_C,
    )
    write_csv(ablation_table, diagnostics / "source_overlap_C_ablation.csv")
    write_csv(ablation_table, source / "source_overlap_C_ablation_source.csv")
    write_json(
        diagnostics / "source_overlap_C_ablation_summary.json",
        ablation_summary,
    )

    # Task 2: both DummyClassifier strategies are fit only to y_train, then
    # evaluated on the locked final test.  Test prevalence is descriptive only.
    dummy_table = evaluate_training_fitted_dummy_classifiers(y_train, y_test)
    dummy_table["evaluation_split"] = "final_publication_source_disjoint_test"
    dummy_payload = {
        "schema_version": "v49-training-fitted-dummy-baselines-1.0",
        "fit_split": "train",
        "evaluation_split": "final_publication_source_disjoint_test",
        "test_class_prevalence_used_for_baseline_selection": False,
        "baselines": dummy_table.to_dict("records"),
    }
    write_json(results / "activation_dummy_baselines.json", dummy_payload)
    write_csv(dummy_table, source / "activation_dummy_baselines_source.csv")

    bootstrap_raw, bootstrap_summary = source_group_cluster_bootstrap(
        y_test,
        p_test,
        test["source_group_id"].astype(str).to_numpy(),
        n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )
    write_csv(
        bootstrap_raw,
        results / "activation_test_source_group_bootstrap_raw.csv",
    )
    write_json(
        results / "activation_test_source_group_bootstrap_summary.json",
        bootstrap_summary,
    )
    bootstrap_source_rows = []
    for metric_name in ["roc_auc", "accuracy_at_0_5", "brier_score"]:
        bootstrap_source_rows.append({
            "metric": metric_name,
            "seed": bootstrap_summary["seed"],
            "requested_resamples": bootstrap_summary["requested_resamples"],
            "valid_auc_resamples": bootstrap_summary["valid_auc_resamples"],
            "invalid_auc_resamples": bootstrap_summary["invalid_auc_resamples"],
            "bootstrap_group_unit": bootstrap_summary["bootstrap_group_unit"],
            "source_group_count": bootstrap_summary["source_group_count"],
            "confidence_interval_method": bootstrap_summary["confidence_interval"][
                "method"
            ],
            "confidence_level": bootstrap_summary["confidence_interval"][
                "confidence_level"
            ],
            "lower_percentile": bootstrap_summary["confidence_interval"][
                "lower_percentile"
            ],
            "upper_percentile": bootstrap_summary["confidence_interval"][
                "upper_percentile"
            ],
            **bootstrap_summary["metrics"][metric_name],
        })
    write_csv(
        pd.DataFrame(bootstrap_source_rows),
        source / "activation_test_source_group_bootstrap_source.csv",
    )

    # Task 3: prove risk and proxy-cost feasibility before any selector ranking
    # is evaluated on the final test.
    feasibility_frames = []
    feasibility_keys = [
        "pool",
        "requested_batch_size",
        "pool_size",
        "pool_source_group_count",
    ]
    cost_fields = [
        "cost_budget_per_item",
        "total_cost_cap",
        "minimum_possible_total_cost_for_exact_batch",
        "full_batch_cost_feasible",
        "max_feasible_cardinality_under_cost_cap",
        "minimum_cost_budget_per_item_for_full_batch",
    ]
    for pool_name, pool_frame in [
        ("final_calibration", calibration_scored),
        ("final_test", test_scored),
    ]:
        risk_audit = risk_feasibility_table(
            pool_frame["risk_score"].to_numpy(),
            pool=pool_name,
            source_group_ids=pool_frame["source_group_id"].astype(str).to_numpy(),
            batch_sizes=DEFAULT_BATCH_SIZES,
            risk_budget_per_item=LEGACY_RISK_BUDGET_PER_ITEM,
        )
        cost_audit = cost_feasibility_table(
            pool_frame["proxy_total_cost"].to_numpy(),
            cost_budget_per_item=cost_per_item,
            pool=pool_name,
            source_group_ids=pool_frame["source_group_id"].astype(str).to_numpy(),
            batch_sizes=DEFAULT_BATCH_SIZES,
        )
        feasibility_frames.append(
            risk_audit.merge(
                cost_audit[feasibility_keys + cost_fields],
                on=feasibility_keys,
                how="inner",
                validate="one_to_one",
            )
        )
    feasibility_audit = pd.concat(feasibility_frames, ignore_index=True)
    write_csv(feasibility_audit, results / "risk_feasibility_audit.csv")
    write_csv(feasibility_audit, source / "risk_feasibility_audit_source.csv")
    batch5_cost = feasibility_audit[
        feasibility_audit["requested_batch_size"].eq(5)
    ]
    if len(batch5_cost) != 2 or not batch5_cost[
        "full_batch_cost_feasible"
    ].astype(bool).all():
        raise RuntimeError(
            "V49_PRIMARY_COST_CONTRACT_INFEASIBLE: exact batch 5 is not "
            "mathematically feasible under the locked proxy-cost cap"
        )

    dummy_prior = dummy_table.set_index("strategy").loc["prior"]
    dummy_most_frequent = dummy_table.set_index("strategy").loc["most_frequent"]
    bootstrap_auc = bootstrap_summary["metrics"]["roc_auc"]
    activation_metrics = {
        "n_train": len(train),
        "n_calibration": len(calibration),
        "n_test": len(test),
        "n_excluded_train_source_overlap": len(excluded),
        "selected_C": selected_C,
        "calibration_roc_auc": roc_auc_score(y_calibration, p_calibration),
        "calibration_brier_score": brier_score_loss(y_calibration, p_calibration),
        "test_roc_auc": roc_auc_score(y_test, p_test),
        "test_accuracy_at_0_5": accuracy_score(y_test, p_test >= 0.5),
        "test_brier_score": brier_score_loss(y_test, p_test),
        "test_positive_fraction": float(y_test.mean()),
        "test_majority_class_fraction_descriptive": float(
            max(y_test.mean(), 1 - y_test.mean())
        ),
        "dummy_prior_accuracy": float(dummy_prior["accuracy"]),
        "dummy_prior_roc_auc": float(dummy_prior["roc_auc"]),
        "dummy_prior_brier": float(dummy_prior["brier_score"]),
        "dummy_most_frequent_accuracy": float(dummy_most_frequent["accuracy"]),
        "dummy_most_frequent_roc_auc": float(dummy_most_frequent["roc_auc"]),
        "dummy_most_frequent_brier": float(dummy_most_frequent["brier_score"]),
        "test_source_group_bootstrap_auc_ci_low": float(
            bootstrap_auc["percentile_2_5"]
        ),
        "test_source_group_bootstrap_auc_ci_high": float(
            bootstrap_auc["percentile_97_5"]
        ),
        "test_source_group_bootstrap_auc_standard_error": float(
            bootstrap_auc["bootstrap_standard_error"]
        ),
        "bootstrap_resamples_requested": int(
            bootstrap_summary["requested_resamples"]
        ),
        "bootstrap_valid_auc_resamples": int(
            bootstrap_summary["valid_auc_resamples"]
        ),
        "bootstrap_invalid_auc_resamples": int(
            bootstrap_summary["invalid_auc_resamples"]
        ),
        "bootstrap_group_unit": bootstrap_summary["bootstrap_group_unit"],
        "artifact_regeneration_max_abs_diff": float(np.max(np.abs(regenerated - p_test))),
        "C_selection_boundary": (
            "training-only StratifiedGroupKFold by normalized source DOI; no test access"
        ),
        "solver": ACTIVATION_SOLVER,
        "max_iter": ACTIVATION_MAX_ITER,
        "tol": ACTIVATION_TOL,
        "final_fit_n_iter": final_fit["n_iter"],
        "all_activation_fits_converged": (
            bool(C_table["all_folds_converged"].all()) and bool(final_fit["converged"])
        ),
        "maximum_cv_n_iter": int(C_table["max_n_iter"].max()),
        "cv_splitter": "StratifiedGroupKFold",
        "cv_random_state": 45,
        "cv_fold_group_overlap_count_max": int(fold_audit["group_overlap_count"].max()),
        "locked_parameters_before_test": V49_LOCK_REL,
        "locked_parameters_before_test_sha256": v49_lock_sha256,
    }
    write_json(results / "activation_model_metrics.json", activation_metrics)

    primary, outputs = _eval(
        test_scored,
        z_test,
        5,
        None,
        cost_per_item,
        cost_scale,
        tuned,
    )
    primary["evaluation_split"] = "test"
    primary["policy_contract"] = "cost_only"
    primary["risk_contract"] = "none"
    primary["risk_role"] = "descriptive_only"
    if not primary["full_batch"].astype(bool).all():
        failed = primary.loc[~primary["full_batch"].astype(bool), "method"].tolist()
        raise RuntimeError(
            "The mathematically feasible primary batch-5 cost contract did not "
            f"produce full batches for methods={failed}"
        )
    write_csv(primary, results / "test_policy_comparison_primary_cost_only.csv")
    # Retain the former unconstrained filename only as a byte-current alias of
    # the now-primary, explicitly cost-only comparison.
    write_csv(primary, results / "test_policy_comparison_unconstrained.csv")

    stress_rows = []
    for batch_size in DEFAULT_BATCH_SIZES:
        stress, _ = _eval(
            test_scored,
            z_test,
            batch_size,
            LEGACY_RISK_BUDGET_PER_ITEM,
            cost_per_item,
            cost_scale,
            tuned,
            BATCH_SENSITIVITY_RANDOM_SEED_OFFSET + batch_size,
            detailed=False,
        )
        stress["evaluation_split"] = "test"
        stress["policy_contract"] = "legacy_risk_stress_0p35"
        stress["contract_role"] = "stress_test_only"
        stress["risk_interpretation"] = RISK_SCORE_INTERPRETATION
        stress_rows += stress.to_dict("records")
    stress_table = pd.DataFrame(stress_rows)
    write_csv(
        stress_table,
        diagnostics / "test_policy_risk_stress_legacy_0p35.csv",
    )
    # The old common-risk name is retained solely as a labelled batch-5
    # compatibility view, never as the primary Figure 3 input.
    write_csv(
        stress_table[stress_table["batch_size"].eq(5)].reset_index(drop=True),
        results / "test_policy_comparison_common_risk.csv",
    )

    batch_rows = []
    full_selection: dict[tuple[str, int], list[str]] = {}
    for batch_size in DEFAULT_BATCH_SIZES:
        table, batch_outputs = _eval(
            test_scored,
            z_test,
            batch_size,
            None,
            cost_per_item,
            cost_scale,
            tuned,
            BATCH_SENSITIVITY_RANDOM_SEED_OFFSET + batch_size,
            detailed=False,
        )
        table["policy_contract"] = "cost_only"
        table["risk_contract"] = "none"
        table["risk_role"] = "descriptive_only"
        batch_rows += table.to_dict("records")
        for method, (selected, _, _) in batch_outputs.items():
            full_selection[(method, batch_size)] = selected["candidate_id"].tolist()
    batch_table = pd.DataFrame(batch_rows)
    write_csv(batch_table, results / "test_batch_size_sensitivity.csv")

    random_rows = []
    for batch_size in DEFAULT_BATCH_SIZES:
        for seed in range(100):
            table, _ = _eval(
                test_scored,
                z_test,
                batch_size,
                None,
                cost_per_item,
                cost_scale,
                tuned,
                seed,
                methods=["random_baseline"],
                detailed=False,
            )
            random_rows.append({
                "seed": seed,
                "policy_contract": "cost_only",
                "risk_contract": "none",
                **table.iloc[0].to_dict(),
            })
    random_table = pd.DataFrame(random_rows)
    write_csv(
        random_table,
        results / "random_baseline_primary_cost_only_raw.csv",
    )
    write_csv(random_table, results / "random_baseline_full_test_distribution.csv")
    random_summary = []
    for batch_size, group in random_table.groupby("batch_size"):
        observed = int(
            batch_table[
                (batch_table["batch_size"] == batch_size)
                & (batch_table["method"] == "top_score")
            ]["hits"].iloc[0]
        )
        random_summary.append({
            "batch_size": int(batch_size),
            "n_runs": len(group),
            "mean_hits": group["hits"].mean(),
            "sd_hits": group["hits"].std(ddof=1),
            "mean_selected_count": group["selected_count"].mean(),
            "full_batch_rate": np.mean(group["selected_count"] == batch_size),
            "mean_proxy_cost_normalized_yield": group["proxy_cost_normalized_yield"].mean(),
            "empirical_probability_hits_ge_top_score": np.mean(group["hits"] >= observed),
        })
    random_summary_table = pd.DataFrame(random_summary)
    random_summary_table["policy_contract"] = "cost_only"
    random_summary_table["risk_contract"] = "none"
    write_csv(
        random_summary_table,
        results / "random_baseline_primary_cost_only_summary.csv",
    )
    write_csv(random_summary_table, results / "random_baseline_full_test_summary.csv")

    raw_rows = []
    for pool_seed in range(20):
        indices = _group_subsample(test_scored, 0.50, pool_seed)
        pool = test_scored.iloc[indices].reset_index(drop=True)
        z_pool = z_test[indices]
        for batch_size in DEFAULT_BATCH_SIZES:
            deterministic, deterministic_outputs = _eval(
                pool,
                z_pool,
                batch_size,
                None,
                cost_per_item,
                cost_scale,
                tuned,
                methods=DETERMINISTIC_METHODS,
                detailed=False,
            )
            for row in deterministic.to_dict("records"):
                ids = deterministic_outputs[row["method"]][0]["candidate_id"].tolist()
                raw_rows.append({
                    "pool_seed": pool_seed,
                    "random_seed": np.nan,
                    "pool_group_count": pool["source_group_id"].nunique(),
                    "pool_size": len(pool),
                    "policy_contract": "cost_only",
                    "risk_contract": "none",
                    **row,
                    "jaccard_to_full_test_selection": _jaccard(
                        ids, full_selection[(row["method"], batch_size)]
                    ),
                })
            for random_seed in range(10):
                seed_value = (
                    POOL_RANDOM_SEED_BASE
                    + pool_seed * 1000
                    + batch_size * 20
                    + random_seed
                )
                random_result, random_outputs = _eval(
                    pool,
                    z_pool,
                    batch_size,
                    None,
                    cost_per_item,
                    cost_scale,
                    tuned,
                    seed_value,
                    methods=["random_baseline"],
                    detailed=False,
                )
                row = random_result.iloc[0].to_dict()
                ids = random_outputs["random_baseline"][0]["candidate_id"].tolist()
                raw_rows.append({
                    "pool_seed": pool_seed,
                    "random_seed": random_seed,
                    "pool_group_count": pool["source_group_id"].nunique(),
                    "pool_size": len(pool),
                    "policy_contract": "cost_only",
                    "risk_contract": "none",
                    **row,
                    "jaccard_to_full_test_selection": _jaccard(
                        ids, full_selection[("random_baseline", batch_size)]
                    ),
                })
    raw_table = pd.DataFrame(raw_rows)
    write_csv(
        raw_table,
        results / "test_pool_resampling_primary_cost_only_raw.csv",
    )
    write_csv(raw_table, results / "test_pool_resampling_raw.csv")
    summary_table = raw_table.groupby(["method", "batch_size"], as_index=False).agg(
        n_records=("pool_seed", "count"),
        n_pools=("pool_seed", "nunique"),
        mean_selected_count=("selected_count", "mean"),
        mean_hits=("hits", "mean"),
        sd_hits=("hits", "std"),
        mean_hit_fraction=("hit_fraction", "mean"),
        sd_hit_fraction=("hit_fraction", "std"),
        mean_proxy_cost_normalized_yield=("proxy_cost_normalized_yield", "mean"),
        mean_jaccard_to_full_test=("jaccard_to_full_test_selection", "mean"),
    )
    summary_table["full_batch_rate"] = [
        np.mean(group["selected_count"] == batch_size)
        for (_, batch_size), group in raw_table.groupby(["method", "batch_size"], sort=True)
    ]
    summary_table["policy_contract"] = "cost_only"
    summary_table["risk_contract"] = "none"
    write_csv(
        summary_table,
        results / "test_pool_resampling_primary_cost_only_summary.csv",
    )
    write_csv(summary_table, results / "test_pool_resampling_summary.csv")

    write_json(
        results / "v49_statistical_summary.json",
        {
            "schema_version": "v49-statistical-summary-1.0",
            "primary_model_point_metrics": {
                "roc_auc": float(roc_auc_score(y_test, p_test)),
                "accuracy_at_0_5": float(accuracy_score(y_test, p_test >= 0.5)),
                "brier_score": float(brier_score_loss(y_test, p_test)),
            },
            "training_fitted_dummy_classifiers": dummy_payload,
            "final_test_source_group_bootstrap": bootstrap_summary,
            "source_overlap_fixed_C_diagnostic": ablation_summary,
            "risk_and_cost_feasibility": feasibility_audit.to_dict("records"),
            "primary_policy_mode": "cost_only",
            "primary_risk_budget": None,
            "selected_policy": tuned,
            "primary_selector_comparison": primary.to_dict("records"),
            "legacy_risk_stress_test": {
                "risk_budget_per_item": LEGACY_RISK_BUDGET_PER_ITEM,
                "risk_score_interpretation": RISK_SCORE_INTERPRETATION,
                "role": "stress_test_only",
                "results": stress_table.to_dict("records"),
            },
            "inference_boundary": (
                "descriptive point estimates and percentile bootstrap uncertainty; "
                "no hypothesis-test or statistical-significance claim"
            ),
        },
    )

    calibration_doi = set(
        calibration.loc[calibration["source_doi_normalized"].ne(""), "source_doi_normalized"]
    )
    test_doi = set(test.loc[test["source_doi_normalized"].ne(""), "source_doi_normalized"])
    train_doi = set(train.loc[train["source_doi_normalized"].ne(""), "source_doi_normalized"])
    excluded_doi = set(excluded["source_doi_normalized"].astype(str))
    training_complete = bool(train["source_doi_normalized"].ne("").all())
    source_group_audit = {
        "schema_version": "v49-source-group-audit-1.0",
        "source_manifest": source_manifest_display,
        "policy_calibration_test_source_doi_overlap_count": len(calibration_doi & test_doi),
        "train_calibration_source_doi_overlap_count": len(train_doi & calibration_doi),
        "train_test_source_doi_overlap_count": len(train_doi & test_doi),
        "calibration_doi_group_count": len(calibration_doi),
        "test_doi_group_count": len(test_doi),
        "training_doi_group_count": len(train_doi),
        "excluded_train_source_overlap_doi_group_count": len(excluded_doi),
        "excluded_train_source_overlap_row_count": int(len(excluded)),
        "active_train_row_count": int(len(train)),
        "active_calibration_row_count": int(len(calibration)),
        "active_test_row_count": int(len(test)),
        "training_rows_with_packaged_source_doi": int(train["source_doi_normalized"].ne("").sum()),
        "training_rows_without_packaged_source_doi": int(train["source_doi_normalized"].eq("").sum()),
        "training_source_doi_complete": training_complete,
        "active_training_doi_coverage": float(train["source_doi_normalized"].ne("").mean()),
        "active_calibration_doi_coverage": float(
            calibration["source_doi_normalized"].ne("").mean()
        ),
        "active_test_doi_coverage": float(test["source_doi_normalized"].ne("").mean()),
        "source_group_basis": "normalized_source_doi",
        "status": "PASS_COMPLETE_ACTIVE_SOURCE_GROUP_DISJOINT",
    }
    if not (
        training_complete
        and source_group_audit["active_calibration_doi_coverage"] == 1.0
        and source_group_audit["active_test_doi_coverage"] == 1.0
        and source_group_audit["train_calibration_source_doi_overlap_count"] == 0
        and source_group_audit["train_test_source_doi_overlap_count"] == 0
        and source_group_audit["policy_calibration_test_source_doi_overlap_count"] == 0
    ):
        raise RuntimeError("V49 active source-group audit failed")
    write_json(results / "source_group_audit.json", source_group_audit)
    leakage = write_audit(relocked, results / "leakage_audit.json")

    trace_dir = root / "traces/action_traces"
    replay_dir = root / "traces/replay_checks"
    shutil.rmtree(trace_dir, ignore_errors=True)
    shutil.rmtree(replay_dir, ignore_errors=True)
    trace_dir.mkdir(parents=True)
    replay_dir.mkdir(parents=True)
    code_path = root / "src/traceable_portfoliobatch/policies.py"
    environment_path = root / "environment-lock.yml"
    for method, (selected, alternatives, steps) in outputs.items():
        config = _policy_config(
            method,
            5,
            cost_per_item,
            None,
            cost_scale,
            tuned,
            44 if method == "random_baseline" else None,
        )
        write_trace(
            trace_dir / f"{method}_test_primary_cost_contract_trace.json",
            run_id=f"{method}_test_primary_cost_contract",
            method=method,
            config=config_dict(config),
            manifest_rel="data/processed/full_public_stability_manifest_scored.csv",
            manifest_sha256=sha256(scored_path),
            source_manifest_rel=source_manifest_display,
            source_manifest_sha256=sha256(source_manifest_path),
            feature_table_rel=DERIVED_FEATURE_TABLE,
            feature_table_sha256=sha256(derived_feature_path),
            model_artifact_rel="models/activation_logistic_train_split.json",
            model_sha256=sha256(artifact_path),
            selected=selected,
            alternatives=alternatives,
            steps=steps,
            metrics=metrics(selected),
            code_rel="src/traceable_portfoliobatch/policies.py",
            code_sha256=sha256(code_path),
            environment_rel="environment-lock.yml",
            environment_sha256=sha256(environment_path),
            evaluation_split="test",
            locked_parameters_rel=V49_LOCK_REL,
            locked_parameters_sha256=v49_lock_sha256,
            locked_parameters=locked,
        )
    from .replay import replay_all

    replay = replay_all(root)
    if replay["status"] != "PASS" or replay["passed_count"] != replay["trace_count"]:
        raise RuntimeError(
            "Exact archived-implementation replay failed: "
            f"passed={replay['passed_count']}/{replay['trace_count']}, "
            f"failed={replay['failed']}"
        )

    thermal = pd.read_csv(processed / "full_public_thermal_features.csv")
    thermal_columns = feature_columns(thermal, THERMAL_META)
    thermal_train = thermal[thermal["model_split"].eq("train")]
    thermal_validation = thermal[thermal["model_split"].eq("val")].reset_index(drop=True)
    alpha, alpha_table = select_thermal_alpha_cv(
        thermal_train, thermal_columns, [0.1, 1.0, 10.0, 100.0]
    )
    write_csv(alpha_table, tuning / "thermal_alpha_train_cv_selection.csv")
    thermal_model = thermal_pipeline(alpha)
    thermal_model.fit(thermal_train[thermal_columns], thermal_train["thermal_T"])
    thermal_prediction = thermal_model.predict(thermal_validation[thermal_columns])
    thermal_artifact = models / "thermal_ridge_train_split.json"
    save_linear_pipeline_artifact(
        thermal_model,
        thermal_columns,
        thermal_artifact,
        {"selected_alpha": alpha, "target_unit": "degrees Celsius"},
    )
    baseline = np.full(len(thermal_validation), thermal_train["thermal_T"].mean())
    thermal_metrics = {
        "n_train": len(thermal_train),
        "n_validation": len(thermal_validation),
        "selected_alpha": alpha,
        "mae_C": mean_absolute_error(thermal_validation["thermal_T"], thermal_prediction),
        "rmse_C": mean_squared_error(thermal_validation["thermal_T"], thermal_prediction) ** 0.5,
        "r2": r2_score(thermal_validation["thermal_T"], thermal_prediction),
        "training_mean_baseline_C": thermal_train["thermal_T"].mean(),
        "training_mean_baseline_rmse_C": mean_squared_error(
            thermal_validation["thermal_T"], baseline
        ) ** 0.5,
    }
    write_json(results / "thermal_regression_metrics.json", thermal_metrics)
    write_csv(
        pd.DataFrame({
            "candidate_id": thermal_validation["candidate_id"],
            "model_split": "val",
            "observed_thermal_T_C": thermal_validation["thermal_T"],
            "predicted_thermal_T_C": thermal_prediction,
            "absolute_error_C": np.abs(
                thermal_prediction - thermal_validation["thermal_T"].to_numpy()
            ),
        }),
        processed / "full_public_thermal_model_scores.csv",
    )

    pca = PCA(n_components=2, random_state=0)
    coordinates = pca.fit_transform(z_test)
    write_csv(
        pd.DataFrame({
            "candidate_id": test_scored["candidate_id"],
            "outcome_binary": test_scored["outcome_binary"],
            "pc1": coordinates[:, 0],
            "pc2": coordinates[:, 1],
        }),
        source / "figure4_pca_coordinates.csv",
    )
    write_json(
        results / "descriptor_pca_variance.json",
        {
            "pc1_percent": 100 * pca.explained_variance_ratio_[0],
            "pc2_percent": 100 * pca.explained_variance_ratio_[1],
        },
    )
    contrast = []
    for index, name in enumerate(columns):
        stable = z_test[y_test == 1, index]
        unstable = z_test[y_test == 0, index]
        pooled = np.sqrt(
            (
                (len(stable) - 1) * stable.var(ddof=1)
                + (len(unstable) - 1) * unstable.var(ddof=1)
            )
            / max(len(stable) + len(unstable) - 2, 1)
        )
        contrast.append({
            "descriptor": name,
            "stable_minus_unstable_smd": (
                0 if pooled == 0 else (stable.mean() - unstable.mean()) / pooled
            ),
        })
    contrast_table = pd.DataFrame(contrast)
    contrast_table["abs"] = contrast_table["stable_minus_unstable_smd"].abs()
    contrast_table = (
        contrast_table.sort_values("abs", ascending=False)
        .head(12)
        .drop(columns="abs")
    )
    write_csv(contrast_table, source / "figure4_descriptor_contrasts.csv")

    portfolio_trace = json.loads(
        (
            trace_dir
            / "portfolio_batch_test_primary_cost_contract_trace.json"
        ).read_text()
    )
    trace_rows = []
    for role, records in [
        ("selected", portfolio_trace["selected_candidates"]),
        ("logged_alternative", portfolio_trace["logged_alternative_candidates"]),
    ]:
        for record in records:
            trace_rows.append({
                "candidate_id": record["candidate_id"],
                "trace_role": role,
                "predicted_success": record["predicted_success"],
                "proxy_total_cost": record["proxy_total_cost"],
                "risk_score": record["risk_score"],
            })
    write_csv(pd.DataFrame(trace_rows), source / "figure4_trace_score_cost.csv")
    write_csv(primary, source / "figure3_test_policy_primary_cost_source.csv")
    write_csv(
        random_summary_table,
        source / "figure3_random_baseline_primary_cost_summary_source.csv",
    )
    write_csv(
        summary_table,
        source / "figure3_pool_resampling_primary_cost_summary_source.csv",
    )
    write_csv(
        raw_table[raw_table["batch_size"].eq(5)].reset_index(drop=True),
        source / "figure3_pool_resampling_primary_cost_raw_points.csv",
    )
    write_csv(
        random_table,
        source / "figure3_random_baseline_primary_cost_raw_points.csv",
    )
    write_csv(
        pd.DataFrame([
            {"metric": "MAE", "value": thermal_metrics["mae_C"], "unit": "degrees C"},
            {"metric": "RMSE", "value": thermal_metrics["rmse_C"], "unit": "degrees C"},
            {
                "metric": "training-mean baseline RMSE",
                "value": thermal_metrics["training_mean_baseline_rmse_C"],
                "unit": "degrees C",
            },
            {"metric": "R2", "value": thermal_metrics["r2"], "unit": "dimensionless"},
        ]),
        source / "supplementary_figure1_source_data.csv",
    )
    write_csv(
        pd.DataFrame([
            {"check": "trace_count", "value": replay["trace_count"]},
            {"check": "trace_replay_passed", "value": replay["passed_count"]},
            {
                "check": "activation_score_artifact_max_abs_diff",
                "value": activation_metrics["artifact_regeneration_max_abs_diff"],
            },
            {
                "check": "activation_score_artifact_portability_tolerance",
                "value": 1e-12,
            },
            {
                "check": "candidate_id_split_overlap",
                "value": leakage["checks"]["candidate_id"]["overlap_count"],
            },
            {
                "check": "raw_refcode_split_overlap",
                "value": leakage["checks"]["raw_refcode"]["overlap_count"],
            },
            {
                "check": "raw_source_row_split_overlap",
                "value": leakage["checks"]["raw_source_row_identity"]["overlap_count"],
            },
            {
                "check": "policy_calibration_test_DOI_overlap",
                "value": source_group_audit["policy_calibration_test_source_doi_overlap_count"],
            },
            {
                "check": "training_source_DOI",
                "value": "complete (100%)",
            },
            {
                "check": "train_calibration_source_DOI_overlap",
                "value": source_group_audit["train_calibration_source_doi_overlap_count"],
            },
            {
                "check": "train_test_source_DOI_overlap",
                "value": source_group_audit["train_test_source_doi_overlap_count"],
            },
            {
                "check": "excluded_train_source_overlap_rows",
                "value": source_group_audit["excluded_train_source_overlap_row_count"],
            },
        ]),
        source / "figure5_trace_replay_source.csv",
    )
    write_csv(
        pd.DataFrame([
            {
                "field": "Primary endpoint",
                "implementation": "MOFSimplify activation / solvent-removal label",
                "boundary": "not water or hydrolytic stability",
            },
            {
                "field": "Model selection",
                "implementation": (
                    f"C={selected_C:g}; five-fold StratifiedGroupKFold by training "
                    "source DOI with one-standard-error rule"
                ),
                "boundary": "training only; fixed random_state=45; zero fold DOI overlap",
            },
            {
                "field": "Source grouping",
                "implementation": (
                    "publication DOI groups with the preserved v45A2R2 calibration/test "
                    "assignment after conservative train-source-overlap exclusion"
                ),
                "boundary": (
                    "publication-source grouping; not chemical-family, topology, structural-"
                    "similarity, or chemical-OOD grouping"
                ),
            },
            {
                "field": "Policy selection",
                "implementation": (
                    "calibration-only 27-configuration cost-constrained tuning; "
                    "risk_budget=None"
                ),
                "boundary": "test labels are not used for model or policy selection",
            },
            {
                "field": "Primary selector constraint",
                "implementation": "locked proxy-cost budget only; full requested batch",
                "boundary": "risk is not a primary hard feasibility gate",
            },
            {
                "field": "Risk",
                "implementation": RISK_SCORE_FORMULA,
                "boundary": (
                    "benchmark-defined uncalibrated proxy; descriptive in the primary "
                    "comparison and constrained only in the legacy 0.35 stress test"
                ),
            },
            {
                "field": "Test use",
                "implementation": "final locked publication-source-disjoint evaluation",
                "boundary": "not used for model selection, policy tuning, or risk-threshold tuning",
            },
            {
                "field": "Statistics",
                "implementation": (
                    "10,000 final-test source-DOI-group bootstrap resamples; 100 random "
                    "full-pool runs; 20 DOI-group pool resamples; "
                    "10 random seeds per resampled pool"
                ),
                "boundary": "percentile uncertainty and descriptive robustness; no hypothesis test",
            },
            {
                "field": "Trace layer",
                "implementation": "stepwise JSON records and exact archived-implementation replay",
                "boundary": "not independent implementation",
            },
        ]),
        source / "table1_benchmark_contract.csv",
    )

    from .figures import build_all

    figures = build_all(root)
    _write_figure_source_manifest(root)

    trace_manifest_rows = []
    for trace_path in sorted(trace_dir.glob("*.json")):
        payload = json.loads(trace_path.read_text())
        replay_path = replay_dir / f"{trace_path.stem}_replay.json"
        replay_payload = json.loads(replay_path.read_text())
        trace_manifest_rows.append({
            "trace_file": trace_path.relative_to(root).as_posix(),
            "run_id": payload["run_id"],
            "method": payload["method"],
            "logged_event_count": len(payload["stepwise_decisions"]),
            "selection_event_count": sum(
                step.get("status") == "selected" for step in payload["stepwise_decisions"]
            ),
            "stop_event_count": sum(
                str(step.get("status", "")).startswith("stopped")
                for step in payload["stepwise_decisions"]
            ),
            "sha256": sha256(trace_path),
            "replay_passed": replay_payload["passed"],
        })
    write_csv(pd.DataFrame(trace_manifest_rows), root / "traces/trace_archive_manifest.csv")
    write_csv(
        pd.DataFrame([
            {
                "endpoint": "activation",
                "split_source": DERIVED_FEATURE_TABLE,
                "notes": (
                    f"{len(train)} train; {len(calibration)} DOI-group policy calibration; "
                    f"{len(test)} DOI-group-disjoint test; {len(excluded)} provenance-only "
                    "train-source-overlap exclusions"
                ),
            },
            {
                "endpoint": "thermal",
                "split_source": "data/processed/full_public_thermal_features.csv",
                "notes": f"{len(thermal_train)} train; {len(thermal_validation)} validation",
            },
        ]),
        root / "data/split_manifest.csv",
    )

    status = "PASS_V49_FOUR_PART_SCIENTIFIC_REPRODUCTION"
    summary = {
        "status": status,
        "activation": activation_metrics,
        "locked_parameters": locked,
        "source_overlap_C_ablation": ablation_summary,
        "dummy_baselines": dummy_payload,
        "test_source_group_bootstrap": bootstrap_summary,
        "risk_feasibility": feasibility_audit.to_dict("records"),
        "test_primary_cost_only_contract": primary.to_dict("records"),
        "test_legacy_risk_stress_contract": stress_table.to_dict("records"),
        "policy_stability": stability.to_dict("records"),
        "source_group_audit": source_group_audit,
        "thermal": thermal_metrics,
        "trace_count": replay["trace_count"],
        "generated_figures": figures,
        "replay": replay,
        "leakage": leakage,
        "reproduction_contract": {
            "frozen_feature_input": "data/processed/full_public_stability_features.csv",
            "derived_relocked_output": DERIVED_FEATURE_TABLE,
            "source_manifest": source_manifest_display,
            "optimizer": {
                "solver": ACTIVATION_SOLVER,
                "max_iter": ACTIVATION_MAX_ITER,
                "tol": ACTIVATION_TOL,
                "all_fits_converged": activation_metrics["all_activation_fits_converged"],
            },
            "generated_artifacts": ("byte-identical for contracted CSV/JSON artifacts in the pinned canonical "
                "Linux environment; exact-discrete, numerical-tolerance and visual-equivalence "
                "checks on other platforms"),
        },
        "claim_boundary": {
            "sequential": "sequential within-batch candidate selection; not closed-loop discovery",
            "source_groups": (
                "active train, calibration and test partitions are fully covered by normalized "
                "source DOI and pairwise DOI-group disjoint; excluded train-source-overlap rows "
                "remain provenance-only"
            ),
            "test_use": (
                "Within the final locked evaluation pipeline, test labels are not used for "
                "model or policy selection."
            ),
            "risk": (
                "benchmark-defined uncalibrated proxy; descriptive in the primary cost-only "
                "comparison and constrained only in the legacy 0.35 stress test"
            ),
            "statistics": (
                "source-group percentile bootstrap and descriptive robustness distributions; "
                "no inferential hypothesis testing"
            ),
        },
    }
    write_json(results / "full_reproduction_summary.json", summary)
    write_generated_artifact_manifest(root)
    return summary
