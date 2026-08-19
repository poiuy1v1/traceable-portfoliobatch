from __future__ import annotations

import argparse
import csv
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import traceable_portfoliobatch  # noqa: E402,F401
import pandas as pd  # noqa: E402

BASELINE_COMMIT = "d078e5a5981d66735be1e1b19838e8318b2c3762"


def _git_show(relative_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def _baseline_json(relative_path: str) -> dict[str, Any]:
    return json.loads(_git_show(relative_path))


def _current_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _baseline_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(_git_show(relative_path)))


def _current_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative_path)


def _display(value: Any) -> str:
    if value is None:
        return "not available in v45A2R2"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def _classification(category: str, old: Any, new: Any) -> str:
    if _same(old, new):
        return "unchanged"
    if category in {"evaluation membership", "source provenance"}:
        return "expected_due_to_source_group_rebuild"
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if abs(float(new) - float(old)) <= 1e-8:
            return "minor_numeric"
    return "material_scientific_change"


def _trace_snapshot(current: bool) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    names = [
        "top_score",
        "uncertainty",
        "cost_aware",
        "portfolio_batch",
        "random_baseline",
    ]
    for method in names:
        rel = f"traces/action_traces/{method}_test_common_contract_trace.json"
        payload = _current_json(rel) if current else _baseline_json(rel)
        result[method] = {
            "selected_candidate_ids": [
                row["candidate_id"] for row in payload["selected_candidates"]
            ],
            "selection_event_count": sum(
                step.get("status") == "selected"
                for step in payload["stepwise_decisions"]
            ),
            "stop_event_count": sum(
                str(step.get("status", "")).startswith("stopped")
                for step in payload["stepwise_decisions"]
            ),
        }
    return result


def build_comparison() -> tuple[pd.DataFrame, str]:
    activation_rel = "results/reproduction/activation_model_metrics.json"
    audit_rel = "results/reproduction/source_group_audit.json"
    lock_rel = "results/reproduction/tuning/locked_parameters.json"
    policy_rel = "results/reproduction/test_policy_comparison_common_risk.csv"
    random_rel = "results/reproduction/random_baseline_full_test_summary.csv"
    pool_rel = "results/reproduction/test_pool_resampling_summary.csv"
    pca_rel = "results/reproduction/descriptor_pca_variance.json"
    replay_rel = "results/reproduction/replay_summary.json"

    old_activation = _baseline_json(activation_rel)
    new_activation = _current_json(activation_rel)
    old_audit = _baseline_json(audit_rel)
    new_audit = _current_json(audit_rel)
    old_lock = _baseline_json(lock_rel)
    new_lock = _current_json(lock_rel)
    old_policy = _baseline_csv(policy_rel).set_index("method")
    new_policy = _current_csv(policy_rel).set_index("method")
    old_random = _baseline_csv(random_rel).set_index("batch_size")
    new_random = _current_csv(random_rel).set_index("batch_size")
    old_pool = _baseline_csv(pool_rel).set_index(["method", "batch_size"])
    new_pool = _current_csv(pool_rel).set_index(["method", "batch_size"])
    old_pca = _baseline_json(pca_rel)
    new_pca = _current_json(pca_rel)
    old_replay = _baseline_json(replay_rel)
    new_replay = _current_json(replay_rel)
    old_traces = _trace_snapshot(current=False)
    new_traces = _trace_snapshot(current=True)

    rows: list[dict[str, Any]] = []

    def add(
        category: str,
        metric: str,
        old: Any,
        new: Any,
        old_source: str,
        new_source: str,
        interpretation: str,
        classification: str | None = None,
    ) -> None:
        delta = ""
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            delta = f"{float(new) - float(old):.12g}"
        rows.append(
            {
                "category": category,
                "metric": metric,
                "v45A2R2": _display(old),
                "v45A3": _display(new),
                "absolute_change_v45A3_minus_v45A2R2": delta,
                "change_classification": classification
                or _classification(category, old, new),
                "interpretation": interpretation,
                "v45A2R2_source": f"{BASELINE_COMMIT}:{old_source}",
                "v45A3_source": new_source,
            }
        )

    evaluation_values = [
        ("training rows", old_activation["n_train"], new_activation["n_train"]),
        ("calibration rows", old_activation["n_calibration"], new_activation["n_calibration"]),
        ("test rows", old_activation["n_test"], new_activation["n_test"]),
        (
            "excluded train-source-overlap rows",
            0,
            new_activation["n_excluded_train_source_overlap"],
        ),
        (
            "training DOI groups",
            "0 packaged (865 only retrospectively reconstructable)",
            new_audit["training_doi_group_count"],
        ),
        (
            "calibration DOI groups",
            old_audit.get("calibration_doi_group_count"),
            new_audit["calibration_doi_group_count"],
        ),
        (
            "test DOI groups",
            old_audit.get("test_doi_group_count"),
            new_audit["test_doi_group_count"],
        ),
        (
            "excluded train-source-overlap DOI groups",
            0,
            new_audit["excluded_train_source_overlap_doi_group_count"],
        ),
    ]
    for metric, old, new in evaluation_values:
        add(
            "evaluation membership",
            metric,
            old,
            new,
            activation_rel if "rows" in metric else audit_rel,
            activation_rel if "rows" in metric else audit_rel,
            "The conservative DOI rule retains all training rows and removes overlapping held rows only from active evaluation.",
        )

    protocol_rows = [
        (
            "training DOI coverage",
            "0/1394 packaged; train-held overlap not evaluable",
            "1394/1394",
            "Training DOI was authenticated and uniquely backfilled without changing held DOI values.",
        ),
        (
            "train-calibration DOI-group overlap",
            "not evaluable from packaged v45A2R2; retrospective official-source audit = 121",
            0,
            "All held rows in training-overlap DOI groups are provenance-only exclusions.",
        ),
        (
            "train-test DOI-group overlap",
            "not evaluable from packaged v45A2R2; retrospective official-source audit = 149",
            0,
            "All held rows in training-overlap DOI groups are provenance-only exclusions.",
        ),
        (
            "calibration-test DOI-group overlap",
            0,
            0,
            "The preserved surviving calibration/test assignment remains DOI-disjoint.",
        ),
    ]
    for metric, old, new, interpretation in protocol_rows:
        add(
            "source provenance",
            metric,
            old,
            new,
            audit_rel,
            audit_rel,
            interpretation,
        )
    add(
        "source provenance",
        "training CV splitter",
        "StratifiedKFold (row-level)",
        "StratifiedGroupKFold (normalized source DOI)",
        "src/traceable_portfoliobatch/modeling.py",
        "results/reproduction/tuning/activation_C_train_internal_SOURCE_GROUP_CV.csv",
        "The fixed five-fold training-only design now prevents a DOI group from crossing a model-selection fold.",
    )
    add(
        "source provenance",
        "maximum model-CV fold DOI-group overlap",
        "not audited",
        0,
        "src/traceable_portfoliobatch/modeling.py",
        "results/reproduction/tuning/activation_train_group_cv_fold_audit.csv",
        "All five folds have zero training-validation DOI-group overlap and both classes.",
    )
    add(
        "source provenance",
        "authoritative lock written before test scoring",
        False,
        True,
        "src/traceable_portfoliobatch/workflow.py",
        "results/reproduction/tuning/v45A3_locked_parameters_before_test.json",
        "The machine-readable model/policy/hash record is written before any final test scoring or test-label metric.",
    )

    add(
        "model selection",
        "selected activation C",
        old_activation["selected_C"],
        new_activation["selected_C"],
        activation_rel,
        activation_rel,
        "The hyperparameter changed under fixed five-fold training-only source-DOI-group CV.",
    )
    for metric in [
        "calibration_roc_auc",
        "calibration_brier_score",
        "test_roc_auc",
        "test_accuracy_at_0_5",
        "test_brier_score",
        "test_majority_class_accuracy",
    ]:
        add(
            "activation performance",
            metric,
            old_activation[metric],
            new_activation[metric],
            activation_rel,
            activation_rel,
            "The active held pools and fitted model changed; the weaker final result must remain visible.",
            (
                "expected_due_to_source_group_rebuild"
                if metric == "test_majority_class_accuracy"
                else None
            ),
        )

    policy_fields = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "diversity_distance_scale",
        "risk_budget_per_item",
        "cost_budget_per_item",
    ]
    for field in policy_fields:
        add(
            "policy lock",
            field,
            old_lock["policy"][field],
            new_lock["policy"][field],
            lock_rel,
            lock_rel,
            "PortfolioBatch policy parameters were reselected on final calibration only.",
            (
                "expected_due_to_source_group_rebuild"
                if field == "cost_budget_per_item"
                and not _same(old_lock["policy"][field], new_lock["policy"][field])
                else None
            ),
        )
    add(
        "policy lock",
        "cost_scale_calibration_median",
        old_lock["cost_scale_calibration_median"],
        new_lock["cost_scale_calibration_median"],
        lock_rel,
        lock_rel,
        "Cost remains a descriptor-complexity proxy, not measured synthesis cost.",
        (
            "expected_due_to_source_group_rebuild"
            if not _same(
                old_lock["cost_scale_calibration_median"],
                new_lock["cost_scale_calibration_median"],
            )
            else None
        ),
    )

    old_thermal = _baseline_json("results/reproduction/thermal_regression_metrics.json")
    new_thermal = _current_json("results/reproduction/thermal_regression_metrics.json")
    for field in ["selected_alpha", "mae_C", "rmse_C", "r2"]:
        add(
            "secondary thermal",
            field,
            old_thermal[field],
            new_thermal[field],
            "results/reproduction/thermal_regression_metrics.json",
            "results/reproduction/thermal_regression_metrics.json",
            "The secondary thermal scope is unchanged; machine-scale drift is diagnostic only.",
        )

    selector_fields = [
        "selected_count",
        "hits",
        "hit_fraction",
        "total_proxy_cost",
        "proxy_cost_normalized_yield",
        "risk_used",
        "mean_predicted_success",
    ]
    for method in new_policy.index:
        for field in selector_fields:
            add(
                "selector comparison",
                f"{method}.{field}",
                old_policy.loc[method, field],
                new_policy.loc[method, field],
                policy_rel,
                policy_rel,
                "Batch-5 common-risk result on the final active test pool; early stop events are preserved.",
            )

    for batch_size in [5, 10, 20]:
        for field in ["mean_hits", "sd_hits"]:
            add(
                "random baseline",
                f"batch_{batch_size}.{field}",
                old_random.loc[batch_size, field],
                new_random.loc[batch_size, field],
                random_rel,
                random_rel,
                "Descriptive distribution across the unchanged 100-seed full-pool design.",
            )

    pool_fields = [
        "n_records",
        "n_pools",
        "mean_selected_count",
        "mean_hits",
        "sd_hits",
        "mean_hit_fraction",
        "sd_hit_fraction",
        "mean_proxy_cost_normalized_yield",
        "mean_jaccard_to_full_test",
        "full_batch_rate",
    ]
    for method, batch_size in new_pool.index:
        for field in pool_fields:
            add(
                "pool resampling",
                f"{method}.batch_{int(batch_size)}.{field}",
                old_pool.loc[(method, batch_size), field],
                new_pool.loc[(method, batch_size), field],
                pool_rel,
                pool_rel,
                "Descriptive DOI-group half-pool resampling; design remains 20 pools and 10 random seeds per pool.",
            )

    for field in ["pc1_percent", "pc2_percent"]:
        add(
            "descriptor PCA",
            field,
            old_pca[field],
            new_pca[field],
            pca_rel,
            pca_rel,
            "PCA is recomputed on the final active test descriptor matrix.",
        )

    for method in new_traces:
        for field in [
            "selected_candidate_ids",
            "selection_event_count",
            "stop_event_count",
        ]:
            add(
                "trace and replay",
                f"{method}.{field}",
                old_traces[method][field],
                new_traces[method][field],
                f"traces/action_traces/{method}_test_common_contract_trace.json",
                f"traces/action_traces/{method}_test_common_contract_trace.json",
                "Trace identity follows the final active test pool and locked parameter record.",
            )
    for field in ["status", "trace_count", "passed_count"]:
        add(
            "trace and replay",
            f"replay.{field}",
            old_replay[field],
            new_replay[field],
            replay_rel,
            replay_rel,
            "Replay is exact archived-implementation replay, not independent implementation validation.",
        )

    frame = pd.DataFrame(rows)
    key_metrics = frame[
        frame["metric"].isin(
            [
                "training rows",
                "calibration rows",
                "test rows",
                "excluded train-source-overlap rows",
                "training DOI groups",
                "calibration DOI groups",
                "test DOI groups",
                "selected activation C",
                "calibration_roc_auc",
                "calibration_brier_score",
                "test_roc_auc",
                "test_accuracy_at_0_5",
                "test_brier_score",
                "test_majority_class_accuracy",
                "pc1_percent",
                "pc2_percent",
                "replay.status",
                "training DOI coverage",
                "train-calibration DOI-group overlap",
                "train-test DOI-group overlap",
                "calibration-test DOI-group overlap",
                "training CV splitter",
                "maximum model-CV fold DOI-group overlap",
                "authoritative lock written before test scoring",
            ]
        )
    ]
    markdown = [
        "# v45A2R2 versus v45A3 scientific comparison",
        "",
        f"Baseline is immutable commit `{BASELINE_COMMIT}`. v45A3 values are read from the current repository artifacts; rerun this report builder after applying the verified hosted-canonical payload.",
        "",
        "The v45A3 source-group rebuild materially weakens the activation benchmark. The final test ROC-AUC is near chance and the 0.5-threshold accuracy is below the final test majority-class baseline. These results are retained without split search or post-test tuning.",
        "",
        "## Key comparison",
        "",
        "| Metric | v45A2R2 | v45A3 | Classification |",
        "|---|---:|---:|---|",
    ]
    for row in key_metrics.to_dict("records"):
        markdown.append(
            f"| {row['metric']} | {row['v45A2R2']} | {row['v45A3']} | {row['change_classification']} |"
        )
    markdown.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- The endpoint remains packaged solvent-removal/activation stability; it is not water stability or a laboratory failure probability.",
            "- Proxy cost remains descriptor complexity, and risk remains a benchmark constraint proxy.",
            "- Within the final locked evaluation pipeline, test labels were not used for model or policy selection.",
            "- Calibration/test membership was preserved for surviving groups; no seed search or test-performance optimization was performed.",
            "- The CSV is the complete comparison register, including every selector, random summary, pool-resampling field, PCA component, selected trace IDs, and replay status.",
            "",
        ]
    )
    return frame, "\n".join(markdown)


def build_sync_report() -> tuple[pd.DataFrame, str]:
    activation = _current_json("results/reproduction/activation_model_metrics.json")
    source = _current_json("results/reproduction/source_group_audit.json")
    lock = _current_json("results/reproduction/tuning/v45A3_locked_parameters_before_test.json")
    replay = _current_json("results/reproduction/replay_summary.json")
    policy = _current_csv(
        "results/reproduction/test_policy_comparison_common_risk.csv"
    ).set_index("method")
    pca = _current_json("results/reproduction/descriptor_pca_variance.json")
    old_activation = _baseline_json("results/reproduction/activation_model_metrics.json")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()

    counts = (
        f"1,394 training, {activation['n_calibration']} calibration, "
        f"{activation['n_test']} test, and {activation['n_excluded_train_source_overlap']} "
        "provenance-only excluded rows"
    )
    groups = (
        f"{source['training_doi_group_count']}/{source['calibration_doi_group_count']}/"
        f"{source['test_doi_group_count']} active train/calibration/test DOI groups"
    )
    metrics = (
        f"calibration ROC-AUC {activation['calibration_roc_auc']:.4f} and Brier "
        f"{activation['calibration_brier_score']:.4f}; test ROC-AUC "
        f"{activation['test_roc_auc']:.4f}, accuracy {activation['test_accuracy_at_0_5']:.4f}, "
        f"Brier {activation['test_brier_score']:.4f}, and majority baseline "
        f"{activation['test_majority_class_accuracy']:.4f}"
    )
    policy_text = (
        f"alpha={lock['policy']['alpha']:g}, beta={lock['policy']['beta']:g}, "
        f"gamma={lock['policy']['gamma']:g}, delta={lock['policy']['delta']:g}, "
        f"diversity scale={lock['policy']['diversity_distance_scale']:g}, "
        f"cost budget/item={lock['policy']['cost_budget_per_item']:.6g}, and "
        f"risk budget/item={lock['policy']['risk_budget_per_item']:g}"
    )
    selector_text = "; ".join(
        f"{method}: {int(row.selected_count)} selected, {int(row.hits)} stable hits, "
        f"cost {row.total_proxy_cost:.4f}, yield {row.proxy_cost_normalized_yield:.4f}, "
        f"risk {row.risk_used:.4f}"
        for method, row in policy.iterrows()
    )
    safe_test_wording = (
        "Within the final locked evaluation pipeline, test labels were not used for "
        "model or policy selection."
    )

    rows: list[dict[str, str]] = []

    def add(
        document: str,
        section: str,
        current: str,
        new: str,
        reason: str,
        source_file: str,
        source_field: str,
        confidence: str = "high",
        requirement: str = "mandatory",
    ) -> None:
        rows.append(
            {
                "document": document,
                "section": section,
                "current_wording_or_value": (
                    "NOT VERBATIM—artifact-backed v45A2R2 state or claim pattern: "
                    + current
                ),
                "new_wording_or_value": new,
                "reason": reason,
                "source_output_file": source_file,
                "source_field": source_field,
                "confidence": confidence,
                "mandatory_or_optional": requirement,
            }
        )

    old_metrics = (
        f"v45A2R2: {old_activation['n_train']}/{old_activation['n_calibration']}/"
        f"{old_activation['n_test']} rows; C={old_activation['selected_C']:g}; test ROC-AUC "
        f"{old_activation['test_roc_auc']:.4f}, accuracy "
        f"{old_activation['test_accuracy_at_0_5']:.4f}, Brier "
        f"{old_activation['test_brier_score']:.4f}"
    )
    add(
        "Main manuscript",
        "Abstract",
        old_metrics,
        f"Report the DOI-complete rebuild ({counts}; {groups}), grouped-CV C={activation['selected_C']:g}, and {metrics}. State that the benchmark weakened after source-group correction.",
        "The primary evaluation universe, model, and all reported activation metrics changed materially.",
        "results/reproduction/activation_model_metrics.json; results/reproduction/source_group_audit.json",
        "n_train, n_calibration, n_test, n_excluded_train_source_overlap, selected_C, performance metrics, DOI group counts",
    )
    add(
        "Main manuscript",
        "Introduction",
        "Source-publication separation was incomplete because training DOI provenance was unavailable.",
        "Explain that official immutable MOFSimplify training provenance enabled DOI-complete grouping and a conservative rebuild that retained training rows while excluding overlapping held rows.",
        "The motivation and evidentiary boundary for the rebuild must be transparent.",
        "data/provenance/mofsimplify_training_source_provenance.json",
        "upstream_repository, upstream_commit, upstream_git_blob_sha1, raw_file_sha256",
    )
    add(
        "Main manuscript",
        "Formal problem statement",
        "Evaluation membership followed the earlier calibration/test split without a complete train-source boundary.",
        "Define normalized source DOI as the publication-group key; retain the original training pool; preserve surviving calibration/test assignments; exclude all held rows whose DOI occurs in training from active evaluation.",
        "The admissible evaluation universe is now an explicit protocol component.",
        "data/processed/full_public_stability_source_manifest.csv",
        "original_model_split, final_evaluation_split, source_group_id, exclusion_reason",
    )
    add(
        "Main manuscript",
        "Methods - source provenance and splits",
        "Training DOI coverage was unavailable; policy calibration/test were DOI-disjoint only from each other.",
        f"State 100% DOI coverage in all active splits, zero train-calibration/train-test/calibration-test DOI overlap, {source['excluded_train_source_overlap_row_count']} excluded held rows from {source['excluded_train_source_overlap_doi_group_count']} DOI groups, and {counts}.",
        "Complete source-group disjointness is now supported and the exclusion mechanism must be reproducible.",
        "results/reproduction/source_group_audit.json; results/reproduction/train_held_source_overlap_exclusions.csv",
        "coverage fields, overlap counts, excluded row/group counts",
    )
    add(
        "Main manuscript",
        "Methods - activation model selection",
        f"Five-fold row-level training CV selected C={old_activation['selected_C']:g}.",
        f"Five-fold StratifiedGroupKFold grouped by normalized training DOI, shuffle=True, fixed random_state=45, and the unchanged one-standard-error rule selected C={activation['selected_C']:g}; every fold had zero group overlap and both classes.",
        "Training DOI availability requires group-aware model selection.",
        "results/reproduction/tuning/activation_C_train_internal_SOURCE_GROUP_CV.csv; results/reproduction/tuning/activation_train_group_cv_fold_audit.csv",
        "selected_one_standard_error, cv_splitter, random_state, n_splits, group_overlap_count, class fractions",
    )
    add(
        "Main manuscript",
        "Methods - policy locking",
        "Policy was calibrated under the earlier 349-row calibration pool.",
        f"Policy was selected only on the preserved final {activation['n_calibration']}-row calibration pool and frozen before test evaluation: {policy_text}.",
        "The calibration pool and selected policy changed; lock-before-test is now machine-readable.",
        "results/reproduction/tuning/v45A3_locked_parameters_before_test.json",
        "policy, cost_scale_calibration_median, test_evaluation_may_start_only_after_this_file_is_written",
    )
    add(
        "Main manuscript",
        "Statistics and reproducibility",
        "Earlier random and resampling results used the larger held-out pool and a row-CV model.",
        f"Retain descriptive-only interpretation: 100 full-pool random runs per batch size, 20 DOI-group half-pool resamples, and 10 random seeds per resampled pool; report 5/5 exact archived-implementation replay. Use this test statement verbatim: \"{safe_test_wording}\"",
        "The design is unchanged but operates on the corrected pool and now has a pre-test lock/source-manifest hash chain.",
        "results/reproduction/tuning/v45A3_locked_parameters_before_test.json; results/reproduction/replay_summary.json",
        "random_baseline_seed_policy, pool_resampling_design, replay status/counts, test_use_claim",
    )
    add(
        "Main manuscript",
        "Results - activation benchmark",
        old_metrics,
        f"Report {metrics}. Explicitly note that test accuracy is below the {activation['test_majority_class_accuracy']:.4f} majority baseline and ROC-AUC is near chance; do not present the weaker result as improved performance.",
        "The source-group rebuild materially weakens and reverses parts of the earlier interpretation.",
        "results/reproduction/activation_model_metrics.json",
        "calibration_roc_auc, calibration_brier_score, test_roc_auc, test_accuracy_at_0_5, test_brier_score, test_majority_class_accuracy",
    )
    add(
        "Main manuscript",
        "Results - selector comparison",
        "Earlier batch-5 results used a 436-row test pool and generally selected 4-5 candidates.",
        f"On the final 193-row test pool under the common risk contract: {selector_text}. All selectors stopped before filling the requested batch; retain the stop events.",
        "Selector outcomes, costs, yields, risks, and full-batch status changed materially.",
        "results/reproduction/test_policy_comparison_common_risk.csv; traces/trace_archive_manifest.csv",
        "selected_count, hits, hit_fraction, total_proxy_cost, proxy_cost_normalized_yield, risk_used, stop_event_count",
    )
    add(
        "Main manuscript",
        "Discussion",
        "The earlier benchmark supported stronger generalization and PortfolioBatch performance claims.",
        "State that source-publication correction produced materially weaker discrimination, near-chance test ROC-AUC, below-majority threshold accuracy, and widespread budget/risk early stopping. Treat this as a limitation and robustness finding, not a result to optimize away.",
        "The scientific conclusion must follow the rebuilt evidence and avoid post-test tuning.",
        "results/reproduction/activation_model_metrics.json; results/reproduction/test_policy_comparison_common_risk.csv",
        "test metrics and selector metrics",
    )
    add(
        "Main manuscript",
        "Figure 1 caption",
        "Workflow caption reflects row-level model CV and no explicit pre-test artifact gate.",
        "Describe training-only source-DOI-group CV, final-calibration-only policy selection, the authoritative parameter lock, and final-test-only evaluation.",
        "The protocol sequence changed and the lock-before-test boundary is central.",
        "figures/Figure1_workflow.*; results/reproduction/tuning/v45A3_locked_parameters_before_test.json",
        "workflow stages and lock hash",
    )
    add(
        "Main manuscript",
        "Figure 2 caption",
        "Training DOI was shown as unavailable; calibration/test comprised 349/436 rows.",
        f"Show {counts}, {groups}, {source['excluded_train_source_overlap_doi_group_count']} excluded overlap DOI groups, 100% active DOI coverage, and zero active pairwise DOI overlap.",
        "Figure 2 now displays the complete publication-group boundary.",
        "figures/Figure2_benchmark_boundary.*; results/reproduction/source_group_audit.json",
        "row counts, group counts, coverage, overlap counts",
    )
    add(
        "Main manuscript",
        "Figure 3 caption",
        "Policy comparison used the earlier 436-row test pool and earlier lock.",
        f"State that only the final 193-row active test pool is used; report the common-risk selector results and that requested batch 5 stopped at 2 selections for every method. Identify 100/20/10 as descriptive robustness designs.",
        "All Figure 3 inputs changed with the corrected evaluation universe.",
        "source_data/figure3_test_policy_common_risk_source.csv; source_data/figure3_random_baseline_summary_source.csv; source_data/figure3_pool_resampling_summary_source.csv",
        "all columns",
    )
    add(
        "Main manuscript",
        "Figure 4 caption",
        "PCA variance was based on the earlier test descriptor matrix.",
        f"State that PCA uses only the final active test rows; PC1/PC2 explain {pca['pc1_percent']:.4f}%/{pca['pc2_percent']:.4f}% and descriptor contrasts are descriptive, not causal.",
        "The test matrix changed, so the PCA and descriptor contrasts changed.",
        "results/reproduction/descriptor_pca_variance.json; source_data/figure4_pca_coordinates.csv",
        "pc1_percent, pc2_percent, candidate_id",
    )
    add(
        "Main manuscript",
        "Figure 5 caption",
        "Replay covered five traces without the DOI-complete source-manifest and pre-test-lock bindings.",
        f"Report {replay['passed_count']}/{replay['trace_count']} exact archived-implementation replay and state that every schema-3 trace binds the source manifest and authoritative pre-test lock; do not call replay an independent implementation.",
        "The trace schema and provenance chain changed.",
        "results/reproduction/replay_summary.json; traces/action_traces/*.json",
        "status, passed_count, trace_count, canonical_source_manifest_sha256, locked_parameters_record_sha256",
    )
    add(
        "Main manuscript",
        "Data availability",
        "Training DOI provenance was not packaged.",
        "Cite the MOFSimplify dataset DOI 10.5281/zenodo.5736562, immutable upstream commit 5693968b3e9b9e26eab3bdb1db908ae2877d4bb7, authenticated Git blob SHA-1, raw SHA256, and reconstruction instructions. State that the third-party raw CSV is not redistributed.",
        "The DOI backfill is reproducible without vendoring upstream raw data.",
        "data/provenance/mofsimplify_training_source_provenance.json",
        "dataset_doi, upstream_commit, upstream_git_blob_sha1, raw_file_sha256, reconstruction",
    )
    add(
        "Main manuscript",
        "Code availability",
        "Earlier private development snapshot/version.",
        f"Reference private-development version 0.45a3-dev and repository commit `{head}` only after the governed main fast-forward and hosted CI pass; keep public release/DOI statements closed.",
        "The scientific implementation, traces, and canonical reference changed.",
        "VERSION; DEVELOPMENT_STATUS.md; hosted CI evidence",
        "version, final main SHA, private-development boundary",
        confidence="medium until final main CI",
    )
    add(
        "Supplementary information",
        "SI captions and benchmark tables",
        "Tables/captions contain v45A2R2 row counts, C=3, policy values, metrics, and selector/random/resampling results.",
        f"Replace every affected v45A2R2 value with {counts}; {groups}; C={activation['selected_C']:g}; {policy_text}; {metrics}; the complete selector, 100-run random, and 20-pool resampling tables from v45A3 outputs.",
        "Nearly every activation benchmark table is numerically stale after the rebuild.",
        "results/reproduction/*.json; results/reproduction/*.csv; results/reproduction/tuning/*.csv",
        "v45A3 activation, policy, random, pool, CV, and exclusion fields",
    )
    add(
        "Main manuscript and SI",
        "Source-group wording",
        "Calibration/test DOI disjointness was described with training DOI unavailable.",
        "Use: \"Active train, calibration, and test partitions had 100% normalized-source-DOI coverage and were pairwise DOI-group disjoint; 434 held rows sharing 270 training DOI groups were retained for provenance only and excluded from evaluation.\"",
        "The complete source boundary now passes and requires exact, bounded wording.",
        "results/reproduction/source_group_audit.json",
        "coverage, overlap counts, exclusion counts",
    )
    add(
        "Main manuscript and SI",
        "Test-use wording",
        "Any statement implying the test split was historically seen only once.",
        safe_test_wording,
        "Earlier development versions existed; only the final locked-pipeline claim is supportable.",
        "results/reproduction/tuning/v45A3_locked_parameters_before_test.json",
        "test_use_claim",
    )
    add(
        "Main manuscript and SI",
        "Endpoint, cost, and risk wording",
        "Any wording that broadens activation into water stability or treats proxies as experimental quantities.",
        "Keep the endpoint as packaged solvent-removal/activation stability. State that proxy cost is descriptor complexity and risk is a benchmark constraint proxy, not measured synthesis cost or calibrated laboratory failure probability.",
        "The v45A3 rebuild does not broaden the scientific endpoint or validate proxy quantities experimentally.",
        "source_data/table1_benchmark_contract.csv; results/reproduction/full_reproduction_summary.json",
        "Primary endpoint, claim_boundary, policy proxy fields",
    )
    add(
        "Supplementary information",
        "Secondary thermal analysis",
        "Existing thermal model scope and metrics.",
        "Retain the existing secondary thermal scope; do not claim source-group-disjoint thermal evaluation. Update bytes/rendering only if regenerated outputs differ, not the scientific interpretation.",
        "The activation DOI rebuild does not establish new thermal publication-group metadata.",
        "results/reproduction/thermal_regression_metrics.json",
        "selected_alpha, MAE, RMSE, R2, scope",
        requirement="optional unless current SI wording overclaims source grouping",
    )

    frame = pd.DataFrame(rows)
    markdown = [
        "# v45A3 reader-facing synchronization report",
        "",
        "No manuscript or SI DOCX/PDF was edited. `current_wording_or_value` records the v45A2R2 repository-supported wording/value or the claim pattern to locate; a human synchronizer must verify exact prose against the authoritative documents.",
        "",
        "## Mandatory scientific message",
        "",
        f"- Final membership: {counts}; {groups}.",
        f"- Source boundary: 100% active DOI coverage; all three active DOI overlaps are zero; {source['excluded_train_source_overlap_row_count']} held rows from {source['excluded_train_source_overlap_doi_group_count']} training-overlap DOI groups are provenance-only.",
        f"- Training-only grouped model CV selected `C={activation['selected_C']:g}`.",
        f"- Performance: {metrics}.",
        "- The final test result is materially weaker than v45A2R2 and must not be hidden or optimized away.",
        f"- Required test wording: “{safe_test_wording}”",
        "",
        "## Synchronization register",
        "",
        "| Document | Section | Mandatory? | New wording/value | Source |",
        "|---|---|---|---|---|",
    ]
    for row in frame.to_dict("records"):
        new = row["new_wording_or_value"].replace("|", "\\|")
        source_file = row["source_output_file"].replace("|", "\\|")
        markdown.append(
            f"| {row['document']} | {row['section']} | {row['mandatory_or_optional']} | {new} | {source_file} |"
        )
    markdown.extend(
        [
            "",
            "Human/public gates remain closed: no Andrew approval, funding finalization, public repository, v1.0.0, release, Zenodo connection, DOI minting, or submission lock is implied.",
            "",
        ]
    )
    return frame, "\n".join(markdown)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the governed external v45A3 comparison and manuscript-sync reports"
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    comparison, comparison_md = build_comparison()
    comparison.to_csv(
        args.output_directory / "V45A2R2_VS_V45A3_SCIENTIFIC_COMPARISON.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    (args.output_directory / "V45A2R2_VS_V45A3_SCIENTIFIC_COMPARISON.md").write_text(
        comparison_md, encoding="utf-8"
    )

    sync, sync_md = build_sync_report()
    sync.to_csv(
        args.output_directory / "V45A3_READER_FACING_SYNC_REPORT.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    (args.output_directory / "V45A3_READER_FACING_SYNC_REPORT.md").write_text(
        sync_md, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "baseline_commit": BASELINE_COMMIT,
                "comparison_rows": len(comparison),
                "sync_rows": len(sync),
                "output_directory": str(args.output_directory),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
