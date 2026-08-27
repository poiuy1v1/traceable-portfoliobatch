from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import traceable_portfoliobatch  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from traceable_portfoliobatch.io_utils import git_blob_sha1, sha256_file
from traceable_portfoliobatch.modeling import (
    predict_from_artifact,
    select_activation_C_train_cv,
)
from traceable_portfoliobatch.regression import (
    CONTRACT_REL,
    MANIFEST_REL,
    build_generated_artifact_manifest,
    build_semantic_snapshot,
    compare_semantic_snapshots,
)
from traceable_portfoliobatch.replay import replay_all
from traceable_portfoliobatch.workflow import _attach_source_groups

DERIVED = ROOT / "data/derived/full_public_stability_features_relocked.csv"


def test_activation_test_scores_reproduce_from_locked_artifact():
    features = pd.read_csv(DERIVED)
    test = features[features.final_evaluation_split.eq("test")].reset_index(drop=True)
    saved = pd.read_csv(ROOT / "data/processed/full_public_stability_manifest_scored.csv")
    saved = saved[saved.final_evaluation_split.eq("test")].reset_index(drop=True)
    probabilities = predict_from_artifact(
        test, ROOT / "models/activation_logistic_train_split.json"
    )
    assert len(saved) == len(test) == 193
    assert abs(probabilities - saved.predicted_success.to_numpy()).max() < 1e-12


def test_model_and_policy_tuning_are_separated():
    locked = json.loads(
        (ROOT / "results/reproduction/tuning/v49_locked_parameters_before_test.json").read_text()
    )
    assert "selected inside train" in locked["selection_split"]
    assert locked["test_use"] == "none during model or policy tuning"
    assert locked["primary_policy_mode"] == "cost_only"
    assert locked["primary_risk_budget"] is None
    assert locked["policy"]["risk_budget_per_item"] is None
    assert locked["legacy_risk_stress_contract"]["risk_budget_per_item"] == 0.35
    assert locked["legacy_risk_stress_contract"]["role"] == "stress_test_only"
    assert locked["training_cv"] == {
        "group_key": "normalized_source_doi",
        "group_overlap_count_max": 0,
        "n_splits": 5,
        "random_state": 45,
        "shuffle": True,
        "splitter": "StratifiedGroupKFold",
    }


def test_authoritative_parameter_lock_is_written_before_test_evaluation():
    source = (ROOT / "src/traceable_portfoliobatch/workflow.py").read_text(encoding="utf-8")
    lock_write = source.index("write_json(v49_lock_path")
    test_labels = source.index('y_test = (test["outcome_binary"]')
    test_scoring = source.index("test_scored, z_test = _score_table(test, model, columns)")
    assert lock_write < test_labels < test_scoring
    assert 'test["outcome_binary"]' not in source[:lock_write]
    lock_path = ROOT / "results/reproduction/tuning/v49_locked_parameters_before_test.json"
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    assert locked["test_evaluation_may_start_only_after_this_file_is_written"] is True
    assert locked["test_use"] == "none during model or policy tuning"


def test_excluded_train_source_overlap_rows_are_never_scored():
    manifest = pd.read_csv(ROOT / "data/processed/full_public_stability_source_manifest.csv")
    excluded = manifest[manifest.final_evaluation_split.eq("excluded_train_source_overlap")]
    scored = pd.read_csv(ROOT / "data/processed/full_public_stability_manifest_scored.csv")
    assert len(excluded) == 434
    assert set(excluded.candidate_id).isdisjoint(set(scored.candidate_id))
    assert set(scored.final_evaluation_split) == {"calibration", "test"}
    assert excluded.exclusion_reason.eq("source DOI group also present in training").all()


def test_v45a2r2_calibration_test_assignment_is_preserved_for_active_rows():
    manifest = pd.read_csv(ROOT / "data/processed/full_public_stability_source_manifest.csv")
    active_held = manifest[manifest.final_evaluation_split.isin(["calibration", "test"])]
    assert active_held.final_evaluation_split.eq(active_held.original_model_split).all()
    assert set(active_held[active_held.original_model_split.eq("calibration")].candidate_id)
    assert set(active_held[active_held.original_model_split.eq("test")].candidate_id)


def test_runner_rejects_changed_original_training_membership():
    features = pd.DataFrame([
        {"candidate_id": "train-a", "model_split": "train"},
        {"candidate_id": "test-b", "model_split": "test"},
    ])
    manifest = pd.DataFrame([
        {
            "candidate_id": "train-a",
            "source_doi": "10.1000/train-a",
            "original_model_split": "train",
            "final_evaluation_split": "calibration",
            "exclusion_reason": "",
        },
        {
            "candidate_id": "test-b",
            "source_doi": "10.1000/test-b",
            "original_model_split": "test",
            "final_evaluation_split": "test",
            "exclusion_reason": "",
        },
    ])
    with pytest.raises(ValueError, match="original training membership"):
        _attach_source_groups(features, manifest)


def test_runner_rejects_nonblank_non_doi_source_group():
    features = pd.DataFrame([{"candidate_id": "train-a", "model_split": "train"}])
    manifest = pd.DataFrame([
        {"candidate_id": "train-a", "source_doi": "not-a-doi"}
    ])
    with pytest.raises(ValueError, match="invalid DOI syntax"):
        _attach_source_groups(features, manifest)


def test_all_active_splits_are_doi_complete_and_group_disjoint():
    audit = json.loads((ROOT / "results/reproduction/source_group_audit.json").read_text())
    assert audit["policy_calibration_test_source_doi_overlap_count"] == 0
    assert audit["train_calibration_source_doi_overlap_count"] == 0
    assert audit["train_test_source_doi_overlap_count"] == 0
    assert audit["training_rows_without_packaged_source_doi"] == 0
    assert audit["training_rows_with_packaged_source_doi"] == 1394
    assert audit["active_training_doi_coverage"] == 1.0
    assert audit["active_calibration_doi_coverage"] == 1.0
    assert audit["active_test_doi_coverage"] == 1.0
    assert audit["excluded_train_source_overlap_row_count"] == 434
    assert audit["excluded_train_source_overlap_doi_group_count"] == 270


def test_canonical_source_manifest_covers_all_splits():
    features = pd.read_csv(ROOT / "data/processed/full_public_stability_features.csv")
    manifest = pd.read_csv(ROOT / "data/processed/full_public_stability_source_manifest.csv")
    assert len(manifest) == len(features) == 2179
    assert manifest.candidate_id.is_unique
    assert set(manifest.candidate_id) == set(features.candidate_id)
    assert manifest["source_doi"].fillna("").ne("").all()
    assert manifest["source_group_basis"].eq("normalized_source_doi").all()
    assert manifest["source_group_id"].eq(manifest["source_doi_normalized"]).all()
    assert manifest["original_model_split"].eq(manifest["model_split"]).all()
    assert manifest["final_evaluation_split"].value_counts().to_dict() == {
        "train": 1394,
        "excluded_train_source_overlap": 434,
        "test": 193,
        "calibration": 158,
    }


def test_C_grid_is_source_group_aware_and_all_fits_converged():
    table = pd.read_csv(
        ROOT / "results/reproduction/tuning/activation_C_train_internal_SOURCE_GROUP_CV.csv"
    )
    selected = float(table.loc[table.selected_one_standard_error, "C"].iloc[0])
    assert table.C.max() >= 1000
    assert selected <= table.C.max()
    assert table.all_folds_converged.astype(bool).all()
    assert (table.max_n_iter < table.max_iter).all()
    assert set(table.cv_splitter) == {"StratifiedGroupKFold"}
    assert set(table.random_state) == {45}
    assert set(table.n_splits) == {5}
    assert set(table.training_doi_group_count) == {865}
    ranked = table.sort_values(
        ["mean_train_cv_roc_auc", "mean_train_cv_brier", "C"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    threshold = (
        float(ranked.loc[0, "mean_train_cv_roc_auc"])
        - float(ranked.loc[0, "sd_train_cv_roc_auc"]) / np.sqrt(5)
    )
    expected_within = table.mean_train_cv_roc_auc.ge(threshold)
    assert table.within_one_standard_error.astype(bool).equals(expected_within)
    assert selected == float(table.loc[expected_within, "C"].min())
    fold = pd.read_csv(
        ROOT / "results/reproduction/tuning/activation_train_group_cv_fold_audit.csv"
    )
    assert len(fold) == 5
    assert fold.group_overlap_count.eq(0).all()
    assert fold.train_positive_fraction.between(0, 1, inclusive="neither").all()
    assert fold.validation_positive_fraction.between(0, 1, inclusive="neither").all()
    metrics = json.loads((ROOT / "results/reproduction/activation_model_metrics.json").read_text())
    assert metrics["all_activation_fits_converged"] is True
    assert metrics["final_fit_n_iter"] < metrics["max_iter"]


def test_group_cv_rejects_refcode_surrogates_even_when_nonblank():
    train = pd.DataFrame(
        {
            "x": [1.0],
            "outcome_binary": [1],
            "source_group_id": ["train_refcode::ABC"],
            "source_doi_normalized": ["train_refcode::ABC"],
            "source_group_basis": ["normalized_source_doi"],
        }
    )
    with pytest.raises(ValueError, match="unresolved"):
        select_activation_C_train_cv(train, ["x"], [1.0])


def test_policy_parameter_stability_reported():
    table = pd.read_csv(
        ROOT / "results/reproduction/tuning/portfolio_policy_cost_only_group_bootstrap_stability.csv"
    )
    assert len(table) == 20
    assert {
        "beta", "gamma", "delta", "equivalent_top_config_count",
        "selected_tuple_frequency", "full_batch", "full_batch_frequency",
        "primary_policy_mode",
    } <= set(table.columns)
    assert table.primary_policy_mode.eq("cost_only").all()
    assert np.allclose(
        table.full_batch_frequency.astype(float),
        float(table.full_batch.astype(bool).mean()),
        rtol=0,
        atol=1e-12,
    )
    if "risk_budget_per_item" in table:
        assert table.risk_budget_per_item.isna().all()


def test_v49_source_overlap_C_ablation_is_the_fixed_post_lock_2x2():
    table = pd.read_csv(
        ROOT / "results/reproduction/diagnostics/source_overlap_C_ablation.csv"
    )
    assert set(zip(table.C, table.evaluation_universe)) == {
        (C, universe)
        for C in (0.001, 3.0)
        for universe in (
            "historical_overlap_inclusive_test",
            "final_publication_source_disjoint_test",
        )
    }
    assert table.groupby("evaluation_universe").n_rows.first().to_dict() == {
        "historical_overlap_inclusive_test": 436,
        "final_publication_source_disjoint_test": 193,
    }
    assert table.diagnostic_only.astype(bool).all()
    assert ~table.affects_selected_C.astype(bool).any()
    summary = json.loads(
        (
            ROOT
            / "results/reproduction/diagnostics/source_overlap_C_ablation_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["selected_C_before_and_after_diagnostic"] == 0.001
    assert summary["affects_selected_C"] is False
    assert {
        "split_effect_at_C_3",
        "split_effect_at_C_0p001",
        "C_effect_on_overlap_inclusive_test",
        "C_effect_on_source_disjoint_test",
    } <= set(summary)


def test_v49_dummy_baselines_are_training_fitted_and_bootstrap_is_group_level():
    dummy = pd.read_csv(ROOT / "source_data/activation_dummy_baselines_source.csv")
    assert set(dummy.strategy) == {"prior", "most_frequent"}
    assert dummy.fit_split.eq("train").all()
    assert dummy.evaluation_split.eq(
        "final_publication_source_disjoint_test"
    ).all()
    assert dummy.training_n.eq(1394).all()
    assert dummy.test_n.eq(193).all()
    assert np.isfinite(dummy[["accuracy", "roc_auc", "brier_score"]]).all().all()

    raw = pd.read_csv(
        ROOT / "results/reproduction/activation_test_source_group_bootstrap_raw.csv"
    )
    summary = json.loads(
        (
            ROOT
            / "results/reproduction/activation_test_source_group_bootstrap_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert len(raw) == summary["requested_resamples"] == 10000
    assert summary["seed"] == 45
    assert summary["bootstrap_group_unit"] == "normalized_source_doi"
    assert raw.sampled_group_occurrences.eq(173).all()
    assert int(raw.invalid_for_auc.astype(bool).sum()) == summary[
        "invalid_auc_resamples"
    ]
    assert summary["valid_auc_resamples"] + summary["invalid_auc_resamples"] == 10000
    assert raw.loc[raw.invalid_for_auc.astype(bool), "roc_auc"].isna().all()


def test_v49_risk_and_cost_feasibility_artifact_has_exact_design_and_roles():
    audit = pd.read_csv(ROOT / "results/reproduction/risk_feasibility_audit.csv")
    assert set(zip(audit.pool, audit.requested_batch_size)) == {
        (pool, batch_size)
        for pool in ("final_calibration", "final_test")
        for batch_size in (5, 10, 20)
    }
    assert np.allclose(
        audit.legacy_risk_budget_per_item.astype(float),
        0.35,
        rtol=0,
        atol=1e-12,
    )
    assert audit.legacy_risk_contract_role.eq("stress_test_only").all()
    assert audit.risk_score_interpretation.eq(
        "benchmark-defined uncalibrated proxy"
    ).all()
    assert ~audit.primary_selected_risk.astype(bool).any()
    assert audit.loc[
        audit.requested_batch_size.eq(5), "five_lowest_risk_values"
    ].str.startswith("[").all()
    assert audit.loc[
        audit.requested_batch_size.ne(5), "five_lowest_risk_values"
    ].isna().all()

    stress = pd.read_csv(
        ROOT / "results/reproduction/diagnostics/test_policy_risk_stress_legacy_0p35.csv"
    )
    assert len(stress) == 15
    assert stress.policy_contract.eq("legacy_risk_stress_0p35").all()
    assert stress.contract_role.eq("stress_test_only").all()
    assert stress.risk_interpretation.eq(
        "benchmark-defined uncalibrated proxy"
    ).all()


def test_random_seeds_nested_within_each_pool_resample_and_contract_table_matches():
    table = pd.read_csv(
        ROOT / "results/reproduction/test_pool_resampling_primary_cost_only_raw.csv"
    )
    random_rows = table[(table.method == "random_baseline") & (table.batch_size == 5)]
    assert random_rows.pool_seed.nunique() == 20
    assert random_rows.groupby("pool_seed").random_seed.nunique().min() == 10
    contract = pd.read_csv(ROOT / "source_data/table1_benchmark_contract.csv")
    statistics = contract.loc[contract.field.eq("Statistics"), "implementation"].iloc[0]
    assert "10 random seeds per resampled pool" in statistics
    assert "20 random seeds per resampled pool" not in statistics


def test_all_stepwise_action_traces_replay_exactly():
    result = replay_all(ROOT)
    assert result["status"] == "PASS" and result["trace_count"] == result["passed_count"] == 5


def test_primary_selector_outputs_report_ids_full_batch_and_stop_events():
    table = pd.read_csv(
        ROOT / "results/reproduction/test_policy_comparison_primary_cost_only.csv"
    )
    assert {
        "selected_candidate_ids",
        "full_batch",
        "full_batch_status",
        "selection_event_count",
        "stop_event_count",
    } <= set(table.columns)
    assert table.selection_event_count.eq(table.selected_count).all()
    assert table.stop_event_count.eq(0).all()
    assert table.full_batch.astype(bool).all()
    assert table.selected_count.eq(5).all()
    assert table.policy_contract.eq("cost_only").all()
    assert table.risk_contract.eq("none").all()
    assert table.risk_budget_per_item.isna().all()
    for row in table.to_dict("records"):
        ids = json.loads(row["selected_candidate_ids"])
        assert len(ids) == int(row["selected_count"])
        trace = json.loads(
            (
                ROOT
                / "traces/action_traces"
                / f"{row['method']}_test_primary_cost_contract_trace.json"
            ).read_text(encoding="utf-8")
        )
        assert ids == [item["candidate_id"] for item in trace["selected_candidates"]]
        assert trace["selection_event_count"] == int(row["selection_event_count"])
        assert trace["stop_event_count"] == int(row["stop_event_count"])
        assert trace["full_batch"] is True


def test_source_manifest_and_pretest_lock_hashes_propagate_to_every_trace():
    source_manifest = ROOT / "data/processed/full_public_stability_source_manifest.csv"
    lock = ROOT / "results/reproduction/tuning/v49_locked_parameters_before_test.json"
    trace_paths = sorted(
        (ROOT / "traces/action_traces").glob("*_test_primary_cost_contract_trace.json")
    )
    assert len(trace_paths) == 5
    for trace_path in trace_paths:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        assert trace["canonical_source_manifest"] == (
            "data/processed/full_public_stability_source_manifest.csv"
        )
        assert trace["canonical_source_manifest_sha256"] == sha256_file(source_manifest)
        assert trace["locked_parameters_record"] == (
            "results/reproduction/tuning/v49_locked_parameters_before_test.json"
        )
        assert trace["locked_parameters_record_sha256"] == sha256_file(lock)
        assert trace["locked_parameters"] == json.loads(lock.read_text(encoding="utf-8"))


def test_primary_cost_only_contract_is_shared():
    table = pd.read_csv(
        ROOT / "results/reproduction/test_policy_comparison_primary_cost_only.csv"
    )
    assert set(table.method) == {
        "top_score", "uncertainty", "cost_aware", "portfolio_batch", "random_baseline"
    }
    assert table.risk_budget_per_item.isna().all()
    assert table.cost_budget_per_item.nunique() == 1
    assert table.policy_contract.eq("cost_only").all()
    assert table.risk_role.eq("descriptive_only").all()


def test_full_runner_summary_is_present_and_bounded():
    result = json.loads((ROOT / "results/reproduction/full_reproduction_summary.json").read_text())
    assert result["status"].startswith("PASS_V49")
    assert result["replay"]["passed_count"] == 5
    assert result["reproduction_contract"]["frozen_feature_input"].endswith(
        "full_public_stability_features.csv"
    )
    assert result["reproduction_contract"]["derived_relocked_output"].startswith("data/derived/")


def test_git_blob_sha_uses_git_object_header(tmp_path: Path):
    path = tmp_path / "example.txt"
    path.write_bytes(b"Hello world\n")
    raw_sha1 = hashlib.sha1(path.read_bytes()).hexdigest()
    expected_blob = hashlib.sha1(b"blob 12\0Hello world\n").hexdigest()
    assert git_blob_sha1(path) == expected_blob
    assert git_blob_sha1(path) != raw_sha1


def _load_backfill_module():
    path = ROOT / "scripts/backfill_train_source_doi.py"
    spec = importlib.util.spec_from_file_location("paper13_backfill", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_scientific_gate_module():
    script = ROOT / "scripts/validate_v49_scientific_gate.py"
    spec = importlib.util.spec_from_file_location("v49_scientific_gate", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backfill_preserves_held_doi_and_writes_canonical_manifest(tmp_path: Path):
    features = pd.DataFrame([
        {"candidate_id": "train-a", "model_split": "train", "raw_refcode": "A"},
        {"candidate_id": "train-b", "model_split": "train", "raw_refcode": "B"},
        {"candidate_id": "test-c", "model_split": "test", "raw_refcode": "C"},
    ])
    source = pd.DataFrame([
        {"candidate_id": "train-a", "model_split": "train", "raw_refcode": "A", "source_doi": ""},
        {"candidate_id": "train-b", "model_split": "train", "raw_refcode": "B", "source_doi": ""},
        {"candidate_id": "test-c", "model_split": "test", "raw_refcode": "C", "source_doi": "10.1000/held"},
    ])
    upstream = pd.DataFrame([
        {"refcode": "A", "doi": "10.1000/a"},
        {"refcode": "B", "doi": "https://doi.org/10.1000/b"},
    ])
    feature_path = tmp_path / "features.csv"
    source_path = tmp_path / "source.csv"
    upstream_path = tmp_path / "train.csv"
    output_path = tmp_path / "source_with_doi.csv"
    audit_path = tmp_path / "audit.json"
    features.to_csv(feature_path, index=False)
    source.to_csv(source_path, index=False)
    upstream.to_csv(upstream_path, index=False)
    module = _load_backfill_module()
    args = argparse.Namespace(
        upstream_train_csv=upstream_path,
        features=feature_path,
        source_manifest=source_path,
        output_source_manifest=output_path,
        audit_output=audit_path,
        require_official_git_blob_sha=False,
        expected_raw_sha256=None,
    )
    audit = module.run_backfill(args)
    output = pd.read_csv(output_path).fillna("")
    assert audit["status"] == "PASS"
    assert output.set_index("candidate_id").loc["test-c", "source_doi"] == "10.1000/held"
    assert output.set_index("candidate_id").loc["train-a", "source_doi"] == "10.1000/a"
    assert output.set_index("candidate_id").loc["train-b", "source_doi"] == "10.1000/b"
    assert output.set_index("candidate_id").loc["test-c", "original_model_split"] == "test"
    assert output.set_index("candidate_id").loc["test-c", "final_evaluation_split"] == "test"
    assert audit["active_source_doi_group_overlaps"] == {
        "train_calibration": 0,
        "train_test": 0,
        "calibration_test": 0,
    }
    assert audit["upstream"]["observed_raw_sha256"]
    assert audit["upstream"]["observed_git_blob_sha1"]
    assert audit["upstream"]["observed_raw_sha256"] != audit["upstream"]["observed_git_blob_sha1"]


def test_backfill_rejects_nonblank_non_doi_upstream_mapping(tmp_path: Path):
    features = pd.DataFrame([
        {"candidate_id": "train-a", "model_split": "train", "raw_refcode": "A"}
    ])
    source = pd.DataFrame([
        {
            "candidate_id": "train-a",
            "model_split": "train",
            "raw_refcode": "A",
            "source_doi": "",
        }
    ])
    upstream = pd.DataFrame([{"refcode": "A", "doi": "not-a-doi"}])
    feature_path = tmp_path / "features.csv"
    source_path = tmp_path / "source.csv"
    upstream_path = tmp_path / "train.csv"
    features.to_csv(feature_path, index=False)
    source.to_csv(source_path, index=False)
    upstream.to_csv(upstream_path, index=False)
    args = argparse.Namespace(
        upstream_train_csv=upstream_path,
        features=feature_path,
        source_manifest=source_path,
        output_source_manifest=tmp_path / "output.csv",
        audit_output=tmp_path / "audit.json",
        require_official_git_blob_sha=False,
        expected_raw_sha256=None,
    )
    with pytest.raises(SystemExit, match="invalid DOI syntax"):
        _load_backfill_module().run_backfill(args)


def test_runner_source_group_attachment_consumes_training_doi():
    features = pd.DataFrame([
        {
            "candidate_id": "train-a", "model_split": "train", "raw_refcode": "A",
            "outcome_binary": 1, "raw_split": "train", "raw_source_file": "train.csv",
            "raw_source_row": 1, "x": 1.0,
        },
        {
            "candidate_id": "test-b", "model_split": "test", "raw_refcode": "B",
            "outcome_binary": 0, "raw_split": "test", "raw_source_file": "test.csv",
            "raw_source_row": 1, "x": 2.0,
        },
    ])
    manifest = pd.DataFrame([
        {"candidate_id": "train-a", "source_doi": "10.1000/train"},
        {"candidate_id": "test-b", "source_doi": "10.1000/test"},
    ])
    attached = _attach_source_groups(features, manifest)
    train = attached.set_index("candidate_id").loc["train-a"]
    assert train.source_group_id == "10.1000/train"
    assert train.source_group_basis == "normalized_source_doi"


def test_figure1_and_source_code_do_not_assign_C_selection_to_calibration():
    source = (ROOT / "src/traceable_portfoliobatch/figures.py").read_text(encoding="utf-8")
    assert "select C and policy" not in source
    assert '("Calibration split", "select policy\\nparameters")' in source
    svg = (ROOT / "figures/Figure1_workflow.svg").read_text(encoding="utf-8")
    assert "select C and policy" not in svg


def test_figure_source_manifest_lists_actual_figure3_raw_inputs():
    manifest = pd.read_csv(ROOT / "source_data/figure_source_data_manifest.csv")
    rows = manifest[manifest.figure_file.str.contains("Figure3_policy_comparison")]
    joined = " ".join(rows.source.astype(str))
    assert "figure3_test_policy_primary_cost_source.csv" in joined
    assert "figure3_random_baseline_primary_cost_raw_points.csv" in joined
    assert "figure3_random_baseline_primary_cost_summary_source.csv" in joined
    assert "figure3_pool_resampling_primary_cost_raw_points.csv" in joined
    assert "figure3_pool_resampling_primary_cost_summary_source.csv" in joined
    assert "common_risk" not in joined


def test_generated_artifact_manifest_is_complete_and_current():
    stored = json.loads((ROOT / MANIFEST_REL).read_text())
    observed = build_generated_artifact_manifest(ROOT)
    assert stored == observed


def test_v49_scientific_gate_passes_independently():
    module = _load_scientific_gate_module()
    result = module.validate_repository_state()
    assert result["status"] == "PASS", result["errors"]
    assert all(result["checks"].values())


def test_cost_only_policy_grid_has_exact_deterministic_winner():
    table = pd.read_csv(
        ROOT / "results/reproduction/tuning/portfolio_policy_cost_only_calibration_selection.csv"
    )
    lock = json.loads(
        (ROOT / "results/reproduction/tuning/v49_locked_parameters_before_test.json").read_text()
    )
    assert len(table) == 27
    assert set(
        zip(table.beta.round(12), table.gamma.round(12), table.delta.round(12))
    ) == {
        (beta, gamma, delta)
        for beta in (0.0, 0.15, 0.30)
        for gamma in (0.10, 0.20, 0.35)
        for delta in (0.0, 0.10, 0.20)
    }
    assert table.primary_policy_mode.eq("cost_only").all()
    if "risk_budget_per_item" in table:
        assert table.risk_budget_per_item.isna().all()
    ranked = table.assign(_full=table.full_batch.astype(bool)).sort_values(
        ["_full", "hits", "proxy_cost_normalized_yield", "total_proxy_cost", "gamma", "beta", "delta"],
        ascending=[False, False, False, True, True, True, True],
        kind="mergesort",
    )
    selected = table[table.selected.astype(bool)]
    assert len(selected) == 1 and selected.index[0] == ranked.index[0]
    for field in ("alpha", "beta", "gamma", "delta", "diversity_distance_scale", "cost_budget_per_item"):
        assert np.isclose(
            float(selected.iloc[0][field]),
            float(lock["policy"][field]),
            rtol=0,
            atol=1e-12,
        )


def test_public_release_version_and_metadata_are_consistent():
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    parts = version.split(".")
    assert len(parts) == 3 and all(part.isdigit() for part in parts)

    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    version_lines = [line for line in cff.splitlines() if line.startswith("version:")]
    assert len(version_lines) == 1
    cff_version = version_lines[0].split(":", 1)[1].strip().strip('"').strip("'")
    assert cff_version == version

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "PUBLIC RELEASE REPOSITORY" in readme


def test_source_data_inventory_covers_regression_contracts():
    inventory = pd.read_csv(ROOT / "data/source_data_manifest.csv")
    paths = set(inventory.file_path.astype(str))
    assert {
        "data/processed/full_public_stability_source_manifest.csv",
        "data/provenance/mofsimplify_training_source_provenance.json",
        "data/derived/full_public_stability_features_relocked.csv",
        "results/reproduction/tuning/v49_locked_parameters_before_test.json",
        "results/reproduction/tuning/activation_C_train_internal_SOURCE_GROUP_CV.csv",
        "results/reproduction/tuning/activation_train_group_cv_fold_audit.csv",
        "results/reproduction/diagnostics/source_overlap_C_ablation.csv",
        "results/reproduction/activation_dummy_baselines.json",
        "results/reproduction/activation_test_source_group_bootstrap_summary.json",
        "results/reproduction/risk_feasibility_audit.csv",
        "source_data/source_overlap_C_ablation_source.csv",
        "source_data/activation_dummy_baselines_source.csv",
        "source_data/activation_test_source_group_bootstrap_source.csv",
        "source_data/risk_feasibility_audit_source.csv",
        MANIFEST_REL,
        "results/reproduction/regression_reference.json",
        CONTRACT_REL,
        "requirements-canonical.txt",
        "Dockerfile.canonical",
    } <= paths


def test_regression_manifest_has_explicit_comparison_classes_and_self_coverage():
    manifest = json.loads((ROOT / MANIFEST_REL).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "3.0"
    classes = {row["regression_class"] for row in manifest["files"].values()}
    assert classes == {"canonical_byte_identity", "visual_equivalence"}
    assert "self-referential" in manifest["comparison_policy"]["self_coverage"]


def test_portability_contract_allows_bounded_numeric_and_diagnostic_drift():
    contract = json.loads((ROOT / CONTRACT_REL).read_text(encoding="utf-8"))
    reference = build_semantic_snapshot(ROOT)
    current = json.loads(json.dumps(reference))
    current["numeric"]["activation"]["test_roc_auc"] += 2.2e-5
    current["numeric"]["random_summary"][0]["mean_hits"] += 0.01
    current["diagnostics"]["final_fit_n_iter"] += 6
    current["diagnostics"]["maximum_cv_n_iter"] += 382
    assert compare_semantic_snapshots(
        reference, current, contract=contract, mode="portability"
    ) == []


def test_portability_contract_rejects_exact_identity_and_large_numeric_drift():
    contract = json.loads((ROOT / CONTRACT_REL).read_text(encoding="utf-8"))
    reference = build_semantic_snapshot(ROOT)
    identity_drift = json.loads(json.dumps(reference))
    identity_drift["exact"]["traces"]["portfolio_batch"]["selected_candidate_ids"][0] = "changed"
    assert compare_semantic_snapshots(
        reference, identity_drift, contract=contract, mode="portability"
    )
    numeric_drift = json.loads(json.dumps(reference))
    numeric_drift["numeric"]["activation"]["test_roc_auc"] += 0.01
    assert compare_semantic_snapshots(
        reference, numeric_drift, contract=contract, mode="portability"
    )


def test_figure5_uses_portability_contract_instead_of_exact_machine_precision_claim():
    source = pd.read_csv(ROOT / "source_data/figure5_trace_replay_source.csv")
    row = source.loc[source.check.eq("activation_score_artifact_portability_tolerance")]
    assert len(row) == 1 and float(row.value.iloc[0]) == 1e-12
    svg = (ROOT / "figures/Figure5_trace_replay_audit.svg").read_text(encoding="utf-8")
    assert "< 1e-12 contract" in svg or "&lt; 1e-12 contract" in svg
    assert "1.1e-16" not in svg


def test_descriptor_contrast_source_contains_twelve_rows():
    contrasts = pd.read_csv(ROOT / "source_data/figure4_descriptor_contrasts.csv")
    assert len(contrasts) == 12


def test_canonical_ci_is_pinned_and_uploads_diagnostics():
    workflow = (ROOT / ".github/workflows/reproduce.yml").read_text(encoding="utf-8")
    assert "python:3.13.5-slim-bookworm@sha256:" in workflow
    assert "requirements-canonical.txt" in workflow
    assert "shell: bash" in workflow
    assert "OPENBLAS_CORETYPE: Haswell" in workflow
    assert "check_canonical_cpu_dispatch.py" in workflow
    assert "determine_numpy_avx512_features.py" in workflow
    assert '--github-env "$GITHUB_ENV"' in workflow
    assert "check_clean_reproduction.py --mode canonical" in workflow
    assert "validate_v49_scientific_gate.py" in workflow
    assert "run_audit_replay.py" in workflow
    assert "check_regression_integrity.py --mode canonical" in workflow
    assert "validate_repository.py" in workflow
    assert "actions/upload-artifact@" in workflow
    assert "if: always()" in workflow
    assert "ConvergenceWarning" in workflow
    # Explicit Git worktree / tracked-cleanliness contract (R3+); do not require
    # the obsolete contiguous literal "git diff --exit-code".
    assert 'WORKSPACE_REAL="$(realpath "$GITHUB_WORKSPACE")"' in workflow
    assert 'safe.directory "$WORKSPACE_REAL"' in workflow
    assert 'git -C "$WORKSPACE_REAL" rev-parse --is-inside-work-tree' in workflow
    assert 'git -C "$WORKSPACE_REAL" rev-parse --show-toplevel' in workflow
    assert 'git -C "$WORKSPACE_REAL" rev-parse HEAD' in workflow
    assert 'git -C "$WORKSPACE_REAL" diff --exit-code -- .' in workflow
    assert 'git -C "$WORKSPACE_REAL" diff --cached --exit-code -- .' in workflow
    assert 'git -C "$WORKSPACE_REAL" status' in workflow
    assert "--porcelain=v1" in workflow
    assert "--untracked-files=no" in workflow
    assert 'HEAD_SHA" != "$GITHUB_SHA"' in workflow or 'OBSERVED_HEAD" != "$GITHUB_SHA"' in workflow
    assert "TRACKED_GIT_CLEANLINESS_PASS" in workflow

    bootstrap_path = ROOT / ".github/workflows/bootstrap-v49-canonical-reference.yml"
    if bootstrap_path.exists():
        bootstrap = bootstrap_path.read_text(encoding="utf-8")
        assert "v49-source-group-leakage-ablation-risk-contract-redefinition" in bootstrap
        assert "workflow_dispatch" not in bootstrap
        assert "contents: read" in bootstrap
        assert "persist-credentials: false" in bootstrap
        assert "--write-reference" in bootstrap
        assert "paper13-v49-canonical-payload-" in bootstrap
        assert "paper13-v49-bootstrap-diagnostics-" in bootstrap
        assert "git commit" not in bootstrap
        assert "git push" not in bootstrap


def test_cross_platform_contract_excludes_iteration_counts_from_scientific_equivalence():
    contract = json.loads((ROOT / CONTRACT_REL).read_text(encoding="utf-8"))
    diagnostic_fields = contract["comparison_classes"]["diagnostics_only"]["fields"]
    assert "diagnostics.final_fit_n_iter" in diagnostic_fields
    assert "diagnostics.maximum_cv_n_iter" in diagnostic_fields
    assert contract["canonical_platform"]["dependency_lock"] == "requirements-canonical.txt"


def test_repository_protocol_docs_use_v49_cost_only_primary_names():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    protocol_docs = {
        path: (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "docs/reproduction_guide.md",
            "docs/data_dictionary.md",
            "docs/trace_schema.md",
        )
    }
    combined = "\n".join([readme, *protocol_docs.values()])

    for text in protocol_docs.values():
        assert "v49_locked_parameters_before_test.json" in text
    assert (
        "test_policy_comparison_primary_cost_only.csv"
        in protocol_docs["docs/reproduction_guide.md"]
    )
    assert (
        "test_policy_comparison_primary_cost_only.csv"
        in protocol_docs["docs/data_dictionary.md"]
    )
    assert (
        "test_policy_risk_stress_legacy_0p35.csv"
        in protocol_docs["docs/reproduction_guide.md"]
    )
    assert (
        "test_policy_risk_stress_legacy_0p35.csv"
        in protocol_docs["docs/data_dictionary.md"]
    )
    assert (
        "*_test_primary_cost_contract_trace.json"
        in protocol_docs["docs/trace_schema.md"]
    )
    assert "cost-only" in readme
    assert "immediate cumulative feasibility" in readme

    for stale_claim in (
        "v45A3_locked_parameters_before_test.json",
        "common test cost/risk contract",
        "test_policy_comparison_common_risk.csv",
        "random_baseline_full_test_distribution.csv",
        "test_pool_resampling_raw.csv",
    ):
        assert stale_claim not in combined
