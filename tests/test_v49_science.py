from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceable_portfoliobatch.v49_science import (  # noqa: E402
    DEFAULT_BATCH_SIZES,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    LEGACY_RISK_BUDGET_PER_ITEM,
    RISK_SCORE_INTERPRETATION,
    cost_feasibility_table,
    evaluate_training_fitted_dummy_classifiers,
    expand_source_group_sample,
    risk_feasibility_table,
    source_group_cluster_bootstrap,
)


def test_dummy_classifiers_fit_training_labels_only():
    y_train = np.array([0, 0, 0, 1])
    y_test = np.array([1, 1, 1, 0])
    table = evaluate_training_fitted_dummy_classifiers(y_train, y_test).set_index("strategy")

    assert set(table.index) == {"prior", "most_frequent"}
    assert table.fit_split.eq("train").all()
    assert table.evaluation_split.eq("test").all()
    assert table.training_positive_fraction.eq(0.25).all()
    assert table.training_majority_class.eq(0).all()
    assert table.accuracy.eq(0.25).all()
    assert table.loc["prior", "predicted_positive_probability"] == pytest.approx(0.25)
    assert table.loc["prior", "brier_score"] == pytest.approx(
        np.mean((y_test - 0.25) ** 2)
    )
    assert table.loc["most_frequent", "predicted_positive_probability"] == 0.0
    assert table.roc_auc.eq(0.5).all()

    # Test prevalence changes evaluation metrics, never the training-fitted class/prior.
    all_positive_test = evaluate_training_fitted_dummy_classifiers(
        y_train, np.ones(7, dtype=int)
    ).set_index("strategy")
    assert all_positive_test.training_majority_class.eq(0).all()
    assert all_positive_test.loc["prior", "predicted_positive_probability"] == 0.25


def test_whole_group_expansion_duplicates_complete_blocks_in_order():
    groups = np.array(["a", "a", "b", "b", "b", "c"])
    indices = expand_source_group_sample(groups, ["b", "a", "b"])
    assert indices.tolist() == [2, 3, 4, 0, 1, 2, 3, 4]

    row_ids = np.array(["a0", "a1", "b0", "b1", "b2", "c0"])
    assert row_ids[indices].tolist() == [
        "b0", "b1", "b2", "a0", "a1", "b0", "b1", "b2"
    ]


def test_cluster_bootstrap_marks_single_class_auc_invalid_and_summarizes_valid_auc():
    y = np.array([0, 1])
    probabilities = np.array([0.2, 0.8])
    groups = np.array(["negative-source", "positive-source"])
    raw, summary = source_group_cluster_bootstrap(
        y,
        probabilities,
        groups,
        n_resamples=40,
        seed=45,
    )

    assert summary["requested_resamples"] == 40
    assert summary["valid_auc_resamples"] + summary["invalid_auc_resamples"] == 40
    assert 0 < summary["invalid_auc_resamples"] < 40
    assert raw.loc[raw.invalid_for_auc, "roc_auc"].isna().all()
    assert raw.loc[raw.invalid_for_auc, "roc_auc_minus_0p5"].isna().all()
    valid_auc = raw.loc[~raw.invalid_for_auc, "roc_auc"].to_numpy()
    auc_summary = summary["metrics"]["roc_auc"]
    expected_low, expected_high = np.percentile(valid_auc, [2.5, 97.5])
    assert auc_summary["bootstrap_standard_error"] == pytest.approx(
        np.std(valid_auc, ddof=1)
    )
    assert auc_summary["percentile_2_5"] == pytest.approx(expected_low)
    assert auc_summary["percentile_97_5"] == pytest.approx(expected_high)


def test_cluster_bootstrap_all_single_class_has_no_auc_summary():
    raw, summary = source_group_cluster_bootstrap(
        [0, 0, 0],
        [0.1, 0.2, 0.3],
        ["a", "b", "c"],
        n_resamples=8,
    )
    assert raw.invalid_for_auc.all()
    assert raw.roc_auc.isna().all()
    assert summary["valid_auc_resamples"] == 0
    assert summary["invalid_auc_resamples"] == 8
    assert summary["point_estimates"]["roc_auc"] is None
    assert summary["metrics"]["roc_auc"] == {
        "point_estimate": None,
        "bootstrap_standard_error": None,
        "percentile_2_5": None,
        "percentile_97_5": None,
        "valid_resamples": 0,
    }


def test_v49_statistical_defaults_are_locked():
    signature = inspect.signature(source_group_cluster_bootstrap)
    assert DEFAULT_BOOTSTRAP_RESAMPLES == 10_000
    assert DEFAULT_BOOTSTRAP_SEED == 45
    assert signature.parameters["n_resamples"].default == 10_000
    assert signature.parameters["seed"].default == 45
    assert DEFAULT_BATCH_SIZES == (5, 10, 20)
    assert LEGACY_RISK_BUDGET_PER_ITEM == 0.35


def test_risk_feasibility_uses_exact_prefix_sums_and_stress_only_label():
    risks = np.full(20, 0.4)
    groups = [f"g{index // 2}" for index in range(20)]
    table = risk_feasibility_table(
        risks,
        pool="test",
        source_group_ids=groups,
    ).set_index("requested_batch_size")

    assert set(table.index) == {5, 10, 20}
    assert table.pool.eq("test").all()
    assert table.pool_size.eq(20).all()
    assert table.pool_source_group_count.eq(10).all()
    assert table.legacy_risk_contract_role.eq("stress_test_only").all()
    assert (~table.primary_selected_risk.astype(bool)).all()
    assert table.risk_score_interpretation.eq(RISK_SCORE_INTERPRETATION).all()
    assert table.legacy_total_risk_cap.to_dict() == pytest.approx(
        {5: 1.75, 10: 3.5, 20: 7.0}
    )
    assert table.minimum_possible_total_risk_for_exact_batch.to_dict() == pytest.approx(
        {5: 2.0, 10: 4.0, 20: 8.0}
    )
    assert not table.full_batch_risk_feasible.any()
    assert table.max_feasible_cardinality_under_legacy_cap.to_dict() == {
        5: 4,
        10: 8,
        20: 17,
    }
    assert table.minimum_risk_budget_per_item_for_full_batch.to_numpy() == pytest.approx(
        np.full(3, 0.4)
    )
    assert json.loads(table.loc[5, "five_lowest_risk_values"]) == [0.4] * 5


def test_cost_feasibility_is_mathematical_and_selector_independent():
    costs = np.arange(1.0, 21.0)
    table = cost_feasibility_table(
        costs,
        cost_budget_per_item=3.0,
        pool="calibration",
        source_group_ids=[f"doi-{index}" for index in range(20)],
    ).set_index("requested_batch_size")

    # Independent sums: 1+...+k = k(k+1)/2; cap = 3k.
    assert table.loc[5, "minimum_possible_total_cost_for_exact_batch"] == 15.0
    assert table.loc[10, "minimum_possible_total_cost_for_exact_batch"] == 55.0
    assert table.loc[20, "minimum_possible_total_cost_for_exact_batch"] == 210.0
    assert table.loc[5, "total_cost_cap"] == 15.0
    assert bool(table.loc[5, "full_batch_cost_feasible"])
    assert not bool(table.loc[10, "full_batch_cost_feasible"])
    assert not bool(table.loc[20, "full_batch_cost_feasible"])
    assert table.loc[10, "max_feasible_cardinality_under_cost_cap"] == 7
    assert table.loc[20, "max_feasible_cardinality_under_cost_cap"] == 10
    assert table.loc[20, "minimum_cost_budget_per_item_for_full_batch"] == 10.5
