from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score


DEFAULT_BOOTSTRAP_SEED = 45
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BATCH_SIZES = (5, 10, 20)
LEGACY_RISK_BUDGET_PER_ITEM = 0.35

RISK_SCORE_FORMULA = (
    "0.5 * (1 - predicted_success) + 0.5 * normalized_binary_entropy"
)
RISK_SCORE_INTERPRETATION = "benchmark-defined uncalibrated proxy"
BOOTSTRAP_GROUP_UNIT = "normalized_source_doi"


def _binary_targets(values: Sequence[object], *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain binary 0/1 values") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError(f"{name} must contain only finite binary 0/1 values")
    return numeric.astype(int)


def _probabilities(values: Sequence[object], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all() or ((array < 0.0) | (array > 1.0)).any():
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    return array


def _source_group_ids(values: Sequence[object], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if pd.isna(array).any():
        raise ValueError(f"{name} must not contain missing values")
    normalized = np.asarray([str(value).strip() for value in array], dtype=object)
    if any(value == "" for value in normalized):
        raise ValueError(f"{name} must not contain blank values")
    return normalized


def _nonnegative_values(values: Sequence[object], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError(f"{name} must contain finite nonnegative values")
    return array


def _positive_batch_sizes(batch_sizes: Sequence[int]) -> tuple[int, ...]:
    sizes = tuple(int(value) for value in batch_sizes)
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("batch_sizes must contain positive integers")
    if any(float(original) != converted for original, converted in zip(batch_sizes, sizes)):
        raise ValueError("batch_sizes must contain integers")
    if len(set(sizes)) != len(sizes):
        raise ValueError("batch_sizes must not contain duplicates")
    return sizes


def _finite_nonnegative_budget(value: float, *, name: str) -> float:
    budget = float(value)
    if not np.isfinite(budget) or budget < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return budget


def evaluate_training_fitted_dummy_classifiers(
    y_train: Sequence[object],
    y_test: Sequence[object],
) -> pd.DataFrame:
    """Fit the two v49 dummy comparators on training labels and evaluate on test.

    Dummy features intentionally contain no information.  In particular, the
    final-test labels are never supplied to ``fit`` and therefore cannot choose
    the predicted class or class-prior probability.
    """

    train_targets = _binary_targets(y_train, name="y_train")
    test_targets = _binary_targets(y_test, name="y_test")
    train_features = np.zeros((len(train_targets), 1), dtype=float)
    test_features = np.zeros((len(test_targets), 1), dtype=float)
    rows: list[dict[str, Any]] = []

    for strategy in ("prior", "most_frequent"):
        classifier = DummyClassifier(strategy=strategy)
        classifier.fit(train_features, train_targets)
        predictions = classifier.predict(test_features).astype(int)
        class_probabilities = classifier.predict_proba(test_features)
        positive_matches = np.flatnonzero(np.asarray(classifier.classes_) == 1)
        if positive_matches.size:
            positive_probability = class_probabilities[:, int(positive_matches[0])]
        else:
            positive_probability = np.zeros(len(test_targets), dtype=float)

        training_majority_class = int(
            np.asarray(classifier.classes_)[int(np.argmax(classifier.class_prior_))]
        )
        roc_auc = (
            float(roc_auc_score(test_targets, positive_probability))
            if np.unique(test_targets).size == 2
            else np.nan
        )
        rows.append({
            "strategy": strategy,
            "fit_split": "train",
            "evaluation_split": "test",
            "training_n": int(len(train_targets)),
            "training_positive_fraction": float(train_targets.mean()),
            "training_majority_class": training_majority_class,
            "test_n": int(len(test_targets)),
            "accuracy": float(accuracy_score(test_targets, predictions)),
            "roc_auc": roc_auc,
            "brier_score": float(brier_score_loss(test_targets, positive_probability)),
            "predicted_positive_probability": float(positive_probability[0]),
        })

    return pd.DataFrame(rows)


def expand_source_group_sample(
    source_group_ids: Sequence[object],
    sampled_group_ids: Sequence[object],
) -> np.ndarray:
    """Expand sampled group occurrences into row indices, preserving blocks.

    A repeated sampled group contributes its complete row block repeatedly and
    within-group row order is the order in ``source_group_ids``.
    """

    groups = _source_group_ids(source_group_ids, name="source_group_ids")
    sampled = _source_group_ids(sampled_group_ids, name="sampled_group_ids")
    blocks = {group: np.flatnonzero(groups == group) for group in pd.unique(groups)}
    missing = sorted(set(sampled) - set(blocks))
    if missing:
        raise ValueError(f"sampled_group_ids contains unknown groups: {missing}")
    return np.concatenate([blocks[group] for group in sampled]).astype(int, copy=False)


def draw_source_group_bootstrap_indices(
    source_group_ids: Sequence[object],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw exactly N source groups with replacement and return expanded rows."""

    groups = _source_group_ids(source_group_ids, name="source_group_ids")
    unique_groups = np.asarray(pd.unique(groups), dtype=object)
    sampled_groups = np.asarray(
        rng.choice(unique_groups, size=len(unique_groups), replace=True),
        dtype=object,
    )
    return sampled_groups, expand_source_group_sample(groups, sampled_groups)


def _metric_summary(
    *,
    point_estimate: float | None,
    bootstrap_values: np.ndarray,
) -> dict[str, float | int | None]:
    finite = np.asarray(bootstrap_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        standard_error = None
        low = None
        high = None
    else:
        standard_error = float(np.std(finite, ddof=1)) if finite.size > 1 else None
        low, high = (float(value) for value in np.percentile(finite, [2.5, 97.5]))
    return {
        "point_estimate": point_estimate,
        "bootstrap_standard_error": standard_error,
        "percentile_2_5": low,
        "percentile_97_5": high,
        "valid_resamples": int(finite.size),
    }


def source_group_cluster_bootstrap(
    y_true: Sequence[object],
    probabilities: Sequence[object],
    source_group_ids: Sequence[object],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bootstrap final-test metrics by whole normalized source-DOI groups."""

    targets = _binary_targets(y_true, name="y_true")
    scores = _probabilities(probabilities, name="probabilities")
    groups = _source_group_ids(source_group_ids, name="source_group_ids")
    if not (len(targets) == len(scores) == len(groups)):
        raise ValueError("y_true, probabilities, and source_group_ids must have equal length")
    if isinstance(n_resamples, bool) or int(n_resamples) != n_resamples or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    requested_resamples = int(n_resamples)
    bootstrap_seed = int(seed)

    unique_groups = np.asarray(pd.unique(groups), dtype=object)
    blocks = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, Any]] = []

    for resample in range(1, requested_resamples + 1):
        sampled_groups = np.asarray(
            rng.choice(unique_groups, size=len(unique_groups), replace=True),
            dtype=object,
        )
        indices = np.concatenate([blocks[group] for group in sampled_groups])
        sample_targets = targets[indices]
        sample_scores = scores[indices]
        invalid_for_auc = np.unique(sample_targets).size != 2
        roc_auc = (
            np.nan
            if invalid_for_auc
            else float(roc_auc_score(sample_targets, sample_scores))
        )
        rows.append({
            "bootstrap_resample": resample,
            "sampled_group_occurrences": int(len(sampled_groups)),
            "sampled_unique_group_count": int(len(set(sampled_groups))),
            "sampled_row_count": int(len(indices)),
            "roc_auc": roc_auc,
            "accuracy_at_0_5": float(
                accuracy_score(sample_targets, sample_scores >= 0.5)
            ),
            "brier_score": float(brier_score_loss(sample_targets, sample_scores)),
            "invalid_for_auc": bool(invalid_for_auc),
            "roc_auc_minus_0p5": np.nan if invalid_for_auc else roc_auc - 0.5,
        })

    raw = pd.DataFrame(rows)
    original_auc = (
        float(roc_auc_score(targets, scores)) if np.unique(targets).size == 2 else None
    )
    point_estimates = {
        "roc_auc": original_auc,
        "accuracy_at_0_5": float(accuracy_score(targets, scores >= 0.5)),
        "brier_score": float(brier_score_loss(targets, scores)),
    }
    metrics = {
        "roc_auc": _metric_summary(
            point_estimate=point_estimates["roc_auc"],
            bootstrap_values=raw["roc_auc"].to_numpy(dtype=float),
        ),
        "accuracy_at_0_5": _metric_summary(
            point_estimate=point_estimates["accuracy_at_0_5"],
            bootstrap_values=raw["accuracy_at_0_5"].to_numpy(dtype=float),
        ),
        "brier_score": _metric_summary(
            point_estimate=point_estimates["brier_score"],
            bootstrap_values=raw["brier_score"].to_numpy(dtype=float),
        ),
    }
    invalid_auc = int(raw["invalid_for_auc"].sum())
    summary: dict[str, Any] = {
        "seed": bootstrap_seed,
        "requested_resamples": requested_resamples,
        "valid_auc_resamples": requested_resamples - invalid_auc,
        "invalid_auc_resamples": invalid_auc,
        "bootstrap_group_unit": BOOTSTRAP_GROUP_UNIT,
        "source_group_count": int(len(unique_groups)),
        "confidence_interval": {
            "method": "percentile",
            "confidence_level": 0.95,
            "lower_percentile": 2.5,
            "upper_percentile": 97.5,
        },
        "used_for_tuning": False,
        "point_estimates": point_estimates,
        "metrics": metrics,
    }
    return raw, summary


def _feasibility_inputs(
    values: Sequence[object],
    *,
    value_name: str,
    source_group_ids: Sequence[object] | None,
    batch_sizes: Sequence[int],
) -> tuple[np.ndarray, int | None, tuple[int, ...]]:
    numeric = np.sort(_nonnegative_values(values, name=value_name), kind="stable")
    group_count: int | None = None
    if source_group_ids is not None:
        groups = _source_group_ids(source_group_ids, name="source_group_ids")
        if len(groups) != len(numeric):
            raise ValueError(f"{value_name} and source_group_ids must have equal length")
        group_count = int(len(pd.unique(groups)))
    return numeric, group_count, _positive_batch_sizes(batch_sizes)


def _exact_budget_quantities(
    sorted_values: np.ndarray,
    *,
    budget_per_item: float,
    requested_batch_size: int,
) -> dict[str, float | int | bool | None]:
    cap = float(budget_per_item * requested_batch_size)
    prefix_sums = np.cumsum(sorted_values, dtype=float)
    exact_batch_defined = len(sorted_values) >= requested_batch_size
    minimum_exact = (
        float(prefix_sums[requested_batch_size - 1]) if exact_batch_defined else None
    )
    feasible_limit = min(requested_batch_size, len(sorted_values))
    max_cardinality = int(np.count_nonzero(prefix_sums[:feasible_limit] <= cap))
    return {
        "cap": cap,
        "minimum_exact": minimum_exact,
        "full_batch_feasible": bool(
            exact_batch_defined and minimum_exact is not None and minimum_exact <= cap
        ),
        "max_feasible_cardinality": max_cardinality,
        "minimum_budget_per_item": (
            float(minimum_exact / requested_batch_size)
            if minimum_exact is not None
            else None
        ),
    }


def risk_feasibility_table(
    risk_scores: Sequence[object],
    *,
    pool: str,
    source_group_ids: Sequence[object] | None = None,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    risk_budget_per_item: float = LEGACY_RISK_BUDGET_PER_ITEM,
) -> pd.DataFrame:
    """Compute selector-independent exact legacy-risk feasibility diagnostics."""

    risks, group_count, sizes = _feasibility_inputs(
        risk_scores,
        value_name="risk_scores",
        source_group_ids=source_group_ids,
        batch_sizes=batch_sizes,
    )
    budget = _finite_nonnegative_budget(
        risk_budget_per_item,
        name="risk_budget_per_item",
    )
    rows: list[dict[str, Any]] = []
    for requested_size in sizes:
        exact = _exact_budget_quantities(
            risks,
            budget_per_item=budget,
            requested_batch_size=requested_size,
        )
        rows.append({
            "pool": str(pool),
            "requested_batch_size": requested_size,
            "pool_size": int(len(risks)),
            "pool_source_group_count": group_count,
            "legacy_risk_budget_per_item": budget,
            "legacy_total_risk_cap": exact["cap"],
            "minimum_possible_total_risk_for_exact_batch": exact["minimum_exact"],
            "full_batch_risk_feasible": exact["full_batch_feasible"],
            "max_feasible_cardinality_under_legacy_cap": exact[
                "max_feasible_cardinality"
            ],
            "minimum_risk_budget_per_item_for_full_batch": exact[
                "minimum_budget_per_item"
            ],
            "five_lowest_risk_values": (
                json.dumps(risks[:5].tolist(), separators=(",", ":"))
                if requested_size == 5
                else ""
            ),
            "risk_score_formula": RISK_SCORE_FORMULA,
            "risk_score_interpretation": RISK_SCORE_INTERPRETATION,
            "legacy_risk_contract_role": "stress_test_only",
            "primary_selected_risk": False,
        })
    return pd.DataFrame(rows)


def cost_feasibility_table(
    costs: Sequence[object],
    *,
    cost_budget_per_item: float,
    pool: str,
    source_group_ids: Sequence[object] | None = None,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
) -> pd.DataFrame:
    """Compute selector-independent exact primary-cost feasibility diagnostics."""

    sorted_costs, group_count, sizes = _feasibility_inputs(
        costs,
        value_name="costs",
        source_group_ids=source_group_ids,
        batch_sizes=batch_sizes,
    )
    budget = _finite_nonnegative_budget(
        cost_budget_per_item,
        name="cost_budget_per_item",
    )
    rows: list[dict[str, Any]] = []
    for requested_size in sizes:
        exact = _exact_budget_quantities(
            sorted_costs,
            budget_per_item=budget,
            requested_batch_size=requested_size,
        )
        rows.append({
            "pool": str(pool),
            "requested_batch_size": requested_size,
            "pool_size": int(len(sorted_costs)),
            "pool_source_group_count": group_count,
            "cost_budget_per_item": budget,
            "total_cost_cap": exact["cap"],
            "minimum_possible_total_cost_for_exact_batch": exact["minimum_exact"],
            "full_batch_cost_feasible": exact["full_batch_feasible"],
            "max_feasible_cardinality_under_cost_cap": exact[
                "max_feasible_cardinality"
            ],
            "minimum_cost_budget_per_item_for_full_batch": exact[
                "minimum_budget_per_item"
            ],
        })
    return pd.DataFrame(rows)
