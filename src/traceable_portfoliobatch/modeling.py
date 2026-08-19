from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .io_utils import write_json

ACTIVATION_META = {
    "candidate_id", "model_split", "raw_split", "raw_source_file",
    "raw_source_row", "raw_refcode", "outcome_binary",
}
THERMAL_META = {
    "candidate_id", "model_split", "raw_split", "raw_source_file",
    "raw_source_row", "raw_refcode", "thermal_T",
}

ACTIVATION_SOLVER = "lbfgs"
ACTIVATION_MAX_ITER = 20_000
ACTIVATION_TOL = 1e-6
ACTIVATION_CV_FOLDS = 5
ACTIVATION_CV_RANDOM_STATE = 45


def feature_columns(frame: pd.DataFrame, meta: Iterable[str]) -> list[str]:
    return [c for c in frame.columns if c not in set(meta)]


def activation_pipeline(C: float) -> Pipeline:
    """Train-split activation model with an explicitly supplied regularization C."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=float(C),
            class_weight="balanced",
            solver=ACTIVATION_SOLVER,
            max_iter=ACTIVATION_MAX_ITER,
            tol=ACTIVATION_TOL,
            random_state=0,
        )),
    ])


def fit_activation_checked(
    pipe: Pipeline,
    x: pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    context: str,
) -> dict[str, int | bool | str]:
    """Fit and fail on a convergence warning or iteration-limit contact."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        try:
            pipe.fit(x, y)
        except ConvergenceWarning as exc:  # pragma: no cover - warning promoted to error
            raise RuntimeError(f"Activation fit did not converge in {context}: {exc}") from exc
    model = pipe.named_steps["model"]
    n_iter = int(np.max(np.asarray(model.n_iter_, dtype=int)))
    max_iter = int(model.max_iter)
    if n_iter >= max_iter:
        raise RuntimeError(
            f"Activation fit reached max_iter in {context}: n_iter={n_iter}, max_iter={max_iter}"
        )
    return {
        "context": context,
        "converged": True,
        "n_iter": n_iter,
        "max_iter": max_iter,
        "solver": str(model.solver),
    }


def select_activation_C_train_cv(
    train: pd.DataFrame,
    columns: list[str],
    grid: Sequence[float],
    *,
    n_splits: int = ACTIVATION_CV_FOLDS,
    random_state: int = ACTIVATION_CV_RANDOM_STATE,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Select C by source-DOI-group CV inside training.

    The splitter, seed and one-standard-error rule are protocol constants.  The
    function deliberately fails rather than falling back to row-level CV when
    the publication-group/class composition cannot support the locked design.
    """
    y = (train["outcome_binary"].astype(float) > 0).astype(int).to_numpy()
    group_columns = {"source_group_id", "source_doi_normalized", "source_group_basis"}
    missing_group_columns = group_columns - set(train.columns)
    if missing_group_columns:
        raise ValueError(
            "Training normalized DOI provenance is required for activation model selection; "
            f"missing={sorted(missing_group_columns)}"
        )
    groups = train["source_group_id"].fillna("").astype(str).str.strip().to_numpy()
    normalized_doi = (
        train["source_doi_normalized"].fillna("").astype(str).str.strip().to_numpy()
    )
    basis = train["source_group_basis"].fillna("").astype(str).str.strip().to_numpy()
    unresolved = (
        (groups == "")
        | (normalized_doi == "")
        | np.char.startswith(groups.astype(str), "unresolved_")
        | np.char.startswith(groups.astype(str), "train_refcode::")
        | np.char.startswith(normalized_doi.astype(str), "unresolved_")
    )
    if unresolved.any():
        raise ValueError(
            f"Training source DOI group is unresolved for {int(unresolved.sum())} rows"
        )
    if not np.all(basis == "normalized_source_doi"):
        raise ValueError("Training source_group_basis must be normalized_source_doi for every row")
    if not np.array_equal(groups, normalized_doi):
        raise ValueError("Training source_group_id must equal source_doi_normalized for every row")
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    splits = list(cv.split(train[columns], y, groups))
    fold_rows: list[dict] = []
    for fold, (tr, va) in enumerate(splits, start=1):
        train_groups = set(groups[tr])
        validation_groups = set(groups[va])
        overlap = train_groups & validation_groups
        if overlap:
            raise RuntimeError(
                f"Source DOI group overlap in activation CV fold {fold}: "
                f"count={len(overlap)}"
            )
        if np.unique(y[tr]).size != 2:
            raise RuntimeError(
                f"Activation CV fold {fold} training portion lacks both outcome classes"
            )
        if np.unique(y[va]).size != 2:
            raise RuntimeError(
                f"Activation CV fold {fold} validation portion lacks both outcome classes"
            )
        fold_rows.append({
            "fold": fold,
            "train_row_count": int(len(tr)),
            "validation_row_count": int(len(va)),
            "train_group_count": int(len(train_groups)),
            "validation_group_count": int(len(validation_groups)),
            "group_overlap_count": int(len(overlap)),
            "train_positive_fraction": float(y[tr].mean()),
            "validation_positive_fraction": float(y[va].mean()),
        })
    fold_audit = pd.DataFrame(fold_rows)
    rows: list[dict] = []
    for C in [float(v) for v in grid]:
        aucs: list[float] = []
        briers: list[float] = []
        n_iters: list[int] = []
        for fold, (tr, va) in enumerate(splits, start=1):
            pipe = activation_pipeline(C)
            fit = fit_activation_checked(
                pipe,
                train.iloc[tr][columns],
                y[tr],
                context=f"activation C={C:g}, source-group fold={fold}",
            )
            probabilities = pipe.predict_proba(train.iloc[va][columns])[:, 1]
            aucs.append(float(roc_auc_score(y[va], probabilities)))
            briers.append(float(brier_score_loss(y[va], probabilities)))
            n_iters.append(int(fit["n_iter"]))
        rows.append({
            "C": C,
            "mean_train_cv_roc_auc": float(np.mean(aucs)),
            "sd_train_cv_roc_auc": float(np.std(aucs, ddof=1)),
            "mean_train_cv_brier": float(np.mean(briers)),
            "sd_train_cv_brier": float(np.std(briers, ddof=1)),
            "fold_n_iter_values": json.dumps(n_iters, separators=(",", ":")),
            "mean_n_iter": float(np.mean(n_iters)),
            "max_n_iter": int(np.max(n_iters)),
            "all_folds_converged": True,
            "training_doi_group_count": int(len(set(groups))),
            "cv_splitter": "StratifiedGroupKFold",
            "random_state": int(random_state),
            "n_splits": int(n_splits),
            "solver": ACTIVATION_SOLVER,
            "max_iter": ACTIVATION_MAX_ITER,
            "tol": ACTIVATION_TOL,
        })
    table = pd.DataFrame(rows).sort_values(
        ["mean_train_cv_roc_auc", "mean_train_cv_brier", "C"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    best_mean = float(table.loc[0, "mean_train_cv_roc_auc"])
    best_sd = float(table.loc[0, "sd_train_cv_roc_auc"])
    threshold = best_mean - best_sd / np.sqrt(n_splits)
    eligible = table[table["mean_train_cv_roc_auc"] >= threshold].sort_values("C")
    selected = float(eligible.iloc[0]["C"])
    table["selected_one_standard_error"] = table["C"].eq(selected)
    table["within_one_standard_error"] = table["mean_train_cv_roc_auc"].ge(threshold)
    return selected, table, fold_audit


def thermal_pipeline(alpha: float) -> Pipeline:
    """Secondary thermal ridge model with an explicitly supplied alpha."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=float(alpha))),
    ])


def select_thermal_alpha_cv(
    train: pd.DataFrame,
    columns: list[str],
    grid: Sequence[float],
) -> tuple[float, pd.DataFrame]:
    """Select ridge alpha by five-fold CV within the thermal training split."""
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(
        base,
        {"model__alpha": [float(v) for v in grid]},
        scoring="neg_root_mean_squared_error",
        cv=cv,
        refit=False,
        n_jobs=1,
    )
    search.fit(train[columns], train["thermal_T"])
    table = pd.DataFrame({
        "alpha": [float(v) for v in search.cv_results_["param_model__alpha"]],
        "mean_cv_rmse_C": -np.asarray(search.cv_results_["mean_test_score"], dtype=float),
        "std_cv_rmse_C": np.asarray(search.cv_results_["std_test_score"], dtype=float),
    }).sort_values(["mean_cv_rmse_C", "alpha"], ascending=[True, True]).reset_index(drop=True)
    table["selected"] = False
    table.loc[0, "selected"] = True
    return float(table.loc[0, "alpha"]), table


def normalized_binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def descriptor_proxy_cost(preprocessed_features: np.ndarray) -> np.ndarray:
    """Positive descriptor-complexity proxy, not experimental cost."""
    z = np.asarray(preprocessed_features, dtype=float)
    return 1.0 + np.mean(np.abs(z), axis=1)


def save_linear_pipeline_artifact(
    pipe: Pipeline,
    feature_names: list[str],
    path: Path,
    extra: dict | None = None,
) -> None:
    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    model = pipe.named_steps["model"]
    payload = {
        "feature_names": feature_names,
        "imputer_strategy": imputer.strategy,
        "imputer_statistics": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "model_class": model.__class__.__name__,
        "coefficients": np.asarray(model.coef_).tolist(),
        "intercept": np.asarray(model.intercept_).tolist(),
        "parameters": model.get_params(),
    }
    if extra:
        payload.update(extra)
    write_json(path, payload)


def predict_from_artifact(frame: pd.DataFrame, artifact_path: Path) -> np.ndarray:
    import json

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    x = frame[payload["feature_names"]].to_numpy(dtype=float)
    med = np.asarray(payload["imputer_statistics"], dtype=float)
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.take(med, np.where(bad)[1])
    mean = np.asarray(payload["scaler_mean"], dtype=float)
    scale = np.asarray(payload["scaler_scale"], dtype=float)
    z = (x - mean) / scale
    coef = np.asarray(payload["coefficients"], dtype=float)
    intercept = np.asarray(payload["intercept"], dtype=float)
    linear = z @ coef.T + intercept
    if payload["model_class"] == "LogisticRegression":
        return 1.0 / (1.0 + np.exp(-linear.ravel()))
    return linear.ravel()


def preprocess_from_artifact(frame: pd.DataFrame, artifact_path: Path) -> np.ndarray:
    import json

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    x = frame[payload["feature_names"]].to_numpy(dtype=float)
    med = np.asarray(payload["imputer_statistics"], dtype=float)
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.take(med, np.where(bad)[1])
    mean = np.asarray(payload["scaler_mean"], dtype=float)
    scale = np.asarray(payload["scaler_scale"], dtype=float)
    return (x - mean) / scale
