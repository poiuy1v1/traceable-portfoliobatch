from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceable_portfoliobatch.modeling import (  # noqa: E402
    descriptor_proxy_cost,
    normalized_binary_entropy,
    predict_from_artifact,
    preprocess_from_artifact,
)
from traceable_portfoliobatch.policies import PolicyConfig, select_batch  # noqa: E402
from traceable_portfoliobatch.workflow import _tune_cost_only_policy  # noqa: E402

EXPECTED_SHA256 = {
    "data/derived/full_public_stability_features_relocked.csv":
        "d334270b4b26b30addb3a7d09fe37ba05b8711b8e6671e64ec5be51bfcaaf44b",
    "models/activation_logistic_train_split.json":
        "219c3fa01be1530ea20057ab9a91839567974eb68ee6680aca7447860994ea30",
    "results/reproduction/tuning/portfolio_policy_cost_only_group_bootstrap_stability.csv":
        "87de5a033e42d622e5b804f5cb491c383a461ba9f3ce11861912e926758e9597",
}
LOCKED_TUPLE = {"alpha": 1.0, "beta": 0.3, "gamma": 0.2, "delta": 0.0}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_design(
    scored: pd.DataFrame,
    z: np.ndarray,
    cost_scale: float,
    cost_per_item: float,
    *,
    replace: bool,
) -> pd.DataFrame:
    groups = scored["source_group_id"].astype(str).unique()
    group_values = scored["source_group_id"].astype(str).to_numpy()
    n_sample = max(2, int(round(0.8 * len(groups))))
    rows: list[dict[str, object]] = []

    for seed in range(20):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(groups, size=n_sample, replace=replace)
        indices = np.concatenate(
            [np.flatnonzero(group_values == group) for group in chosen]
        )
        subset = scored.iloc[indices].reset_index(drop=True)
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
        config = PolicyConfig(
            batch_size=5,
            cost_budget=cost_per_item * 5,
            risk_budget=None,
            alpha=1.0,
            beta=float(tuned["beta"]),
            gamma=float(tuned["gamma"]),
            delta=float(tuned["delta"]),
            cost_scale=cost_scale,
            diversity_distance_scale=2.0,
        )
        selected, _, _ = select_batch(
            subset,
            z_subset,
            "portfolio_batch",
            config,
            record_details=False,
        )
        rows.append(
            {
                "seed": seed,
                **tuned,
                "calibration_group_count": len(set(chosen)),
                "sampled_group_occurrences": len(chosen),
                "subset_row_count": len(subset),
                "unique_candidate_count": int(subset["candidate_id"].nunique()),
                "selected_count": len(selected),
                "selected_unique_count": int(selected["candidate_id"].nunique()),
                "selected_ids": ",".join(selected["candidate_id"].astype(str)),
                "equivalent_top_config_count": int(equivalent.sum()),
                "full_batch": bool(table.loc[0, "full_batch"]),
            }
        )

    out = pd.DataFrame(rows)
    out["selected_tuple_frequency"] = (
        out.groupby(["beta", "gamma", "delta"])["seed"].transform("count") / 20.0
    )
    out["full_batch_frequency"] = float(out["full_batch"].mean())
    return out


def _prepare_scored(root: Path) -> tuple[pd.DataFrame, np.ndarray, float, float]:
    for rel, expected in EXPECTED_SHA256.items():
        observed = sha256(root / rel)
        if observed != expected:
            raise SystemExit(
                f"FROZEN_INPUT_IDENTITY_FAIL {rel} {observed} != {expected}"
            )

    features = pd.read_csv(
        root / "data/derived/full_public_stability_features_relocked.csv"
    )
    calibration = features.loc[
        features["final_evaluation_split"].eq("calibration")
    ].reset_index(drop=True)
    if (
        len(calibration) != 158
        or calibration["source_group_id"].nunique() != 136
        or calibration["candidate_id"].duplicated().any()
    ):
        raise SystemExit("FROZEN_CALIBRATION_IDENTITY_FAIL")

    artifact = root / "models/activation_logistic_train_split.json"
    z = preprocess_from_artifact(calibration, artifact)
    p = predict_from_artifact(calibration, artifact)
    u = normalized_binary_entropy(p)
    c = descriptor_proxy_cost(z)
    r = 0.5 * (1.0 - p) + 0.5 * u

    fields = [
        field
        for field in [
            "candidate_id",
            "outcome_binary",
            "source_group_id",
            "source_doi",
            "source_doi_normalized",
            "raw_refcode",
            "final_evaluation_split",
        ]
        if field in calibration
    ]
    scored = calibration[fields].copy()
    scored["predicted_success"] = p
    scored["uncertainty"] = u
    scored["proxy_total_cost"] = c
    scored["risk_score"] = r
    cost_scale = float(np.median(c))
    return scored, z, cost_scale, 1.5 * cost_scale


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the v1.0.1 duplicate-candidate defect and the bounded "
            "v1.0.2 publication-source-group subsampling correction."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_root = (args.output_root or root).resolve()

    scored, z, cost_scale, cost_per_item = _prepare_scored(root)

    old = _run_design(
        scored, z, cost_scale, cost_per_item, replace=True
    )
    archived = pd.read_csv(
        root
        / "results/reproduction/tuning/portfolio_policy_cost_only_group_bootstrap_stability.csv"
    )
    old_for_compare = old.rename(columns={"seed": "bootstrap_seed"})
    shared = [
        "bootstrap_seed",
        "beta",
        "gamma",
        "delta",
        "calibration_group_count",
        "equivalent_top_config_count",
        "full_batch",
        "selected_tuple_frequency",
        "full_batch_frequency",
    ]
    for column in shared:
        if column == "full_batch":
            ok = np.array_equal(
                old_for_compare[column].astype(bool).to_numpy(),
                archived[column].astype(bool).to_numpy(),
            )
        else:
            ok = np.allclose(
                old_for_compare[column].astype(float).to_numpy(),
                archived[column].astype(float).to_numpy(),
                rtol=0,
                atol=1e-12,
            )
        if not ok:
            raise SystemExit(f"ARCHIVED_REPRODUCTION_MISMATCH {column}")
    old["duplicate_selected_candidate_present"] = (
        old["selected_unique_count"] < old["selected_count"]
    )

    corrected = _run_design(
        scored, z, cost_scale, cost_per_item, replace=False
    )
    corrected["sampling_fraction"] = 0.8
    corrected["sampling_with_replacement"] = False
    corrected["duplicate_selected_candidate_present"] = (
        corrected["selected_unique_count"] < corrected["selected_count"]
    )

    locked = (
        corrected["beta"].eq(LOCKED_TUPLE["beta"])
        & corrected["gamma"].eq(LOCKED_TUPLE["gamma"])
        & corrected["delta"].eq(LOCKED_TUPLE["delta"])
    )
    summary = {
        "schema_version": "v1.0.2-policy-stability-correction-1.0",
        "base_release": "v1.0.1",
        "old_archived_reproduction": "PASS",
        "old_design": "80% publication-source-group sampling with replacement",
        "old_exact_locked_tuple_count": int(
            (
                old["beta"].eq(LOCKED_TUPLE["beta"])
                & old["gamma"].eq(LOCKED_TUPLE["gamma"])
                & old["delta"].eq(LOCKED_TUPLE["delta"])
            ).sum()
        ),
        "old_delta_zero_count": int(old["delta"].eq(0).sum()),
        "old_full_batch_count": int(old["full_batch"].sum()),
        "old_delta_zero_duplicate_selected_candidate_count": int(
            (
                old["delta"].eq(0)
                & old["duplicate_selected_candidate_present"]
            ).sum()
        ),
        "corrected_design": (
            "80% publication-source-group subsampling without replacement"
        ),
        "source_group_count": 136,
        "corrected_sampled_groups_per_subsample": 109,
        "seeds": "0..19",
        "corrected_exact_locked_tuple_count": int(locked.sum()),
        "corrected_delta_zero_count": int(corrected["delta"].eq(0).sum()),
        "corrected_full_batch_count": int(corrected["full_batch"].sum()),
        "corrected_unique_five_candidate_count": int(
            (
                corrected["selected_unique_count"].eq(5)
                & corrected["selected_count"].eq(5)
            ).sum()
        ),
        "scope": (
            "secondary calibration policy-stability analysis only; the original "
            "158-row calibration lock, model, final-test results, primary selectors "
            "and five archived traces are unchanged"
        ),
    }
    expected_summary = {
        "old_exact_locked_tuple_count": 1,
        "old_delta_zero_count": 14,
        "old_full_batch_count": 20,
        "old_delta_zero_duplicate_selected_candidate_count": 10,
        "corrected_exact_locked_tuple_count": 3,
        "corrected_delta_zero_count": 18,
        "corrected_full_batch_count": 20,
        "corrected_unique_five_candidate_count": 20,
    }
    for key, expected in expected_summary.items():
        if summary[key] != expected:
            raise SystemExit(
                f"CORRECTION_EXPECTATION_FAIL {key}={summary[key]} expected={expected}"
            )

    paths = {
        "old_audit": output_root
        / "results/reproduction/diagnostics/v1_0_1_policy_stability_duplicate_candidate_audit.csv",
        "corrected_csv": output_root
        / "results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_stability.csv",
        "corrected_summary": output_root
        / "results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_summary.json",
        "source_data": output_root
        / "source_data/policy_stability_unique_candidate_subsampling_source.csv",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    old.to_csv(paths["old_audit"], index=False, lineterminator="\n")
    corrected.to_csv(paths["corrected_csv"], index=False, lineterminator="\n")
    corrected.to_csv(paths["source_data"], index=False, lineterminator="\n")
    _write_json(paths["corrected_summary"], summary)

    manifest = {
        "schema_version": "v1.0.2-correction-manifest-1.0",
        "base_release": "v1.0.1",
        "frozen_input_sha256": EXPECTED_SHA256,
        "locked_evidence_unchanged": {
            "activation_model": True,
            "train_calibration_test_membership": True,
            "selected_C": 0.001,
            "policy_locked_on_original_calibration": True,
            "final_test_metrics": True,
            "primary_five_selectors": True,
            "primary_five_traces": True,
            "thermal_analysis": True,
        },
        "outputs": {
            str(path.relative_to(output_root)).replace("\\", "/"): {
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths.values()
        },
    }
    manifest_path = output_root / "results/reproduction/v1_0_2_correction_manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
