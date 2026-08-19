from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import write_json


def _overlap(train: pd.Series, held: pd.Series) -> dict:
    a = set(train.dropna().astype(str))
    b = set(held.dropna().astype(str))
    overlap = sorted(a & b)
    return {"overlap_count": len(overlap), "examples": overlap[:20]}


def run_exact_split_audit(features: pd.DataFrame) -> dict:
    split_column = (
        "final_evaluation_split"
        if "final_evaluation_split" in features
        else "model_split"
    )
    active = {
        split: features[features[split_column].eq(split)].copy()
        for split in ["train", "calibration", "test"]
    }
    train = active["train"]
    held = pd.concat([active["calibration"], active["test"]], ignore_index=True)
    checks = {
        "candidate_id": _overlap(train["candidate_id"], held["candidate_id"]),
        "raw_refcode": _overlap(train["raw_refcode"], held["raw_refcode"]),
    }
    train_source = train["raw_source_file"].astype(str) + "::" + train["raw_source_row"].astype(str)
    held_source = held["raw_source_file"].astype(str) + "::" + held["raw_source_row"].astype(str)
    checks["raw_source_row_identity"] = _overlap(train_source, held_source)
    checks["raw_source_row_identity"]["definition"] = "raw_source_file::raw_source_row composite key"

    def identity(frame: pd.DataFrame, key: str) -> pd.Series:
        if key == "raw_source_row_identity":
            return (
                frame["raw_source_file"].astype(str)
                + "::"
                + frame["raw_source_row"].astype(str)
            )
        return frame[key]

    pairwise: dict[str, dict] = {}
    pairs = [
        ("train", "calibration"),
        ("train", "test"),
        ("calibration", "test"),
    ]
    keys = [
        "candidate_id",
        "raw_refcode",
        "raw_source_row_identity",
        "source_doi_normalized",
        "source_group_id",
    ]
    for left, right in pairs:
        pairwise[f"{left}_vs_{right}"] = {
            key: _overlap(identity(active[left], key), identity(active[right], key))
            for key in keys
        }

    missing_provenance = int(
        features[["raw_source_file", "raw_source_row", "raw_refcode"]].isna().any(axis=1).sum()
    )
    exact_pass = all(checks[k]["overlap_count"] == 0 for k in checks)
    active_doi_coverage = {
        split: float(frame["source_doi_normalized"].fillna("").astype(str).str.strip().ne("").mean())
        for split, frame in active.items()
    }
    pairwise_pass = all(
        detail["overlap_count"] == 0
        for pair in pairwise.values()
        for detail in pair.values()
    )
    excluded = features[features[split_column].eq("excluded_train_source_overlap")].copy()
    return {
        "schema_version": "v45A3-full-source-leakage-audit-1.0",
        "scope": "active train/calibration/test pairwise identity and source DOI audit",
        "split_column": split_column,
        "candidate_level_pass": checks["candidate_id"]["overlap_count"] == 0,
        "raw_refcode_pass": checks["raw_refcode"]["overlap_count"] == 0,
        "raw_source_row_pass": checks["raw_source_row_identity"]["overlap_count"] == 0,
        "all_exact_identity_checks_pass": exact_pass,
        "all_active_pairwise_checks_pass": pairwise_pass,
        "missing_provenance_rows": missing_provenance,
        "checks": checks,
        "pairwise_active_split_checks": pairwise,
        "active_doi_coverage": active_doi_coverage,
        "excluded_train_source_overlap": {
            "row_count": int(len(excluded)),
            "doi_group_count": int(excluded["source_group_id"].nunique()),
            "candidate_id_examples": excluded["candidate_id"].astype(str).head(20).tolist(),
        },
        "chemical_group_leakage": {
            "status": "not_evaluated",
            "reason": (
                "topology_family, metal_node and linker_family labels are not available "
                "for the full train, calibration and test feature tables"
            ),
        },
    }


def write_audit(features: pd.DataFrame, path: Path) -> dict:
    result = run_exact_split_audit(features)
    write_json(path, result)
    return result
