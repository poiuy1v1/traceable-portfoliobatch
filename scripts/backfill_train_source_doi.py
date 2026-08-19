from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import traceable_portfoliobatch  # noqa: E402,F401
import pandas as pd  # noqa: E402

from traceable_portfoliobatch.io_utils import git_blob_sha1, sha256_file, write_csv, write_json

EXPECTED_UPSTREAM_GIT_BLOB_SHA1 = "dc67ae0a5b72863cdc8c6850257dab40ede74b1c"
UPSTREAM_COMMIT = "5693968b3e9b9e26eab3bdb1db908ae2877d4bb7"
UPSTREAM_RELATIVE_PATH = "model/solvent/ANN/dropped_connectivity_dupes/train.csv"
UPSTREAM_BLOB_URL = (
    "https://github.com/hjkgrp/MOFSimplify/blob/"
    f"{UPSTREAM_COMMIT}/{UPSTREAM_RELATIVE_PATH}"
)


def normalize_doi(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.rstrip("/.,; ")


def is_valid_normalized_doi(value: object) -> bool:
    text = normalize_doi(value)
    return re.fullmatch(r"10\.\d{4,9}/\S+", text) is not None


def normalize_refcode(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill training-row source DOI values in the canonical all-split "
            "source manifest from the official MOFSimplify train.csv. Existing "
            "calibration/test DOI values are preserved exactly."
        )
    )
    parser.add_argument("upstream_train_csv", type=Path)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/full_public_stability_features.csv"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/processed/full_public_stability_source_manifest.csv"),
    )
    parser.add_argument(
        "--output-source-manifest",
        type=Path,
        default=Path("data/processed/full_public_stability_source_manifest_with_train_doi.csv"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("results/reproduction/training_source_doi_backfill_audit.json"),
    )
    parser.add_argument(
        "--require-official-git-blob-sha",
        "--require-official-blob-sha",
        dest="require_official_git_blob_sha",
        action="store_true",
        help=(
            "Require Git blob-object SHA-1 semantics to match the recorded official "
            "GitHub blob. This is not the same as raw-file SHA-1."
        ),
    )
    parser.add_argument(
        "--expected-raw-sha256",
        default=None,
        help="Optionally require a separately recorded raw-file SHA-256 value.",
    )
    return parser


def run_backfill(args: argparse.Namespace) -> dict:
    upstream_path = args.upstream_train_csv.resolve()
    if not upstream_path.is_file():
        raise SystemExit(f"Upstream training CSV does not exist: {upstream_path}")

    observed_raw_sha256 = sha256_file(upstream_path)
    observed_git_blob_sha1 = git_blob_sha1(upstream_path)
    if args.require_official_git_blob_sha and observed_git_blob_sha1 != EXPECTED_UPSTREAM_GIT_BLOB_SHA1:
        raise SystemExit(
            "Upstream Git blob SHA-1 mismatch: "
            f"expected {EXPECTED_UPSTREAM_GIT_BLOB_SHA1}, observed {observed_git_blob_sha1}. "
            "Git blob IDs include the 'blob <size>\\0' object header."
        )
    if args.expected_raw_sha256 and observed_raw_sha256.lower() != args.expected_raw_sha256.lower():
        raise SystemExit(
            f"Upstream raw SHA-256 mismatch: expected {args.expected_raw_sha256}, "
            f"observed {observed_raw_sha256}"
        )

    upstream = pd.read_csv(upstream_path, low_memory=False)
    required_upstream = {"refcode", "doi"}
    missing_upstream = required_upstream - set(upstream.columns)
    if missing_upstream:
        raise SystemExit(f"Upstream file is missing required columns: {sorted(missing_upstream)}")

    mapping = upstream[["refcode", "doi"]].copy()
    mapping["raw_refcode"] = mapping["refcode"].map(normalize_refcode)
    mapping["source_doi_backfill"] = mapping["doi"].map(normalize_doi)
    mapping = mapping[["raw_refcode", "source_doi_backfill"]]
    if mapping["raw_refcode"].eq("").any():
        raise SystemExit("Upstream file contains blank refcode values")
    if mapping["source_doi_backfill"].eq("").any():
        examples = mapping.loc[mapping["source_doi_backfill"].eq(""), "raw_refcode"].head(10).tolist()
        raise SystemExit(f"Upstream file contains blank DOI values; examples={examples}")
    invalid_mapping_doi = ~mapping["source_doi_backfill"].map(is_valid_normalized_doi)
    if invalid_mapping_doi.any():
        examples = mapping.loc[
            invalid_mapping_doi, ["raw_refcode", "source_doi_backfill"]
        ].head(10).to_dict("records")
        raise SystemExit(f"Upstream file contains invalid DOI syntax; examples={examples}")
    conflicts = mapping.groupby("raw_refcode")["source_doi_backfill"].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise SystemExit(
            f"Conflicting DOI values for {len(conflicts)} refcodes; "
            f"examples={conflicts.index[:10].tolist()}"
        )
    mapping = mapping.drop_duplicates("raw_refcode", keep="first")

    features = pd.read_csv(args.features, low_memory=False)
    required_features = {"candidate_id", "model_split", "raw_refcode"}
    missing_features = required_features - set(features.columns)
    if missing_features:
        raise SystemExit(f"Feature table is missing required columns: {sorted(missing_features)}")
    if features["candidate_id"].duplicated().any():
        raise SystemExit("Feature table contains duplicate candidate_id values")

    manifest = pd.read_csv(args.source_manifest, low_memory=False)
    required_manifest = {"candidate_id", "model_split", "raw_refcode", "source_doi"}
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise SystemExit(f"Source manifest is missing required columns: {sorted(missing_manifest)}")
    if manifest["candidate_id"].duplicated().any():
        raise SystemExit("Source manifest contains duplicate candidate_id values")

    feature_ids = set(features["candidate_id"].astype(str))
    manifest_ids = set(manifest["candidate_id"].astype(str))
    missing_from_manifest = sorted(feature_ids - manifest_ids)
    extra_in_manifest = sorted(manifest_ids - feature_ids)
    if missing_from_manifest or extra_in_manifest:
        raise SystemExit(
            "Canonical source manifest does not cover the feature table exactly: "
            f"missing={len(missing_from_manifest)}, extra={len(extra_in_manifest)}"
        )

    manifest = manifest.copy()
    manifest["raw_refcode"] = manifest["raw_refcode"].map(normalize_refcode)
    manifest["source_doi"] = manifest["source_doi"].fillna("").astype(str)
    held_mask = ~manifest["model_split"].eq("train")
    held_doi_before = manifest.loc[held_mask, ["candidate_id", "source_doi"]].copy()

    train_lookup = (
        features.loc[features["model_split"].eq("train"), ["candidate_id", "raw_refcode"]]
        .assign(raw_refcode=lambda x: x["raw_refcode"].map(normalize_refcode))
        .merge(mapping, on="raw_refcode", how="left", validate="many_to_one")
    )
    missing_train = train_lookup["source_doi_backfill"].isna() | train_lookup["source_doi_backfill"].eq("")
    if missing_train.any():
        examples = train_lookup.loc[missing_train, "raw_refcode"].head(10).tolist()
        raise SystemExit(
            f"DOI backfill incomplete: {int(missing_train.sum())} training rows remain unresolved; "
            f"examples={examples}"
        )

    candidate_to_doi = train_lookup.set_index("candidate_id")["source_doi_backfill"]
    train_mask = manifest["model_split"].eq("train")
    existing_train = manifest.loc[train_mask, "source_doi"].map(normalize_doi)
    incoming_train = manifest.loc[train_mask, "candidate_id"].map(candidate_to_doi).map(normalize_doi)
    conflicts_existing = existing_train.ne("") & existing_train.ne(incoming_train)
    if conflicts_existing.any():
        examples = manifest.loc[train_mask].loc[conflicts_existing, "candidate_id"].head(10).tolist()
        raise SystemExit(f"Existing training DOI conflicts with upstream mapping; examples={examples}")
    manifest.loc[train_mask, "source_doi"] = incoming_train.to_numpy()

    held_after = manifest.loc[held_mask, ["candidate_id", "source_doi"]].copy()
    if not held_doi_before.reset_index(drop=True).equals(held_after.reset_index(drop=True)):
        raise SystemExit("Held-out DOI values changed during training DOI backfill")

    manifest["source_doi_normalized"] = manifest["source_doi"].map(normalize_doi)
    invalid_manifest_doi = ~manifest["source_doi_normalized"].map(is_valid_normalized_doi)
    if invalid_manifest_doi.any():
        examples = manifest.loc[
            invalid_manifest_doi, ["candidate_id", "source_doi_normalized"]
        ].head(10).to_dict("records")
        raise SystemExit(
            "Canonical source manifest contains invalid DOI syntax after backfill; "
            f"examples={examples}"
        )
    manifest["source_group_id"] = manifest["source_doi_normalized"]
    blank = manifest["source_group_id"].eq("")
    manifest.loc[blank, "source_group_id"] = (
        "unresolved_candidate::" + manifest.loc[blank, "candidate_id"].astype(str)
    )
    manifest["source_group_basis"] = "normalized_source_doi"
    manifest.loc[blank, "source_group_basis"] = "unresolved_unique_candidate_surrogate"

    unresolved = int(manifest["source_doi_normalized"].eq("").sum())
    if unresolved:
        raise SystemExit(f"Canonical source manifest still contains {unresolved} unresolved DOI values")

    # v45A3 keeps the existing training pool and the existing calibration/test
    # assignment.  Held rows whose publication DOI is represented in training
    # remain in the provenance manifest but are removed from active evaluation.
    manifest["original_model_split"] = manifest["model_split"].astype(str)
    train_groups = set(
        manifest.loc[
            manifest["original_model_split"].eq("train"), "source_doi_normalized"
        ]
    )
    held_mask = manifest["original_model_split"].isin(["calibration", "test"])
    excluded_mask = held_mask & manifest["source_doi_normalized"].isin(train_groups)
    manifest["final_evaluation_split"] = manifest["original_model_split"]
    manifest.loc[excluded_mask, "final_evaluation_split"] = (
        "excluded_train_source_overlap"
    )
    manifest["exclusion_reason"] = ""
    manifest.loc[excluded_mask, "exclusion_reason"] = (
        "source DOI group also present in training"
    )

    active_groups = {
        split: set(
            manifest.loc[
                manifest["final_evaluation_split"].eq(split), "source_doi_normalized"
            ]
        )
        for split in ["train", "calibration", "test"]
    }
    active_overlaps = {
        "train_calibration": len(active_groups["train"] & active_groups["calibration"]),
        "train_test": len(active_groups["train"] & active_groups["test"]),
        "calibration_test": len(active_groups["calibration"] & active_groups["test"]),
    }
    if any(active_overlaps.values()):
        raise SystemExit(f"Active source DOI overlap remains after exclusion: {active_overlaps}")

    write_csv(manifest, args.output_source_manifest)
    audit = {
        "status": "PASS",
        "upstream": {
            "commit": UPSTREAM_COMMIT,
            "relative_path": UPSTREAM_RELATIVE_PATH,
            "blob_url": UPSTREAM_BLOB_URL,
            "expected_git_blob_sha1": EXPECTED_UPSTREAM_GIT_BLOB_SHA1,
            "observed_git_blob_sha1": observed_git_blob_sha1,
            "observed_raw_sha256": observed_raw_sha256,
            "expected_raw_sha256": args.expected_raw_sha256,
        },
        "rows": {
            "all": int(len(manifest)),
            "train": int(train_mask.sum()),
            "calibration_test": int(held_mask.sum()),
            "training_rows_with_doi": int(
                manifest.loc[train_mask, "source_doi_normalized"].ne("").sum()
            ),
            "held_rows_preserved": int(held_mask.sum()),
            "unresolved": unresolved,
            "ambiguous_or_conflicting": 0,
            "excluded_train_source_overlap": int(excluded_mask.sum()),
            "final_train": int(manifest["final_evaluation_split"].eq("train").sum()),
            "final_calibration": int(
                manifest["final_evaluation_split"].eq("calibration").sum()
            ),
            "final_test": int(manifest["final_evaluation_split"].eq("test").sum()),
        },
        "active_source_doi_group_overlaps": active_overlaps,
        "excluded_train_source_overlap_doi_group_count": int(
            manifest.loc[excluded_mask, "source_doi_normalized"].nunique()
        ),
        "inputs": {
            "features": str(args.features),
            "source_manifest": str(args.source_manifest),
        },
        "output_source_manifest": str(args.output_source_manifest),
        "runner_command": (
            "python scripts/run_full_reproduction.py --source-manifest "
            f"{args.output_source_manifest.as_posix()}"
        ),
    }
    write_json(args.audit_output, audit)
    return audit


def main() -> None:
    args = _parser().parse_args()
    audit = run_backfill(args)
    print(audit)


if __name__ == "__main__":
    main()
