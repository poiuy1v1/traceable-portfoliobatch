from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

CORRECTED = Path(
    "results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_stability.csv"
)
SOURCE = Path("source_data/policy_stability_unique_candidate_subsampling_source.csv")
SUMMARY = Path(
    "results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_summary.json"
)
LEGACY_AUDIT = Path(
    "results/reproduction/diagnostics/v1_0_1_policy_stability_duplicate_candidate_audit.csv"
)
MANIFEST = Path("results/reproduction/v1_0_2_correction_manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    required = [CORRECTED, SOURCE, SUMMARY, LEGACY_AUDIT, MANIFEST]
    missing = [path.as_posix() for path in required if not (root / path).is_file()]
    if missing:
        raise RuntimeError(f"MISSING_V1_0_2_CORRECTION_FILES:{missing}")

    corrected = pd.read_csv(root / CORRECTED)
    source = pd.read_csv(root / SOURCE)
    legacy = pd.read_csv(root / LEGACY_AUDIT)
    summary = read_json(root / SUMMARY)
    manifest = read_json(root / MANIFEST)

    if (root / CORRECTED).read_bytes() != (root / SOURCE).read_bytes():
        raise RuntimeError("CORRECTED_SOURCE_DATA_BYTE_MISMATCH")

    required_columns = {
        "seed",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "calibration_group_count",
        "sampled_group_occurrences",
        "subset_row_count",
        "unique_candidate_count",
        "selected_count",
        "selected_unique_count",
        "selected_ids",
        "full_batch",
        "sampling_fraction",
        "sampling_with_replacement",
        "duplicate_selected_candidate_present",
    }
    missing_columns = sorted(required_columns - set(corrected.columns))
    if missing_columns:
        raise RuntimeError(f"CORRECTED_SCHEMA_MISSING:{missing_columns}")

    exact = (
        corrected["beta"].astype(float).eq(0.3)
        & corrected["gamma"].astype(float).eq(0.2)
        & corrected["delta"].astype(float).eq(0.0)
    )
    observed = {
        "old_exact_locked_tuple_count": int(
            (
                legacy["beta"].astype(float).eq(0.3)
                & legacy["gamma"].astype(float).eq(0.2)
                & legacy["delta"].astype(float).eq(0.0)
            ).sum()
        ),
        "old_delta_zero_count": int(legacy["delta"].astype(float).eq(0.0).sum()),
        "old_full_batch_count": int(legacy["full_batch"].astype(bool).sum()),
        "old_delta_zero_duplicate_selected_candidate_count": int(
            (
                legacy["delta"].astype(float).eq(0.0)
                & legacy["duplicate_selected_candidate_present"].astype(bool)
            ).sum()
        ),
        "corrected_exact_locked_tuple_count": int(exact.sum()),
        "corrected_delta_zero_count": int(
            corrected["delta"].astype(float).eq(0.0).sum()
        ),
        "corrected_full_batch_count": int(corrected["full_batch"].astype(bool).sum()),
        "corrected_unique_five_candidate_count": int(
            (
                corrected["selected_count"].astype(int).eq(5)
                & corrected["selected_unique_count"].astype(int).eq(5)
            ).sum()
        ),
    }
    expected = {
        "old_exact_locked_tuple_count": 1,
        "old_delta_zero_count": 14,
        "old_full_batch_count": 20,
        "old_delta_zero_duplicate_selected_candidate_count": 10,
        "corrected_exact_locked_tuple_count": 3,
        "corrected_delta_zero_count": 18,
        "corrected_full_batch_count": 20,
        "corrected_unique_five_candidate_count": 20,
    }
    if observed != expected:
        raise RuntimeError(f"V1_0_2_CORRECTION_SUMMARY_FAIL:{observed}")

    design_ok = (
        len(corrected) == 20
        and set(corrected["seed"].astype(int)) == set(range(20))
        and corrected["sampling_fraction"].astype(float).eq(0.8).all()
        and not corrected["sampling_with_replacement"].astype(bool).any()
        and corrected["calibration_group_count"].astype(int).eq(109).all()
        and corrected["sampled_group_occurrences"].astype(int).eq(109).all()
        and corrected["selected_count"].astype(int).eq(5).all()
        and corrected["selected_unique_count"].astype(int).eq(5).all()
        and not corrected["duplicate_selected_candidate_present"].astype(bool).any()
        and corrected["full_batch"].astype(bool).all()
    )
    if not design_ok:
        raise RuntimeError("V1_0_2_CORRECTED_DESIGN_FAIL")

    for key, value in observed.items():
        if summary.get(key) != value:
            raise RuntimeError(f"V1_0_2_SUMMARY_JSON_FAIL:{key}")

    for relative, expected_hash in manifest["frozen_input_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"V1_0_2_FROZEN_INPUT_HASH_FAIL:{relative}")
    for relative, metadata in manifest["outputs"].items():
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(metadata["size"])
            or sha256(path) != metadata["sha256"]
        ):
            raise RuntimeError(f"V1_0_2_OUTPUT_HASH_FAIL:{relative}")

    return {
        "status": "PASS",
        "corrected_rows": int(len(corrected)),
        "observed_summary": observed,
        "frozen_input_files": int(len(manifest["frozen_input_sha256"])),
        "output_files": int(len(manifest["outputs"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the v1.0.2 bounded policy-stability correction."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.repo_root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")


if __name__ == "__main__":
    main()
