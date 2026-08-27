from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import traceable_portfoliobatch  # noqa: E402,F401
from traceable_portfoliobatch.regression import (  # noqa: E402
    MANIFEST_REL,
    REFERENCE_REL,
    build_generated_artifact_manifest,
)
from validate_v49_scientific_gate import validate_repository_state  # noqa: E402

REQUIRED = [
    "README.md",
    "CITATION.cff",
    "LICENSE-CODE",
    "LICENSE-DATA",
    "THIRD_PARTY_NOTICES.md",
    "environment-lock.yml",
    "requirements-canonical.txt",
    "VERSION",
    "RELEASE_NOTES_v1.0.2.md",
    "data/source_data_manifest.csv",
    "data/processed/full_public_stability_features.csv",
    "data/processed/full_public_stability_source_manifest.csv",
    "data/provenance/mofsimplify_training_source_provenance.json",
    "data/derived/full_public_stability_features_relocked.csv",
    "data/processed/full_public_stability_manifest_scored.csv",
    "models/activation_logistic_train_split.json",
    "models/thermal_ridge_train_split.json",
    "results/reproduction/tuning/locked_parameters.json",
    "results/reproduction/tuning/v49_locked_parameters_before_test.json",
    "results/reproduction/tuning/portfolio_policy_cost_only_calibration_selection.csv",
    "results/reproduction/tuning/portfolio_policy_cost_only_group_bootstrap_stability.csv",
    "results/reproduction/diagnostics/v1_0_1_policy_stability_duplicate_candidate_audit.csv",
    "results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_stability.csv",
    "results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_summary.json",
    "results/reproduction/v1_0_2_correction_manifest.json",
    "source_data/policy_stability_unique_candidate_subsampling_source.csv",
    "results/reproduction/diagnostics/source_overlap_C_ablation.csv",
    "results/reproduction/activation_dummy_baselines.json",
    "results/reproduction/activation_test_source_group_bootstrap_summary.json",
    "results/reproduction/risk_feasibility_audit.csv",
    "results/reproduction/test_policy_comparison_primary_cost_only.csv",
    "results/reproduction/diagnostics/test_policy_risk_stress_legacy_0p35.csv",
    "results/reproduction/v49_statistical_summary.json",
    MANIFEST_REL,
    REFERENCE_REL,
    "results/reproduction/regression_contract.json",
    "scripts/run_full_reproduction.py",
    "scripts/recompute_v1_0_2_policy_stability.py",
    "scripts/validate_v1_0_2_policy_stability.py",
    "scripts/run_audit_replay.py",
    "scripts/check_regression_integrity.py",
    "scripts/check_clean_reproduction.py",
    "scripts/validate_v49_scientific_gate.py",
    "tests/test_v1_0_2_policy_stability_correction.py",
    ".github/workflows/reproduce.yml",
]



def _read_top_level_cff_scalar(path: Path, key: str) -> str | None:
    """Read one top-level scalar from CITATION.cff without adding a YAML dependency."""
    prefix = f"{key}:"
    values: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.startswith(prefix):
            continue
        value = raw_line[len(prefix):].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        values.append(value)
    return values[0] if len(values) == 1 else None

CORRECTION_INVENTORY = {
    "results/reproduction/diagnostics/v1_0_1_policy_stability_duplicate_candidate_audit.csv",
    "results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_stability.csv",
    "results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_summary.json",
    "results/reproduction/v1_0_2_correction_manifest.json",
    "source_data/policy_stability_unique_candidate_subsampling_source.csv",
}


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    checks: dict[str, bool] = {"required_files": not missing}

    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    citation_version = _read_top_level_cff_scalar(ROOT / "CITATION.cff", "version")
    checks["public_release_metadata_v1_0_2"] = (
        version == "1.0.2"
        and citation_version == "1.0.2"
        and "PUBLIC RELEASE REPOSITORY" in readme
        and "v1.0.2 policy-stability correction" in readme
        and "subsampling without replacement" in readme
    )

    if (ROOT / "data/source_data_manifest.csv").is_file():
        inventory = pd.read_csv(ROOT / "data/source_data_manifest.csv")
        inventory_paths = set(inventory["file_path"].astype(str))
        checks["correction_inventory_declared"] = CORRECTION_INVENTORY <= inventory_paths
    else:
        checks["correction_inventory_declared"] = False

    corrected_path = (
        ROOT
        / "results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_stability.csv"
    )
    summary_path = (
        ROOT
        / "results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_summary.json"
    )
    if corrected_path.is_file() and summary_path.is_file():
        corrected = pd.read_csv(corrected_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checks["corrected_stability_contract"] = (
            len(corrected) == 20
            and corrected["sampling_with_replacement"].eq(False).all()  # noqa: E712
            and corrected["calibration_group_count"].eq(109).all()
            and corrected["selected_count"].eq(5).all()
            and corrected["selected_unique_count"].eq(5).all()
            and not corrected["duplicate_selected_candidate_present"].any()
            and summary.get("old_archived_reproduction") == "PASS"
            and summary.get("corrected_exact_locked_tuple_count") == 3
            and summary.get("corrected_delta_zero_count") == 18
            and summary.get("corrected_full_batch_count") == 20
            and summary.get("corrected_unique_five_candidate_count") == 20
        )
    else:
        checks["corrected_stability_contract"] = False

    if (ROOT / MANIFEST_REL).is_file():
        stored_manifest = json.loads((ROOT / MANIFEST_REL).read_text(encoding="utf-8"))
        checks["generated_artifact_manifest_current"] = (
            stored_manifest == build_generated_artifact_manifest(ROOT)
        )
    else:
        checks["generated_artifact_manifest_current"] = False

    gate = validate_repository_state()
    checks["v49_scientific_gate"] = gate["status"] == "PASS"

    checks = {name: bool(value) for name, value in checks.items()}
    status = "PASS" if all(checks.values()) else "FAIL"
    print(
        json.dumps(
            {
                "status": status,
                "missing": missing,
                "checks": checks,
                "v49_scientific_gate": gate,
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
