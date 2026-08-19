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
    "DEVELOPMENT_STATUS.md",
    "CITATION.cff",
    "LICENSE-CODE",
    "LICENSE-DATA",
    "THIRD_PARTY_NOTICES.md",
    "environment-lock.yml",
    "requirements-canonical.txt",
    "VERSION",
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
    "scripts/run_audit_replay.py",
    "scripts/check_regression_integrity.py",
    "scripts/check_clean_reproduction.py",
    "scripts/validate_v49_scientific_gate.py",
    ".github/workflows/reproduce.yml",
]

REQUIRED_SOURCE_INVENTORY = {
    "data/processed/full_public_stability_source_manifest.csv",
    "data/derived/full_public_stability_features_relocked.csv",
    "results/reproduction/tuning/v49_locked_parameters_before_test.json",
    "results/reproduction/diagnostics/source_overlap_C_ablation.csv",
    "results/reproduction/activation_dummy_baselines.json",
    "results/reproduction/activation_test_source_group_bootstrap_summary.json",
    "results/reproduction/risk_feasibility_audit.csv",
    "source_data/source_overlap_C_ablation_source.csv",
    "source_data/activation_dummy_baselines_source.csv",
    "source_data/activation_test_source_group_bootstrap_source.csv",
    "source_data/risk_feasibility_audit_source.csv",
    MANIFEST_REL,
    REFERENCE_REL,
    "results/reproduction/regression_contract.json",
    "requirements-canonical.txt",
    "Dockerfile.canonical",
}

PRIMARY_FIGURE3_SOURCES = {
    "figure3_test_policy_primary_cost_source.csv",
    "figure3_random_baseline_primary_cost_raw_points.csv",
    "figure3_random_baseline_primary_cost_summary_source.csv",
    "figure3_pool_resampling_primary_cost_raw_points.csv",
    "figure3_pool_resampling_primary_cost_summary_source.csv",
}


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    checks: dict[str, bool] = {"required_files": not missing}

    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    development = (ROOT / "DEVELOPMENT_STATUS.md").read_text(encoding="utf-8")
    checks["development_version_not_release"] = (
        version == "0.49-dev"
        and "DEVELOPMENT-ONLY PRIVATE REPOSITORY" in development
        and "Do not create" in development
        and "v1.0.0" in development
    )

    if (ROOT / "data/source_data_manifest.csv").is_file():
        inventory = pd.read_csv(ROOT / "data/source_data_manifest.csv")
        inventory_paths = set(inventory["file_path"].astype(str))
        checks["source_data_inventory_v49_contracts"] = REQUIRED_SOURCE_INVENTORY <= inventory_paths
    else:
        checks["source_data_inventory_v49_contracts"] = False

    if (ROOT / "source_data/figure_source_data_manifest.csv").is_file():
        figure_manifest = pd.read_csv(ROOT / "source_data/figure_source_data_manifest.csv")
        sources = " ".join(
            figure_manifest.loc[
                figure_manifest["figure_file"].astype(str).str.contains("Figure3_policy_comparison"),
                "source",
            ].astype(str)
        )
        checks["figure3_uses_cost_only_primary_sources"] = (
            all(source in sources for source in PRIMARY_FIGURE3_SOURCES)
            and "common_risk" not in sources
        )
    else:
        checks["figure3_uses_cost_only_primary_sources"] = False

    primary_traces = sorted(
        (ROOT / "traces/action_traces").glob("*_test_primary_cost_contract_trace.json")
    )
    replay = (
        json.loads((ROOT / "results/reproduction/replay_summary.json").read_text(encoding="utf-8"))
        if (ROOT / "results/reproduction/replay_summary.json").is_file()
        else {}
    )
    checks["primary_trace_count_5"] = len(primary_traces) == 5
    checks["replay_all_pass"] = (
        replay.get("status") == "PASS"
        and replay.get("trace_count") == 5
        and replay.get("passed_count") == 5
    )

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
