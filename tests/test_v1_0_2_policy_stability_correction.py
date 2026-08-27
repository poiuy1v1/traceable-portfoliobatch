from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_v1_0_2_policy_stability_correction_reproduces_committed_evidence(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/recompute_v1_0_2_policy_stability.py"),
            "--repo-root",
            str(ROOT),
            "--output-root",
            str(tmp_path),
        ],
        check=True,
    )

    relative_paths = [
        "results/reproduction/diagnostics/v1_0_1_policy_stability_duplicate_candidate_audit.csv",
        "results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_stability.csv",
        "results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_summary.json",
        "source_data/policy_stability_unique_candidate_subsampling_source.csv",
        "results/reproduction/v1_0_2_correction_manifest.json",
    ]
    for rel in relative_paths:
        assert (tmp_path / rel).read_bytes() == (ROOT / rel).read_bytes()

    corrected = pd.read_csv(
        ROOT
        / "results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_stability.csv"
    )
    summary = json.loads(
        (
            ROOT
            / "results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert len(corrected) == 20
    assert corrected["sampling_with_replacement"].eq(False).all()  # noqa: E712
    assert corrected["calibration_group_count"].eq(109).all()
    assert corrected["selected_count"].eq(5).all()
    assert corrected["selected_unique_count"].eq(5).all()
    assert not corrected["duplicate_selected_candidate_present"].any()
    assert summary["corrected_exact_locked_tuple_count"] == 3
    assert summary["corrected_delta_zero_count"] == 18
    assert summary["corrected_full_batch_count"] == 20
    assert summary["corrected_unique_five_candidate_count"] == 20
    validation = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_v1_0_2_policy_stability.py"),
            "--repo-root",
            str(ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert '"status": "PASS"' in validation.stdout

