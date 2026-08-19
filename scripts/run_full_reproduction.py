from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceable_portfoliobatch.workflow import run_full_reproduction


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Paper13 reproduction workflow")
    parser.add_argument(
        "--source-manifest",
        default=None,
        help=(
            "Canonical all-split source manifest. Defaults to "
            "data/processed/full_public_stability_source_manifest.csv."
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_full_reproduction(ROOT, source_manifest_rel=args.source_manifest),
            indent=2,
            sort_keys=True,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
