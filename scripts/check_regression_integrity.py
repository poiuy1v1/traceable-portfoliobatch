from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceable_portfoliobatch.io_utils import write_json
from traceable_portfoliobatch.regression import (
    CONTRACT_REL,
    REFERENCE_REL,
    build_semantic_snapshot,
    compare_semantic_snapshots,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Paper13 scientific regression contract")
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--mode", choices=["canonical", "portability"], default="portability")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    reference_path = ROOT / REFERENCE_REL
    contract_path = ROOT / CONTRACT_REL
    if not contract_path.exists():
        raise SystemExit(f"Regression contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    current = build_semantic_snapshot(ROOT)
    if args.write_reference:
        write_json(reference_path, current)
        result = {
            "status": "REFERENCE_WRITTEN",
            "path": REFERENCE_REL,
            "schema_version": current["schema_version"],
        }
    else:
        if not reference_path.exists():
            raise SystemExit(f"Regression reference is missing: {reference_path}")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        errors = compare_semantic_snapshots(reference, current, contract=contract, mode=args.mode)
        result = {
            "status": "PASS" if not errors else "FAIL",
            "mode": args.mode,
            "contract": CONTRACT_REL,
            "error_count": len(errors),
            "errors": errors[:100],
            "diagnostics": current.get("diagnostics", {}),
        }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if result["status"] in {"PASS", "REFERENCE_WRITTEN"} else 1)


if __name__ == "__main__":
    main()
