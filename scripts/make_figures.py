from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from traceable_portfoliobatch.figures import build_all

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="Compatibility flag; source data are still read and assets regenerated.")
args = parser.parse_args()
paths = build_all(ROOT)
missing = [p for p in paths if not (ROOT / p).exists()]
status = "PASS" if not missing else "FAIL"
print(json.dumps({"status": status, "regenerated": paths, "missing": missing, "dry_run_flag_received": args.dry_run}, indent=2))
raise SystemExit(0 if status == "PASS" else 1)
