from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from traceable_portfoliobatch.replay import replay_all

parser = argparse.ArgumentParser()
parser.add_argument("--trace", type=Path)
parser.add_argument("--dry-run", action="store_true", help="Retained for CLI compatibility; replay is still executed.")
args = parser.parse_args()
trace = args.trace
if trace and not trace.is_absolute():
    trace = ROOT / trace
summary = replay_all(ROOT, trace)
summary["dry_run_flag_received"] = bool(args.dry_run)
print(json.dumps(summary, indent=2))
raise SystemExit(0 if summary["status"] == "PASS" else 1)
