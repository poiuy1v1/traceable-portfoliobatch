from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_REL = Path("results/reproduction/generated_artifact_manifest.json")
REFERENCE_REL = Path("results/reproduction/regression_reference.json")
CONTRACT_REL = Path("results/reproduction/regression_contract.json")

sys.path.insert(0, str(ROOT / "src"))
from traceable_portfoliobatch.regression import compare_png_visuals


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated Paper13 reproduction checks")
    parser.add_argument("--mode", choices=["canonical", "portability"], default="portability")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    expected_manifest_path = ROOT / MANIFEST_REL
    expected_reference_path = ROOT / REFERENCE_REL
    contract_path = ROOT / CONTRACT_REL
    for path in [expected_manifest_path, expected_reference_path, contract_path]:
        if not path.exists():
            raise SystemExit(f"Required regression file is missing: {path}")

    expected_manifest = json.loads(expected_manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="paper13-clean-reproduction-") as temp_dir:
        copy_root = Path(temp_dir) / "repo"
        shutil.copytree(
            ROOT,
            copy_root,
            ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"),
        )
        reproduction = _run([sys.executable, "scripts/run_full_reproduction.py"], cwd=copy_root)
        if reproduction.returncode != 0:
            result = {
                "status": "FAIL",
                "mode": args.mode,
                "stage": "clean_copy_reproduction",
                "returncode": reproduction.returncode,
                "stdout": reproduction.stdout[-12000:],
                "stderr": reproduction.stderr[-12000:],
            }
            _finish(result, args.json_output, 1)

        semantic = _run(
            [
                sys.executable,
                "scripts/check_regression_integrity.py",
                "--mode",
                args.mode,
            ],
            cwd=copy_root,
        )
        semantic_result = _parse_last_json(semantic.stdout)
        if semantic.returncode != 0:
            result = {
                "status": "FAIL",
                "mode": args.mode,
                "stage": "clean_copy_semantic_regression",
                "returncode": semantic.returncode,
                "stdout": semantic.stdout[-12000:],
                "stderr": semantic.stderr[-12000:],
                "semantic": semantic_result,
            }
            _finish(result, args.json_output, 1)

        observed_manifest = json.loads((copy_root / MANIFEST_REL).read_text(encoding="utf-8"))
        expected_files = expected_manifest.get("files", {})
        observed_files = observed_manifest.get("files", {})
        file_set_errors: list[str] = []
        if set(expected_files) != set(observed_files):
            file_set_errors.append(
                f"generated file-set mismatch: missing={sorted(set(expected_files)-set(observed_files))}, "
                f"extra={sorted(set(observed_files)-set(expected_files))}"
            )

        byte_errors: list[str] = []
        visual_results: dict[str, dict] = {}
        for path in sorted(set(expected_files) & set(observed_files)):
            expected = expected_files[path]
            observed = observed_files[path]
            regression_class = expected.get("regression_class", "canonical_byte_identity")
            if regression_class != observed.get("regression_class"):
                file_set_errors.append(
                    f"{path}: regression class {regression_class!r} != {observed.get('regression_class')!r}"
                )
                continue
            if regression_class == "canonical_byte_identity" and args.mode == "canonical":
                if expected != observed:
                    byte_errors.append(f"{path}: expected {expected}, observed {observed}")
            elif regression_class == "visual_equivalence" and path.endswith(".png"):
                visual_results[path] = compare_png_visuals(ROOT / path, copy_root / path, contract)

        visual_errors = [f"{p}: {r}" for p, r in visual_results.items() if r["status"] != "PASS"]
        stderr_nonempty = bool(reproduction.stderr.strip())
        result = {
            "status": (
                "PASS"
                if not file_set_errors and not byte_errors and not visual_errors
                and semantic_result.get("status") == "PASS"
                and not stderr_nonempty
                else "FAIL"
            ),
            "mode": args.mode,
            "reproduction_returncode": reproduction.returncode,
            "reproduction_stderr_empty": not stderr_nonempty,
            "file_set_error_count": len(file_set_errors),
            "canonical_byte_error_count": len(byte_errors),
            "visual_error_count": len(visual_errors),
            "semantic_error_count": semantic_result.get("error_count"),
            "file_set_errors": file_set_errors[:50],
            "canonical_byte_errors": byte_errors[:50],
            "visual_errors": visual_errors[:50],
            "visual_results": visual_results,
            "semantic": semantic_result,
            "reproduction_stdout_tail": reproduction.stdout[-4000:],
            "reproduction_stderr_tail": reproduction.stderr[-4000:],
        }
    _finish(result, args.json_output, 0 if result["status"] == "PASS" else 1)


def _parse_last_json(text: str) -> dict:
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.rfind("\n{")
        if start >= 0:
            return json.loads(text[start + 1 :])
        raise


def _finish(result: dict, output: Path | None, code: int) -> None:
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
