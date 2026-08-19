$ErrorActionPreference = "Stop"
$env:PYTHONHASHSEED = "0"
$env:SOURCE_DATE_EPOCH = "1785369600"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

python scripts\collect_runtime_diagnostics.py --output windows_runtime_diagnostics.json
python scripts\check_clean_reproduction.py --mode portability --json-output windows_portability_check.json
python -m pytest -q -W error::sklearn.exceptions.ConvergenceWarning
python scripts\validate_repository.py

Write-Host "Windows portability contract PASS"
