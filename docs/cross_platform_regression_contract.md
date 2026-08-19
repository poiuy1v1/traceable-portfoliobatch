# Cross-platform regression contract

This development snapshot separates four kinds of reproducibility evidence.

1. **Exact discrete identity**: split sizes, selected `C`, selector names,
   selected candidate IDs, selected counts, stable-hit counts, trace event
   counts, schemas and random/resampling run counts must match on every
   supported platform.
2. **Numerical tolerance**: floating-point metrics may differ within the
   field-specific absolute/relative limits in
   `results/reproduction/regression_contract.json`.
3. **Canonical byte identity**: CSV and JSON scientific artifacts must be
   byte-identical in the pinned canonical Linux container. This is not claimed
   for Windows or for PDF/SVG renderer metadata.
4. **Visual equivalence**: figure dimensions and bounded rendered-pixel
   differences are checked for PNG assets. PDF/SVG byte hashes are retained as
   diagnostics, not platform-neutral scientific claims.

Optimizer iteration counts and machine-precision artifact reload differences
are archived as diagnostics. They are not used as scientific claims or
cross-platform pass/fail criteria.

## Canonical environment

The canonical reference is generated with the pinned image recorded in
`Dockerfile.canonical`, Python 3.13.5, the full transitive package lock in
`requirements-canonical.txt`, single-threaded numerical-library environment
variables and fixed hash/source-date settings.

## Portability mode

On Windows or a non-canonical Linux host, run:

```text
python scripts/check_clean_reproduction.py --mode portability
```

A portability PASS requires exact discrete identity, field-specific numerical
agreement and visual-equivalence checks. It does not require byte-identical
floating-point CSV/JSON output or byte-identical PDF/SVG files.

## Manifest convention

A manifest never includes its own digest, nor the digest of the archive that
contains it. The enclosing package-level checksum manifest covers those files.
This explicit convention replaces impossible self-referential ZIP hashing.
