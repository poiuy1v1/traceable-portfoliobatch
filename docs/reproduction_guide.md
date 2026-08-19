# Reproduction and regression guide

## Environment

For the canonical reference, build `Dockerfile.canonical` or use the exact transitive lock in `requirements-canonical.txt`. `requirements-lock.txt` remains the compact direct-dependency list. Non-canonical hosts are portability environments.

## Full reproduction

```bash
python scripts/run_full_reproduction.py
```

The runner:

1. reads the frozen processed feature table without overwriting it;
2. selects logistic-regression `C` inside training using five-fold
   `StratifiedGroupKFold` by normalized source DOI, fixed `random_state=45`, an
   expanded grid and a one-standard-error rule;
3. requires every activation fit to converge and archives `n_iter_`, `max_iter` and convergence state as diagnostics;
4. fits the selected train model and scores only the preserved active
   calibration rows;
5. selects PortfolioBatch parameters only on calibration over the fixed
   27-configuration cost-only grid and writes a 20-run DOI-group-bootstrap
   stability table;
6. writes and hashes
   `results/reproduction/tuning/v49_locked_parameters_before_test.json`
   before any final test score or label metric;
7. runs the post-lock fixed-C source-overlap diagnostic, train-fitted dummy
   baselines and the 10,000-replicate final-test DOI-group bootstrap;
8. audits mathematical risk/cost feasibility, runs the primary shared
   cost-only comparison with no hard risk cap, and runs the legacy `0.35` risk
   cap only as a separate unretuned stress test;
9. runs batch-size, 100-run full-pool random and DOI-group pool-resampling
   analyses, nesting 10 random seeds within each of 20 resampled test pools;
10. computes candidate-ID, raw-refcode, source-row, normalized DOI and
   source-group checks pairwise across active train/calibration/test;
11. writes five `*_test_primary_cost_contract_trace.json` stepwise test traces
    under `risk_budget=None` and exact archived-implementation replays;
12. selects thermal ridge alpha by train-internal cross-validation and evaluates
    the fixed validation set; and
13. regenerates figures, source-data tables, a classed artifact manifest and a
    semantic regression snapshot.

The primary final-test comparison is
`results/reproduction/test_policy_comparison_primary_cost_only.csv`. The legacy
risk stress test is isolated at
`results/reproduction/diagnostics/test_policy_risk_stress_legacy_0p35.csv`; it
is not a second tuned primary analysis.

### Interpretation chronology

The lock chronology above is a reproducibility statement about the **final v49
implementation**. It is not a claim that the final test set was historically
untouched: earlier development iterations had already inspected final-test
behavior before the cost-only selector comparison was designated as the v49
primary protocol. Consequently, selector outcomes should be interpreted as
retrospective descriptive benchmark results rather than as a preregistered
confirmatory comparison. The fixed-C source-overlap analysis, random empirical
tails and pool-resampling summaries are diagnostic/descriptive and are not used
to retune C, policy parameters, budgets or protocol choice.

The relocked activation table is written to:

```text
data/derived/full_public_stability_features_relocked.csv
```

The frozen input remains:

```text
data/processed/full_public_stability_features.csv
```

## Regression checks

Canonical Linux container:

```bash
python scripts/check_clean_reproduction.py --mode canonical
python scripts/check_regression_integrity.py --mode canonical
```

Windows or another non-canonical host:

```bash
python scripts/check_clean_reproduction.py --mode portability
python scripts/check_regression_integrity.py --mode portability
```

The comparison classes and field-specific tolerances are defined in
`results/reproduction/regression_contract.json`. Exact discrete identities must
match everywhere. Floating-point metrics use declared portability tolerances.
CSV/JSON byte identity is required only in the pinned canonical container. PNG
assets use a bounded rendered-pixel comparison; PDF/SVG hashes are diagnostic.
Optimizer iteration counts are recorded but do not define cross-platform
scientific equivalence.

Runtime diagnostics can be collected with:

```bash
python scripts/collect_runtime_diagnostics.py --output runtime_diagnostics.json
```

## Controlled training DOI backfill

Download the official upstream MOFSimplify file:

```text
model/solvent/ANN/dropped_connectivity_dupes/train.csv
```

Then run:

```bash
python scripts/backfill_train_source_doi.py /path/to/train.csv \
  --require-official-git-blob-sha
```

The helper validates the recorded Git blob-object SHA-1 using Git object
semantics; this is intentionally distinct from a raw-file SHA-1. An optional
raw-file SHA-256 may be enforced with `--expected-raw-sha256`.

The helper preserves calibration/test DOI values and writes a DOI-complete
all-split source manifest. It fails on blank, unresolved, ambiguous or
conflicting DOI mappings. It then applies the preserved conservative
source-overlap rule: training
membership is unchanged, held rows from a DOI also present in training are
retained as `excluded_train_source_overlap`, and all other calibration/test
assignments remain unchanged. The audit JSON includes the exact runner command.

The official raw CSV is intentionally not packaged. Reconstruct it only from
the immutable source recorded in
`data/provenance/mofsimplify_training_source_provenance.json` and require both
the Git-blob SHA-1 and raw SHA-256 checks.

## Decision semantics

Replay is exact re-execution with the archived implementation and locked
environment. It checks hashes, selected identifiers, stepwise events,
alternatives and metrics. It is not an independent software implementation.
The replay set is the five primary cost-contract traces in
`traces/action_traces/*_test_primary_cost_contract_trace.json`; the legacy risk
stress test is not archived as a primary trace set.
