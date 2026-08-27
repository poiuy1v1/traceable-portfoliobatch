# Traceable batch decision layer for MOF activation-stability screening

> **PUBLIC RELEASE REPOSITORY**  
> This repository contains the versioned reproducibility package for the study.
> The immutable `v1.0.0` scientific release is archived by Zenodo at DOI
> `10.5281/zenodo.22016145`; `v1.0.1` corrected release metadata only and is
> archived at DOI `10.5281/zenodo.22030668`. Version `v1.0.2` publishes a bounded
> correction to the secondary calibration policy-stability analysis. The Zenodo
> concept DOI is `10.5281/zenodo.22016144`; use the version-specific DOI shown on
> the corresponding Zenodo record when citing a release.

This repository implements a retrospective MOFSimplify-derived activation /
solvent-removal benchmark. It separates a policy-agnostic trace layer from the
PortfolioBatch selector and uses a source-DOI-group-disjoint calibration/test
evaluation.

## Interpretation boundary

This is a **retrospective benchmark audit**, not a preregistered confirmatory
evaluation. Earlier development iterations had already inspected final-test
behavior before the final cost-only selector contract was designated as
primary. Within the frozen implementation, test labels are not used for model
fitting or policy-parameter selection, but the final selector table is
descriptive rather than a historically untouched confirmatory test.

PortfolioBatch is retained as an **illustrative deterministic policy** for
trace/replay auditing. The benchmark does not support a claim that
PortfolioBatch is superior to the simpler selectors. The reported stability
labels are benchmark annotations inherited from MOFSimplify, the reported cost
is a descriptor-complexity proxy rather than laboratory or monetary cost, and
the benchmark risk proxy is not a calibrated failure probability.

Sequential selection enforces **immediate cumulative feasibility** at each
step. Pool-level existence of an exact-size feasible batch does not guarantee
that every greedy or random path completes the requested size. In the archived
`k=5` random robustness runs, 99/100 full-pool runs and 198/200 DOI-group
pool-resampled runs completed all five slots.

## Evaluation protocol

1. Fit preprocessing and the activation model on the official 1,394-row
   training split.
2. Select logistic-regression regularization by five-fold train-internal
   `StratifiedGroupKFold` cross-validation by normalized source DOI with
   `shuffle=True` and `random_state=45`.
3. Preserve the final calibration/test assignment while excluding held rows
   whose DOI also occurs in training from calibration and final evaluation.
4. Select PortfolioBatch parameters only on the source-group calibration set.
5. Evaluate the frozen selectors under the primary shared cost-only contract.
6. Archive five primary decision traces and replay them with the same locked
   implementation and environment.

## v1.0.2 policy-stability correction

The immutable `v1.0.1` archive sampled 80% of calibration DOI groups with
replacement. Repeated group occurrences duplicated candidate rows in a
secondary policy-stability analysis. Version `v1.0.2` retains that historical
output for provenance and adds the corrected design:

- 109 of 136 publication-source DOI groups per subsample;
- subsampling without replacement, seeds 0–19;
- exact recovery of the original calibration-locked tuple: 3/20;
- `delta = 0`: 18/20;
- five-candidate completion and unique selected identities: 20/20.

This correction does not change the model, split identities, selected
`C=0.001`, the policy tuple locked on the original 158-row calibration pool,
final-test metrics, primary selector outputs, five archived traces, figures or
thermal analysis.

Reproduce the bounded correction with:

```bash
python scripts/recompute_v1_0_2_policy_stability.py --repo-root . --output-root .
python scripts/validate_v1_0_2_policy_stability.py --repo-root .
```

See `docs/v1_0_2_policy_stability_correction.md` and the correction manifest at
`results/reproduction/v1_0_2_correction_manifest.json`.

## Reproducibility assets

- exact environments: `environment-lock.yml`, `requirements-canonical.txt`,
  `Dockerfile.canonical`;
- frozen and derived data: `data/processed/`, `data/derived/`;
- model records: `models/`;
- reproduction outputs and statistical diagnostics: `results/reproduction/`;
- figure/table and corrected stability source data: `source_data/`;
- source code and scripts: `src/`, `scripts/`;
- tests: `tests/`;
- decision traces and archived-implementation replay checks: `traces/`.

Replay evidence remains exact archived-implementation replay, not independent
implementation validation.

## Funding and competing interests

The authors have confirmed that no relevant funding was received for this work
and that they have no competing interests.

## Licensing and third-party attribution

Code is under MIT (`LICENSE-CODE`). Packaged derived data and figure source data
are under CC BY 4.0 (`LICENSE-DATA`). See `THIRD_PARTY_NOTICES.md` and
`data/upstream/upstream_manifest.csv` for the upstream MOFSimplify dataset DOI,
version, license and attribution boundary.
