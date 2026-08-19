# Traceable batch decision layer for MOF activation-stability screening

> **PUBLIC RELEASE REPOSITORY**  
> This repository contains the versioned reproducibility package for the study.
> The scientific code, data, models, results and traces remain frozen to the
> audited scientific payload. The immutable `v1.0.0` release is archived by
> Zenodo at DOI `10.5281/zenodo.22016145`. Patch release `v1.0.1` corrects
> public-release metadata only and does not alter the scientific payload.

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

## Reproducibility assets

- exact environments: `environment-lock.yml`, `requirements-canonical.txt`,
  `Dockerfile.canonical`;
- frozen and derived data: `data/processed/`, `data/derived/`;
- model records: `models/`;
- reproduction outputs and statistical diagnostics: `results/reproduction/`;
- figure/table source data: `source_data/`;
- source code and scripts: `src/`, `scripts/`;
- tests: `tests/`;
- decision traces and archived-implementation replay checks: `traces/`.

The scientific snapshot has passed an exact private-checkout canonical
reproduction, scientific invariant gate, unit/integration tests, five-trace
archived-implementation replay, semantic regression, repository validation and
tracked-tree cleanliness check. Replay evidence remains exact
archived-implementation replay, not independent implementation validation.

## Funding and competing interests

The authors have confirmed that no relevant funding was received for this work
and that they have no competing interests. The manuscript records these
statements directly.

## Release status

The repository is public and uses immutable GitHub releases. Version-specific
software archives are created through the connected Zenodo integration. Use the
DOI shown on the corresponding Zenodo record when citing a specific release.
The `v1.0.0` scientific payload remains preserved at its immutable release and
Zenodo record; `v1.0.1` changes release metadata only.

## Licensing and third-party attribution

Code is under MIT (`LICENSE-CODE`). Packaged derived data and figure source data
are under CC BY 4.0 (`LICENSE-DATA`). See `THIRD_PARTY_NOTICES.md` and
`data/upstream/upstream_manifest.csv` for the upstream MOFSimplify dataset DOI,
version, license and attribution boundary.
