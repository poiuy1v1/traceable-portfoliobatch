# Version 1.0.2

This release publishes a bounded correction to the secondary calibration
policy-stability analysis.

In v1.0.1, 80% of publication-source DOI groups were sampled with replacement
and repeated group occurrences duplicated their candidate rows. Version 1.0.2
retains the immutable v1.0.1 evidence for provenance, adds an explicit
reproduction of the defect, and replaces the interpreted stability result with
80% publication-source-group subsampling without replacement.

Corrected evidence:

- 109 of 136 DOI groups per subsample, seeds 0–19;
- exact locked-tuple recovery: 3/20;
- `delta = 0`: 18/20;
- full five-candidate completion: 20/20;
- five unique selected candidates: 20/20.

The correction is limited to this secondary stability analysis. The model,
train/calibration/final-evaluation identities, selected `C=0.001`, policy tuple
locked on the original calibration pool, final-test metrics, primary selector
outputs, five archived traces, figures and thermal analysis are unchanged.

The version-specific Zenodo DOI is provided on the Zenodo record created from
this immutable GitHub release.
