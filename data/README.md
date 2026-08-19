# Data

The package contains processed MOFSimplify-derived descriptor tables, a
canonical all-split source manifest, a derived relocked split, generated scores
and result tables. It does not redistribute the complete upstream archive.
Upstream DOI, version, archive checksum, license and file identities are
recorded in `upstream/upstream_manifest.csv` and `THIRD_PARTY_NOTICES.md`.

## Frozen and derived assets

- Frozen activation input:
  `processed/full_public_stability_features.csv`
- Canonical source manifest:
  `processed/full_public_stability_source_manifest.csv`
- Derived relocked split:
  `derived/full_public_stability_features_relocked.csv`

The reproduction runner must not overwrite the frozen activation input.

## Complete activation source-DOI boundary

The canonical manifest contains DOI values for every training, calibration and
test provenance row. It preserves the v45A2R2 assignment in
`original_model_split` and records active v45A3 membership in
`final_evaluation_split`. Held rows whose DOI also appears in training are
retained as `excluded_train_source_overlap` and are not scored. Active
training, calibration and test DOI coverage is 100%, with all three pairwise
DOI-group overlaps equal to zero.

The authenticated upstream raw CSV is not redistributed. Its immutable commit,
Git-blob SHA-1, raw SHA-256, dataset DOI and reconstruction instructions are in
`provenance/mofsimplify_training_source_provenance.json`.
