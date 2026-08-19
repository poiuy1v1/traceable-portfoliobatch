# Data dictionary

## Activation scored manifest

- `candidate_id`: stable identifier within the packaged benchmark.
- `original_model_split`: preserved v45A2R2 membership.
- `final_evaluation_split`: `calibration` or `test` for scored rows;
  `excluded_train_source_overlap` rows exist only in provenance tables.
- `source_doi`, `source_doi_normalized`, `source_group_id`: authenticated
  publication DOI identity used for all activation source groups.
- `raw_source_file`, `raw_source_row`, `raw_refcode`: upstream provenance.
- `outcome_binary`: retrospective solvent-removal / activation label.
- `predicted_success`: train-only logistic model score.
- `uncertainty`: normalized binary entropy of the score.
- `proxy_total_cost`: descriptor-complexity proxy cost defined as 1 plus mean absolute train-standardized descriptor value; it is not laboratory or monetary cost.
- `risk_score`: `0.5*(1-predicted_success)+0.5*uncertainty`; a
  benchmark-defined, uncalibrated proxy retained descriptively in the primary
  cost-only analysis and used as a hard cap only in the separate legacy `0.35`
  stress test.

## Locked parameters

`results/reproduction/tuning/v49_locked_parameters_before_test.json` is the
authoritative pre-test record of the training-selected logistic C,
calibration-selected cost-only PortfolioBatch weights, feature/hash identities,
cost scale, the locked per-item cost budget, the null primary risk budget, the
legacy `0.35` stress-only contract and the 100/20/10 random design. The
compatibility record `locked_parameters.json` stores its SHA-256. Within the
final locked evaluation pipeline, test labels are not used for model or policy
selection.

## Robustness outputs

- `test_batch_size_sensitivity.csv`: full-test results for batch sizes 5, 10, 20.
- `test_policy_comparison_primary_cost_only.csv`: primary five-method final-test
  comparison under the shared locked cost budget and no hard risk cap.
- `random_baseline_primary_cost_only_raw.csv` and
  `random_baseline_primary_cost_only_summary.csv`: 100 primary-contract random
  feasible selections per batch size and their summaries.
- `test_pool_resampling_primary_cost_only_raw.csv` and
  `test_pool_resampling_primary_cost_only_summary.csv`: 20 source-DOI-group 50%
  test-pool resamples, with 10 random seeds nested within each pool and
  requested batch size.
- `diagnostics/test_policy_risk_stress_legacy_0p35.csv`: separate, unretuned
  legacy hard-risk-cap stress test; it is not a primary comparison output.

## Reader-facing terminology

- `hits`: machine-readable count of benchmark-positive solvent-removal labels
  among the selected candidates. Reader-facing prose should use
  **benchmark-positive labels**, not "successful experiments" or an unqualified
  "stable hits" claim.
- `proxy_total_cost`, `cost_budget_per_item`, and
  `proxy_cost_normalized_yield`: quantities defined on the
  **descriptor-complexity proxy-cost** scale. Reader-facing prose should not
  imply synthesis cost, reagent price, experimental time or monetary expense.
- `risk_score`: uncalibrated benchmark proxy. Reader-facing prose should not
  call it a failure probability.

## Retrospective interpretation

The final v49 selector comparison is a retrospective descriptive benchmark.
Earlier development iterations had already inspected final-test behavior before
the v49 cost-only protocol was designated as primary. The pre-test lock remains
important because it prevents test labels from fitting the model or selecting
policy parameters within the final implementation, but it does not create an
untouched-test history.
