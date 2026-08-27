# v1.0.2 policy-stability correction

Version 1.0.2 corrects one bounded secondary robustness analysis. In v1.0.1,
publication-source DOI groups were sampled with replacement and every row block
for every sampled occurrence was concatenated. A source group drawn more than
once could therefore duplicate the same physical candidate in a nominal
five-candidate tuning batch.

The corrected design uses 80% publication-source-group subsampling **without
replacement**: 109 of 136 DOI groups, seeds 0–19, retaining all rows belonging
to each selected group. Candidate identities are unique in all 20 input subsets
and in all 20 selected five-candidate batches.

Corrected results:

- exact recovery of the original calibration-locked tuple: 3/20;
- `delta = 0`: 18/20;
- full five-candidate completion: 20/20;
- five unique selected candidate identities: 20/20.

The v1.0.1 with-replacement table is retained unchanged for historical
provenance and is superseded for interpretation by:

- `results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_stability.csv`;
- `results/reproduction/tuning/portfolio_policy_cost_only_group_subsampling_summary.json`;
- `source_data/policy_stability_unique_candidate_subsampling_source.csv`.

Run:

```bash
python scripts/recompute_v1_0_2_policy_stability.py --repo-root . --output-root .
```

The script first reproduces the archived v1.0.1 table and duplicate-candidate
mechanism, then regenerates the corrected evidence and a hash manifest. This
correction does not change the activation model, split membership, selected
`C`, the policy tuple locked on the original 158-row calibration pool, final-test
metrics, primary selector outputs, five primary traces, or thermal analysis.
