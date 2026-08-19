# Known limitations

- The endpoint is solvent-removal / activation stability, not water,
  hydrolytic, acid/base or long-term cycling stability.
- On the final publication-source-disjoint test set, the activation model has
  weak, near-chance discrimination (ROC-AUC approximately 0.522), its DOI-group
  bootstrap interval spans 0.5, and it does not clearly exceed the
  training-fitted prior dummy baseline. It is not a state-of-the-art stability
  predictor.
- Binary entropy is an exploration proxy, not calibrated uncertainty.
- `proxy_total_cost` is a descriptor-complexity proxy cost derived from
  train-standardized descriptors, not measured laboratory, reagent, time or
  monetary cost.
- The risk proxy is a benchmark-defined, uncalibrated quantity, not a calibrated
  failure probability. The legacy `0.35` hard cap is retained only as a stress
  test and is mathematically infeasible for an exact five-candidate batch in the
  locked calibration and final-test pools.
- Active activation training, calibration and test rows are fully DOI-covered
  and pairwise publication-source-DOI-group-disjoint. This claim does not extend
  to chemical-family, topology or structural-similarity independence, and it
  does not extend to the secondary thermal analysis.
- Chemical-family leakage cannot be evaluated because complete topology,
  metal-node and linker-family annotations are unavailable.
- The study is retrospective. Earlier development iterations inspected final-
  test behavior before the v49 primary cost-only protocol was designated. The
  final v49 lock prevents test-label use for model fitting and policy-parameter
  selection within that implementation, but it does not establish a
  preregistered, untouched-test or historically single-use evaluation.
- Policy parameters vary across group-bootstrap calibration samples; the exact
  locked tuple is not uniquely stable and should be treated as one deterministic
  benchmark instantiation on a broad calibration plateau.
- PortfolioBatch is retained as an illustrative deterministic policy for
  trace/replay auditing. v49 does not support a claim that it outperforms the
  simpler selectors.
- The machine field `hits` counts benchmark-positive solvent-removal labels.
  It is not a count of prospectively validated wet-lab successes. The underlying
  MOFSimplify solvent-removal labels are literature-derived, NLP-assigned
  benchmark annotations rather than candidate-level independently revalidated
  experimental ground truth. In the original MOFSimplify Technical Validation,
  manual review of a random 100-MOF subset reported 78 clearly correct labels,
  two incorrect assignments, and 20 cases where the extracted sentences did not
  support a definitive solvent-removal-stability judgment. These validation
  counts characterize that audited subset and should not be treated as a known
  error rate for every Paper13 benchmark label.
- Selector comparisons are descriptive and depend on one public dataset.
- Pool resampling measures candidate-pool sensitivity but does not substitute
  for a prospective laboratory campaign.
- Exact replay uses the same archived implementation; it is not an independent
  reimplementation.
- Byte-identical CSV/JSON reproduction is a canonical-container property. Other
  platforms are assessed by exact discrete identity, field-specific numerical
  tolerances and visual equivalence.
- Optimizer iteration counts and sub-machine-precision reload differences may
  vary across numerical runtimes and are diagnostics rather than scientific
  claims.
