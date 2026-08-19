# Descriptive selector diagnostics for v49R1

This document is a **post-lock descriptive audit** computed from already locked
v49 outputs. It does not change model fitting, policy tuning, budgets, selector
implementation, test membership or the primary protocol. The frequencies below
are not p-values and are not used to choose parameters or methods.

## Full-pool random reference, batch size 5

The 100 cost-only random runs have mean benchmark-positive-label count
`3.00` (SD `1.128`) and mean descriptor-complexity
proxy-cost-normalized yield `0.362525` (SD `0.144090`). The
benchmark-positive-label count distribution is `{"0": 2, "1": 7, "2": 22, "3": 35, "4": 26, "5": 8}`.

| method | benchmark-positive labels | random fraction with labels >= method | proxy-cost-normalized yield | random fraction with yield >= method |
| --- | ---: | ---: | ---: | ---: |
| top_score | 4 | 0.340000 | 0.394390 | 0.330000 |
| uncertainty | 4 | 0.340000 | 0.535457 | 0.120000 |
| cost_aware | 4 | 0.340000 | 0.571126 | 0.080000 |
| portfolio_batch | 3 | 0.690000 | 0.323540 | 0.630000 |

These are empirical tail **frequencies** from 100 archived random runs. They are
not formal significance tests.

## Paired DOI-group pool resampling

For each of 20 DOI-group pool resamples, each deterministic method is compared
with the mean of the 10 random seeds from that same pool. This controls the
comparison descriptively for the sampled candidate pool, but it is still not a
hypothesis test.

| batch | method | mean labels | mean paired difference vs pool-random mean | pools > / = / < random mean | mean proxy yield | mean paired proxy-yield difference | pools > / = / < random mean yield |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 5 | top_score | 3.250000 | 0.280000 | 12 / 2 / 6 | 0.325151 | -0.031565 | 8 / 0 / 12 |
| 5 | uncertainty | 4.100000 | 1.130000 | 18 / 0 / 2 | 0.555579 | 0.198863 | 20 / 0 / 0 |
| 5 | cost_aware | 3.100000 | 0.130000 | 9 / 2 / 9 | 0.447938 | 0.091222 | 15 / 0 / 5 |
| 5 | portfolio_batch | 3.050000 | 0.080000 | 8 / 1 / 11 | 0.358801 | 0.002085 | 9 / 0 / 11 |
| 10 | top_score | 5.500000 | -0.635000 | 5 / 1 / 14 | 0.278504 | -0.090981 | 1 / 0 / 19 |
| 10 | uncertainty | 8.550000 | 2.415000 | 20 / 0 / 0 | 0.571932 | 0.202448 | 20 / 0 / 0 |
| 10 | cost_aware | 6.050000 | -0.085000 | 9 / 0 / 11 | 0.435876 | 0.066391 | 14 / 0 / 6 |
| 10 | portfolio_batch | 5.750000 | -0.385000 | 9 / 0 / 11 | 0.353332 | -0.016153 | 8 / 0 / 12 |
| 20 | top_score | 10.850000 | -1.100000 | 5 / 2 / 13 | 0.299548 | -0.057259 | 3 / 0 / 17 |
| 20 | uncertainty | 14.750000 | 2.800000 | 19 / 0 / 1 | 0.490722 | 0.133915 | 20 / 0 / 0 |
| 20 | cost_aware | 11.600000 | -0.350000 | 8 / 1 / 11 | 0.406082 | 0.049275 | 16 / 0 / 4 |
| 20 | portfolio_batch | 11.750000 | -0.200000 | 11 / 0 / 9 | 0.370413 | 0.013606 | 11 / 0 / 9 |

## Conservative reading

- The full-test `4/5` label counts from top-score, uncertainty and cost-aware are
  not rare relative to the archived random distribution (`34%` of random runs
  reached at least four benchmark-positive labels).
- Cost-aware has the largest full-test proxy-cost-normalized yield, but the
  empirical random tail frequency is `0.08`; this is descriptive, not a p-value.
- Across DOI-group pool resamples, uncertainty has the highest mean
  benchmark-positive-label count at batch sizes 5, 10 and 20, and exceeds the
  same-pool random mean in 18/20, 20/20 and 19/20 pools, respectively.
- PortfolioBatch does not show a consistent descriptive advantage over simpler
  selectors.

These observations support treating PortfolioBatch as an illustrative
deterministic trace/replay policy rather than as a demonstrated superior
optimizer.
