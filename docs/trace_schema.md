# Stepwise action-trace schema

The v49 primary trace set consists of exactly five files named
`traces/action_traces/*_test_primary_cost_contract_trace.json`. Their selector
configuration uses the locked cost budget and `risk_budget=null`. The legacy
`0.35` risk-cap stress test is separate and is not archived as a primary trace
set.

Each JSON trace contains:

- schema version and run identifier;
- evaluation split and selector identity;
- frozen selector configuration;
- hashes for the DOI-complete canonical source manifest, derived final-split
  feature table, scored candidate manifest, model artifact, selector code and
  exact environment lock;
- the authoritative v49 pre-test lock path
  (`results/reproduction/tuning/v49_locked_parameters_before_test.json`),
  SHA-256 and full locked payload;
- runtime Python metadata;
- selected candidates and logged alternatives;
- one record per greedy decision event;
- exact stepwise-decision, selection-event and stop-event counts;
- requested-batch completion as both `full_batch` and an exact
  `full_batch_status` value;
- summary metrics and the exact replay command.

Each selected event records:

- candidate identifier;
- feasibility counts plus cost and risk exclusion counts (primary risk
  exclusions are zero because its risk budget is null);
- cost and descriptive risk accumulation before and after selection, plus the
  remaining cost budget and null remaining primary risk budget;
- predicted success, entropy proxy, raw and normalized proxy cost, risk score and diversity bonus;
- score, uncertainty, cost and diversity utility components;
- the five highest-ranked feasible candidates at that event.

Exact replay first verifies that the frozen trace configuration is derived from
the embedded authoritative pre-test lock, including policy terms, budgets,
cost scale, batch size and the locked seed rule. It then compares the complete
selected and alternative records, every stepwise cost/risk state, priority
component and stop event, plus the explicit event counts; floating-point JSON
round trips are accepted only within `1e-12`.

The schema documents the implemented greedy marginal procedure. It does not
claim global batch optimization or independent replay implementation. Trace
content and the exact/discrete event fields are also covered by the semantic
regression reference.

## Reader-facing summary terminology

The trace JSON retains the machine field `hits` for schema compatibility. In
reader-facing text, this value should be described as the number of
**benchmark-positive solvent-removal labels** among selected candidates. It is
not evidence of prospectively validated experimental success. Likewise,
`proxy_total_cost` and related budget/yield fields live on a
**descriptor-complexity proxy-cost** scale rather than a measured synthesis-cost
scale. PortfolioBatch is an illustrative deterministic policy in this
retrospective benchmark, not a demonstrated superior optimizer.
