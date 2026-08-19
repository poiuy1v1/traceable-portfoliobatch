# Action traces and replay checks

Five primary test decisions are archived: top-score, uncertainty, cost-aware,
PortfolioBatch and a seeded random baseline. Every trace uses the same locked
cost/risk contract and contains stepwise greedy decision records. Replay checks
re-execute the archived implementation and compare hashes, selected IDs,
stepwise selected IDs, logged alternatives and metrics.
