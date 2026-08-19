# Source modules

The reusable implementation is in `src/traceable_portfoliobatch/`:

- `modeling.py`: train-only activation logistic model, convergence-enforced fitting, thermal ridge baseline, model artifacts and score regeneration;
- `policies.py`: deterministic baselines and the PortfolioBatch selector;
- `tracing.py` / `replay.py`: policy-agnostic action traces with hashes and exact replay;
- `leakage.py`: exact candidate-ID, raw-refcode and source-row split audit;
- `io_utils.py`: canonical JSON/CSV serialization and raw/Git-object hashing helpers;
- `regression.py`: generated-artifact byte manifests and scientific semantic snapshots;
- `workflow.py`: end-to-end reproduction from frozen inputs to derived outputs;
- `figures.py`: deterministic regeneration of all main and supplementary figures.
