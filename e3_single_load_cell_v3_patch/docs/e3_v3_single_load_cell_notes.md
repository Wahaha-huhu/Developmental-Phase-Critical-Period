# E3 v3 single-load cell runner

This patch implements the worked-example cell structure:

1. load the checkpoint once;
2. score base probes and controls;
3. inject the factual signal;
4. score uptake;
5. clone the injected model state;
6. run clean-continuation retention;
7. restore the injected state;
8. run each poison-budget degradation branch from the same injected state;
9. log summary, item-level metrics, degradation curves, and signal audits.

The runner is deliberately self-contained so it does not depend on older E3 helper code that may have contained the train/probe split bug.

Important sanity checks:

- factual probes query taught facts, not held-out unseen facts;
- control facts are disjoint from taught facts;
- probe template is not used in training templates;
- CSV schemas are fixed and quoted;
- `np.trapezoid`/`np.trapz` compatibility is handled in the analyzer.

The continuation corpus defaults to a deterministic synthetic generic corpus. For the final thesis run, replace it with a fixed real held-out corpus if available, but keep it identical across all stages.
