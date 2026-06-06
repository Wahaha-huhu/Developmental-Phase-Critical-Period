# C1 token-matched continuation notes

This update resolves the C1 continuation budget in tokens rather than arbitrary local optimizer steps.

For each arm:

```text
target_continuation_tokens = max(0, endpoint_step - inject_step) * pythia_tokens_per_step
local_tokens_per_step = batch_size * sequence_length * grad_accum_steps
continuation_steps = ceil(target_continuation_tokens / local_tokens_per_step)
```

This preserves the design decision that the in-window arm receives the token-distance from its
injection step to the common consolidation endpoint, while the post-hoc arm receives near-zero
continuation. The continuation corpus remains the same fixed held-out Pile sample across arms;
only the number of consumed tokens differs because the arms begin at different global steps.

The generated CSV is an audit table and should be included in thesis/reproducibility artifacts.

Important caveat: matching token count is a faithful approximation of development length, not a
bit-exact replay of the original Pythia dataloader or optimizer state.
