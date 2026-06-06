# C1 runner NaN-stability update

This replacement runner targets the observed failure where injection step 1 was finite and injection step 2 became NaN.

Changes:
- supports `force_float32: true` at top level, `runtime.force_float32`, or `model.force_float32`;
- uses conservative AdamW defaults (`eps=1e-6`, `betas=(0.9,0.95)`, `foreach=False`);
- checks parameters immediately after optimizer steps when `finite_param_check_tensors` is set;
- supports structured label-gate prompts ending in `Decision:` without appending a second `Response:` marker;
- uses `REFUSE`/`COMPLY` labels when supplied as `signal.refusal_label` / `signal.compliance_label`;
- supports `train_templates` / `probe_templates` with `{item}` and `{request}` fields.

Run the smoke test first. If it still NaNs with fp32 + lr=1e-8 + batch_size=1, the issue is likely in the loop/model state rather than the alignment task strength.
