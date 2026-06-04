# E2 multi-model functional-grounding review

## Verdict

E2 now provides a stronger and more stable functional-grounding result than the earlier 70M/160M pilot. The run covers Pythia-70M, 160M, 410M, and 1B on the same ten checkpoints used in E1. The most reliable functional signals are fixed-text next-token NLL and syntax-regularity log-probability margins.

The result should be framed as **observational functional grounding**, not causal evidence. It supports the claim that the E1 weight-space reorganisation overlaps with rapid functional change, but it does not establish a sensitive or critical period.

## Coverage

Each model has 410 metric rows over 10 checkpoints. The checkpoint grid is:

step0, step128, step512, step1000, step2000, step3000, step8000, step16000, step64000, step143000.

The probe set consists of:
- fixed-text next-token NLL;
- arithmetic small multiple-choice probes;
- sequence-completion probes;
- syntax-regularity probes.

## Main findings

### Fixed-text NLL

Fixed-text NLL improves rapidly by step2000 across all four models. The fraction of total step0-to-final NLL reduction achieved by step2000 is:

| Model | Step0 NLL | Step2000 NLL | Final NLL | Fraction of total NLL reduction by step2000 |
|---|---:|---:|---:|---:|
| Pythia-70M | 11.036 | 5.333 | 4.930 | 0.934 |
| Pythia-160M | 11.059 | 5.207 | 4.620 | 0.909 |
| Pythia-410M | 10.995 | 5.119 | 4.034 | 0.844 |
| Pythia-1B | 11.042 | 4.892 | 3.805 | 0.850 |

The largest single NLL improvement occurs at 0->128 for all models. This is earlier than the strongest E1 spectral reorganisation, so fixed-text NLL should be interpreted as coarse early language-model improvement rather than a precise readout of the E1 boundary.

### Syntax regularities

Syntax-regularity margins align more closely with the E1 window. Averaged across models, the largest margin improvements occur in:

- 128->512: mean delta 2.212
- 512->1000: mean delta 1.678
- 1000->2000: mean delta 1.828

The margin therefore improves mostly in the 128->2000 interval, overlapping the E1 candidate developmental window.

The fraction of final syntax-margin improvement achieved by step2000 is:

| Model | Step0 | Step2000 | Final | Fraction by step2000 |
|---|---:|---:|---:|---:|
| Pythia-70M | -0.073 | 4.200 | 4.998 | 0.843 |
| Pythia-160M | -0.055 | 4.839 | 7.001 | 0.694 |
| Pythia-410M | -0.693 | 5.712 | 7.576 | 0.775 |
| Pythia-1B | -0.208 | 7.440 | 7.728 | 0.964 |

### Arithmetic and sequence completion

Arithmetic and sequence-completion accuracies remain noisy. The item counts are small, and accuracy often fluctuates across checkpoints. These probes should not be used as headline E2 evidence unless expanded in a later run.

## Thesis interpretation

E2 should be reported as **moderate functional grounding**:
- fixed-text NLL shows strong early improvement across all model sizes;
- syntax margins improve consistently within the E1 128--2000 window;
- arithmetic and sequence-completion probes are noisy and should be treated as exploratory.

The result strengthens the thesis narrative:

E1: reproducible early weight-space reorganisation.
E2: the same early period overlaps with rapid functional/log-likelihood improvement.
E3: the next step is a causal injection-and-retention test.
