# Toy mechanism experiment plan: feature-subspace consolidation

## Purpose
Provide a controlled mechanism that is sufficient to reproduce the thesis pattern:

1. early spectral outlier / subspace instability,
2. stable-rank collapse / feature-subspace consolidation,
3. stage-dependent durability of an injected signal.

The toy is explanatory support, not direct evidence about Pythia's internal mechanism.

## Recommended toy

### Data generator
Use a latent-topic next-token prediction task.

- Vocabulary size: 128 or 256.
- Latent topics: 8 or 16.
- Each sequence begins with a topic/task token.
- Subsequent tokens are sampled from a topic-specific distribution generated from a low-rank topic-token matrix.
- The true data distribution is therefore low-rank, and the true latent subspace is known.

### Model
Start with a 2-layer small transformer:

- hidden size: 128 or 256
- layers: 2
- heads: 4
- context length: 32 or 64
- causal LM loss

Also include a simpler MLP/matrix-factorization version if time allows. The MLP version is less LLM-like but easier to interpret.

## Known transition
Track true subspace recovery:

- compute the top-k subspace of the model's token/readout or hidden representation,
- compare with the true latent topic-token subspace using principal-angle cosine,
- define the toy transition as the interval where true subspace recovery rapidly increases.

This gives ground truth for whether the spectral indicators are locating a real structure-formation transition.

## Spectral indicators
Reuse the E1 metrics:

- stable rank,
- effective rank,
- spectral norm,
- MP outlier count where applicable,
- alpha/heavy-tail proxy,
- top-k singular-vector stability.

The toy is successful if these indicators peak near the known subspace-recovery transition.

## Intervention
At checkpoints before/during/after the toy transition:

- inject a small factual association or synthetic mapping,
- measure uptake,
- continue training on the original latent-topic distribution,
- measure normalized retention,
- optionally run a conflicting-association degradation attack.

## Expected pattern

- Before transition: signal may enter but is less integrated because stable features are not formed.
- During transition: signal is most durably integrated into the forming subspace.
- After transition: signal can still be learned, but retention/degradation-resistance is lower because it must overwrite a consolidated basis.

## Why this is better than the previous toy pilot
The previous toy pilot memorized training examples without held-out generalization. This design avoids that by making the ground truth a known latent subspace, not just test accuracy. Even if behavioural accuracy is noisy, subspace recovery provides a direct mechanistic transition target.

## Minimal deliverables

1. `toy_training_curve.csv`: loss and held-out NLL over training.
2. `toy_subspace_recovery.csv`: principal-angle recovery against true latent factors.
3. `toy_spectral_metrics.csv`: E1-style indicators.
4. `toy_intervention_retention.csv`: uptake and retention across stages.
5. `toy_mechanism_report.md`: whether indicators locate the known transition.
