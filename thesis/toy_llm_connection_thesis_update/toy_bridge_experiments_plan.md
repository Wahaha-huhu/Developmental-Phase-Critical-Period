# Optional follow-up experiments to strengthen the toy-to-LLM bridge

The current low-rank toy should be presented as a controlled mechanism vignette, not as an architectural replica of Pythia. If time and GPU/CPU resources permit after E3, the following follow-ups would make the connection stronger.

## 1. Textualised low-rank transformer bridge
Generate the same low-rank teacher matrix, but express each observation as a short token sequence:

`Entity E17 relation R04 value: <bin_23>`

or

`Question: relation R04 of entity E17? Answer: <bin_23>`

The target value can be discretised into bins or represented as one of a fixed set of value tokens. Train a small transformer on the textualised observations and evaluate held-out entity--relation pairs. This would test whether the same latent-subspace dynamics survive in a sequence-model setting.

Expected use: optional bridge between the clean matrix-factorisation toy and Pythia.

## 2. Frequency/SNR ablation
Run the low-rank toy under several rare-feature sampling rates, for example 0.05, 0.12, 0.25, and balanced 0.50. The prediction is that stronger frequency imbalance produces more separated acquisition: frequent held-out entries improve earlier, rare entries improve later, and spectral indicators show a more extended consolidation period.

Expected use: supports the claim that ordered phase-like dynamics can arise from frequency/SNR structure.

## 3. Rank/noise ablation
Vary teacher rank and observation noise. Higher rank or higher noise should delay subspace recovery and broaden the transition. If spectral boundary estimates move with these controlled difficulty parameters, that strengthens the interpretation that the indicators respond to structure formation rather than arbitrary training time.

Expected use: supports the mechanism that phase timing depends on latent-structure difficulty.

## 4. Exception-injection ablation
Inject exceptions before, during, and after the toy's subspace-recovery transition. Compare retention of exceptions aligned with the learned subspace versus exceptions orthogonal/conflicting with it. A stronger mechanism result would show that exceptions aligned with forming structure integrate better, while conflicting exceptions are more easily overwritten after consolidation.

Expected use: connects the toy more directly to E3's factual-injection intervention.
