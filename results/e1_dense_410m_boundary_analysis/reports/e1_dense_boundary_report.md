# E1 dense boundary analysis report

- Metrics: `results/e1_dense_410m_early_cheap/raw/e1_spectral_metrics.csv`
- Rows: 21792
- Models: EleutherAI/pythia-410m-deduped
- Checkpoints: 19 (0 to 8000)
- Modules: 4
- Metrics: 12

## Top boundary-vote intervals

### EleutherAI/pythia-410m-deduped

- `1000->2000`: 18 votes, mean strength 0.4385
- `512->1000`: 8 votes, mean strength 2.708e+11
- `256->512`: 6 votes, mean strength 2.292e+11
- `2000->3000`: 4 votes, mean strength 0.1129
- `128->256`: 3 votes, mean strength 1.389e+11

## Top aggregate-strength intervals

### EleutherAI/pythia-410m-deduped

- `512->1000`: mean signed strength 4.514e+10, mean abs relative delta 4.514e+10, n=48
- `256->512`: mean signed strength 2.865e+10, mean abs relative delta 2.865e+10, n=48
- `128->256`: mean signed strength 8.681e+09, mean abs relative delta 8.681e+09, n=48
- `64->128`: mean signed strength 1.736e+09, mean abs relative delta 1.736e+09, n=48
- `1000->2000`: mean signed strength 1.213, mean abs relative delta 1.297, n=48
