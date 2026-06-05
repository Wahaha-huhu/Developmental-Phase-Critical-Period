# E1 dense boundary analysis report

- Metrics: `results/e1_dense_1b_early_cheap/raw/e1_spectral_metrics.csv`
- Rows: 14528
- Models: EleutherAI/pythia-1b-deduped
- Checkpoints: 19 (0 to 8000)
- Modules: 4
- Metrics: 12

## Top boundary-vote intervals

### EleutherAI/pythia-1b-deduped

- `1000->2000`: 16 votes, mean strength 0.4053
- `512->1000`: 9 votes, mean strength 3.403e+11
- `256->512`: 6 votes, mean strength 6.667e+11
- `128->256`: 4 votes, mean strength 5.469e+11
- `2000->3000`: 3 votes, mean strength 0.09514

## Top aggregate-strength intervals

### EleutherAI/pythia-1b-deduped

- `256->512`: mean signed strength 8.333e+10, mean abs relative delta 8.333e+10, n=48
- `512->1000`: mean signed strength 6.38e+10, mean abs relative delta 6.38e+10, n=48
- `128->256`: mean signed strength 4.557e+10, mean abs relative delta 4.557e+10, n=48
- `64->128`: mean signed strength 1.302e+09, mean abs relative delta 1.302e+09, n=48
- `1000->2000`: mean signed strength 1.099, mean abs relative delta 1.233, n=48
