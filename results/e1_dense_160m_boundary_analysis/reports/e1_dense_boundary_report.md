# E1 dense boundary analysis report

- Metrics: `results/e1_dense_160m_all_checkpoints/raw/e1_spectral_metrics.csv`
- Rows: 88656
- Models: EleutherAI/pythia-160m-deduped
- Checkpoints: 154 (0 to 143000)
- Modules: 4
- Metrics: 12

## Top boundary-vote intervals

### EleutherAI/pythia-160m-deduped

- `1000->2000`: 19 votes, mean strength 0.4165
- `512->1000`: 9 votes, mean strength 6.204e+11
- `128->256`: 3 votes, mean strength 7.222e+11
- `64->128`: 3 votes, mean strength 1.111e+11
- `2000->3000`: 3 votes, mean strength 0.06405

## Top aggregate-strength intervals

### EleutherAI/pythia-160m-deduped

- `512->1000`: mean signed strength 1.163e+11, mean abs relative delta 1.163e+11, n=48
- `128->256`: mean signed strength 4.514e+10, mean abs relative delta 4.514e+10, n=48
- `64->128`: mean signed strength 6.944e+09, mean abs relative delta 6.944e+09, n=48
- `1000->2000`: mean signed strength 0.8243, mean abs relative delta 0.9217, n=48
- `256->512`: mean signed strength 0.7739, mean abs relative delta 0.7986, n=48
