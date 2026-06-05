# Thesis E3 statistical-strengthening update

This package updates the thesis after the segmented-vs-monotone and uptake-controlled analyses of E3.

## Files

- `chapters/03_methodology_e3_statistical_tests_update.tex`  
  Adds the statistical testing methodology: window-vs-late bootstrap, segmented-vs-monotone comparison, and uptake-controlled regression.

- `chapters/04_results_e3_statistical_strengthening_update.tex`  
  Adds the E3 statistical results, including 160M-only and pooled 160M+410M findings.

- `chapters/05_discussion_e3_statistical_scope_update.tex`  
  Clarifies the interpretation and limits: short-horizon sensitive period, not strict/lifetime critical period.

- `figures/e3_statistics/`  
  Figure files copied from the pooled 160M+410M statistical analysis.

- `tables/e3_statistics/`  
  CSV tables from the pooled statistical analysis.

## Recommended integration

1. Paste the methodology snippet into Chapter 3 after the E3 intervention protocol.
2. Paste the results snippet into Chapter 4 after the main E3 stage-wise results.
3. Paste the discussion snippet into Chapter 5 under limitations / interpretation.
4. Copy the `figures/e3_statistics/` folder into your thesis `figures/` directory, preserving paths used in LaTeX.
5. Copy the `tables/e3_statistics/` folder into your thesis artifact directory.

## Headline conclusion

The strongest supported claim is:

> Uptake-normalised clean retention is higher inside the independently identified E1/E2 reorganisation window than at late checkpoints, remains higher after controlling for uptake, and is better fit by a segmented model with a break around step1000 than by a smooth monotone log-step trend.

Use “short-horizon sensitive-window evidence”, not “strict critical period proven”.
