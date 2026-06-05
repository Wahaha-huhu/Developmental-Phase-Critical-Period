# E3 thesis update package

This package contains thesis-ready E3 updates after the final Pythia-160M run and reduced Pythia-410M scale check.

## Files

- `03_methodology_e3_final_design_update.tex`
  - Insert into Chapter 3. Explains the E3 design, synthetic factual signal, fixed Pile continuation corpus, conflicting-fact degradation branch, uptake-normalised metrics, and scope.

- `04_results_e3_final_update.tex`
  - Insert into Chapter 4 after E1/E2. Reports the final 160M result and 410M scale-check result.

- `05_discussion_e3_scope_update.tex`
  - Insert into the Discussion chapter. Clarifies the claim as short-horizon sensitive-period evidence, not a full biological critical period.

- `figures/e3/`
  - Figure files copied from the E3 review packages.

- `tables/e3/`
  - Summary CSVs copied from the E3 review packages.

## Required LaTeX figure paths

Copy `figures/e3/` into your thesis `figures/e3/` folder. The `.tex` files refer to paths like:

```latex
\includegraphics{figures/e3/e3_normalized_retention_margin_by_stage.png}
```

## Suggested thesis structure

- Chapter 3: add `03_methodology_e3_final_design_update.tex` after E1/E2 methodology.
- Chapter 4: add `04_results_e3_final_update.tex` after the E1/E2 results.
- Chapter 5 or Discussion: add `05_discussion_e3_scope_update.tex` after mechanism/toy discussion.

## Main result wording

Use: "short-horizon sensitive-period evidence".

Avoid: "critical period proven" or "lifetime persistence established".
