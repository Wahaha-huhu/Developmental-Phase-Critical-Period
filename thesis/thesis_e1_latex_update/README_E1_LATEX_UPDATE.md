# E1 LaTeX update

This package contains LaTeX updates after the multi-model E1 result across Pythia-70M, 160M, 410M, and 1B.

Main file:
- `04_results_e1_updated.tex`: replaces the previous Results chapter. It updates E1 from a planned/preliminary experiment into a completed observational result, adds coverage and peak-interval tables, and keeps E2/E3+ scoped as planned next steps.

Optional supporting file:
- `01_introduction_e1_updated.tex`: lightly updates the introduction so it no longer calls E1 merely preliminary and instead frames E1 as the completed observational basis for later experiments.

Figure references used in `04_results_e1_updated.tex`:
- `figures/e1_multimodel_stable_rank_relative_early_8000.png`
- `figures/e1_multimodel_interval_mean_stable_rank_drop.png`

These figures are from `e1_multimodel_1b_review_summary.zip`; copy them into your thesis `figures/` directory before compiling. The LaTeX uses `\IfFileExists`, so the chapter will still compile with placeholders if the figures are not present.
