#!/usr/bin/env python3
from pathlib import Path

ROOTS = [
    "solid_results",
    "solid_results/step0_artifact_overlay",
    "solid_results/step1_dense_durability",
    "solid_results/e1_dense_indicator_panels_160m",
    "solid_results/e1_dense_indicator_panels_1p4b",
    "solid_results/logs",
    "solid_results/configs",
]
for r in ROOTS:
    Path(r).mkdir(parents=True, exist_ok=True)
    print("created", r)
