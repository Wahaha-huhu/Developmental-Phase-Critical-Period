#!/usr/bin/env python3
"""Create the thesis-clean solid_results directory layout."""
from pathlib import Path

DIRS = [
    "solid_results",
    "solid_results/configs",
    "solid_results/logs",
    "solid_results/figures",
    "solid_results/tables",
    "solid_results/reports",
    "solid_results/e1_dense_indicator_panels_160m",
    "solid_results/e1_dense_indicator_panels_1b",
]


def main() -> None:
    for d in DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"created/exists: {d}")


if __name__ == "__main__":
    main()
