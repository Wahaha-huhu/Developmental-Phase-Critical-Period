#!/usr/bin/env python3
"""Repair older E3 raw CSVs that were appended with changing schemas.

The first E3 MVP writer could create CSVs whose header had 10 columns but
post_degradation rows had 12 values. This script rewrites the summary and
item metrics CSVs with stable schemas and saves a .bak copy first.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

METRICS_FIELDNAMES = [
    "experiment_id", "model", "stage", "step", "seed", "signal_type",
    "event", "metric", "value", "timestamp", "poison_budget", "degradation_steps",
]

ITEM_FIELDNAMES = [
    "experiment_id", "model", "stage", "step", "seed", "signal_type",
    "event", "timestamp", "poison_budget", "degradation_steps",
    "item_id", "fact_id", "prompt", "gold_value", "answer_index",
    "predicted_index", "correct", "gold_logprob", "best_wrong_logprob",
    "gold_logprob_margin", "choice_lengths", "choices",
]


def repair(path: Path, fieldnames: list[str]) -> None:
    if not path.exists():
        print(f"Missing, skipping: {path}")
        return
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    with backup.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            old_header = next(reader)
        except StopIteration:
            return
        rows = []
        for row in reader:
            if not row:
                continue
            # Most old rows are header length; degradation rows have extra fields.
            cols = list(old_header)
            if len(row) > len(cols):
                for name in fieldnames:
                    if len(cols) >= len(row):
                        break
                    if name not in cols:
                        cols.append(name)
                while len(cols) < len(row):
                    cols.append(f"extra_{len(cols)}")
            if len(row) < len(cols):
                row = row + [""] * (len(cols) - len(row))
            d = dict(zip(cols, row))
            rows.append({k: d.get(k, "") for k in fieldnames})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Repaired {path} (backup: {backup})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/e3_critical_period_intervention")
    args = ap.parse_args()
    root = Path(args.root)
    repair(root / "raw/e3_factual_metrics.csv", METRICS_FIELDNAMES)
    repair(root / "raw/e3_factual_item_metrics.csv", ITEM_FIELDNAMES)


if __name__ == "__main__":
    main()
