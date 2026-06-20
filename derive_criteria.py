"""
Derive data-grounded suitability ranges for the 'Cancer Treatment (General)' use-case.

Motivation
----------
The original ideal ranges were hand-picked literature values that did not match the
``Successful_Treatment`` label in this dataset: only ~82% of "all-in-range" syntheses are
actually successful, and the ranges miss ~half of the genuinely successful ones. That
mismatch is why the rule-based recommender could say "no adjustments required" next to a
model verdict of "Unsuccessful".

This script recovers the ranges that the success label actually implies, by taking the
5th-95th percentile of each property over the SUCCESSFUL subset. The result is reproducible,
citable, and makes "all in range" line up with the model's notion of success.

Run:
    python derive_criteria.py
It prints a ready-to-paste criteria dict plus precision/recall diagnostics.
Only the use-case backed by the ``Successful_Treatment`` label can be validated this way;
the other UI use-cases remain literature-based targets (documented as such).
"""

import numpy as np
import pandas as pd

DATA_FILE = "aunp_synthesis_realistic_v1.csv"
TARGETS = [
    "Particle_Size_nm", "Particle_Width_nm", "Drug_Loading_Efficiency",
    "Targeting_Efficiency", "Cytotoxicity",
]
LOW_PCTILE, HIGH_PCTILE = 5, 95  # central 90% of the successful population


def derive_ranges(df: pd.DataFrame, low=LOW_PCTILE, high=HIGH_PCTILE) -> dict:
    """Return {property: (low, high)} from the percentiles of the successful subset."""
    succ = df[df["Successful_Treatment"] == 1]
    ranges = {}
    for t in TARGETS:
        lo, hi = np.percentile(succ[t], [low, high])
        ranges[t] = (round(float(lo), 1), round(float(hi), 1))
    return ranges


def diagnostics(df: pd.DataFrame, ranges: dict) -> dict:
    """Precision = P(success | all-in-range); recall = fraction of successes captured."""
    mask = np.ones(len(df), dtype=bool)
    for t, (lo, hi) in ranges.items():
        mask &= df[t].between(lo, hi)
    in_range = df[mask]
    n_succ = int(df["Successful_Treatment"].sum())
    precision = float(in_range["Successful_Treatment"].mean()) if mask.sum() else 0.0
    recall = float(in_range["Successful_Treatment"].sum() / n_succ) if n_succ else 0.0
    return {
        "in_range_rows": int(mask.sum()),
        "precision": precision,
        "recall": recall,
        "base_rate": float(df["Successful_Treatment"].mean()),
        "n_successful": n_succ,
    }


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)
    ranges = derive_ranges(df)
    diag = diagnostics(df, ranges)

    print(f"Derived from {LOW_PCTILE}th-{HIGH_PCTILE}th percentile of "
          f"{diag['n_successful']} successful syntheses in {DATA_FILE}\n")
    print("ideal_range per property:")
    for t, (lo, hi) in ranges.items():
        print(f"    {t:28s} ({lo}, {hi})")

    print("\nDiagnostics (how well 'all-in-range' matches the success label):")
    print(f"    all-in-range rows : {diag['in_range_rows']} "
          f"({100*diag['in_range_rows']/len(df):.1f}% of data)")
    print(f"    precision P(success|in-range) : {100*diag['precision']:.1f}%  "
          f"(base success rate {100*diag['base_rate']:.1f}%)")
    print(f"    recall (successes captured)   : {100*diag['recall']:.1f}%")
