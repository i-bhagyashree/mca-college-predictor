#!/usr/bin/env python3
"""Estimate a student's merit RANK from their graduation PERCENTAGE.

Students know their percentage, not their rank. This script builds the percentage → rank
relation that the website uses, and measures how accurate it is.

1. Percentage → rank for the latest year (the "last year" method)
   Only lists that print merit percentages next to ranks can be used. In this data that is
   2026 Round 2 only: every row prints opening/closing rank AND opening/closing percentage,
   giving ~250 (rank, %) points between rank 1 and the end of the list. Higher % always means
   a better rank there (checked below), so they are points on ONE merit list. A student's rank
   is read off by linear interpolation between neighbouring points.
   Accuracy: leave-one-out (hide one point, interpolate it from the others).

2. Rank next year (used by the ML page)
   The same percentage gives a different rank next year if a different number of students
   applies. The list size (highest Round-1 closing rank) is known for every year 2017–2026,
   so the forecast of next year's list size is benchmarked forward-chaining (predict year t
   from years < t) over several methods, including a linear-regression trend. The method with
   the lowest error is used, and the likely range comes from that method's past errors.
   Predicted rank next year = (your position in the latest list) × (forecast list size).

   Not possible with this data: learning how the percentage → rank relation itself changes
   between years, because 2017–2025 lists print no percentages. This is stated on the site.

Output: data/processed/rank_from_percentage.json
Usage:  python pipeline/rank_from_percentage.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from list_size import SIZE_METHODS, list_sizes, size_benchmark  # noqa: E402,F401  (re-exported for tests)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(PROJECT, "data", "processed")


def curve_points(rows):
    """(percentage, rank) points of the latest list that prints percentages.
    Returns year, round, ascending percentages and their (mean) ranks."""
    scored = [r for r in rows if r["closing_score"] is not None and not r["result_awaited"]]
    year = max(r["year"] for r in scored)
    rnd = sorted({r["round"] for r in scored if r["year"] == year})[-1]
    by_pct = {}
    for r in scored:
        if r["year"] != year or r["round"] != rnd:
            continue
        for rank, pct in ((r["opening_rank"], r["opening_score"]), (r["closing_rank"], r["closing_score"])):
            if rank is not None and pct is not None:
                by_pct.setdefault(round(float(pct), 4), []).append(rank)
    pct = np.array(sorted(by_pct))
    rank = np.array([float(np.mean(by_pct[p])) for p in pct])
    return year, rnd, pct, rank


def rank_for(p, pct, rank):
    """Interpolated rank for percentage p (pct ascending, rank descending)."""
    return float(np.interp(p, pct, rank))


def monotonic_violations(pct, rank):
    """Points where a HIGHER percentage has a WORSE (larger) rank."""
    return int(np.sum(np.diff(rank) > 1e-9))


def leave_one_out(pct, rank):
    err = []
    for i in range(1, len(pct) - 1):                 # end points cannot be interpolated
        m = np.ones(len(pct), bool)
        m[i] = False
        err.append(rank_for(pct[i], pct[m], rank[m]) - rank[i])
    return np.array(err)


def main():
    rows = json.load(open(os.path.join(PROC, "mca_cutoffs_clean.json"), encoding="utf-8"))["rows"]
    year, rnd, pct, rank = curve_points(rows)
    viol = monotonic_violations(pct, rank)
    loo = leave_one_out(pct, rank)
    sizes = list_sizes(rows)
    sb = size_benchmark(sizes)
    doc = {
        "curve": {"year": year, "round": rnd, "points": [[round(float(p), 4), round(float(r), 1)] for p, r in zip(pct, rank)],
                  "n_points": int(len(pct)), "monotonic_violations": viol,
                  "pct_min": float(pct[0]), "pct_max": float(pct[-1]), "rank_max": float(rank[0])},
        "accuracy": {"method": "leave-one-out interpolation", "n": int(len(loo)),
                     "mae_ranks": float(np.mean(np.abs(loo))), "median_abs_ranks": float(np.median(np.abs(loo))),
                     "p90_abs_ranks": float(np.quantile(np.abs(loo), 0.9))},
        "scale_year": year, "scale": sizes[year], "list_sizes": {str(y): s for y, s in sizes.items()},
        "size_forecast": sb,
    }
    out = os.path.join(PROC, "rank_from_percentage.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    a = doc["accuracy"]
    print(f"curve: {year} {rnd}, {len(pct)} (percentage, rank) points, {pct[0]:.2f}%–{pct[-1]:.2f}%, "
          f"monotonic violations: {viol}")
    print(f"percentage → rank, leave-one-out: MAE {a['mae_ranks']:.1f} ranks, median {a['median_abs_ranks']:.1f}, "
          f"90% within {a['p90_abs_ranks']:.1f}")
    print("list size by year:", sizes)
    for k, v in sorted(sb["methods"].items(), key=lambda kv: kv[1]["mean_abs_pct_error"]):
        print(f"  {v['label']:46s} mean error {v['mean_abs_pct_error']:.0%}")
    print(f"→ {sb['next_year']} list size used by the ML Predictor: {sb['next_size']} ({SIZE_METHODS[sb['used']][0]}), "
          f"likely range {sb['range'][0]}–{sb['range'][1]}; lowest-error method: {SIZE_METHODS[sb['best']][0]}")
    print(f"→ {os.path.relpath(out, PROJECT)}")


if __name__ == "__main__":
    main()
