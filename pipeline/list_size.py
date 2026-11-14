#!/usr/bin/env python3
"""How many candidates will the next merit list have? (the 'list size')

The list size of a year = highest Round-1 closing rank on ordinary seats. It turns a relative
position (e.g. "top 50%") into a rank, so it is used by:
  * rank_from_percentage.py   the student's predicted next-year rank (ML Predictor page)
  * predict_next_year.py      the forecast closing ranks of every college (same scale, so the
                              student's rank and the college cut-offs can be compared directly)

Several methods are benchmarked forward-chaining (each year predicted only from earlier years).

Which method the ML side uses: ML_SIZE_METHOD = "weighted_all", a recency-weighted average of ALL
years since ML_FIRST_YEAR. Project decision: the ML Predictor must base the student's rank on all
years 2017–2026, not on last year alone (that is what the Rank Predictor page does). Its half-life
is chosen leak-free, by how well each half-life predicted the earlier years only. In the backtest
it is less accurate than "same as last year" (the lowest-error method); both numbers are reported
in the output and on the website.

Usage: imported; `python pipeline/list_size.py` prints the benchmark.
"""
import json
import os

import numpy as np

ML_FIRST_YEAR = 2017                 # first year the ML side learns from: all 10 years (project decision)
ML_SIZE_METHOD = "weighted_all"      # list-size method used by the ML Predictor (see docstring)
HALF_LIVES = [0.5, 1.0, 2.0, 4.0]    # candidate half-lives (years) for the weighted average

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(PROJECT, "data", "processed", "mca_cutoffs_clean.json")


def benchmark_row(r):
    return (not r["result_awaited"] and r["round"] == "R1" and r["seat_class"] in ("X", None)
            and not r["after_conversion"])


def list_sizes(rows, first_year=ML_FIRST_YEAR):
    s = {}
    for r in rows:
        if benchmark_row(r) and r["year"] >= first_year:
            s[r["year"]] = max(s.get(r["year"], 0), r["closing_rank"])
    return dict(sorted(s.items()))


def _trend(h, log=False):
    if len(h) < 3:
        return float(h[-1])
    y = np.log(h[-3:]) if log else h[-3:]
    a, b = np.polyfit(np.arange(3), y, 1)
    nxt = a * 3 + b
    return float(np.exp(nxt) if log else max(1.0, nxt))


def _weighted(h, half_life):
    w = np.power(0.5, np.arange(len(h))[::-1] / half_life)
    return float(np.sum(w * h) / np.sum(w))


def weighted_half_life(h):
    """Half-life that best predicted the earlier years of h (uses only h itself: no look-ahead)."""
    best, best_err = 1.0, None
    for hl in HALF_LIVES:
        errs = [abs(_weighted(h[:j], hl) - h[j]) / h[j] for j in range(2, len(h))]
        if errs and (best_err is None or np.mean(errs) < best_err):
            best, best_err = hl, float(np.mean(errs))
    return best


def weighted_all(h):
    return _weighted(h, weighted_half_life(h))


SIZE_METHODS = {
    "last_year": ("Same as last year", lambda h: float(h[-1])),
    "weighted_all": ("Recency-weighted average of all years", weighted_all),
    "mean_all": ("Average of all earlier years", lambda h: float(np.mean(h))),
    "mean_3": ("Average of the last 3 years", lambda h: float(np.mean(h[-3:]))),
    "median_3": ("Median of the last 3 years", lambda h: float(np.median(h[-3:]))),
    "linear_trend_3": ("Linear-regression trend, last 3 years", lambda h: _trend(h)),
    "log_trend_3": ("Growth-rate (log-linear) trend, last 3 years", lambda h: _trend(h, log=True)),
}


def size_benchmark(sizes, first_test_index=2, used=ML_SIZE_METHOD):
    years = list(sizes)
    n = np.array([sizes[y] for y in years], float)
    res = {}
    for k, (label, f) in SIZE_METHODS.items():
        tests = []
        for i in range(first_test_index, len(n)):
            pred = f(n[:i])
            tests.append({"year": years[i], "predicted": round(pred), "actual": int(n[i]),
                          "abs_pct_error": abs(pred - n[i]) / n[i], "ratio_actual_to_pred": n[i] / pred})
        res[k] = {"label": label, "tests": tests, "next": round(f(n)),
                  "mean_abs_pct_error": float(np.mean([t["abs_pct_error"] for t in tests]))}
    best = min(res, key=lambda k: res[k]["mean_abs_pct_error"])
    used = used if used in res else best
    ratios = np.array([t["ratio_actual_to_pred"] for t in res[used]["tests"]])
    lo, hi = np.quantile(ratios, [0.1, 0.9])
    nxt = SIZE_METHODS[used][1](n)
    out = {"methods": res, "best": best, "used": used, "next_year": years[-1] + 1, "next_size": round(nxt),
           "range_ratio": [float(lo), float(hi)], "range": [round(nxt * lo), round(nxt * hi)],
           "n_tests": len(ratios), "years": [years[0], years[-1]],
           "biggest_jump": float(max(max(a / b, b / a) for a, b in zip(n[1:], n[:-1])))}
    if used == "weighted_all":
        out["half_life"] = weighted_half_life(n)
    return out


def next_list_size(rows=None):
    rows = rows if rows is not None else json.load(open(CLEAN, encoding="utf-8"))["rows"]
    return size_benchmark(list_sizes(rows))


if __name__ == "__main__":
    b = next_list_size()
    for k, v in sorted(b["methods"].items(), key=lambda kv: kv[1]["mean_abs_pct_error"]):
        print(f"  {v['label']:46s} mean error {v['mean_abs_pct_error']:.0%}  next {v['next']}"
              + ("   <- used by the ML Predictor" if k == b["used"] else "") + ("   (lowest error)" if k == b["best"] else ""))
    print(f"→ {b['next_year']}: {b['next_size']} (likely {b['range'][0]}–{b['range'][1]})")
