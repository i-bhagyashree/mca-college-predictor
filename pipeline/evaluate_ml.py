#!/usr/bin/env python3
"""Hold-out test of the ML predictor: does it give the right answers on years it has not seen?

For each test year T (default: the last three years in the data):
  1. Train the site's model (ResidualGBM from predict_next_year.py) on years < T only.
  2. Predict every college × category × gender cut-off for T.
  3. Compare with the REAL cut-offs of T, and with the Historical method (last year's real
     cut-off), which is what the Historical Predictor page uses.

Measured
  * Cut-off error in ranks (MAE, median). Errors are in T's real rank scale, so they measure
    the model itself; how many students apply is a separate risk, reported in rank_from_percentage.py.
  * Likely-range coverage: share of real cut-offs inside the 10–90% range. The range is built
    only from errors on years before T (no peeking).
  * Student-level accuracy (what a student sees): 50 simulated students per seat pool, spread
    over the whole merit list. For each (student, college) pair:
        seat decision  "you would get this seat" yes/no, predicted vs real
        exact label    Safe / Likely / Reach / Unlikely, predicted vs real (same rules as the site)
    Reported for all colleges, and for competitive colleges only (real cut-off < 90% of the
    list), because "open seat" colleges take everyone and make any method look good.

Outputs: reports/ml_test/REPORT.md, ml_test_results.json, predictions_vs_actual_<T>.csv
Usage:   python pipeline/evaluate_ml.py [test years...]
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_next_year import ResidualGBM, dataset, features, load  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "reports", "ml_test")
STUDENTS = np.linspace(0.02, 1.0, 50)        # student positions in the merit list (rank ÷ list size)
OPEN = 0.9                                    # same "open seats" threshold as the website


def band(q, p):
    """Website rules (bandOf): rank / closing rank."""
    x = q / p
    return np.where(x <= 0.8, 0, np.where(x <= 1.0, 1, np.where(x <= 1.15, 2, 3)))   # safe, likely, reach, unlikely


def fit_predict(X, Y, T, t, Xe):
    tr = T < t
    Xtr, Xte = X[tr].copy(), Xe.copy()
    empty = np.all(np.isnan(Xtr), axis=0)
    Xtr[:, empty] = 0.0
    Xte[:, empty] = 0.0
    return np.clip(ResidualGBM().fit(Xtr, Y[tr]).predict(Xte), 0.01, 1.5)


def eval_rows(pools, scale, colleges, t):
    keys, Xe = features(pools, scale, colleges, t)
    sel = [i for i, k in enumerate(keys) if t in pools[k]]
    keys = [keys[i] for i in sel]
    return keys, Xe[sel], np.array([pools[k][t]["cl"] / scale[t] for k in keys])


def residuals_before(pools, scale, colleges, data, t, first=2020):
    """Held-out errors of the model on years first..t-1 (each trained on its own past)."""
    X, Y, T = data
    res = []
    for u in sorted(scale):
        if first <= u < t:
            keys, Xe, y = eval_rows(pools, scale, colleges, u)
            res.append(y - fit_predict(X, Y, T, u, Xe))
    return np.concatenate(res)


def student_scores(actual, pred, competitive):
    m = competitive[:, None] if competitive is not None else np.ones((len(actual), 1), bool)
    q = STUDENTS[None, :]
    a, p = actual[:, None], pred[:, None]
    mask = np.broadcast_to(m, (len(actual), len(STUDENTS)))
    seat = ((q <= a) == (q <= p))[mask]
    lab = (band(q, a) == band(q, p))[mask]
    return float(seat.mean()), float(lab.mean())


def evaluate_year(pools, scale, colleges, data, t):
    keys, Xe, y = eval_rows(pools, scale, colleges, t)
    ml = fit_predict(*data, t, Xe)
    hist = Xe[:, 0]                                 # last real cut-off (relative), = Historical method
    res = residuals_before(pools, scale, colleges, data, t)
    lo_q, hi_q = np.quantile(res, [0.1, 0.9])
    inside = (y >= ml + lo_q) & (y <= ml + hi_q)
    S = scale[t]
    comp = y < OPEN
    out = {"test_year": t, "trained_on": f"{min(scale)}–{t - 1}", "n": len(keys), "n_competitive": int(comp.sum()),
           "range_built_from_errors_of": f"2020–{t - 1}", "coverage_10_90": float(inside.mean())}
    for name, pr in (("ml", ml), ("historical", hist)):
        err = np.abs(pr - y) * S
        s_all, l_all = student_scores(y, pr, None)
        s_cmp, l_cmp = student_scores(y, pr, comp)
        out[name] = {"rank_mae": float(err.mean()), "rank_median": float(np.median(err)),
                     "within_50": float(np.mean(err <= 50)), "within_100": float(np.mean(err <= 100)),
                     "seat_accuracy_all": s_all, "label_accuracy_all": l_all,
                     "seat_accuracy_competitive": s_cmp, "label_accuracy_competitive": l_cmp}
    out["ml_closer_share"] = float(np.mean(np.abs(ml - y) < np.abs(hist - y)))
    rows = []
    for k, a, p, h in zip(keys, y, ml, hist):
        cid, s, g = k
        rows.append({"college_id": cid, "college": (colleges.get(cid) or {}).get("name", cid), "category": s, "gender": g,
                     "last_year_actual": round(h * S), "ml_predicted": round(p * S),
                     "ml_range": f"{max(1, round((p + lo_q) * S))}–{round((p + hi_q) * S)}",
                     "actual": round(a * S), "ml_error": round((p - a) * S), "historical_error": round((h - a) * S),
                     "inside_range": bool((a >= p + lo_q) and (a <= p + hi_q))})
    return out, rows


def pct(x):
    return f"{x:.0%}"


def report(results):
    L = ["# ML predictor: hold-out test\n",
         "Generated by `pipeline/evaluate_ml.py`. For each test year the model is trained only on earlier years and its "
         "predictions are compared with the real cut-offs of that year. *Historical* = last year's real cut-off, the method "
         "of the Historical Predictor page. Every number is computed, not estimated.\n",
         "## Cut-off accuracy (ranks)\n",
         "| Test year | Trained on | Seat pools | ML error (avg) | Historical error (avg) | ML median | Historical median | ML within ±100 | Historical within ±100 | ML closer in | Real cut-off inside ML range |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m, h = r["ml"], r["historical"]
        L.append(f"| {r['test_year']} | {r['trained_on']} | {r['n']} | {m['rank_mae']:.0f} | {h['rank_mae']:.0f} | {m['rank_median']:.0f} | "
                 f"{h['rank_median']:.0f} | {pct(m['within_100'])} | {pct(h['within_100'])} | {pct(r['ml_closer_share'])} of pools | {pct(r['coverage_10_90'])} |")
    L += ["\nThe likely range targets 80% (10th–90th percentile of past errors).\n",
          "## What a student sees (50 simulated students per seat pool)\n",
          "*Seat decision* = \"you would get this seat\" yes/no was right. *Exact label* = Safe/Likely/Reach/Unlikely was right. "
          "*Competitive* = colleges whose real cut-off closed before 90% of the list (open-seat colleges excluded).\n",
          "| Test year | ML seat decision | Historical seat decision | ML exact label | Historical exact label | ML seat decision (competitive) | Historical (competitive) | ML label (competitive) | Historical label (competitive) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m, h = r["ml"], r["historical"]
        L.append(f"| {r['test_year']} | {pct(m['seat_accuracy_all'])} | {pct(h['seat_accuracy_all'])} | {pct(m['label_accuracy_all'])} | "
                 f"{pct(h['label_accuracy_all'])} | {pct(m['seat_accuracy_competitive'])} | {pct(h['seat_accuracy_competitive'])} | "
                 f"{pct(m['label_accuracy_competitive'])} | {pct(h['label_accuracy_competitive'])} |")
    avg = lambda f: np.mean([f(r) for r in results])
    ml_mae, hi_mae = avg(lambda r: r["ml"]["rank_mae"]), avg(lambda r: r["historical"]["rank_mae"])
    L += ["\n## Verdict\n",
          f"- Averaged over {', '.join(str(r['test_year']) for r in results)}: ML cut-off error **{ml_mae:.0f} ranks** vs "
          f"Historical **{hi_mae:.0f}** ({(ml_mae - hi_mae) / hi_mae:+.0%}). ML was closer for "
          f"{pct(avg(lambda r: r['ml_closer_share']))} of seat pools.",
          f"- Seat decision right: ML {pct(avg(lambda r: r['ml']['seat_accuracy_all']))}, Historical "
          f"{pct(avg(lambda r: r['historical']['seat_accuracy_all']))}; on competitive colleges ML "
          f"{pct(avg(lambda r: r['ml']['seat_accuracy_competitive']))}, Historical "
          f"{pct(avg(lambda r: r['historical']['seat_accuracy_competitive']))}.",
          f"- Exact label right: ML {pct(avg(lambda r: r['ml']['label_accuracy_all']))}, Historical "
          f"{pct(avg(lambda r: r['historical']['label_accuracy_all']))}.",
          f"- Likely range contained the real cut-off {pct(avg(lambda r: r['coverage_10_90']))} of the time (target 80%).",
          "- Per-college predictions vs reality: `predictions_vs_actual_<year>.csv`.\n"]
    return "\n".join(L)


def main():
    pools, scale, colleges = load()
    years = sorted(scale)
    tests = [int(a) for a in sys.argv[1:]] or years[-3:]
    K, X, Y, T = dataset(pools, scale, colleges, years[1:])
    data = (X, Y, T)
    os.makedirs(OUT, exist_ok=True)
    results = []
    for t in tests:
        r, rows = evaluate_year(pools, scale, colleges, data, t)
        results.append(r)
        with open(os.path.join(OUT, f"predictions_vs_actual_{t}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda x: (x["college"], x["category"], x["gender"])))
        m, h = r["ml"], r["historical"]
        print(f"{t}: {r['n']} pools | rank MAE ML {m['rank_mae']:.0f} vs hist {h['rank_mae']:.0f} | "
              f"seat ok ML {m['seat_accuracy_all']:.1%} vs {h['seat_accuracy_all']:.1%} | competitive ML "
              f"{m['seat_accuracy_competitive']:.1%} vs {h['seat_accuracy_competitive']:.1%} | label ML "
              f"{m['label_accuracy_all']:.1%} vs {h['label_accuracy_all']:.1%} | range coverage {r['coverage_10_90']:.0%}")
    with open(os.path.join(OUT, "ml_test_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    with open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write(report(results))
    print(f"→ {os.path.relpath(OUT, PROJECT)}/REPORT.md")


if __name__ == "__main__":
    main()
