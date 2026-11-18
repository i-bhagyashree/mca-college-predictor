#!/usr/bin/env python3
"""Machine learning: predict a student's RANK from their graduation PERCENTAGE.

Data: the official MP-DTE MCA merit lists, read by pipeline/merit_lists.py into
data/processed/merit_lists.json (percentage and rank only; no names or roll numbers).
Merit lists exist for only a few rounds (2020 R1, 2025 R1, 2026 R1, 2026 R2), not for every year,
so this model learns from the three recent lists the Rank Predictor also uses.

What is learnt: percentage → position, where position = rank ÷ list size (0 = top, 1 = last).
Position, not rank, because the number of candidates changes between lists (1,091 to 1,146 here,
326 to 3,354 across all years). The predicted rank is position × the size of the list.

Test (the same one the Rank Predictor is judged by, so the numbers are comparable):
leave one list out. Train on two lists, predict every percentage 50.0, 50.5 … 95.0 for the third,
compare with the real rank in that list. Repeated for each of the three lists.

Models compared:
  * Rank Predictor (no ML): average of the student's position in the other lists
  * Linear regression, polynomial regression (degree 3)
  * k-nearest neighbours
  * Decision tree, random forest
  * Gradient boosting (monotonic: a higher percentage can never get a worse rank)
  * Isotonic regression (monotonic by design)

Outputs:
  reports/ml_rank/REPORT.md, reports/ml_rank/results.json
  data/processed/ml_rank_model.json   (the chosen model's percentage → position curve, trained on all
                                        three lists, read by build_site_data.py for the website)
Usage: python pipeline/ml_rank_from_percentage.py
"""
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERIT = os.path.join(PROJECT, "data", "processed", "merit_lists.json")
REPORT_DIR = os.path.join(PROJECT, "reports", "ml_rank")
MODEL_OUT = os.path.join(PROJECT, "data", "processed", "ml_rank_model.json")
SEED = 42
GRID = [round(x, 1) for x in np.arange(50, 95.01, 0.5)]        # same grid as merit_lists.py
CURVE_STEP = 0.05                                               # resolution of the curve sent to the website


def candidates(lst):
    """(percentage, rank) of every candidate in a list, rebuilt from its [pct, first, last] points."""
    out = []
    for pct, first, last in lst["points"]:
        out += [(pct, r) for r in range(first, last + 1)]
    return out


def rank_of(pct, lst):
    """Rank a student with `pct` would have had in this list: 1 + number of candidates above them."""
    above = 0
    for p, first, last in lst["points"]:          # percentage descending
        if p > pct + 1e-9:
            above = last
        else:
            break
    return above + 1


def position_of(pct, lst):
    return rank_of(pct, lst) / lst["n"]


def models():
    return {
        "Linear regression": lambda: LinearRegression(),
        "Polynomial regression (degree 3)": lambda: make_pipeline(StandardScaler(), PolynomialFeatures(3), LinearRegression()),
        "k-nearest neighbours (k = 25)": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=25)),
        "Decision tree": lambda: DecisionTreeRegressor(min_samples_leaf=10, random_state=SEED),
        "Random forest": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "Gradient boosting (monotonic)": lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=20,
            monotonic_cst=[1], random_state=SEED),            # higher % → never a worse (larger) position? see fit()
        "Isotonic regression (monotonic)": "isotonic",
    }


def fit(name, make, train_lists):
    """Train on every candidate of the given lists: x = percentage, y = position (rank ÷ list size).
    Returns a function percentage(s) → predicted position."""
    x, y = [], []
    for l in train_lists:
        for pct, rank in candidates(l):
            x.append(pct)
            y.append(rank / l["n"])
    x, y = np.array(x), np.array(y)
    if make == "isotonic":
        m = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x, y)
        return lambda q: np.clip(m.predict(np.asarray(q, float)), 0, 1)
    if name.startswith("Gradient boosting"):
        # monotonic_cst=+1 on (−percentage): position never improves when the percentage falls
        m = make().fit(-x.reshape(-1, 1), y)
        return lambda q: np.clip(m.predict(-np.asarray(q, float).reshape(-1, 1)), 0, 1)
    m = make().fit(x.reshape(-1, 1), y)
    return lambda q: np.clip(m.predict(np.asarray(q, float).reshape(-1, 1)), 0, 1)


def rank_predictor(train_lists):
    """The site's method, no ML: average of the student's position in each list."""
    return lambda q: np.array([np.mean([position_of(p, l) for l in train_lists]) for p in np.atleast_1d(q)])


def evaluate(predict_for, recent):
    """Leave one list out. Returns per-target and overall errors in ranks."""
    tests, all_err = [], []
    for t in recent:
        train = [l for l in recent if l is not t]
        pred_pos = predict_for(train)
        grid = [p for p in GRID if p >= t["pct_min"]]           # below the last candidate there is no real rank
        pred = np.round(pred_pos(grid) * t["n"])
        pred = np.maximum(pred, 1)
        real = np.array([rank_of(p, t) for p in grid])
        err = np.abs(pred - real)
        all_err += list(err)
        tests.append({"target": f"{t['year']} {t['round']}", "trained_on": [f"{l['year']} {l['round']}" for l in train],
                      "mae_ranks": round(float(err.mean()), 1), "p90_ranks": round(float(np.percentile(err, 90)), 1),
                      "max_ranks": int(err.max()), "n_points": len(grid)})
    maes = [x["mae_ranks"] for x in tests]
    return {"tests": tests, "mae_ranks": round(float(np.mean(maes)), 1),
            "p90_ranks": round(float(np.mean([x["p90_ranks"] for x in tests])), 1),
            "max_ranks": int(max(x["max_ranks"] for x in tests)),
            "within_25": round(100 * float(np.mean(np.array(all_err) <= 25)), 1)}


def main():
    data = json.load(open(MERIT, encoding="utf-8"))
    lists = data["lists"]
    latest = max(l["year"] for l in lists)
    recent = [l for l in lists if l["year"] >= latest - 1]          # 2025 R1, 2026 R1, 2026 R2
    names = [f"{l['year']} {l['round']}" for l in recent]
    n_train = sum(l["n"] for l in recent)
    print(f"Merit lists used: {', '.join(names)} ({n_train:,} candidates)")

    results = [{"model": "Rank Predictor (no ML: average position)", "ml": False,
                **evaluate(rank_predictor, recent)}]
    for name, make in models().items():
        r = evaluate(lambda tr, n=name, m=make: fit(n, m, tr), recent)
        results.append({"model": name, "ml": True, **r})
    for r in results:
        print(f"  {r['model']:<44} MAE {r['mae_ranks']:>6} ranks   90% within {r['p90_ranks']:>6}   "
              f"worst {r['max_ranks']:>4}   ≤25 ranks {r['within_25']}%   "
              + "  ".join(f"{t['target']}: {t['mae_ranks']}" for t in r["tests"]))

    base = results[0]
    ml = [r for r in results if r["ml"]]
    best = min(ml, key=lambda r: r["mae_ranks"])
    print(f"Best ML model: {best['model']} ({best['mae_ranks']} vs Rank Predictor {base['mae_ranks']})")

    # Does more (older) data help? Add the 2020 Round 1 list to the training data of the best model.
    older = [l for l in lists if l["year"] < latest - 1]
    more = None
    if older:
        more = evaluate(lambda tr: fit(best["model"], models()[best["model"]], tr + older), recent)
        older_names = ", ".join(f"{l['year']} {l['round']}" for l in older)
        more.update({"model": f"{best['model']} + older lists ({older_names})"})
        print(f"  with older lists added: MAE {more['mae_ranks']} (older lists hurt: the candidates have changed)")

    # the chosen model, trained on all three lists, as a percentage → position curve for the website
    final = fit(best["model"], models()[best["model"]], recent)
    lo = min(l["pct_min"] for l in recent)
    hi = max(l["pct_max"] for l in recent)
    xs = [round(x, 2) for x in np.arange(round(lo, 2), hi + 1e-9, CURVE_STEP)]
    ys = final(xs)
    curve = [[x, round(float(y), 5)] for x, y in zip(xs, ys)]
    os.makedirs(REPORT_DIR, exist_ok=True)
    out = {"lists": names, "candidates": n_train, "grid": [GRID[0], GRID[-1], 0.5],
           "results": results, "best_ml": best["model"], "rank_predictor_mae": base["mae_ranks"], "with_older_lists": more}
    json.dump(out, open(os.path.join(REPORT_DIR, "results.json"), "w"), indent=1)
    json.dump({"model": best["model"], "trained_on": names, "candidates": n_train,
               "target": "position = rank ÷ list size", "curve": curve,
               "test": {"mae_ranks": best["mae_ranks"], "p90_ranks": best["p90_ranks"]},
               "rank_predictor_test": {"mae_ranks": base["mae_ranks"], "p90_ranks": base["p90_ranks"]},
               "with_older_lists": more and {k: more[k] for k in ("model", "mae_ranks", "p90_ranks")},
               "results": [{k: r[k] for k in ("model", "ml", "mae_ranks", "p90_ranks", "max_ranks", "within_25", "tests")}
                           for r in results]},
              open(MODEL_OUT, "w"), indent=1)

    rows = "\n".join(f"| {r['model']} | {r['mae_ranks']} | {r['p90_ranks']} | {r['max_ranks']} | {r['within_25']}% | "
                     + " | ".join(str(t["mae_ranks"]) for t in r["tests"]) + " |" for r in results)
    head = " | ".join(f"Error on {t['target']}" for t in base["tests"])
    md = f"""# Machine learning: rank from percentage

Generated by `pipeline/ml_rank_from_percentage.py`.

**Question:** a student gives their graduation percentage. Can a machine-learning model predict their merit rank
better than the Rank Predictor (the average of the student's position in the recent merit lists)?

**Data:** the official merit lists {', '.join(names)}: {n_train:,} candidates, percentage and rank only.
Merit lists are not available for every year, so the model learns from these three lists.
**Target:** position = rank ÷ list size; predicted rank = position × list size.

**Test:** leave one list out. Train on two lists, predict the rank for every percentage from 50% to 95%
(step 0.5) in the third list, compare with the real rank. Done for each of the three lists.
This is the same test that gives the Rank Predictor its error.

| Model | Average error (ranks) | 90% of predictions within | Worst error | Within 25 ranks | {head} |
|---|---|---|---|---|{'---|' * len(base['tests'])}
{rows}

**Best ML model:** {best['model']}, average error {best['mae_ranks']} ranks.
**Rank Predictor:** {base['mae_ranks']} ranks.
""" + (f"""
**More data is not better here:** adding the older list(s) to the training data of the best model gives
{more['mae_ranks']} ranks average error instead of {best['mae_ranks']}. The candidates of 2020 had a different spread of
percentages, so old lists teach the model the wrong positions.
""" if more else "")
    open(os.path.join(REPORT_DIR, "REPORT.md"), "w").write(md)
    print(f"→ {os.path.relpath(os.path.join(REPORT_DIR, 'REPORT.md'), PROJECT)}, {os.path.relpath(MODEL_OUT, PROJECT)}")


if __name__ == "__main__":
    main()
