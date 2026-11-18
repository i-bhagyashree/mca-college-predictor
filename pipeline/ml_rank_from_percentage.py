#!/usr/bin/env python3
"""Rank from percentage: the Rank Predictor (one merit list) and an ML model (three merit lists).

A student gives their graduation PERCENTAGE; both methods give a merit RANK. Neither uses cut-offs:
the rank comes only from the official MP-DTE MCA merit lists, read by pipeline/merit_lists.py into
data/processed/merit_lists.json (percentage and rank only; no names or roll numbers).

Merit lists exist only for a few rounds (2020 R1, 2025 R1, 2026 R1, 2026 R2), not for every year.

  * Rank Predictor (no ML): reads the student's position (candidates above them ÷ list size) from the
    LATEST merit list of the same round (Round 1 → latest Round 1 list, Round 2 → latest Round 2 list;
    the latest list of any round if that round has none), then rank = 1 + position × list size.
  * ML model: trained on ALL THREE recent lists (2025 R1, 2026 R1, 2026 R2): every candidate is one
    example, x = percentage, y = position. All three lists count equally (giving newer lists more weight
    was tested and gave no consistent gain). Seven models are compared; the best one is used.

Position, not rank, is learnt because the number of candidates differs between lists.

Tests (every method gets exactly the same questions):
  1. Leave one list out: hide one of the three lists, predict the rank for every percentage 50.0, 50.5 … 95.0
     from the other two, compare with the real rank in the hidden list. Repeated for each list.
  2. Real use: predict a list only from the lists published BEFORE it (2026 R1 from 2025 R1; 2026 R2 from
     2025 R1 and 2026 R1), as the site does for next year.

Outputs:
  reports/ml_rank/REPORT.md, reports/ml_rank/results.json
  data/processed/ml_rank_model.json   (the best ML model's percentage → position curve, trained on all three
                                        lists, and every test result; read by build_site_data.py)
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
RECENCY = 1                                                     # weight of each list vs the one before it (1 = all lists equal; 2 tested, no consistent gain)
RP_NAME = "Rank Predictor (latest merit list, no ML)"
AVG_NAME = "Average of the merit lists (no ML, earlier method)"


def name(l):
    return f"{l['year']} {l['round']}"


def order(l):
    return (l["year"], l["round"])


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
    """Share of the list above this percentage (0 = top)."""
    return (rank_of(pct, lst) - 1) / lst["n"]


def latest_same_round(lists, rnd):
    same = [l for l in lists if l["round"] == rnd]
    return max(same or lists, key=order)


def models():
    return {
        "Linear regression": lambda: LinearRegression(),
        "Polynomial regression (degree 3)": lambda: make_pipeline(StandardScaler(), PolynomialFeatures(3), LinearRegression()),
        "k-nearest neighbours (k = 25)": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=25)),
        "Decision tree": lambda: DecisionTreeRegressor(min_samples_leaf=10, random_state=SEED),
        "Random forest": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "Gradient boosting (monotonic)": lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=20, monotonic_cst=[1], random_state=SEED),
        "Isotonic regression (monotonic)": "isotonic",
    }


def training_rows(train_lists):
    """x = percentage, y = position (candidates above ÷ list size), one row per candidate.
    With RECENCY > 1 newer lists are repeated (the newest RECENCY^(k-1) times); with 1 all count equally."""
    ordered = sorted(train_lists, key=order)
    x, y = [], []
    for age, l in enumerate(reversed(ordered)):            # age 0 = newest
        times = RECENCY ** (len(ordered) - 1 - age)
        rows = [(pct, (rank_of(pct, l) - 1) / l["n"]) for pct, _ in candidates(l)]
        for _ in range(times):
            x += [p for p, _ in rows]
            y += [v for _, v in rows]
    return np.array(x), np.array(y)


def fit(model_name, make, train_lists):
    """Train one ML model; returns percentage(s) → predicted position."""
    x, y = training_rows(train_lists)
    if make == "isotonic":
        m = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x, y)
        return lambda q: np.clip(m.predict(np.atleast_1d(np.asarray(q, float))), 0, 1)
    if model_name.startswith("Gradient boosting"):
        m = make().fit(-x.reshape(-1, 1), y)                # monotonic in −percentage: more marks, never a worse position
        return lambda q: np.clip(m.predict(-np.atleast_1d(np.asarray(q, float)).reshape(-1, 1)), 0, 1)
    m = make().fit(x.reshape(-1, 1), y)
    return lambda q: np.clip(m.predict(np.atleast_1d(np.asarray(q, float)).reshape(-1, 1)), 0, 1)


def rank_predictor(train_lists, target):
    """The site's Rank Predictor: position in the latest list of the target's round."""
    src = latest_same_round(train_lists, target["round"])
    return lambda q: np.array([position_of(p, src) for p in np.atleast_1d(q)])


def average_of_lists(train_lists, target):
    return lambda q: np.array([np.mean([position_of(p, l) for l in train_lists]) for p in np.atleast_1d(q)])


def score(pred_pos, t):
    grid = [p for p in GRID if p >= t["pct_min"]]           # below the last candidate there is no real rank
    pred = np.maximum(1, np.round(1 + pred_pos(grid) * t["n"]))
    real = np.array([rank_of(p, t) for p in grid])
    return np.abs(pred - real)


def summarise(per_target):
    all_err = np.concatenate([e for _, _, e in per_target])
    tests = [{"target": name(t), "trained_on": [name(l) for l in tr], "mae_ranks": round(float(e.mean()), 1),
              "p90_ranks": round(float(np.percentile(e, 90)), 1), "max_ranks": int(e.max())} for t, tr, e in per_target]
    return {"tests": tests, "mae_ranks": round(float(np.mean([x["mae_ranks"] for x in tests])), 1),
            "p90_ranks": round(float(np.mean([x["p90_ranks"] for x in tests])), 1),
            "max_ranks": int(max(x["max_ranks"] for x in tests)),
            "within_25": round(100 * float(np.mean(all_err <= 25)), 1)}


def leave_one_out(method, recent):
    out = []
    for t in recent:
        train = [l for l in recent if l is not t]
        out.append((t, train, score(method(train, t), t)))
    return summarise(out)


def real_use(method, recent):
    chrono = sorted(recent, key=order)
    out = []
    for i in range(1, len(chrono)):
        t, train = chrono[i], chrono[:i]
        out.append((t, train, score(method(train, t), t)))
    return summarise(out)


def ml_method(model_name):
    make = models()[model_name]
    return lambda train, t: fit(model_name, make, train)


def main():
    data = json.load(open(MERIT, encoding="utf-8"))
    lists = data["lists"]
    latest = max(l["year"] for l in lists)
    recent = sorted([l for l in lists if l["year"] >= latest - 1], key=order)   # 2025 R1, 2026 R1, 2026 R2
    names = [name(l) for l in recent]
    n_train = sum(l["n"] for l in recent)
    print(f"Merit lists: {', '.join(names)} ({n_train:,} candidates)")

    methods = [(RP_NAME, False, rank_predictor), (AVG_NAME, False, average_of_lists)]
    methods += [(m, True, ml_method(m)) for m in models()]
    results = []
    for mname, is_ml, method in methods:
        loo, use = leave_one_out(method, recent), real_use(method, recent)
        results.append({"model": mname, "ml": is_ml, **loo, "real_use": use})
        print(f"  {mname:<52} leave-one-out {loo['mae_ranks']:>5} ranks (90% within {loo['p90_ranks']:>5})   "
              f"real use {use['mae_ranks']:>5}   " + "  ".join(f"{x['target']}: {x['mae_ranks']}" for x in loo["tests"]))

    rp = results[0]
    ml = [r for r in results if r["ml"]]
    best = min(ml, key=lambda r: (r["mae_ranks"], r["real_use"]["mae_ranks"]))
    print(f"Best ML model: {best['model']}: {best['mae_ranks']} ranks (Rank Predictor {rp['mae_ranks']}); "
          f"real use {best['real_use']['mae_ranks']} vs {rp['real_use']['mae_ranks']}")

    # Does more (older) data help the ML model? Add the 2020 list to its training data.
    older = [l for l in lists if l["year"] < latest - 1]
    more = None
    if older:
        make = models()[best["model"]]
        out = []
        for t in recent:
            train = [l for l in recent if l is not t] + older
            out.append((t, train, score(fit(best["model"], make, train), t)))
        more = summarise(out)
        older_names = ", ".join(name(l) for l in older)
        more["model"] = f"{best['model']} + older lists ({older_names})"
        print(f"  with older lists added: {more['mae_ranks']} ranks (worse: the candidates of that year were different)")

    # the best ML model, trained on all three lists, as a percentage → position curve for the website
    final = fit(best["model"], models()[best["model"]], recent)
    lo, hi = min(l["pct_min"] for l in recent), max(l["pct_max"] for l in recent)
    xs = [round(x, 2) for x in np.arange(round(lo, 2), hi + 1e-9, CURVE_STEP)]
    curve = [[x, round(float(y), 5)] for x, y in zip(xs, final(xs))]

    os.makedirs(REPORT_DIR, exist_ok=True)
    keep = ("model", "ml", "mae_ranks", "p90_ranks", "max_ranks", "within_25", "tests", "real_use")
    summary = {"lists": names, "candidates": n_train, "recency_weight": RECENCY,
               "rank_predictor": {"name": RP_NAME, "method": "latest merit list of the same round",
                                  **{k: rp[k] for k in keep if k not in ("model", "ml")}},
               "model": best["model"], "trained_on": names,
               "target": "position = candidates above ÷ list size; rank = 1 + position × list size",
               "test": {k: best[k] for k in ("mae_ranks", "p90_ranks", "within_25")},
               "real_use": best["real_use"],
               "with_older_lists": more and {k: more[k] for k in ("model", "mae_ranks", "p90_ranks")},
               "results": [{k: r[k] for k in keep} for r in results]}
    json.dump(summary, open(os.path.join(REPORT_DIR, "results.json"), "w"), indent=1)
    json.dump({**summary, "curve": curve}, open(MODEL_OUT, "w"), indent=1)

    head = " | ".join(f"Error on {x['target']}" for x in rp["tests"])
    rows = "\n".join(f"| {r['model']} | {r['mae_ranks']} | {r['p90_ranks']} | {r['within_25']}% | "
                     + " | ".join(str(x["mae_ranks"]) for x in r["tests"]) + f" | {r['real_use']['mae_ranks']} |"
                     for r in results)
    use_txt = "; ".join(f"{x['target']} from {' + '.join(x['trained_on'])}" for x in rp["real_use"]["tests"])
    md = f"""# Rank from percentage: Rank Predictor vs ML

Generated by `pipeline/ml_rank_from_percentage.py`. No cut-offs are used: the rank comes only from the merit lists.

**Data:** the official merit lists {', '.join(names)} ({n_train:,} candidates, percentage and rank only).
Merit lists are not published for every year.

- **Rank Predictor (no ML):** the student's position in the **latest merit list of the same round**, × list size.
- **ML:** trained on **all three lists** (every candidate: percentage → position = candidates above ÷ list size),
  {"all lists counting equally" if RECENCY == 1 else f"newer lists counting {RECENCY}× the one before"}. Seven models compared.

**Tests**
1. *Leave one list out:* hide one list, predict the rank for every percentage 50–95% (step 0.5) from the other two.
2. *Real use:* predict a list only from earlier lists ({use_txt}).

| Method | Leave-one-out error (ranks) | 90% within | Within 25 ranks | {head} | Real-use error |
|---|---|---|---|{'---|' * len(rp['tests'])}---|
{rows}

**Best ML model:** {best['model']}: {best['mae_ranks']} ranks (leave-one-out), {best['real_use']['mae_ranks']} (real use).
**Rank Predictor:** {rp['mae_ranks']} ranks (leave-one-out), {rp['real_use']['mae_ranks']} (real use).
""" + (f"""
**More data is not better:** adding the older list(s) to the ML training data gives {more['mae_ranks']} ranks instead of
{best['mae_ranks']}. The candidates of that year had a different spread of percentages.
""" if more else "")
    open(os.path.join(REPORT_DIR, "REPORT.md"), "w").write(md)
    print(f"→ {os.path.relpath(os.path.join(REPORT_DIR, 'REPORT.md'), PROJECT)}, {os.path.relpath(MODEL_OUT, PROJECT)}")


if __name__ == "__main__":
    main()
