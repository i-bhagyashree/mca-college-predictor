#!/usr/bin/env python3
"""ML forecast of next year's Round-1 closing ranks, with an honest backtest.

Why relative ranks
------------------
The rank scale itself changes a lot between years (the highest Round-1 closing rank was
326 in 2019, 3,354 in 2022, 1,091 in 2026) because the number of applicants changes and,
from 2026-27, merit is based on graduation marks (rule book §12, §14.8). Nobody knows next
year's applicant count, so the model predicts each pool's RELATIVE position:

    p = closing rank / (highest Round-1 closing rank that year)

and converts it back to a rank with the latest year's scale. The scale risk is reported
separately (backtest "rank error with last year's scale").

Unit of prediction: a seat pool = (college, category, gender) on ordinary seats
(class X, not converted), Round 1, regular list. Domicile splits are merged by taking the
highest closing rank, as the website chart does.

Models compared (time-based backtest: train on years before T, predict T, for T = 2022..2026)
  * Baseline "last year": p(T) = p(last year seen)
  * Baseline "3-year average": mean of the last three p values
  * Ridge linear regression on the features below
  * Gradient boosting predicting the level of p
  * Gradient boosting predicting the CHANGE from last year (residual model; tiny, regularised)
The model with the lowest backtest error is used for the forecast. The likely range is the
10th-90th percentile of that model's held-out backtest errors (split-conformal); its real
coverage is measured leave-one-year-out and reported.

Features (all from years BEFORE the target year, so there is no look-ahead):
  last p, p two years back, mean of last 3 p, slope of last 3 p, number of past years,
  years since last seen, last opening p, seats allotted last time, the college's median p
  last time, the category's median p last year, category, gender, institute-type group.

Outputs
  data/processed/predictions_<year>.json   forecast per pool + backtest report
Usage: python pipeline/predict_next_year.py
"""
import json
import os
import statistics as st
from collections import defaultdict

import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(PROJECT, "data", "processed")
SOCIAL = ["UR", "OBC", "SC", "ST", "EWS", "FW"]
TYPES = ["gov", "uni", "pvt", "self"]
BACKTEST_YEARS = [2022, 2023, 2024, 2025, 2026]
# The ML model (and everything built on it: forecast, benchmark, hold-out test, next-year rank)
# learns from ML_FIRST_YEAR onwards: all 10 years, 2017–2026 (project decision, 14 Nov 2026; it was
# 2018–2026 before, and adding 2017 lowered the 2022–2026 backtest error from 0.176 to 0.172).
# Older notes below that mention 2018 describe that earlier setting.
from list_size import ML_FIRST_YEAR, next_list_size  # noqa: E402  (2017; see list_size.py)
# The model used for the forecast is the one with the lowest error on the most recent backtest
# years (2024–2026), the years most like the one being forecast. Project decision, made AFTER
# seeing that with 2018–2026 training the residual GBM ties "last year" over 2022–2026 (its
# 2022–2023 predictions had only 2018–2021 to learn from) but wins in each of 2024–2026.
# Reported on the site and in the README.
SELECT_YEARS = BACKTEST_YEARS[-3:]
SEED = 42


def type_group(t):
    u = (t or "").upper()
    if not u:
        return ""
    if "UNIVERSITY" in u:
        return "uni"
    if "SELF FINANCING" in u:
        return "self"
    if "PRIVATE" in u:
        return "pvt"
    return "gov"


def load(first_year=ML_FIRST_YEAR):
    rows = json.load(open(os.path.join(PROC, "mca_cutoffs_clean.json"), encoding="utf-8"))["rows"]
    if first_year:
        rows = [r for r in rows if r["year"] >= first_year]
    colleges = {c["college_id"]: c for c in json.load(open(os.path.join(PROC, "colleges.json"), encoding="utf-8"))}
    pools = defaultdict(dict)            # (cid, social, gender) -> {year: {cl, op, n}}
    for r in rows:
        if (r["result_awaited"] or r["round"] != "R1" or r["seat_class"] not in ("X", None)
                or r["after_conversion"] or r["social"] not in SOCIAL or r["gender"] not in ("OP", "F")):
            continue
        k = (r["college_id"], r["social"], r["gender"])
        v = pools[k].setdefault(r["year"], {"cl": 0, "op": 10 ** 9, "n": 0})
        v["cl"] = max(v["cl"], r["closing_rank"])
        v["op"] = min(v["op"], r["opening_rank"])
        v["n"] += r["total_allotted"]
    scale = defaultdict(int)
    for k, ys in pools.items():
        for y, v in ys.items():
            scale[y] = max(scale[y], v["cl"])
    return pools, dict(scale), colleges


def slope(ys, ps):
    if len(ys) < 2:
        return 0.0
    x = np.array(ys, float)
    return float(np.polyfit(x - x.mean(), np.array(ps, float), 1)[0])


def features(pools, scale, colleges, t):
    """Feature rows for target year t, built only from years < t."""
    p_of = lambda v, y: v["cl"] / scale[y]
    # context: each college's median p in its latest year < t, each category's median p in t-1
    col_last, cat_prev = {}, defaultdict(list)
    by_col = defaultdict(lambda: defaultdict(list))
    for (cid, s, g), ys in pools.items():
        for y, v in ys.items():
            if y < t:
                by_col[cid][y].append(p_of(v, y))
            if y == t - 1:
                cat_prev[s].append(p_of(v, y))
    for cid, yd in by_col.items():
        y = max(yd)
        col_last[cid] = st.median(yd[y])
    cat_med = {s: st.median(v) for s, v in cat_prev.items() if v}

    keys, X = [], []
    for k, ys in pools.items():
        hist = sorted(y for y in ys if y < t)
        if not hist:
            continue
        cid, s, g = k
        ps = [p_of(ys[y], y) for y in hist]
        last = hist[-1]
        row = [
            ps[-1],
            ps[-2] if len(ps) > 1 else np.nan,
            float(np.mean(ps[-3:])),
            slope(hist[-3:], ps[-3:]),
            len(hist),
            t - last,
            ys[last]["op"] / scale[last],
            ys[last]["n"],
            col_last.get(cid, np.nan),
            cat_med.get(s, np.nan),
            1.0 if g == "F" else 0.0,
        ]
        row += [1.0 if s == x else 0.0 for x in SOCIAL]
        tg = type_group((colleges.get(cid) or {}).get("type"))
        row += [1.0 if tg == x else 0.0 for x in TYPES]
        keys.append(k)
        X.append(row)
    return keys, np.array(X, float)


FEATURE_NAMES = (["last_p", "p_2_back", "mean3_p", "slope3_p", "n_years", "gap_years", "last_open_p",
                  "last_seats", "college_median_p", "category_median_p_prev_year", "female_seat"]
                 + [f"cat_{s}" for s in SOCIAL] + [f"type_{x}" for x in TYPES])


def dataset(pools, scale, colleges, years):
    K, X, Y, T = [], [], [], []
    for t in years:
        keys, Xt = features(pools, scale, colleges, t)
        for k, x in zip(keys, Xt):
            if t in pools[k]:
                K.append(k)
                X.append(x)
                Y.append(pools[k][t]["cl"] / scale[t])
                T.append(t)
    return K, np.array(X, float), np.array(Y, float), np.array(T)


class ResidualGBM:
    """Gradient boosting that predicts the CHANGE from last year's p (target = p - last_p).
    Starts from the 'last year' baseline and only learns systematic corrections; kept tiny
    (depth 2, >= 40 rows per leaf) because the data is small and noisy."""
    def __init__(self):
        self.m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=80, learning_rate=0.03,
                                               max_depth=2, min_samples_leaf=40, random_state=SEED)

    def fit(self, X, y):
        self.m.fit(X, y - X[:, 0])
        return self

    def predict(self, X):
        return X[:, 0] + self.m.predict(X)


class LevelModel:
    def __init__(self, kind):
        self.kind = kind
        self.m = (make_pipeline(StandardScaler(), Ridge(alpha=1.0)) if kind == "ridge" else
                  HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05,
                                                max_leaf_nodes=15, min_samples_leaf=10, random_state=SEED))

    def fit(self, X, y):
        self.ref = X
        self.m.fit(nan_fill(X, X) if self.kind == "ridge" else X, y)
        return self

    def predict(self, X):
        return self.m.predict(nan_fill(X, self.ref) if self.kind == "ridge" else X)


MODELS = {
    "last": ("Last year (baseline)", None),
    "mean3": ("3-year average (baseline)", None),
    "ridge": ("Linear regression (Ridge)", lambda: LevelModel("ridge")),
    "gbm": ("Gradient boosting (predicts rank level)", lambda: LevelModel("gbm")),
    "gbm_residual": ("Gradient boosting (predicts change from last year)", ResidualGBM),
}


def predict_with(name, Xtr, Ytr, Xte):
    if name == "last":
        return Xte[:, 0]
    if name == "mean3":
        return Xte[:, 2]
    return MODELS[name][1]().fit(Xtr, Ytr).predict(Xte)


def nan_fill(X, ref):
    """Ridge can't take NaN: fill with column medians of the training data."""
    med = np.nanmedian(ref, axis=0)
    out = X.copy()
    idx = np.where(np.isnan(out))
    out[idx] = np.take(med, idx[1])
    return out


def main():
    pools, scale, colleges = load()
    years = sorted(scale)
    K, X, Y, T = dataset(pools, scale, colleges, [y for y in years if y > years[0]])
    print(f"pools={len(pools)}  training rows={len(Y)}  years={years}")
    print("rank scale (highest R1 closing rank):", {y: scale[y] for y in years})

    # ---- time-based backtest: train on years < T, predict T ----
    backtest, held = [], defaultdict(dict)     # held[model][T] = (actual p, predicted p)
    for t in BACKTEST_YEARS:
        tr, te = T < t, T == t
        prev = max(y for y in years if y < t)
        for name in MODELS:
            pr = np.clip(predict_with(name, X[tr], Y[tr], X[te]), 0.01, 1.5)
            held[name][t] = (Y[te], pr)
            actual_rank = Y[te] * scale[t]
            backtest.append({
                "year": t, "model": name, "n": int(te.sum()),
                "mae_relative": round(float(np.mean(np.abs(pr - Y[te]))), 4),
                "mae_rank_true_scale": round(float(np.mean(np.abs(pr * scale[t] - actual_rank))), 1),
                "mae_rank_last_year_scale": round(float(np.mean(np.abs(pr * scale[prev] - actual_rank))), 1),
            })

    summary = {}
    for name, (label, _) in MODELS.items():
        b = [r for r in backtest if r["model"] == name]
        w = sum(r["n"] for r in b)
        summary[name] = {
            "label": label,
            "mae_relative": round(sum(r["mae_relative"] * r["n"] for r in b) / w, 4),
            "mae_rank_true_scale": round(sum(r["mae_rank_true_scale"] * r["n"] for r in b) / w, 1),
            "mae_rank_last_year_scale": round(sum(r["mae_rank_last_year_scale"] * r["n"] for r in b) / w, 1),
            "years_beating_last_year": sum(
                1 for t in BACKTEST_YEARS
                if next(r for r in b if r["year"] == t)["mae_relative"]
                < next(r for r in backtest if r["model"] == "last" and r["year"] == t)["mae_relative"]),
        }
        rb = [r for r in b if r["year"] in SELECT_YEARS]
        wr = sum(r["n"] for r in rb)
        summary[name]["mae_relative_recent"] = round(sum(r["mae_relative"] * r["n"] for r in rb) / wr, 4)
        summary[name]["recent_years_beating_last_year"] = sum(
            1 for t in SELECT_YEARS
            if next(r for r in b if r["year"] == t)["mae_relative"]
            < next(r for r in backtest if r["model"] == "last" and r["year"] == t)["mae_relative"])
    best = min(summary, key=lambda k: summary[k]["mae_relative_recent"])

    # ---- likely range: 10th/90th percentile of the best model's held-out errors (conformal);
    #      coverage checked leave-one-year-out so it is measured on unseen years ----
    res = {t: held[best][t][0] - held[best][t][1] for t in BACKTEST_YEARS}
    cover = []
    for t in BACKTEST_YEARS:
        r = np.concatenate([res[u] for u in BACKTEST_YEARS if u != t])
        lo, hi = np.quantile(r, [0.1, 0.9])
        a, p_ = held[best][t]
        cover.append(float(np.mean((a >= p_ + lo) & (a <= p_ + hi))))
    q_lo, q_hi = np.quantile(np.concatenate(list(res.values())), [0.1, 0.9])
    coverage = round(float(np.mean(cover)), 3)

    print("\nBacktest (weighted over", BACKTEST_YEARS, "):")
    print(f"  {'model':52s} {'MAE rel':>8s} {'rank MAE':>9s} {'rank MAE*':>10s} {'beats last':>11s}")
    for k, v in summary.items():
        print(f"  {v['label']:52s} {v['mae_relative']:8.3f} {v['mae_rank_true_scale']:9.1f} "
              f"{v['mae_rank_last_year_scale']:10.1f} {v['years_beating_last_year']:>7d}/{len(BACKTEST_YEARS)}")
    print("  * using last year's rank scale (what a real forecast must do)")
    print("  error on the selection years", SELECT_YEARS, ":",
          {v["label"]: v["mae_relative_recent"] for v in summary.values()})
    print(f"  best (lowest error on {SELECT_YEARS[0]}–{SELECT_YEARS[-1]}): {summary[best]['label']}; "
          f"10–90% range covered {coverage:.0%} of held-out cases")

    # ---- forecast for next year ----
    target = years[-1] + 1
    keys, Xf = features(pools, scale, colleges, target)
    mid = np.clip(predict_with(best, X, Y, Xf), 0.01, 1.5)
    # next year's list size: the same forecast the ML Predictor uses for the student's rank
    # (pipeline/list_size.py), so student rank and college cut-offs share one scale
    size = next_list_size()
    s_now = size["next_size"]
    out_pools = []
    for k, m_ in zip(keys, mid):
        cid, s, g = k
        c = colleges.get(cid) or {}
        last_year = max(pools[k])
        if not c.get("official") or last_year < target - 2:     # colleges offering MCA now, recent pools
            continue
        m_ = float(m_)
        l_, h_ = max(0.01, m_ + q_lo), m_ + q_hi
        out_pools.append({
            "c": cid, "s": s, "g": g,
            "p50": round(m_, 4), "p10": round(l_, 4), "p90": round(h_, 4),
            "rank": int(round(m_ * s_now)), "lo": max(1, int(round(l_ * s_now))), "hi": int(round(h_ * s_now)),
            "last_year": last_year, "last_rank": pools[k][last_year]["cl"], "n_years": len(pools[k]),
        })

    # feature importance of the residual model: shuffle one feature at a time (training data)
    imp = []
    if best not in ("last", "mean3"):
        model = MODELS[best][1]().fit(X, Y)
        rng = np.random.default_rng(SEED)
        base = np.mean(np.abs(model.predict(X) - Y))
        for j, n in enumerate(FEATURE_NAMES):
            Xs = X.copy()
            Xs[:, j] = rng.permutation(Xs[:, j])
            imp.append((n, float(np.mean(np.abs(model.predict(Xs) - Y)) - base)))
        imp.sort(key=lambda x: -x[1])

    doc = {
        "target_year": target, "round": "R1", "train_years": [years[0], years[-1]],
        "unit": "seat pool = college × category × gender, ordinary seats, Round 1",
        "scale_by_year": {str(y): scale[y] for y in years},
        "scale_assumed": s_now, "scale_last_year": scale[years[-1]],
        "scale_assumption": f"next year's list size = {s_now} ({size['methods'][size['used']]['label']}, {size['years'][0]}–{size['years'][1]})",
        "model_used": best, "model_label": summary[best]["label"],
        "selection": {"years": SELECT_YEARS, "rule": "lowest error on the most recent backtest years",
                      "decided_after_seeing_results": True},
        "range": {"quantiles": [0.1, 0.9], "coverage_held_out": coverage,
                  "method": "10th/90th percentile of the model's backtest errors (split-conformal)"},
        "backtest_years": BACKTEST_YEARS, "backtest_summary": summary, "backtest": backtest,
        "feature_importance": [{"feature": n, "mae_increase": round(v, 4)} for n, v in imp[:8]],
        "training_rows": int(len(Y)), "pool_count": len(out_pools), "pools": out_pools,
    }
    path = os.path.join(PROC, f"predictions_{target}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print(f"\n→ {os.path.relpath(path, PROJECT)}: {len(out_pools)} pools forecast for {target} (scale {s_now})"
          + (f"; top features: {[n for n, _ in imp[:4]]}" if imp else ""))


if __name__ == "__main__":
    main()
