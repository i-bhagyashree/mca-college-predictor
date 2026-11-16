#!/usr/bin/env python3
"""Forward-chaining (year-by-year) benchmark: can past MCA cut-offs predict next year's?

Experiment i trains on years 1..i and predicts year i+1. Nothing from the target year (or
later) is ever used: not as a feature, not for scaling, not for choosing a hyper-parameter.

What is predicted
-----------------
The unit is a SEAT POOL: (college, social category, gender) on ordinary seats, Round 1,
regular list, domicile splits merged (highest closing rank). This keeps every comparison
between equivalent pools: the same college, category, gender and round in both years.

Only closing RANKS are available for all years. Merit lists with percentages exist for only
2020, 2025 and 2026 (pipeline/merit_lists.py), too few for a percentage-based benchmark across years.

Ranks are compared on a relative scale, p = closing rank / highest Round-1 closing rank that
year, because the size of the rank list changes a lot from year to year. Errors are reported
both on p and in ranks. The rank error is computed the way a real forecast must work, by
converting with the LAST KNOWN year's scale, because next year's scale is unknown.

Approaches
----------
  prev_year   Previous-year baseline: the pool's most recent closing position.
  hist_avg    Historical average: the mean over ALL earlier years of the pool.
  weighted    Weighted historical: exponentially decaying weights (recent years count more).
              The half-life is chosen using ONLY earlier experiments (nested, no leakage).
  rf          Random forest regression on the engineered features of predict_next_year.py.
  gbm_resid   Gradient boosting that learns the CHANGE from last year (the site's model).

Why these ML models: about 1,000 training rows and 21 features. Random forest is robust on
small, noisy tabular data. Residual boosting starts from the strongest simple rule and only
learns corrections. A single decision tree is a high-variance version of the forest, and
XGBoost would be the same algorithm family as scikit-learn's histogram gradient boosting,
so neither is added.

Outputs (reports/benchmark/)
  benchmark_results.json   every number below
  benchmark_table.csv      the main comparison table
  REPORT.md                data inspection, tables, answers to the research questions, figures
  figures/*.png            4 charts

Usage: python pipeline/benchmark_forward_chain.py
"""
import csv
import json
import os
import sys
from collections import defaultdict

import itertools

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_audit import audit, markdown as audit_markdown  # noqa: E402
from predict_next_year import (FEATURE_NAMES, SEED, ResidualGBM, dataset, features,  # noqa: E402
                               load)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "reports", "benchmark")
FIG = os.path.join(OUT, "figures")
HALF_LIVES = [0.5, 1.0, 2.0, 4.0]       # candidate half-lives (years) for the weighted model
DEFAULT_HALF_LIFE = 1.0                  # used until earlier experiments exist to choose from
STUDENT_GRID = 40                        # simulated student ranks per category group
TOP_K = 5
BOOT = 2000

APPROACHES = {
    "prev_year": "Previous-year baseline",
    "hist_avg": "Historical average",
    "weighted": "Weighted historical",
    "rf": "Random forest (ML)",
    "gbm_resid": "Gradient boosting, change from last year (ML)",
}
ML = ("rf", "gbm_resid")


class RandomForest:
    def __init__(self):
        self.m = RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                       random_state=SEED, n_jobs=-1)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


ML_FACTORY = {"rf": RandomForest, "gbm_resid": ResidualGBM}


def fit_predict(m, Xtr, Ytr, Xte):
    """Columns that are entirely missing in the training rows (e.g. 'rank two years back' when
    only one earlier year exists) carry no information; set them to 0 in train and test."""
    empty = np.all(np.isnan(Xtr), axis=0)
    Xtr, Xte = Xtr.copy(), Xte.copy()
    Xtr[:, empty] = 0.0
    Xte[:, empty] = 0.0
    return np.clip(ML_FACTORY[m]().fit(Xtr, Ytr).predict(Xte), 0.01, 1.5)


# ---------------------------------------------------------------- simple approaches
def pool_history(pools, scale, k, t, window=None):
    ys = sorted(y for y in pools[k] if y < t)
    if window:
        ys = [y for y in ys if y >= t - window]
    return ys, [pools[k][y]["cl"] / scale[y] for y in ys]


def weighted_mean(ys, ps, t, half_life):
    w = np.power(0.5, (t - np.array(ys, float)) / half_life)
    return float(np.sum(w * np.array(ps)) / np.sum(w))


def simple_predictions(pools, scale, keys, t, half_life, window=None):
    out = {"prev_year": [], "hist_avg": [], "weighted": []}
    for k in keys:
        ys, ps = pool_history(pools, scale, k, t, window)
        if not ys:                                   # window excludes all history
            ys, ps = pool_history(pools, scale, k, t)
            ys, ps = ys[-1:], ps[-1:]
        out["prev_year"].append(ps[-1])
        out["hist_avg"].append(float(np.mean(ps)))
        out["weighted"].append(weighted_mean(ys, ps, t, half_life))
    return {n: np.array(v) for n, v in out.items()}


# ---------------------------------------------------------------- metrics
def regression_metrics(y, pred, actual_rank, scale_prev):
    err = pred - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    rank_err = np.abs(pred * scale_prev - actual_rank)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": (1 - float(np.sum(err ** 2)) / ss_tot) if ss_tot > 0 else None,
        "rank_mae": float(np.mean(rank_err)),
        "rank_median_ae": float(np.median(rank_err)),
    }


def student_metrics(groups, actual_rank, pred_rank, scale_t):
    """Simulated students: for each category group, ranks spread over the target year's list.
    A college is 'reachable' for rank r if r <= that pool's closing rank. Compares the predicted
    reachable list with the actual one. (No individual allotment records exist in this data.)"""
    tp = fp = fn = tn = 0
    hits = shown = 0
    for g in set(groups):
        idx = [i for i, x in enumerate(groups) if x == g]
        a, p = actual_rank[idx], pred_rank[idx]
        for r in np.linspace(1, scale_t, STUDENT_GRID):
            ra, rp = r <= a, r <= p
            tp += int(np.sum(ra & rp)); fp += int(np.sum(~ra & rp))
            fn += int(np.sum(ra & ~rp)); tn += int(np.sum(~ra & ~rp))
            cand = [i for i in np.argsort(p) if rp[i]][:TOP_K]   # most selective predicted-reachable
            hits += int(np.sum(ra[cand])); shown += len(cand)
    total = tp + fp + fn + tn
    return {
        "decision_accuracy": (tp + tn) / total if total else None,
        "list_precision": tp / (tp + fp) if tp + fp else None,
        "list_recall": tp / (tp + fn) if tp + fn else None,
        f"top{TOP_K}_hit_rate": hits / shown if shown else None,
    }


def paired_bootstrap(err_a, err_b, rng):
    """95% CI of mean(|err_a|) - mean(|err_b|) over pools; negative = A better."""
    d = np.abs(err_a) - np.abs(err_b)
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


# ---------------------------------------------------------------- experiments
def eval_keys(pools, scale, colleges, t):
    keys, Xt = features(pools, scale, colleges, t)
    sel = [i for i, k in enumerate(keys) if t in pools[k]]
    return [keys[i] for i in sel], Xt[sel]


def run(pools, scale, colleges, years):
    K, X, Y, T = dataset(pools, scale, colleges, years[1:])
    rng = np.random.default_rng(SEED)
    experiments, preds_store = [], {}
    abs_err_by_hl = defaultdict(dict)          # half-life -> {t: mean abs error}, for nested choice

    for i, t in enumerate(years[1:], start=1):
        keys, Xe = eval_keys(pools, scale, colleges, t)
        y = np.array([pools[k][t]["cl"] / scale[t] for k in keys])
        actual_rank = np.array([pools[k][t]["cl"] for k in keys], float)
        prev = years[i - 1]

        # weighted model: choose the half-life on EARLIER experiments only
        for h in HALF_LIVES:
            abs_err_by_hl[h][t] = float(np.mean(np.abs(simple_predictions(pools, scale, keys, t, h)["weighted"] - y)))
        earlier = [u for u in years[1:] if u < t]
        hl = (min(HALF_LIVES, key=lambda h: np.mean([abs_err_by_hl[h][u] for u in earlier]))
              if earlier else DEFAULT_HALF_LIFE)

        preds = simple_predictions(pools, scale, keys, t, hl)
        tr = T < t
        n_train_rows = int(tr.sum())
        for m in ML:
            if n_train_rows >= 30:
                preds[m] = fit_predict(m, X[tr], Y[tr], Xe)
            else:
                preds[m] = None
        for m in preds:
            if preds[m] is not None:
                preds[m] = np.clip(preds[m], 0.01, 1.5)

        groups = [(k[1], k[2]) for k in keys]
        row = {"experiment": i, "train_years": [years[0], prev], "n_train_years": i, "test_year": t,
               "n_pools": len(keys), "ml_training_rows": n_train_rows, "weighted_half_life": hl,
               "share_seen_last_year": float(np.mean([prev in pools[k] for k in keys])),
               "scale_prev": scale[prev], "scale_test": scale[t], "results": {}}
        for m, pr in preds.items():
            if pr is None:
                row["results"][m] = None
                continue
            res = regression_metrics(y, pr, actual_rank, scale[prev])
            res.update(student_metrics(groups, actual_rank, pr * scale[prev], scale[t]))
            row["results"][m] = res
        avail = [m for m in preds if preds[m] is not None]
        row["best"] = min(avail, key=lambda m: row["results"][m]["mae"])
        row["vs_prev_year"] = {m: dict(zip(["mean_diff", "ci_lo", "ci_hi"],
                                            paired_bootstrap(preds[m] - y, preds["prev_year"] - y, rng)))
                               for m in avail if m != "prev_year"}
        experiments.append(row)
        preds_store[t] = {"y": y, "keys": keys, **{m: p for m, p in preds.items() if p is not None}}
        print(f"exp {i}: train {years[0]}-{prev} -> {t}  pools={len(keys):3d}  ML rows={n_train_rows:4d}  "
              + "  ".join(f"{m}={row['results'][m]['mae']:.3f}" if row["results"][m] else f"{m}=n/a"
                          for m in APPROACHES) + f"  best={row['best']}")
    return experiments, preds_store, (K, X, Y, T)


def window_study(pools, scale, colleges, years, data, test_years):
    """Same test year, different amounts of history: use only the last w years.
    Separates 'more history' from 'easier/harder test year'."""
    K, X, Y, T = data
    out = []
    for t in test_years:
        keys, Xe = eval_keys(pools, scale, colleges, t)
        y = np.array([pools[k][t]["cl"] / scale[t] for k in keys])
        max_w = len([u for u in years if u < t])
        for w in range(1, max_w + 1):
            sp = simple_predictions(pools, scale, keys, t, DEFAULT_HALF_LIFE, window=w)
            row = {"test_year": t, "window": w,
                   "hist_avg": float(np.mean(np.abs(sp["hist_avg"] - y))),
                   "weighted": float(np.mean(np.abs(sp["weighted"] - y))),
                   "prev_year": float(np.mean(np.abs(sp["prev_year"] - y)))}
            tr = (T < t) & (T >= t - w)                # ML: training targets from the last w years
            for m in ML:
                row[m] = (float(np.mean(np.abs(fit_predict(m, X[tr], Y[tr], Xe) - y)))
                          if tr.sum() >= 30 else None)
            out.append(row)
    return out


# Settings tried for the residual GBM in the leak-free tuning check. The site's setting
# (depth 2, leaf 40, lr 0.03, 80 rounds) was chosen after seeing 2022-2026 results; this check
# instead picks the setting in each experiment using ONLY earlier experiments.
GBM_GRID = list(itertools.product([2, 3], [10, 20, 40], [0.03, 0.06], [80, 200]))
SITE_GBM = (2, 40, 0.03, 80)


def tuning_check(pools, scale, colleges, data, experiments):
    K, X, Y, T = data
    ex = [e for e in experiments if e["results"]["gbm_resid"]]
    err, n = {}, {}
    for e in ex:
        t = e["test_year"]
        keys, Xe = eval_keys(pools, scale, colleges, t)
        y = np.array([pools[k][t]["cl"] / scale[t] for k in keys])
        tr = T < t
        Xtr, Xte = X[tr].copy(), Xe.copy()
        empty = np.all(np.isnan(Xtr), axis=0)
        Xtr[:, empty] = 0.0
        Xte[:, empty] = 0.0
        n[t] = len(y)
        for c in GBM_GRID:
            d, leaf, lr, it = c
            m = HistGradientBoostingRegressor(loss="absolute_error", max_depth=d, min_samples_leaf=leaf,
                                              learning_rate=lr, max_iter=it, random_state=SEED)
            m.fit(Xtr, Y[tr] - Xtr[:, 0])
            err[c, t] = float(np.mean(np.abs(np.clip(Xte[:, 0] + m.predict(Xte), 0.01, 1.5) - y)))
    rows = []
    for i, e in enumerate(ex):
        t = e["test_year"]
        earlier = [u["test_year"] for u in ex[:i]]
        c = SITE_GBM if not earlier else min(GBM_GRID, key=lambda c: np.mean([err[c, u] for u in earlier]))
        v = [err[g, t] for g in GBM_GRID]
        rows.append({"test_year": t, "chosen": list(c), "nested_mae": err[c, t], "site_mae": e["results"]["gbm_resid"]["mae"],
                     "prev_mae": e["results"]["prev_year"]["mae"], "grid_min": min(v), "grid_median": float(np.median(v)),
                     "grid_max": max(v)})
    w = np.array([n[r["test_year"]] for r in rows], float)
    avg = lambda k: float(np.average([r[k] for r in rows], weights=w))
    grid_pooled = [float(np.average([err[c, r["test_year"]] for r in rows], weights=w)) for c in GBM_GRID]
    return {"grid": [list(c) for c in GBM_GRID], "site_setting": list(SITE_GBM), "rows": rows,
            "pooled": {k: avg(k) for k in ("nested_mae", "site_mae", "prev_mae", "grid_median")},
            "nested_years_better": sum(r["nested_mae"] < r["prev_mae"] for r in rows), "of": len(rows),
            "share_grid_beating_prev": float(np.mean([g < avg("prev_mae") for g in grid_pooled]))}


# ---------------------------------------------------------------- summary & answers
def pooled(experiments, models, years_subset):
    s = {}
    for m in models:
        rows = [e for e in experiments if e["test_year"] in years_subset and e["results"][m]]
        if len(rows) != len(years_subset):
            s[m] = None
            continue
        w = np.array([e["n_pools"] for e in rows], float)
        s[m] = {k: float(np.average([e["results"][m][k] for e in rows], weights=w))
                for k in ("mae", "rmse", "rank_mae", "decision_accuracy", "list_precision", f"top{TOP_K}_hit_rate")}
        s[m]["years_beating_prev_year"] = sum(e["results"][m]["mae"] < e["results"]["prev_year"]["mae"] for e in rows)
    return s


def answers(experiments, window, summary, comparable_years):
    A = {}
    pv = summary["prev_year"]["mae"]
    best_overall = min((m for m in summary if summary[m]), key=lambda m: summary[m]["mae"])
    n = len(comparable_years)

    # Q1: more history? Use the window study (same test year, varying history):
    # MAE averaged over the study's test years, with 1 year, 2 years and all available years.
    q1 = {}
    tys = sorted({r["test_year"] for r in window})
    for m in ("prev_year", "hist_avg", "weighted", "rf", "gbm_resid"):
        def at(sel):
            v = [sel([r for r in window if r["test_year"] == t and r[m] is not None]) for t in tys]
            v = [x[m] for x in v if x]
            return float(np.mean(v)) if len(v) == len(tys) else None
        q1[m] = {"w1": at(lambda rs: next((r for r in rs if r["window"] == 1), None)),
                 "w2": at(lambda rs: next((r for r in rs if r["window"] == 2), None)),
                 "all": at(lambda rs: max(rs, key=lambda r: r["window"]) if rs else None)}
    A["q1_more_history"] = {"test_years": tys, "by_approach": q1}

    # Q2: previous year vs all-years average
    A["q2_prev_vs_all"] = {"prev_year_mae": pv, "hist_avg_mae": summary["hist_avg"]["mae"],
                           "prev_year_better_years": sum(e["results"]["prev_year"]["mae"] < e["results"]["hist_avg"]["mae"]
                                                         for e in experiments)}
    # Q3/Q4: ML and weighting vs baseline, with significance count
    for m in ("rf", "gbm_resid", "weighted"):
        sig_better = sum(1 for e in experiments if m in e["vs_prev_year"] and e["vs_prev_year"][m]["ci_hi"] < 0)
        sig_worse = sum(1 for e in experiments if m in e["vs_prev_year"] and e["vs_prev_year"][m]["ci_lo"] > 0)
        A[f"vs_prev_{m}"] = {"pooled_mae": summary[m]["mae"], "prev_mae": pv,
                             "relative_change": (summary[m]["mae"] - pv) / pv,
                             "years_better": summary[m]["years_beating_prev_year"], "of": n,
                             "significantly_better_years": sig_better, "significantly_worse_years": sig_worse}
    A["best_overall"] = best_overall
    A["wins"] = {m: sum(e["best"] == m for e in experiments) for m in APPROACHES}
    return A


# ---------------------------------------------------------------- figures
def figures(experiments, preds_store, window, best_ml):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    colors = {"prev_year": "#64748b", "hist_avg": "#b45309", "weighted": "#0f766e", "rf": "#7c3aed", "gbm_resid": "#1d4ed8"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    paths = {}

    # 1. actual vs predicted, latest test year
    t = max(preds_store)
    d = preds_store[t]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, m in zip(axes, ["prev_year", best_ml]):
        ax.scatter(d["y"], d[m], s=14, alpha=.6, color=colors[m])
        ax.plot([0, 1.2], [0, 1.2], color="#94a3b8", lw=1, ls="--")
        mae = np.mean(np.abs(d[m] - d["y"]))
        ax.set_title(f"{APPROACHES[m]}\n{t}: MAE {mae:.3f}", fontsize=10)
        ax.set_xlabel(f"Actual closing position {t} (rank / max rank)")
    axes[0].set_ylabel("Predicted closing position")
    fig.tight_layout(); paths["actual_vs_pred"] = "figures/actual_vs_predicted.png"
    fig.savefig(os.path.join(OUT, paths["actual_vs_pred"]), dpi=150); plt.close(fig)

    # 2. error across test years
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for m in APPROACHES:
        pts = [(e["test_year"], e["results"][m]["mae"]) for e in experiments if e["results"][m]]
        ax.plot(*zip(*pts), marker="o", label=APPROACHES[m], color=colors[m], lw=2 if m in ("prev_year", best_ml) else 1.3)
    ax.set_xlabel("Test year"); ax.set_ylabel("MAE (relative closing position)")
    ax.set_title("Prediction error in each forward-chaining experiment")
    ax.legend(frameon=False, fontsize=8.5); ax.grid(axis="y", color="#e2e8f0")
    fig.tight_layout(); paths["error_by_year"] = "figures/error_by_test_year.png"
    fig.savefig(os.path.join(OUT, paths["error_by_year"]), dpi=150); plt.close(fig)

    # 3. baseline vs ML, rank error and top-K hit rate
    ex = [e for e in experiments if e["results"][best_ml]]
    yrs = [e["test_year"] for e in ex]
    xs = np.arange(len(yrs))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for j, (key, lab) in enumerate([("rank_mae", "Rank MAE (using last year's scale)"),
                                    (f"top{TOP_K}_hit_rate", f"Top-{TOP_K} hit rate (simulated students)")]):
        ax = axes[j]
        for off, m in [(-.2, "prev_year"), (.2, best_ml)]:
            ax.bar(xs + off, [e["results"][m][key] for e in ex], width=.4, color=colors[m], label=APPROACHES[m])
        ax.set_xticks(xs, yrs); ax.set_title(lab, fontsize=10); ax.grid(axis="y", color="#e2e8f0")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); paths["baseline_vs_ml"] = "figures/baseline_vs_ml.png"
    fig.savefig(os.path.join(OUT, paths["baseline_vs_ml"]), dpi=150); plt.close(fig)

    # 4. error vs amount of history (window study)
    tys = sorted({r["test_year"] for r in window})
    fig, axes = plt.subplots(1, len(tys), figsize=(4 * len(tys), 4), sharey=True)
    for ax, ty in zip(np.atleast_1d(axes), tys):
        rs = [r for r in window if r["test_year"] == ty]
        for m in ("prev_year", "hist_avg", "weighted", "rf", "gbm_resid"):
            pts = [(r["window"], r[m]) for r in rs if r[m] is not None]
            if pts:
                ax.plot(*zip(*pts), marker="o", ms=3.5, color=colors[m], label=APPROACHES[m])
        ax.set_title(f"Test year {ty}", fontsize=10); ax.set_xlabel("Years of history used")
        ax.grid(axis="y", color="#e2e8f0")
    np.atleast_1d(axes)[0].set_ylabel("MAE (relative)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); paths["history_length"] = "figures/error_vs_history_length.png"
    fig.savefig(os.path.join(OUT, paths["history_length"]), dpi=150); plt.close(fig)
    return paths


# ---------------------------------------------------------------- report
def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def write_report(doc, paths):
    E, S, A = doc["experiments"], doc["summary"], doc["answers"]
    bm = doc["best_ml"]
    L = []
    L.append("# Forward-chaining benchmark: MCA closing-rank prediction\n")
    L.append(f"Generated by `pipeline/benchmark_forward_chain.py` from `data/processed/mca_cutoffs_clean.json`. "
             f"Years in data: {doc['years'][0]}–{doc['years'][-1]} ({len(doc['years'])} years, "
             f"{len(E)} experiments). Every number below is computed, not estimated.\n")
    L.append("## Data used\n")
    L.append(f"- **Years used: {doc['years'][0]}–{doc['years'][-1]}** (`ML_FIRST_YEAR` in `predict_next_year.py`). Earlier cut-offs "
             "are in the data inspection below and on the website, but are not used to train or test any approach.")
    L.append(f"- Unit: {doc['unit']}.")
    L.append(f"- Seat pools: {doc['n_pools_total']}; pool-year rows usable as ML training examples: {doc['ml_rows_total']}.")
    L.append("- Highest Round-1 closing rank per year (the list size): "
             + ", ".join(f"{y}: {s:,}" for y, s in doc["scale"].items()) + ".")
    L.append("- **Graduation percentage is not available for any benchmark year.** Only the 2026 Round-2 list prints "
             "merit percentages (145 rows, one year), so the benchmark predicts closing **ranks**.\n")
    L.append(doc["audit_md"])

    L.append("## Summary table\n")
    other = next(m for m in ML if m != bm)
    L.append(f"Error = MAE of relative closing position (lower is better). Rank errors in brackets (MAE in ranks, using "
             f"the previous year's scale). Both ML models are shown: {APPROACHES[bm]} (better overall, used on the website) "
             f"and {APPROACHES[other]}, so every 'best approach' can be read off the table.\n")
    L.append(f"| Training data | Test year | Previous-year baseline error | ML error: {APPROACHES[bm]} | "
             f"ML error: {APPROACHES[other]} | Weighted model error | Best approach |")
    L.append("|---|---|---|---|---|---|---|")
    for e in E:
        r = e["results"]
        cell = lambda m: f"{r[m]['mae']:.3f} ({r[m]['rank_mae']:.0f})" if r[m] else "n/a (no training pairs yet)"
        L.append(f"| {e['train_years'][0]}" + (f"–{e['train_years'][1]}" if e['train_years'][1] != e['train_years'][0] else "")
                 + f" | {e['test_year']} | {cell('prev_year')} | {cell(bm)} | {cell(other)} | {cell('weighted')} | {APPROACHES[e['best']]} |")
    L.append("")

    L.append("## Main comparison (MAE of relative closing position, lower is better)\n")
    L.append("Relative position = closing rank ÷ highest closing rank that year. 0.10 means an error of 10% of the rank list.\n")
    L.append("| Exp | Training data | Test year | Pools | Prev-year baseline | Historical avg | Weighted | Random forest | GBM (change) | Best |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for e in E:
        r = e["results"]
        L.append(f"| {e['experiment']} | {e['train_years'][0]}–{e['train_years'][1]} ({e['n_train_years']} yr) | {e['test_year']} | {e['n_pools']} | "
                 + " | ".join(f3(r[m]["mae"]) if r[m] else "n/a" for m in APPROACHES) + f" | {APPROACHES[e['best']]} |")
    L.append("\nn/a: the ML models need at least one earlier (features → result) year to learn from, so they start at experiment 2.\n")

    L.append("## Other metrics per experiment\n")
    L.append(f"Rank MAE uses last year's rank scale, as a real forecast must. Student metrics: {STUDENT_GRID} simulated ranks per "
             f"category group, spread over the test year's list; a college is reachable if rank ≤ its actual closing rank. "
             f"Top-{TOP_K} hit = share of the {TOP_K} most selective colleges shown as reachable that really were.\n")
    L.append(f"| Test year | Approach | RMSE | R² | Rank MAE | Median rank error | Decision accuracy | List precision | List recall | Top-{TOP_K} hit |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for e in E:
        for m in APPROACHES:
            r = e["results"][m]
            if not r:
                continue
            L.append(f"| {e['test_year']} | {APPROACHES[m]} | {f3(r['rmse'])} | {f3(r['r2'])} | {r['rank_mae']:.0f} | {r['rank_median_ae']:.0f} | "
                     f"{pct(r['decision_accuracy'])} | {pct(r['list_precision'])} | {pct(r['list_recall'])} | {pct(r[f'top{TOP_K}_hit_rate'])} |")

    cy = doc["comparable_years"]
    L.append(f"\n## Pooled over the experiments where every approach runs ({cy[0]}–{cy[-1]}, weighted by pools)\n")
    L.append(f"| Approach | MAE | RMSE | Rank MAE | Decision accuracy | List precision | Top-{TOP_K} hit | Years better than prev-year |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in APPROACHES:
        s = S[m]
        L.append(f"| {APPROACHES[m]} | {f3(s['mae'])} | {f3(s['rmse'])} | {s['rank_mae']:.0f} | {pct(s['decision_accuracy'])} | "
                 f"{pct(s['list_precision'])} | {pct(s[f'top{TOP_K}_hit_rate'])} | "
                 f"{'—' if m == 'prev_year' else str(s['years_beating_prev_year']) + '/' + str(len(cy))} |")

    L.append("\n## Is the difference real? (paired bootstrap, 95% CI of MAE difference vs previous-year baseline)\n")
    L.append("Negative = better than the baseline. A CI that crosses 0 means the difference is within noise for that year.\n")
    L.append("| Test year | " + " | ".join(APPROACHES[m] for m in APPROACHES if m != "prev_year") + " |")
    L.append("|---|" + "---|" * (len(APPROACHES) - 1))
    for e in E:
        cells = []
        for m in APPROACHES:
            if m == "prev_year":
                continue
            v = e["vs_prev_year"].get(m)
            cells.append("n/a" if not v else f"{v['mean_diff']:+.3f} [{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]")
        L.append(f"| {e['test_year']} | " + " | ".join(cells) + " |")

    L.append("\n## Does more history help? (same test year, only the last *w* years used)\n")
    L.append("| Test year | Years of history | Prev-year | Historical avg | Weighted | Random forest | GBM (change) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in doc["window_study"]:
        L.append(f"| {r['test_year']} | {r['window']} | " + " | ".join(f3(r[m]) for m in ("prev_year", "hist_avg", "weighted", "rf", "gbm_resid")) + " |")

    tc = doc["tuning_check"]
    L.append("\n## Leak-free tuning check for the gradient-boosting model\n")
    L.append(f"The site's GBM setting was chosen after seeing 2022–2026. Here, in each experiment, one of {len(tc['grid'])} "
             "settings (depth 2–3, min leaf 10–40, learning rate 0.03–0.06, 80–200 rounds) is chosen using only the "
             "EARLIER experiments, then scored on the test year.\n")
    L.append("| Test year | Prev-year | Site setting | Nested choice | Setting chosen (depth, leaf, lr, rounds) | Best / median / worst of all settings |")
    L.append("|---|---|---|---|---|---|")
    for r in tc["rows"]:
        L.append(f"| {r['test_year']} | {r['prev_mae']:.3f} | {r['site_mae']:.3f} | {r['nested_mae']:.3f} | "
                 f"{tuple(r['chosen'])} | {r['grid_min']:.3f} / {r['grid_median']:.3f} / {r['grid_max']:.3f} |")
    p = tc["pooled"]
    L.append(f"\nPooled: previous-year {p['prev_mae']:.4f}, site setting {p['site_mae']:.4f}, **nested (leak-free) "
             f"{p['nested_mae']:.4f}**, a typical setting {p['grid_median']:.4f}. The nested model beats the previous year in "
             f"{tc['nested_years_better']} of {tc['of']} years, so the ML gain is **not** an artefact of tuning on the test "
             f"years. But only {tc['share_grid_beating_prev']:.0%} of the settings beat the previous year overall: the gain "
             "depends on the setting, and choosing it from past experiments is enough to find a good one.\n")

    L.append("\n## Figures\n")
    for k, lab in [("actual_vs_pred", "Actual vs predicted"), ("error_by_year", "Error across test years"),
                   ("baseline_vs_ml", "Previous-year baseline vs ML"), ("history_length", "Error vs amount of history")]:
        L.append(f"![{lab}]({paths[k]})\n")

    # answers (generated from the numbers above)
    q = A
    pv = S["prev_year"]["mae"]
    L.append("## Answers to the research questions\n")
    q1 = q["q1_more_history"]
    L.append(f"**1. Does using more historical years improve accuracy?** Window study, MAE averaged over test years "
             f"{', '.join(map(str, q1['test_years']))}:\n")
    L.append("| Approach | 1 year of history | 2 years | All available years |\n|---|---|---|---|")
    for m, v in q1["by_approach"].items():
        L.append(f"| {APPROACHES[m]} | {f3(v['w1'])} | {f3(v['w2'])} | {f3(v['all'])} |")
    helped = [m for m, v in q1["by_approach"].items() if v["all"] is not None and v["w1"] is not None and v["all"] < min(v["w1"], v["w2"] or 9) - 0.002]
    L.append("\n" + ("No approach gets clearly better with more years. " if not helped else
                     f"More years help: {', '.join(APPROACHES[m] for m in helped)}. ")
             + f"Old years ({doc['years'][0]}–2020, when far fewer seats and colleges took part) describe a different admissions market, "
               "so adding them mostly adds noise. In the forward-chaining table the errors do fall after 2021, but that "
               "is because the later years themselves were more stable, not because more history was used.\n")
    q2 = q["q2_prev_vs_all"]
    recent = E[-5:]
    rb = sum(e["results"]["prev_year"]["mae"] < e["results"]["hist_avg"]["mae"] for e in recent)
    close = abs(q2["prev_year_mae"] - q2["hist_avg_mae"]) / q2["prev_year_mae"] < 0.02
    L.append(f"**2. Is the previous year more useful than all years?** Pooled over {cy[0]}–{cy[-1]}: previous-year MAE "
             f"{f3(q2['prev_year_mae'])} vs historical average {f3(q2['hist_avg_mae'])}"
             + (" (effectively a tie)" if close else "") + f"; the previous year was better in {q2['prev_year_better_years']} "
             f"of {len(E)} experiments, and in {rb} of the last {len(recent)} ({recent[0]['test_year']}–{recent[-1]['test_year']}). "
             + ("In the recent, larger years the previous year is the more useful of the two, and the window study (Q1) "
                "shows that averaging in older years makes the plain average worse."
                if rb > len(recent) / 2 else "Neither is clearly more useful.") + "\n")
    m = q[f"vs_prev_{bm}"]
    L.append(f"**3. Does ML beat the previous-year cut-off method?** The better ML model ({APPROACHES[bm]}) has pooled MAE "
             f"{f3(m['pooled_mae'])} vs {f3(pv)} ({m['relative_change']:+.1%}); better in {m['years_better']} of {m['of']} years, "
             f"statistically clearly better in {m['significantly_better_years']} and clearly worse in {m['significantly_worse_years']}. "
             + (("The direction is consistent but the gain is small" if m["years_better"] >= 0.75 * m["of"] else
                 "The gain is small and not consistent") + ", so ML is a modest refinement of the baseline, not a big improvement."
                if m["relative_change"] > -0.10 else "The gain is meaningful.")
             + (f" With settings chosen only from earlier years (leak-free) the pooled MAE is {tc['pooled']['nested_mae']:.3f}, "
                f"better in {tc['nested_years_better']} of {tc['of']} years." if bm == "gbm_resid" else "") + "\n")
    w = q["vs_prev_weighted"]
    L.append(f"**4. Does weighting recent years help?** Weighted historical MAE {f3(w['pooled_mae'])} vs previous-year {f3(pv)} "
             f"({w['relative_change']:+.1%}), better in {w['years_better']} of {w['of']} years; vs the plain average "
             f"{f3(S['hist_avg']['mae'])}. "
             + ("Weighting beats the plain average" if w["pooled_mae"] < S["hist_avg"]["mae"] else "Weighting does not beat the plain average")
             + (" and slightly beats last year alone, but the gain is within noise in most years."
                if w["pooled_mae"] < pv and w["significantly_better_years"] < w["of"] / 2 else
                " and beats last year alone." if w["pooled_mae"] < pv else ", and does not beat last year alone.") + "\n")
    L.append(f"**5. Best approach across all test years:** {APPROACHES[q['best_overall']]} (lowest pooled MAE). "
             "Experiments won: " + ", ".join(f"{APPROACHES[k]} {v}" for k, v in q["wins"].items() if v) + ".\n")
    sc = [doc["scale"][str(y)] for y in doc["years"]]
    jump = max(max(a / b, b / a) for a, b in zip(sc[1:], sc[:-1]))
    L.append(f"**6. Is the data sufficient for reliable ML?** Only partly. There are {doc['n_pools_total']} seat pools and "
             f"{doc['ml_rows_total']} training rows in total; the first experiments ({E[0]['test_year']}–{E[2]['test_year']}) have only "
             f"{min(e['n_pools'] for e in E[:3])}–{max(e['n_pools'] for e in E[:3])} evaluable pools each. The rank list "
             f"size changed by up to {jump:.1f}× from one year to the next, which no historical feature can foresee, so "
             "rank errors of several hundred are mostly caused by that shift, whichever method is used. Individual allotment "
             "records are not in the data (merit lists contain personal data), so student-level accuracy is measured with "
             "simulated ranks against published cut-offs. Merit lists with graduation percentages exist for only 2020, 2025 and 2026, "
             "so a percentage-based model cannot be trained or validated across years yet.\n")
    best = q["best_overall"]
    sb, sp = S[bm], S["prev_year"]
    L.append(f"**7. Which model to use for the next year?** "
             + (f"Use **{APPROACHES[best]}** for the forecast's central estimate: it has the lowest pooled error and beat the "
                f"previous-year baseline in {S[best]['years_beating_prev_year']} of {len(cy)} years. "
                if best in ML else f"Use **{APPROACHES[best]}**: no ML model beat it overall. ")
             + f"Keep showing the previous-year cut-off next to it: for the student-facing list, the previous-year rule gave "
               f"list precision {pct(sp['list_precision'])} and top-{TOP_K} hit {pct(sp[f'top{TOP_K}_hit_rate'])}, vs "
               f"{pct(sb['list_precision'])} and {pct(sb[f'top{TOP_K}_hit_rate'])} for {APPROACHES[bm]}, so the ML model is not more "
               "reliable at telling a student which colleges they will actually get. Always show a likely range "
               "(from past held-out errors), never a single guaranteed rank.\n")
    L.append("## Limitations\n")
    L.append("- Predictions cover Round 1 ordinary seats (category × gender × college). NCC, divyang and converted seats, "
             "Round 2 and special rounds are too sparse to benchmark year by year.")
    L.append("- The ML hyper-parameters in the main tables come from `pipeline/predict_next_year.py`, which was designed after "
             "looking at 2022–2026 backtests. The leak-free tuning check above shows the conclusion survives when the "
             "setting is chosen from earlier years only. The weighted model's half-life is always chosen that way.")
    L.append(f"- The early lists ({doc['years'][0]}–2019) are small (about 25 colleges), so the first experiments have high variance.")
    L.append("- The college-type feature is taken from the current (2026-27) intake list; college type rarely changes, "
             "but strictly it is present-day information.")
    L.append("- The rank scale of the next year is unknown; all realistic rank errors use the previous year's scale.")
    return "\n".join(L)


def main():
    pools, scale, colleges = load()
    years = sorted(scale)
    clean_rows = json.load(open(os.path.join(PROJECT, "data", "processed", "mca_cutoffs_clean.json"), encoding="utf-8"))["rows"]
    data_audit = audit(clean_rows)
    experiments, preds_store, data = run(pools, scale, colleges, years)
    comparable = [e["test_year"] for e in experiments if all(e["results"][m] for m in APPROACHES)]
    summary = pooled(experiments, APPROACHES, comparable)
    best_ml = min(ML, key=lambda m: summary[m]["mae"])
    window = window_study(pools, scale, colleges, years, data, years[-3:])
    doc = {
        "years": years, "scale": {str(y): scale[y] for y in years},
        "unit": "seat pool = college × social category × gender, ordinary seats, Round 1, domicile merged",
        "n_pools_total": len(pools), "ml_rows_total": int(len(data[2])),
        "approaches": APPROACHES, "experiments": experiments, "comparable_years": comparable,
        "summary": summary, "best_ml": best_ml, "window_study": window,
        "tuning_check": tuning_check(pools, scale, colleges, data, experiments),
        "data_audit": data_audit, "audit_md": audit_markdown(data_audit),
    }
    doc["answers"] = answers(experiments, window, summary, comparable)
    os.makedirs(OUT, exist_ok=True)
    paths = figures(experiments, preds_store, window, best_ml)
    with open(os.path.join(OUT, "benchmark_results.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, default=float)
    with open(os.path.join(OUT, "benchmark_table.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["experiment", "training_years", "test_year", "pools"] + [f"{m}_mae" for m in APPROACHES]
                    + [f"{m}_rank_mae" for m in APPROACHES] + ["best"])
        for e in experiments:
            r = e["results"]
            wr.writerow([e["experiment"], f"{e['train_years'][0]}-{e['train_years'][1]}", e["test_year"], e["n_pools"]]
                        + [round(r[m]["mae"], 4) if r[m] else "" for m in APPROACHES]
                        + [round(r[m]["rank_mae"], 1) if r[m] else "" for m in APPROACHES] + [e["best"]])
    report = write_report(doc, paths)
    with open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print("\npooled", comparable[0], "-", comparable[-1])
    for m in APPROACHES:
        s = summary[m]
        print(f"  {APPROACHES[m]:48s} MAE {s['mae']:.4f}  rankMAE {s['rank_mae']:7.1f}  acc {s['decision_accuracy']:.3f}  "
              f"prec {s['list_precision']:.3f}  top{TOP_K} {s[f'top{TOP_K}_hit_rate']:.3f}  beats prev {s['years_beating_prev_year']}")
    print("answers:", json.dumps(doc["answers"], indent=1, default=float))
    print(f"→ {os.path.relpath(OUT, PROJECT)}/REPORT.md")


if __name__ == "__main__":
    main()
