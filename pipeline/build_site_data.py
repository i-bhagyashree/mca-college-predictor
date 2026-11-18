#!/usr/bin/env python3
"""Layer 3 (website assets): clean data -> assets/data/mca-data.js.

Generated file: never edit it by hand. Fix the source or a script and re-run.

It is written as a .js file (window.MCA_DATA = {...}) rather than .json so that
index.html also works when opened straight from disk (file://), where browsers
block fetch() of local JSON.

Rows are stored compactly (short keys); index.html documents the keys.
raw_cells are NOT shipped to the browser, but source PDF + page + S.No. are, so
every number on the site can be traced back to its PDF page.

Usage: python pipeline/build_site_data.py
"""
import json
import os
import sys
from datetime import date

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(PROJECT, "data", "processed")
OUT = os.path.join(PROJECT, "assets", "data", "mca-data.js")


def main():
    clean = json.load(open(os.path.join(PROC, "mca_cutoffs_clean.json"), encoding="utf-8"))
    colleges = json.load(open(os.path.join(PROC, "colleges.json"), encoding="utf-8"))
    sources = sorted({r["source_pdf"] for r in clean["rows"]})
    src_idx = {s: i for i, s in enumerate(sources)}
    rows = []
    for r in clean["rows"]:
        rows.append({
            "c": r["college_id"], "y": r["year"], "r": r["round"], "aw": int(r["result_awaited"]),
            "o": r["opening_rank"], "cl": r["closing_rank"],
            "os": r["opening_score"], "cs": r["closing_score"],
            "s": r["social"], "k": r["seat_class"], "g": r["gender"],
            "d": r["domicile"], "cv": int(r["after_conversion"]),
            "n": r["total_allotted"], "cat": r["category_raw"],
            "src": src_idx[r["source_pdf"]], "p": r["page"], "sn": r["sno"],
        })
    data = {
        "generated": date.today().isoformat(),
        "intake_session": clean["intake_session"],
        "sources": [os.path.basename(s) for s in sources],
        "colleges": [{k: c[k] for k in ("college_id", "official", "name", "city", "type",
                                        "university", "intake_2026", "years_in_cutoffs", "names_seen")}
                     | ({"fees": {k: c["fees"][k] for k in ("st", "sc", "obc", "ur", "fetched")}} if c.get("fees") else {})
                     | ({"afrc": [{"s": f["session"], "fee": f["semester_fee_rs"], "raw": f["raw_fee_text"]}
                                  for f in c["afrc_fees"]]} if c.get("afrc_fees") else {})
                     | ({"seats": {k: c["intake_detail"][k] for k in (
                         "session", "total", "tfw_seats", "nri_seats", "ews", "aicte_code", "minority", "fee_fixed",
                         "affiliation_previous_year", "affiliation_current_year",
                         "affiliation_required_before_merit", "remark")}} if c.get("intake_detail") else {})
                     for c in colleges],
        "rows": rows,
    }
    # ML forecast (pipeline/predict_next_year.py), if it has been run
    import glob
    preds = sorted(glob.glob(os.path.join(PROC, "predictions_*.json")))
    if preds:
        f = json.load(open(preds[-1], encoding="utf-8"))
        data["forecast"] = {
            "target_year": f["target_year"], "round": f["round"], "unit": f["unit"],
            "model_label": f["model_label"], "model_used": f["model_used"],
            "scale_assumed": f["scale_assumed"], "scale_by_year": f["scale_by_year"],
            "coverage": f["range"]["coverage_held_out"], "backtest_years": f["backtest_years"],
            "summary": f["backtest_summary"], "train_years": f.get("train_years"), "selection": f.get("selection"),
            "recent": [r for r in f["backtest"] if r["model"] in ("last", f["model_used"])
                       and r["year"] >= f["backtest_years"][-2]],
            "importance": f["feature_importance"][:6], "training_rows": f["training_rows"],
            "pools": [{k: p[k] for k in ("c", "s", "g", "rank", "lo", "hi", "last_year", "last_rank", "n_years")}
                      for p in f["pools"]],
        }
    # Year-by-year benchmark (pipeline/benchmark_forward_chain.py), if it has been run
    bpath = os.path.join(PROJECT, "reports", "benchmark", "benchmark_results.json")
    if os.path.exists(bpath):
        b = json.load(open(bpath, encoding="utf-8"))
        data["benchmark"] = {
            "approaches": b["approaches"], "best_ml": b["best_ml"], "comparable_years": b["comparable_years"],
            "experiments": [{"t": e["test_year"], "from": e["train_years"][0], "to": e["train_years"][1],
                             "n": e["n_pools"], "best": e["best"],
                             "mae": {m: (round(r["mae"], 4) if r else None) for m, r in e["results"].items()}}
                            for e in b["experiments"]],
            "summary": {m: ({k: round(v, 4) for k, v in s.items()} if s else None) for m, s in b["summary"].items()},
            # extra detail for the ML Analysis page
            "scale": b["scale"], "n_pools_total": b["n_pools_total"], "ml_rows_total": b["ml_rows_total"],
            "detail": [{"t": e["test_year"], "rows": e["ml_training_rows"],
                        "m": {m: ({k: round(r[k], 4) for k in ("mae", "rmse", "rank_mae", "decision_accuracy", "list_precision", "top5_hit_rate")} if r else None)
                              for m, r in e["results"].items()}}
                       for e in b["experiments"]],
            "answers": b["answers"],
            "tuning": {"nested_mae": round(b["tuning_check"]["pooled"]["nested_mae"], 4),
                       "nested_years_better": b["tuning_check"]["nested_years_better"], "of": b["tuning_check"]["of"]},
        }
    # ML hold-out predictions vs what really happened (pipeline/evaluate_ml.py), if it has been run:
    # shown on the ML Analysis page ("Past predictions vs reality" tab)
    tpath = os.path.join(PROJECT, "reports", "ml_test", "ml_test_results.json")
    if os.path.exists(tpath):
        import csv
        years = []
        for r in json.load(open(tpath, encoding="utf-8")):
            cpath = os.path.join(PROJECT, "reports", "ml_test", f"predictions_vs_actual_{r['test_year']}.csv")
            if not os.path.exists(cpath):
                continue
            rows_ = []
            for c in csv.DictReader(open(cpath, encoding="utf-8")):
                lo, hi = (int(x) for x in c["ml_range"].replace("\u2013", "-").split("-"))
                rows_.append({"c": c["college_id"], "s": c["category"], "g": c["gender"], "last": int(c["last_year_actual"]),
                              "pred": int(c["ml_predicted"]), "lo": lo, "hi": hi, "act": int(c["actual"]),
                              "in": c["inside_range"] == "True"})
            years.append({"t": r["test_year"], "trained_on": r["trained_on"], "n": r["n"], "coverage": round(r["coverage_10_90"], 4),
                          "ml_mae": round(r["ml"]["rank_mae"], 1), "hist_mae": round(r["historical"]["rank_mae"], 1),
                          "ml_seat": round(r["ml"]["seat_accuracy_all"], 4), "hist_seat": round(r["historical"]["seat_accuracy_all"], 4),
                          "ml_closer": round(r["ml_closer_share"], 4),
                          # every metric of both methods, for the "Rank Predictor vs ML" tab
                          "ml": {k: round(v, 4) for k, v in r["ml"].items()},
                          "hist": {k: round(v, 4) for k, v in r["historical"].items()},
                          "rows": rows_})
        if years:
            data["ml_test"] = years
    # Official merit lists (pipeline/merit_lists.py): percentage ↔ rank only, no names or roll numbers
    mpath = os.path.join(PROC, "merit_lists.json")
    if os.path.exists(mpath):
        m = json.load(open(mpath, encoding="utf-8"))
        data["merit"] = {
            "lists": [{k: l[k] for k in ("year", "round", "date", "n", "pct_max", "pct_min", "pct_median", "points")}
                      for l in m["lists"] if not l["issues"]],
            "stability": m["stability"],
            "combined": m.get("combined"),
        }
    # ML rank from percentage (pipeline/ml_rank_from_percentage.py): models trained on the merit lists
    mlpath = os.path.join(PROC, "ml_rank_model.json")
    if os.path.exists(mlpath):
        data["ml_rank"] = json.load(open(mlpath, encoding="utf-8"))
    # Percentage → rank (pipeline/rank_from_percentage.py), if it has been run
    rpath = os.path.join(PROC, "rank_from_percentage.json")
    if os.path.exists(rpath):
        r = json.load(open(rpath, encoding="utf-8"))
        sf = r["size_forecast"]
        data["pct_rank"] = {
            "year": r["curve"]["year"], "round": r["curve"]["round"], "points": r["curve"]["points"],
            "n_points": r["curve"]["n_points"], "accuracy": r["accuracy"],
            "scale_year": r["scale_year"], "scale": r["scale"],
            "next": {"year": sf["next_year"], "size": sf["next_size"], "range": sf["range"],
                     "range_ratio": sf["range_ratio"], "method": sf["methods"][sf["used"]]["label"],
                     "method_error": sf["methods"][sf["used"]]["mean_abs_pct_error"], "n_tests": sf["n_tests"],
                     "best_method": sf["methods"][sf["best"]]["label"], "best_error": sf["methods"][sf["best"]]["mean_abs_pct_error"],
                     "best_size": sf["methods"][sf["best"]]["next"], "years": sf["years"], "half_life": sf.get("half_life"),
                     "biggest_jump": sf["biggest_jump"],
                     "others": sorted(({"label": v["label"], "error": round(v["mean_abs_pct_error"], 3)}
                                       for v in sf["methods"].values()), key=lambda x: x["error"])},
        }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # Validate first (allow_nan=False: NaN or infinity stops the build), write to a temporary file, and only
    # then replace the published file, so a failed run never leaves a partial or empty data file behind.
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if not data.get("rows") or not data.get("colleges"):
        sys.exit("✗ no cut-off rows or colleges: the published data file was NOT replaced")
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("// GENERATED by pipeline/build_site_data.py — do not edit by hand.\n")
        fh.write("window.MCA_DATA = " + body + ";\n")
    os.replace(tmp, OUT)
    print(f"→ {os.path.relpath(OUT, PROJECT)}  ({os.path.getsize(OUT)/1024:.0f} KB, "
          f"{len(rows):,} rows, {len(colleges)} colleges)")


if __name__ == "__main__":
    main()
