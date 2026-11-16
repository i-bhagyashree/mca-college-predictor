#!/usr/bin/env python3
"""Percentage ↔ rank from the official MP-DTE MCA merit lists.

The merit lists (data/raw/mca/<year>/<year>__merit__<round>.pdf) print, for every candidate,
the merit RANK and the qualifying-exam PERCENTAGE, plus name, roll number and category.
They contain personal data, so they stay out of Git and off the website (.gitignore).
This script keeps ONLY the (percentage, rank) relation:

    for every distinct percentage p in a list: [p, first rank with p, last rank with p]

That is enough to read a student's rank from their percentage exactly, for the same year
and round as the cut-offs it is compared with. No names, roll numbers or categories leave
this script.

It also answers a question the site needs: how well does LAST year's list predict THIS year's
rank? For each pair of consecutive first-round lists, the rank of each percentage is read from
the older list (as a position, rank ÷ list size, times the newer list's size) and compared with
the real rank in the newer list.

Output: data/processed/merit_lists.json
Usage:  python pipeline/merit_lists.py
"""
import glob
import json
import os
import re

import numpy as np
import pdfplumber

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT, "data", "raw", "mca")
OUT = os.path.join(PROJECT, "data", "processed", "merit_lists.json")

ROW = re.compile(r"^(\d{1,5})\s+(\d{1,3}(?:\.\d+)?)\s+\d{10,13}\s")      # rank, percentage, roll no.
DATE = re.compile(r"Date\s*:\s*(\d{2}\.\d{2}\.\d{4})")
ROUNDS = {"first-round": "R1", "second-round": "R2"}


def parse(path):
    """[(rank, pct)] in list order, plus the list date. Only rank and percentage are read."""
    rows, date = [], None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if date is None and (m := DATE.search(text)):
                date = m.group(1)
            for line in text.splitlines():
                m = ROW.match(line.strip())
                if m:
                    rows.append((int(m.group(1)), float(m.group(2))))
    return rows, date


def check(rows):
    """Problems found in a parsed list (empty = clean)."""
    ranks = [r for r, _ in rows]
    issues = []
    if ranks != list(range(1, len(ranks) + 1)):
        missing = sorted(set(range(1, max(ranks) + 1)) - set(ranks))
        issues.append(f"ranks not 1..N (missing {missing[:10]})")
    up = sum(1 for (_, a), (_, b) in zip(rows, rows[1:]) if b > a + 1e-9)
    if up:
        issues.append(f"{up} places where a worse rank has a higher percentage")
    if any(not (0 < p <= 100) for _, p in rows):
        issues.append("percentage outside 0–100")
    return issues


def points(rows):
    """[[pct, first_rank, last_rank], ...], percentage descending (rank ascending)."""
    out = []
    for rank, pct in rows:
        p = round(pct, 4)
        if out and out[-1][0] == p:
            out[-1][2] = rank
        else:
            out.append([p, rank, rank])
    return out


def rank_of(pct, rows):
    """Rank a student with `pct` would have had in this list (placed after everyone above them)."""
    return 1 + sum(1 for _, p in rows if p > pct + 1e-9)


def carry_over(old, new, grid):
    """Error of reading THIS year's rank from LAST year's list (position × this year's size)."""
    n_old, n_new = len(old), len(new)
    errs = []
    for p in grid:
        if p < min(q for _, q in new):                 # below this year's last candidate: no real rank
            continue
        pred = rank_of(p, old) / n_old * n_new
        real = rank_of(p, new)
        errs.append((p, round(pred), real))
    abs_err = np.array([abs(a - b) for _, a, b in errs])
    return {"mae_ranks": round(float(abs_err.mean()), 1), "p90_ranks": round(float(np.percentile(abs_err, 90)), 1),
            "max_ranks": int(abs_err.max()), "n": len(errs),
            "examples": [{"pct": p, "predicted": a, "real": b} for p, a, b in errs if p in (60, 65, 70, 75, 80, 85)]}


def main():
    lists = []
    for path in sorted(glob.glob(os.path.join(RAW, "*", "*__merit__*.pdf"))):
        year = int(os.path.basename(os.path.dirname(path)))
        rnd = ROUNDS[os.path.basename(path).split("__merit__")[1].removesuffix(".pdf")]
        rows, date = parse(path)
        issues = check(rows)
        pcts = [p for _, p in rows]
        lists.append({"year": year, "round": rnd, "date": date, "n": len(rows),
                      "pct_max": max(pcts), "pct_min": min(pcts), "pct_median": round(float(np.median(pcts)), 2),
                      "issues": issues, "source_pdf": os.path.relpath(path, PROJECT), "points": points(rows), "_rows": rows})
        print(f"{year} {rnd}: {len(rows):,} candidates, {pcts[0]}% … {pcts[-1]}%, "
              f"{len(lists[-1]['points'])} distinct percentages, date {date}"
              + (f"  ISSUES: {issues}" if issues else "  (ranks 1..N, percentage never rises)"))

    # how stable is percentage → position from one year's Round 1 list to the next one we have?
    r1 = [l for l in lists if l["round"] == "R1"]
    grid = [round(x, 1) for x in np.arange(50, 95.01, 0.5)]
    stability = []
    for old, new in zip(r1, r1[1:]):
        s = carry_over(old["_rows"], new["_rows"], grid)
        s.update({"from": old["year"], "to": new["year"]})
        stability.append(s)
        print(f"  {old['year']} R1 list → {new['year']} R1 ranks: off by {s['mae_ranks']} ranks on average "
              f"(90% within {s['p90_ranks']}, worst {s['max_ranks']})")
    same_year = [l for l in lists if l["year"] == max(x["year"] for x in lists)]
    if len(same_year) == 2:
        a, b = same_year
        s = carry_over(a["_rows"], b["_rows"], grid)
        s.update({"from": f"{a['year']} {a['round']}", "to": f"{b['year']} {b['round']}"})
        stability.append(s)
        print(f"  {a['year']} R1 list → R2 ranks (same year): off by {s['mae_ranks']} ranks on average")

    # The site predicts with the RECENT lists combined (2025 R1, 2026 R1, 2026 R2): your position in each
    # (rank ÷ list size) is averaged. Leave-one-out test: predict each recent list from the other two.
    recent = [l for l in lists if l["year"] >= max(x["year"] for x in lists) - 1]
    combined = []
    for t in recent:
        others = [o for o in recent if o is not t]
        errs = []
        for p in grid:
            if p < min(q for _, q in t["_rows"]):
                continue
            pred = np.mean([rank_of(p, o["_rows"]) / len(o["_rows"]) for o in others]) * len(t["_rows"])
            errs.append(abs(pred - rank_of(p, t["_rows"])))
        combined.append({"target": f"{t['year']} {t['round']}", "from": [f"{o['year']} {o['round']}" for o in others],
                         "mae_ranks": round(float(np.mean(errs)), 1), "p90_ranks": round(float(np.percentile(errs, 90)), 1)})
        print(f"  combined {' + '.join(combined[-1]['from'])} → {combined[-1]['target']}: "
              f"off by {combined[-1]['mae_ranks']} on average (90% within {combined[-1]['p90_ranks']})")
    combined_summary = {"lists": [f"{l['year']} {l['round']}" for l in recent], "tests": combined,
                        "mae_ranks": round(float(np.mean([c["mae_ranks"] for c in combined])), 1),
                        "p90_ranks": round(float(np.mean([c["p90_ranks"] for c in combined])), 1)}

    for l in lists:
        del l["_rows"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"note": "Percentage ↔ rank only. Names, roll numbers and categories are not stored.",
                   "lists": lists, "stability": stability, "combined": combined_summary}, fh, ensure_ascii=False, indent=1)
    print(f"→ {os.path.relpath(OUT, PROJECT)}")


if __name__ == "__main__":
    main()
