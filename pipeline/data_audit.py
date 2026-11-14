#!/usr/bin/env python3
"""Data inspection for the benchmark: what is in data/processed/mca_cutoffs_clean.json.

Answers, from the data itself (nothing hand-typed):
  * which years, rounds and lists exist, and how many rows each has
  * which columns exist and how often each is filled
  * colleges per year and how their names were matched to official College IDs
  * category / seat class / gender / domicile / conversion coverage
  * duplicate records, impossible ranks (opening > closing), missing ranks
  * where graduation-percentage scores are available
  * how many rows survive the benchmark's filter (Round 1, ordinary seats, regular list)

Imported by benchmark_forward_chain.py (its REPORT.md includes this section) and runnable alone:
  python pipeline/data_audit.py          prints the markdown
"""
import json
import os
from collections import Counter, defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(PROJECT, "data", "processed", "mca_cutoffs_clean.json")
SOCIAL = ["UR", "OBC", "SC", "ST", "EWS", "FW"]

# The benchmark's seat pool within one published list. Several printed rows can map to one pool
# (domicile splits; 2025's split by the candidate's own eligible category); they are merged.
POOL = ("year", "round", "result_awaited", "college_id", "social", "seat_class", "gender", "after_conversion")
SKIP_COLUMNS = ("raw_cells", "source_pdf", "page", "sno")


def benchmark_row(r):
    """The filter used by predict_next_year.load(): Round 1, regular list, ordinary seats."""
    return (not r["result_awaited"] and r["round"] == "R1" and r["seat_class"] in ("X", None)
            and not r["after_conversion"] and r["social"] in SOCIAL and r["gender"] in ("OP", "F"))


def audit(rows):
    years = sorted({r["year"] for r in rows})
    a = {"n_rows": len(rows), "years": years}

    lists = Counter((r["year"], r["round"], r["result_awaited"]) for r in rows)
    a["lists"] = [{"year": y, "round": rd, "result_awaited": aw, "rows": n}
                  for (y, rd, aw), n in sorted(lists.items())]

    cols = [c for c in rows[0] if c not in SKIP_COLUMNS]
    a["columns"] = {c: sum(r.get(c) not in (None, "") for r in rows) / len(rows) for c in cols}

    by_year = defaultdict(list)
    for r in rows:
        by_year[r["year"]].append(r)
    per_year = {}
    for y in years:
        rs = by_year[y]
        per_year[y] = {
            "rows": len(rs),
            "colleges": len({r["college_id"] or r["institute_name_raw"] for r in rs}),
            "unmatched_names": len({r["institute_name_raw"] for r in rs if not r["college_id"]}),
            "social": dict(Counter(r["social"] for r in rs)),
            "female_rows": sum(r["gender"] == "F" for r in rs),
            "special_class_rows": sum(r["seat_class"] not in ("X", None) for r in rs),
            "converted_rows": sum(bool(r["after_conversion"]) for r in rs),
            "domicile_rows": sum(r["domicile"] is not None for r in rs),
            "score_rows": sum(r["closing_score"] is not None for r in rs),
            "benchmark_rows": sum(benchmark_row(r) for r in rs),
        }
    a["per_year"] = per_year
    a["id_match"] = dict(Counter(r.get("id_match") or "none" for r in rows))

    pool = Counter(tuple(r.get(k) for k in POOL) for r in rows)
    a["split_pools"] = sum(1 for n in pool.values() if n > 1)
    a["split_rows"] = sum(n for n in pool.values() if n > 1)
    a["rounds_by_year"] = {y: sorted({r["round"] for r in by_year[y]}) for y in years}
    full = Counter(json.dumps({k: v for k, v in r.items() if k not in SKIP_COLUMNS}, sort_keys=True) for r in rows)
    a["exact_duplicate_rows"] = sum(n - 1 for n in full.values() if n > 1)
    a["missing_closing_rank"] = sum(r["closing_rank"] is None for r in rows)
    a["missing_opening_rank"] = sum(r["opening_rank"] is None for r in rows)
    a["opening_gt_closing"] = sum(1 for r in rows if r["opening_rank"] is not None and r["closing_rank"] is not None
                                  and r["opening_rank"] > r["closing_rank"])
    a["category_values"] = sorted({r["category_raw"] for r in rows if r["category_raw"]})
    return a


def markdown(a):
    L = ["## Data inspection (from `data/processed/mca_cutoffs_clean.json`)\n"]
    L.append(f"{a['n_rows']:,} cut-off rows, years {a['years'][0]}–{a['years'][-1]} ({len(a['years'])} admission years; "
             "the brief assumed 8, the data has more, so there are more experiments).\n")
    L.append("**Lists published per year** (R1 = first round, R2 = second round, SP = special rounds; "
             "'result awaited' = separate lists for candidates whose degree result was pending):\n")
    L.append("| Year | Round | Result-awaited list | Rows |\n|---|---|---|---|")
    for x in a["lists"]:
        L.append(f"| {x['year']} | {x['round']} | {'yes' if x['result_awaited'] else 'no'} | {x['rows']} |")
    L.append("\n**Per year:**\n")
    L.append("| Year | Rows | Colleges | Names without College ID | Female-seat rows | NCC/divyang rows | "
             "Converted rows | Domicile given | %-score rows | Rows used by benchmark |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for y, p in a["per_year"].items():
        L.append(f"| {y} | {p['rows']} | {p['colleges']} | {p['unmatched_names']} | {p['female_rows']} | "
                 f"{p['special_class_rows']} | {p['converted_rows']} | {p['domicile_rows']} | {p['score_rows']} | "
                 f"{p['benchmark_rows']} |")
    L.append("\n**Columns and how often they are filled:** "
             + ", ".join(f"`{c}` {v:.0%}" for c, v in a["columns"].items()) + ".\n")
    L.append("**Quality checks:** "
             f"exact duplicate rows: {a['exact_duplicate_rows']}; {a['split_rows']} rows are printed splits of "
             f"{a['split_pools']} seat pools (by domicile, or in 2025 by the candidate's own eligible category); they "
             f"are not duplicates and are merged by taking the highest closing rank; missing closing rank: {a['missing_closing_rank']}; "
             f"missing opening rank: {a['missing_opening_rank']}; opening rank > closing rank: {a['opening_gt_closing']}. "
             "College-name → official ID matching (provisional = a college not in the current intake list, "
             "kept under its own ID): "
             + ", ".join(f"{k} {v}" for k, v in sorted(a["id_match"].items())) + " rows.\n")
    L.append("**What the data does and does not contain:**\n")
    L.append("- Available every year: college, closing and opening **rank**, category (social/class/gender), seats "
             "allotted, round. Domicile is printed only in some years; it is merged, not compared.")
    L.append("- **Graduation-percentage cut-offs** appear only in "
             + ", ".join(f"{y} ({p['score_rows']} rows)" for y, p in a["per_year"].items() if p["score_rows"])
             + ". A percentage-based model therefore cannot be trained or tested across years.")
    L.append("- **Individual allotments** (which student got which college) are not in the data: the merit lists that "
             "contain them hold personal data and are deliberately kept out. Allotment accuracy is therefore measured "
             "with simulated student ranks against the published cut-offs.")
    r2 = [y for y, rs in a["rounds_by_year"].items() if "R2" in rs]
    no_r2 = [y for y in a["years"] if y not in r2 and y > r2[0]] if r2 else []
    sp = [y for y, rs in a["rounds_by_year"].items() if any(x.startswith("SP") for x in rs)]
    L.append(f"- Round 2 lists exist for {', '.join(map(str, r2))}"
             + (f" (none for {', '.join(map(str, no_r2))})" if no_r2 else "")
             + (f"; special rounds only for {', '.join(map(str, sp))}" if sp else "")
             + ". Only Round 1 exists in every year, so the benchmark uses Round 1 to compare like with like.\n")
    return "\n".join(L)


def main():
    rows = json.load(open(CLEAN, encoding="utf-8"))["rows"]
    print(markdown(audit(rows)))


if __name__ == "__main__":
    main()
