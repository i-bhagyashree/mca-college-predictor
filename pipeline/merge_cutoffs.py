#!/usr/bin/env python3
"""Layer 2 (merge): every per-PDF JSON -> one master list of rows.

Reads data/extracted/mca/<year>/*.json (from mca_pdf_to_json.py, after
fix_overflow_columns.py) and writes data/processed/mca_cutoffs_master.json.

Each row keeps every field from Layer 1 (including page, table_index and
raw_cells, so any number can be traced back to its PDF page) and is stamped
with year, round_title, source_pdf and the PDF's sha256.

The master is rebuilt from scratch on every run, so re-running after a
re-extraction never duplicates rows (idempotent).

Usage: python pipeline/merge_cutoffs.py
"""
import glob
import json
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJECT, "data", "extracted", "mca")
OUT = os.path.join(PROJECT, "data", "processed", "mca_cutoffs_master.json")


def main():
    files = sorted(glob.glob(os.path.join(SRC, "*", "*.json")))
    if not files:
        sys.exit("no extracted JSON: run mca_pdf_to_json.py first")
    rows, sources = [], []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        if d.get("problems"):
            sys.exit(f"✗ {os.path.relpath(f, PROJECT)} has {len(d['problems'])} unresolved problems; fix them first")
        for r in d["rows"]:
            out = dict(r)
            out["year"] = d["year"]
            out["round_title"] = d["round_title"]
            out["source_pdf"] = d["source_pdf"]
            out["source_sha256"] = d["sha256"]
            rows.append(out)
        sources.append({"source_pdf": d["source_pdf"], "sha256": d["sha256"], "year": d["year"],
                        "round_title": d["round_title"], "rows": d["row_count"],
                        "repairs": d.get("repairs", [])})
        print(f"  {os.path.basename(f):60s} +{d['row_count']:4d}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"sources": sources, "row_count": len(rows), "rows": rows},
                  fh, ensure_ascii=False, separators=(",", ":"))
    years = sorted({r["year"] for r in rows})
    print(f"→ {os.path.relpath(OUT, PROJECT)}: {len(rows):,} rows, {len(files)} PDFs, years {years[0]}–{years[-1]}")


if __name__ == "__main__":
    main()
