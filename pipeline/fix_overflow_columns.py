#!/usr/bin/env python3
"""Repair step (run after mca_pdf_to_json.py): re-attach columns that overflowed
onto the next page.

Why this exists
---------------
In 2017__cutoff__special-round-2.pdf the table is too wide for the page. Page 1
holds S.No. .. JK MIGRANTS; the last column, DOMICILE, is printed alone on page 2
as a one-column table with the same row grid. The extractor sees page 2 as
"table but no data rows" and the domicile values are lost.

What it does
------------
For each extracted JSON listed in OVERFLOWS it re-reads the PDF and checks, for
the overflow page:
  * it holds exactly one table with exactly one column;
  * that column's header classifies (same letter-bag rules as the extractor)
    to the expected field, and the field is not already filled;
  * it has the same number of value rows as the data rows on the source page;
  * every value row's top/bottom y matches the matching data row within 1pt.
Only if all checks pass does it set the field on each record and record the
repair under "repairs" in the file. Any failed check stops with an error.
Idempotent: it re-derives everything from the PDF on every run.

Usage: python pipeline/fix_overflow_columns.py
"""
import json
import os
import sys
import warnings

import pdfplumber

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mca_pdf_to_json import LINES, PROJECT, classify, clean  # noqa: E402

warnings.filterwarnings("ignore")

# json path (relative to project) -> list of {source page, overflow page, field}
OVERFLOWS = {
    "data/extracted/mca/2017/2017__cutoff__special-round-2.json": [
        {"source_page": 1, "overflow_page": 2, "field": "domicile"},
    ],
}


def row_boxes(page):
    """[(cells, (top, bottom))] for the single table on a page."""
    tables = page.find_tables(LINES)
    if len(tables) != 1:
        raise ValueError(f"expected 1 table, found {len(tables)}")
    t = tables[0]
    cells = t.extract()
    boxes = [(r.bbox[1], r.bbox[3]) for r in t.rows]
    return list(zip(cells, boxes))


def repair(json_rel, spec):
    path = os.path.join(PROJECT, json_rel)
    doc = json.load(open(path, encoding="utf-8"))
    pdf_path = os.path.join(PROJECT, doc["source_pdf"])
    field = spec["field"]
    with pdfplumber.open(pdf_path) as pdf:
        src = row_boxes(pdf.pages[spec["source_page"] - 1].dedupe_chars(tolerance=1))
        ovf = row_boxes(pdf.pages[spec["overflow_page"] - 1].dedupe_chars(tolerance=1))

    if any(len(c) != 1 for c, _ in ovf):
        raise ValueError("overflow table has more than one column")
    header_idx = [i for i, (c, _) in enumerate(ovf) if classify(c[0]) == field]
    if len(header_idx) != 1:
        raise ValueError(f"no single header classifying to {field!r} on overflow page")
    h = header_idx[0]
    values = ovf[h + 1:]

    data = [(c, b) for c, b in src if (c[0] or "").strip().isdigit()]
    if len(values) != len(data):
        raise ValueError(f"{len(values)} overflow values vs {len(data)} data rows")

    recs = {r["sno"]: r for r in doc["rows"] if r["page"] == spec["source_page"]}
    applied = []
    for (vcells, vbox), (dcells, dbox) in zip(values, data):
        if abs(vbox[0] - dbox[0]) > 1 or abs(vbox[1] - dbox[1]) > 1:
            raise ValueError(f"row grid mismatch at S.No. {dcells[0]}: {vbox} vs {dbox}")
        rec = recs[int(dcells[0])]
        if rec.get(field) not in (None, clean(vcells[0])):
            raise ValueError(f"S.No. {rec['sno']} already has {field}={rec[field]!r}")
        rec[field] = clean(vcells[0])
        rec.setdefault("raw_cells_overflow", {})[field] = vcells[0]
        applied.append(rec["sno"])

    if field not in doc["columns"]:
        doc["columns"].append(field)
    repairs = [r for r in doc.get("repairs", []) if r.get("field") != field]
    repairs.append({"script": "pipeline/fix_overflow_columns.py", "field": field,
                    "from_page": spec["overflow_page"], "to_page": spec["source_page"],
                    "snos": applied,
                    "reason": f"{field} column printed alone on page {spec['overflow_page']}"})
    doc["repairs"] = repairs
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print(f"✓ {json_rel}: {field} set on S.No. {applied}")


def main():
    for json_rel, specs in OVERFLOWS.items():
        if not os.path.exists(os.path.join(PROJECT, json_rel)):
            sys.exit(f"✗ {json_rel} not found: run mca_pdf_to_json.py first")
        for spec in specs:
            try:
                repair(json_rel, spec)
            except ValueError as e:
                sys.exit(f"✗ {json_rel} page {spec['overflow_page']}: {e}")


if __name__ == "__main__":
    main()
