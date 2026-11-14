#!/usr/bin/env python3
"""Step 3 (extract): MCA cut-off PDF -> one JSON per PDF (Layer 1).

Approach (from pdf_to_json.py in i-shivamsoni/mp-dte-btech-predictor, MIT):
  * pdfplumber 'lines' strategy: the ruling lines become cell borders, so wrapped
    names stay in one cell and empty cells stay as '' instead of vanishing.
  * dedupe_chars(tolerance=1): removes doubled glyphs (22002255 -> 2025).
  * Headers are rotated 90°, which scrambles the ORDER of a header's letters but
    never WHICH letters are present. So we compare letter bags (multisets).

Stricter than the B.Tech parser, on purpose:
  * every record keeps source_pdf, page (1-based), table_index and raw_cells;
  * ALL tables on a page are processed, not only the largest;
  * two columns mapping to the same field is an error for that file, not a rename;
  * ranks are parsed only from plain digits (commas allowed). Anything else stays
    as text and is flagged. Scores (percentages) are kept as decimals;
  * missing values are null.

Usage:
  python pipeline/mca_pdf_to_json.py                       # every PDF under data/raw/mca
  python pipeline/mca_pdf_to_json.py data/raw/mca/2025/2025__cutoff__first-round.pdf
"""
import glob
import hashlib
import json
import os
import re
import sys
import warnings
from collections import Counter

import pdfplumber

warnings.filterwarnings("ignore")
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT, "data", "raw", "mca")
OUT = os.path.join(PROJECT, "data", "extracted", "mca")
LINES = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}

RANK_FIELDS = {"opening_rank", "closing_rank"}
SCORE_FIELDS = {"opening_score", "closing_score"}
COUNT_FIELDS = {"total_allotted"}


def letters(cell):
    return re.sub(r"[^A-Za-z]", "", cell or "").upper()


def has(bag, keyword):
    """True if every letter of keyword (with multiplicity) is in the bag."""
    return not (Counter(keyword) - bag)


# ---------------------------------------------------------------------------
# Header rules, checked against the real MCA headers 2017-2026 (inspect_pdf.py).
# Most specific first: short keywords hide inside long headers (the letters of
# NAME are all inside OPENING ... RANK; JK MIGRANTS contains R-A-N-K).
# Each rule: (field, [keywords that must ALL fit]).
# ---------------------------------------------------------------------------
RULES = [
    ("exam_type",          ["EXAM"]),
    ("remarks",            ["REMARK"]),
    # 2017 only: J&K quota flags (values Y/N).
    ("jk_resident",        ["JK", "RESIDENT"]),
    ("jk_migrant",         ["JK", "MIGRANT"]),
    # 2019 headers misspell OPENING as OPEINNG: same letters, so still fine.
    ("opening_rank",       ["OPENING", "RANK"]),
    ("closing_rank",       ["CLOSING", "RANK"]),
    # 2026 second round adds qualifying-exam merit (decimals, e.g. 71.7561).
    ("opening_score",      ["OPENING", "PERCENT"]),
    ("closing_score",      ["CLOSING", "PERCENT"]),
    ("opening_score",      ["OPENING", "MERIT"]),
    ("closing_score",      ["CLOSING", "MERIT"]),
    # 2025 adds ELIGIBLE CATEGORY next to ALLOTTED CATEGORY: must come first.
    ("eligible_category",  ["ELIGIBLE", "CATEGORY"]),
    ("allotted_category",  ["CATEGORY"]),
    # 2017 prints "ALLOTED CATEGOEY" (no R). TOTAL ALLOTTED has no C/G, so it can't match.
    ("allotted_category",  ["ALLOT", "CATEGO"]),
    ("domicile",           ["DOMICILE"]),
    ("total_allotted",     ["TOTAL", "ALLOT"]),
    ("institute_name",     ["NAME"]),
    ("institute_type",     ["TYPE"]),
    ("course",             ["COURSE"]),
    ("branch",             ["BRANCH"]),
]

# Per-file header fixes for typos printed in the PDF itself. Each entry is applied
# only if that column's header letters are exactly `expect`, so a re-issued PDF
# with a corrected header is not silently overridden.
HEADER_OVERRIDES = {
    # The 2019 PDF prints "INSTITUTE NAME" over both column 1 and column 2.
    # Column 2 holds Private / G.I. / S.F.I. on every row: it is the institute type.
    "2019__cutoff__first-round.pdf": {2: {"field": "institute_type", "expect": "EMANETUTTISNI"}},
}


def classify(cell):
    comp = letters(cell)
    if not comp:
        return None
    bag = Counter(comp)
    for field, kws in RULES:
        if all(has(bag, k) for k in kws):
            return field
    if comp in ("FW", "WF"):
        return "fw"
    if len(comp) <= 4 and set(comp) <= set("SNO"):        # S.No. -> '.oN.S' -> 'ONS'
        return "sno"
    return None


class HeaderError(Exception):
    pass


def find_header(row, overrides=None):
    """Return {column index: field} if this row is a header, else None.
    A header must map an institute column AND a rank or score column.
    `overrides` = {col: {"field", "expect"}} from HEADER_OVERRIDES."""
    fields = {j: classify(c) for j, c in enumerate(row)}
    for j, o in (overrides or {}).items():
        if j < len(row) and letters(row[j]) == o["expect"] and fields.get(j) is not None:
            fields[j] = o["field"]
    present = set(fields.values())
    if not ({"institute_name", "institute_type"} & present and
            (RANK_FIELDS | SCORE_FIELDS) & present):
        return None
    mapping, seen = {}, {}
    for j, f in fields.items():
        if f is None:
            continue
        if f in seen:
            raise HeaderError(f"columns {seen[f]} and {j} both map to '{f}': "
                              f"{row[seen[f]]!r} / {row[j]!r}. Add a more specific rule.")
        seen[f] = j
        mapping[j] = f
    return mapping


# ---------------------------------------------------------------------------
# Value parsing. Returns (value, problem or None).
# ---------------------------------------------------------------------------
def clean(v):
    s = re.sub(r"\s+", " ", (v or "").replace("\n", " ")).strip()
    return s or None


def parse_int(v):
    s = clean(v)
    if s is None:
        return None, None
    if re.fullmatch(r"\d{1,3}(,\d{3})+|\d+", s):
        return int(s.replace(",", "")), None
    return s, f"not a plain integer: {s!r}"


def parse_score(v):
    s = clean(v)
    if s is None:
        return None, None
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s), None
    return s, f"not a plain decimal: {s!r}"


def parse_value(field, v):
    if field in RANK_FIELDS or field in COUNT_FIELDS:
        return parse_int(v)
    if field in SCORE_FIELDS:
        return parse_score(v)
    return clean(v), None


# ---------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_index(year_dir):
    """saved_name -> official title, from the fetcher's _index.txt."""
    out = {}
    p = os.path.join(year_dir, "_index.txt")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            parts = [x.strip() for x in line.split("|")]
            if len(parts) == 4 and not line.startswith("#"):
                out[parts[0]] = parts[2]
    return out


def parse_pdf(path):
    rel = os.path.relpath(path, PROJECT)
    overrides = HEADER_OVERRIDES.get(os.path.basename(path))
    used_override = False
    rows, pages, problems = [], [], []
    unmapped = Counter()
    mapping, columns = None, []
    title_lines = []
    with pdfplumber.open(path) as pdf:
        for pno, raw_page in enumerate(pdf.pages, start=1):
            page = raw_page.dedupe_chars(tolerance=1)
            text = page.extract_text() or ""
            if pno == 1:
                title_lines = [l.strip() for l in text.splitlines()[:6] if l.strip()]
            tables = page.extract_tables(LINES)
            if not text.strip():
                pages.append({"page": pno, "status": "no text (needs OCR)", "rows": 0})
                continue
            if not tables:
                pages.append({"page": pno, "status": "no table", "rows": 0})
                continue
            n_before = len(rows)
            for ti, table in enumerate(tables):
                for row in table:
                    first = (row[0] or "").strip()
                    # Data rows are never headers: a long college name can contain
                    # every letter of OPENING+RANK.
                    hdr = None if first.isdigit() else find_header(row, overrides)   # may raise HeaderError
                    if hdr:
                        if overrides and any(hdr.get(j) == o["field"] for j, o in overrides.items()):
                            used_override = True
                        if mapping and hdr != mapping:
                            problems.append({"page": pno, "table": ti,
                                             "problem": "header changed mid-file", "raw": row})
                        mapping = hdr
                        for f in hdr.values():
                            if f not in columns:
                                columns.append(f)
                        continue
                    if not first.isdigit():
                        if first == "" and any(clean(c) for c in row):
                            problems.append({"page": pno, "table": ti,
                                             "problem": "row without S.No. (continuation line?)",
                                             "raw": row})
                        continue
                    if mapping is None:
                        problems.append({"page": pno, "table": ti,
                                         "problem": "data row before any header", "raw": row})
                        continue
                    rec = {"sno": int(first)}
                    for j, val in enumerate(row):
                        field = mapping.get(j)
                        if field == "sno":
                            continue
                        if field is None:
                            if clean(val):
                                rec[f"col{j}"] = clean(val)     # keep, never drop
                                unmapped[j] += 1
                            continue
                        value, prob = parse_value(field, val)
                        rec[field] = value
                        if prob:
                            problems.append({"page": pno, "table": ti, "sno": rec["sno"],
                                             "field": field, "problem": prob})
                    rec["source_pdf"] = rel
                    rec["page"] = pno
                    rec["table_index"] = ti
                    rec["raw_cells"] = row
                    rows.append(rec)
            got = len(rows) - n_before
            pages.append({"page": pno, "status": "extracted" if got else "table but no data rows",
                          "rows": got})
    if overrides and not used_override:
        problems.append({"problem": "HEADER_OVERRIDES entry for this file was never applied; "
                                    "check whether the PDF changed"})
    return {"rows": rows, "pages": pages, "problems": problems, "columns": columns,
            "unmapped_columns": dict(unmapped), "title_lines": title_lines}


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(RAW, "*", "*.pdf")))
    if not paths:
        sys.exit(f"No PDFs found under {RAW}. Run pipeline/fetch_mca_cutoffs.py first.")
    summary = []
    for path in paths:
        name = os.path.basename(path)
        m = re.search(r"(\d{4})", name) or re.search(r"/(\d{4})/", path)
        year = int(m.group(1)) if m else None
        print(f"\n### {name}")
        try:
            res = parse_pdf(path)
        except HeaderError as e:
            print(f"  ✗ STOPPED: {e}")
            summary.append((name, 0, ["header error"]))
            continue
        for p in res["pages"]:
            print(f"  page {p['page']:3d}: {p['status']:26s} rows={p['rows']}")
        out = {
            "source_pdf": os.path.relpath(path, PROJECT),
            "sha256": sha256(path),
            "year": year,
            "round_title": load_index(os.path.dirname(path)).get(name, ""),
            "document_title_lines": res["title_lines"],
            "course": "MCA",
            "columns": res["columns"],
            "row_count": len(res["rows"]),
            "pages": res["pages"],
            "header_overrides": HEADER_OVERRIDES.get(name, {}),
            "unmapped_columns": res["unmapped_columns"],
            "problems": res["problems"],
            "rows": res["rows"],
        }
        dest_dir = os.path.join(OUT, str(year))
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, name[:-4] + ".json")
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        flags = []
        if res["unmapped_columns"]:
            flags.append(f"UNMAPPED COLS {res['unmapped_columns']}")
        if res["problems"]:
            flags.append(f"{len(res['problems'])} problems")
        if not res["rows"]:
            flags.append("ZERO ROWS")
        empty = [p["page"] for p in res["pages"] if p["status"] == "table but no data rows"]
        if empty:
            flags.append(f"table but no data rows on pages {empty} (overflow column? see fix_overflow_columns.py)")
        needs_ocr = [p["page"] for p in res["pages"] if p["status"].startswith("no text")]
        if needs_ocr:
            flags.append(f"OCR needed on pages {needs_ocr}")
        print(f"  → {os.path.relpath(dest, PROJECT)}  rows={len(res['rows'])}  columns={res['columns']}")
        summary.append((name, len(res["rows"]), flags))

    print(f"\nTOTAL: {len(summary)} PDFs, {sum(s[1] for s in summary):,} rows")
    bad = [s for s in summary if s[2]]
    print("ANOMALIES:" if bad else "No anomalies.")
    for name, n, flags in bad:
        print(f"  ⚠ {name}: " + "; ".join(flags))


if __name__ == "__main__":
    main()
