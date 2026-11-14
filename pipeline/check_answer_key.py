#!/usr/bin/env python3
"""Step 4 (check): compare extracted JSON against a hand-written answer key.

Write the answer key BEFORE running the extractor, by reading the PDF yourself.
Never edit the answer key to make the output pass.

tests/fixtures/answer_key.csv columns:
  pdf      file name of the PDF, e.g. 2025__cutoff__first-round.pdf
  page     1-based page number where you read the row
  sno      the S.No. printed in the row
  field    e.g. institute_name, opening_rank, allotted_category
  expected exactly what the PDF shows ('' for an empty cell)

Usage: python pipeline/check_answer_key.py [tests/fixtures/answer_key.csv]
Exit code 1 if anything mismatches.
"""
import csv
import glob
import json
import os
import re
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED = os.path.join(PROJECT, "data", "extracted", "mca")


def norm(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:g}"
    s = re.sub(r"\s+", " ", str(v)).strip()
    if re.fullmatch(r"\d+\.\d+", s):
        return f"{float(s):g}"
    if re.fullmatch(r"\d{1,3}(,\d{3})+", s):
        return s.replace(",", "")
    return s


def main():
    key_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT, "tests", "fixtures", "answer_key.csv")
    with open(key_path, newline="", encoding="utf-8") as fh:
        key = [r for r in csv.DictReader(fh) if r["pdf"] and not r["pdf"].startswith("#")]
    if not key:
        sys.exit(f"{key_path} has no rows yet. Fill it in from the PDF first.")

    cache, ok, bad = {}, 0, []
    for k in key:
        pdf = k["pdf"]
        if pdf not in cache:
            hits = glob.glob(os.path.join(EXTRACTED, "*", pdf[:-4] + ".json"))
            cache[pdf] = json.load(open(hits[0], encoding="utf-8"))["rows"] if hits else None
        rows = cache[pdf]
        where = f"{pdf} p{k['page']} S.No.{k['sno']} {k['field']}"
        if rows is None:
            bad.append(f"{where}: no extracted JSON for this PDF")
            continue
        match = [r for r in rows if r["page"] == int(k["page"]) and r["sno"] == int(k["sno"])]
        if len(match) != 1:
            bad.append(f"{where}: found {len(match)} rows (expected 1)")
            continue
        got = match[0].get(k["field"], "<field missing>")
        if norm(got) == norm(k["expected"]):
            ok += 1
        else:
            bad.append(f"{where}: expected {k['expected']!r}, got {got!r}   raw={match[0]['raw_cells']}")

    print(f"{ok}/{len(key)} fields match")
    for b in bad:
        print("  ✗", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
