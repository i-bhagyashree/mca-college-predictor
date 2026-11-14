#!/usr/bin/env python3
"""Collect: college fees from the MP-DTE "Colleges Fee Details" page, for MCA.

Page: https://dte.mponline.gov.in/Portal/Services/OnlineCounselling/NW/Utilities/Citizen/frmCandidateFeeDetails.aspx
The page heading says: "Fee Information as filled by the institute." It does NOT say
whether the amount is per year or for the whole course, so we store it as printed.
For MCA it lists only some institutes (government / aided / university ones); private
college fees are fixed by AFRC and are not on this page.

ASP.NET WebForms flow, replayed per institute:
  GET -> postback drpCourse=MCA (fills the institute list)
      -> postback drpInstitute=<id> -> POST btnShow -> parse the fee table.

Saves to data/raw/fees/:
  fees_MCA_<date>/<institute value>.html   each response as received (traceability)
  fees_MCA_<date>.json                     parsed rows (+ sha256 of each HTML)

Usage: python pipeline/fetch_fees.py [COURSE_CODE]      # default MCA
"""
import hashlib
import json
import os
import re
import sys
import time
from datetime import date

import requests
from bs4 import BeautifulSoup

URL = ("https://dte.mponline.gov.in/Portal/Services/OnlineCounselling/NW/Utilities/"
       "Citizen/frmCandidateFeeDetails.aspx")
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(PROJECT, "data", "raw", "fees")
HEAD = {"User-Agent": "Mozilla/5.0 (compatible; mca-college-explorer student project)"}
FIELD = {"S.No.": "sno", "Institute Name": "institute_name", "Branch": "branch",
         "ST Fee": "fee_st", "SC Fee": "fee_sc", "OBC Fee": "fee_obc", "UR Fee": "fee_ur"}
FEE_FIELDS = {"fee_st", "fee_sc", "fee_obc", "fee_ur"}


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def hidden(soup):
    return {i["name"]: i.get("value", "") for i in soup.find_all("input")
            if i.get("name", "").startswith("__")}


def post(session, soup, extra):
    f = hidden(soup)
    f.update({"__EVENTTARGET": "", "__EVENTARGUMENT": ""})
    f.update(extra)
    r = session.post(URL, data=f, timeout=60)
    r.raise_for_status()
    return r.text


def parse_fee_table(html):
    """Rows of the table whose header row starts with S.No. / Institute Name."""
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all("table"):
        trs = t.find_all("tr", recursive=False)
        if not trs:
            continue
        header = [norm(c.get_text(" ")) for c in trs[0].find_all(["td", "th"], recursive=False)]
        if header[:2] != ["S.No.", "Institute Name"]:
            continue
        cols = [FIELD.get(h, h) for h in header]
        unknown = [h for h in header if h not in FIELD]
        rows, problems = [], []
        for tr in trs[1:]:
            vals = [norm(td.get_text(" ")) for td in tr.find_all("td", recursive=False)]
            if len(vals) != len(cols) or not vals[0].isdigit():
                continue
            rec = {}
            for c, v in zip(cols, vals):
                if c in FEE_FIELDS:
                    if re.fullmatch(r"\d+", v):
                        rec[c] = int(v)
                    else:                                   # keep as text, never guess
                        rec[c] = v or None
                        if v:
                            problems.append(f"{c} not a plain number: {v!r}")
                else:
                    rec[c] = v
            rows.append(rec)
        return rows, unknown, problems
    return [], [], ["no fee table in response"]


def main():
    course = sys.argv[1] if len(sys.argv) > 1 else "MCA"
    stamp = date.today().isoformat()
    rawdir = os.path.join(OUTDIR, f"fees_{course}_{stamp}")
    os.makedirs(rawdir, exist_ok=True)

    s = requests.Session()
    s.headers.update(HEAD)
    first = BeautifulSoup(s.get(URL, timeout=60).text, "html.parser")
    soup = BeautifulSoup(post(s, first, {"__EVENTTARGET": "drpCourse", "drpCourse": course,
                                         "drpInstitute": "00"}), "html.parser")
    institutes = [(o.get("value"), norm(o.get_text())) for o in soup.select("select[name=drpInstitute] option")
                  if o.get("value") not in (None, "", "00")]
    if not institutes:
        sys.exit("✗ no institutes listed for this course: page layout changed?")
    print(f"{len(institutes)} institutes listed for {course}")

    out_rows, issues = [], []
    for value, label in institutes:
        # fresh postback chain per institute, so every request carries valid ViewState
        page = BeautifulSoup(post(s, first, {"__EVENTTARGET": "drpCourse", "drpCourse": course,
                                             "drpInstitute": "00"}), "html.parser")
        page = BeautifulSoup(post(s, page, {"__EVENTTARGET": "drpInstitute", "drpCourse": course,
                                            "drpInstitute": value}), "html.parser")
        html = post(s, page, {"drpCourse": course, "drpInstitute": value, "btnShow": "Get Fee Details"})
        fname = re.sub(r"[^A-Za-z0-9_-]", "_", value) + ".html"
        with open(os.path.join(rawdir, fname), "w", encoding="utf-8") as fh:
            fh.write(html)
        rows, unknown, problems = parse_fee_table(html)
        if unknown:
            issues.append(f"{label}: unknown columns {unknown}")
        issues += [f"{label}: {p}" for p in problems]
        for r in rows:
            r.update({"institute_option": label, "institute_value": value,
                      "raw_html": os.path.relpath(os.path.join(rawdir, fname), PROJECT),
                      "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest()})
            out_rows.append(r)
        fees = rows[0] if rows else {}
        print(f"  {'✓' if rows else '✗'} {label[:70]:70s} UR {fees.get('fee_ur', '–')}")
        time.sleep(0.5)                                    # be gentle with the portal

    out = {"source_url": URL, "course_code": course, "fetched": stamp,
           "note": "Fee Information as filled by the institute (period not stated on the page).",
           "institutes_listed": len(institutes), "row_count": len(out_rows),
           "issues": issues, "rows": out_rows}
    path = os.path.join(OUTDIR, f"fees_{course}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"→ {os.path.relpath(path, PROJECT)}: {len(out_rows)} rows")
    if issues:
        print("ISSUES:")
        for i in issues:
            print("  ⚠", i)


if __name__ == "__main__":
    main()
