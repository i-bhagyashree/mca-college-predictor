#!/usr/bin/env python3
"""Collect: the MP-DTE 'Tentative List of Institutes and Intake' (seat matrix) for MCA.

Adapted from fetch_intake.py in i-shivamsoni/mp-dte-btech-predictor (MIT).
The portal serves an HTML table (no PDF parsing needed). It gives every college's
official College ID, which clean_cutoffs.py uses to join cut-off names across years.

ASP.NET WebForms flow: GET -> postback to select course -> POST "View" with all
filters set to All -> parse the result table.

Saves to data/raw/intake/:
  intake_MCA_<session>.html   the response exactly as received (for traceability)
  intake_MCA_<session>.json   parsed rows

Usage: python pipeline/fetch_intake.py [COURSE_CODE]     # default MCA
"""
import hashlib
import json
import os
import re
import sys

import requests
from bs4 import BeautifulSoup

URL = ("https://dte.mponline.gov.in/Portal/Services/OnlineCounselling/"
       "Administration/InitialIntake.aspx")
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(PROJECT, "data", "raw", "intake")
HEAD = {"User-Agent": "Mozilla/5.0 (compatible; mca-college-explorer student project)"}

FIELD = {
    "S. No.": "sno", "College ID": "college_id", "Institute Name": "institute_name",
    "Branch": "branch", "Total Intake": "total_intake",
    "Tuition Fee Waiver seats": "tfw_seats", "IPS Seats": "ips_seats",
    "NRI Seats": "nri_seats", "EWS Seats": "ews_seats", "PIO/FN Seats": "pio_fn_seats",
    "NTPC Seats": "ntpc_seats", "Institute Type": "institute_type",
    "AICTE Code": "aicte_code", "Remark": "remark",
    "AFRC/Govt. Fee Status": "fee_status", "Minority Status": "minority_status",
    "University": "university",
}
INT_FIELDS = {"sno", "total_intake", "tfw_seats", "ips_seats", "nri_seats",
              "ews_seats", "pio_fn_seats", "ntpc_seats"}


def hidden(soup):
    return {i["name"]: i.get("value", "") for i in soup.find_all("input") if i.get("name")}


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def to_int(s):
    s = norm(s)
    return int(s) if s.isdigit() else (s or None)


def fetch(course):
    s = requests.Session()
    s.headers.update(HEAD)
    soup = BeautifulSoup(s.get(URL, timeout=30).text, "html.parser")
    f = hidden(soup)
    f.update({"__EVENTTARGET": "ddlCourse", "__EVENTARGUMENT": "", "ddlCourse": course,
              "drpbranch": "00", "drpCity": "0", "DrpInstType": "00", "drpSeatType": "0"})
    f.pop("btnView", None)
    soup = BeautifulSoup(s.post(URL, data=f, timeout=60).text, "html.parser")
    f = hidden(soup)
    f.update({"__EVENTTARGET": "", "__EVENTARGUMENT": "", "ddlCourse": course,
              "drpbranch": "1", "drpCity": "0", "DrpInstType": "0",
              "drpSeatType": "0", "btnView": "View"})
    r = s.post(URL, data=f, timeout=180)
    r.raise_for_status()
    return r.text


def parse(soup):
    """Anchor on the 'Institute Name' header cell and read its own table."""
    cell = soup.find(lambda t: t.name in ("td", "th") and norm(t.get_text()) == "Institute Name")
    if not cell:
        return [], [], []
    tbl = cell.find_parent("table")
    header = [norm(c.get_text(" ")) for c in cell.find_parent("tr").find_all(["th", "td"], recursive=False)]
    cols = [FIELD.get(h, h) for h in header]
    unknown = [h for h in header if h not in FIELD]
    rows = []
    for tr in tbl.find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) != len(cols):
            continue
        vals = [norm(td.get_text(" ")) for td in tds]
        if not vals[0].isdigit():
            continue
        rows.append({cols[j]: (to_int(v) if cols[j] in INT_FIELDS else (v or None))
                     for j, v in enumerate(vals)})
    return cols, rows, unknown


def main():
    course = sys.argv[1] if len(sys.argv) > 1 else "MCA"
    html = fetch(course)
    soup = BeautifulSoup(html, "html.parser")
    full = soup.get_text(" ", strip=True)
    m = re.search(r"TENTATIVE LIST OF INSTITUTES AND INTAKE\s*(20\d{2}-\d{2})", full, re.I)
    session = m.group(1) if m else "unknown"
    cols, rows, unknown = parse(soup)
    if not rows:
        sys.exit("✗ no rows parsed: page layout changed? Inspect the saved HTML.")

    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(OUTDIR, f"intake_{course}_{session}")
    with open(base + ".html", "w", encoding="utf-8") as fh:
        fh.write(html)
    out = {
        "source_url": URL, "course_code": course,
        "title": norm(m.group(0)) if m else "Tentative List of Institutes and Intake",
        "session": session,
        "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
        "columns": cols, "unknown_columns": unknown,
        "row_count": len(rows), "rows": rows,
    }
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    ids = {r.get("college_id") for r in rows}
    print(f"{out['title']}: rows={len(rows)} colleges={len(ids)}")
    if unknown:
        print(f"⚠ unknown columns kept under their header text: {unknown}")
    print(f"→ {os.path.relpath(base, PROJECT)}.json/.html")


if __name__ == "__main__":
    main()
