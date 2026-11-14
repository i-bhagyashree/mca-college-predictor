#!/usr/bin/env python3
"""Collect: MCA fees fixed by AFRC (Admission & Fee Regulatory Committee, MP).

Page: https://web.afrcmp.org/feesinformation/frm_showinstitutes.aspx
AFRC fixes the fees of PRIVATE professional colleges, so this fills the gap left by the
DTE fee page (which lists only government / aided / university colleges).

Flow (ASP.NET UpdatePanel, partial "Delta" postbacks; adapted from
scripts/extract_afrc_fees.py in i-shivamsoni/mp-dte-btech-predictor, MIT):
  GET -> stream "Technical Education" (rblstream=T) -> program M.C.A. (ddlProgram=5)
      -> year -> Show (all divisions) -> page through grdShowInstitutes.

The grid's own header names the fee column; we store that header text with every row
(for B.Tech it reads "Semester wise Fees (in Rs)"), so the unit comes from the page,
not from us. The fee cell is kept exactly as printed (raw_fee_text) and parsed into a
number only when it is a single plain amount.

Saves to data/raw/afrc/:
  afrc_MCA_<session>__p<N>.txt   each response as received
  afrc_MCA_<session>.json        parsed rows

Usage: python pipeline/fetch_afrc_fees.py [YEAR ...]      # default: 2026 2025 2024
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

URL = "https://web.afrcmp.org/feesinformation/frm_showinstitutes.aspx"
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(PROJECT, "data", "raw", "afrc")
STREAM, STREAM_TARGET = "T", "rblstream$2"        # Technical Education
PROGRAM, PROGRAM_LABEL = "5", "M.C.A."
HEAD = {"User-Agent": "Mozilla/5.0 (compatible; mca-college-explorer student project)",
        "Referer": URL, "Origin": "https://web.afrcmp.org"}


def clean(s):
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).strip()


class AFRC:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update(HEAD)
        self.soup, self.state = None, {}

    def controls(self):
        d = dict(self.state)
        for inp in self.soup.find_all("input"):
            n, t = inp.get("name"), (inp.get("type") or "").lower()
            if not n or t in ("submit", "button", "image", "file"):
                continue
            if t in ("radio", "checkbox") and not inp.has_attr("checked"):
                continue
            d[n] = inp.get("value", "")
        for sel in self.soup.find_all("select"):
            o = sel.find("option", selected=True) or sel.find("option")
            if sel.get("name"):
                d[sel["name"]] = o.get("value", "") if o else ""
        return d

    def ajax(self, extra):
        d = self.controls()
        d.update(extra)
        d["__ASYNCPOST"] = "true"
        r = self.s.post(URL, data=d, timeout=60,
                        headers={"X-MicrosoftAjax": "Delta=true", "X-Requested-With": "XMLHttpRequest"})
        r.raise_for_status()
        if "pageRedirect" in r.text or "oops.aspx" in r.text.lower():
            raise RuntimeError("AFRC returned an error page / redirect")
        self.state.update(dict(re.findall(r"\|hiddenField\|([^|]+)\|([^|]*)\|", r.text)))
        self.soup = BeautifulSoup(r.text, "html.parser")
        time.sleep(0.4)                                   # be gentle with the portal
        return r.text

    def start(self):
        r = self.s.get(URL, timeout=60)
        r.raise_for_status()
        self.soup = BeautifulSoup(r.text, "html.parser")
        self.state = {k: v for k, v in self.controls().items() if k.startswith("__")}
        self.ajax({"ScriptManager1": f"UPmain|{STREAM_TARGET}", "__EVENTTARGET": STREAM_TARGET,
                   "__EVENTARGUMENT": "", "rblstream": STREAM})
        progs = {o.get("value"): clean(o.get_text()) for o in self.soup.select("#ddlProgram option")}
        if progs.get(PROGRAM) != PROGRAM_LABEL:
            raise RuntimeError(f"program {PROGRAM} is now {progs.get(PROGRAM)!r}, expected {PROGRAM_LABEL!r}")
        self.ajax({"ScriptManager1": "UPmain|ddlProgram", "__EVENTTARGET": "ddlProgram", "__EVENTARGUMENT": "",
                   "rblstream": STREAM, "ddlProgram": PROGRAM})
        # Selecting "--ALL--" divisions refreshes the district/place lists, so the "0" (ALL)
        # values we post with Show pass ASP.NET event validation (same step as the reference).
        self.ajax({"ScriptManager1": "UPmain|DdlDivision", "__EVENTTARGET": "DdlDivision", "__EVENTARGUMENT": "",
                   "rblstream": STREAM, "ddlProgram": PROGRAM, "DdlDivision": "0"})

    def common(self, year):
        return {"rblstream": STREAM, "ddlProgram": PROGRAM, "ddlYear": str(year),
                "DdlDivision": "0", "DdlDistrict": "0", "DdlPlace": "0", "ddlShowInstitute": "0"}

    def show(self, year):
        return self.ajax({"ScriptManager1": "UPmain|btnShow", "__EVENTTARGET": "", "__EVENTARGUMENT": "",
                          "btnShow": "Show", **self.common(year)})

    def page(self, year, n):
        return self.ajax({"ScriptManager1": "UPmain|grdShowInstitutes", "__EVENTTARGET": "grdShowInstitutes",
                          "__EVENTARGUMENT": f"Page${n}", **self.common(year)})


def parse_grid(soup):
    """(header cells, data rows, max page) from grdShowInstitutes."""
    t = soup.find(id="grdShowInstitutes")
    if not t:
        return [], [], 1
    header, rows, pages = [], [], [1]
    for tr in t.find_all("tr"):
        ths = tr.find_all("th")
        if ths and not header:
            header = [clean(th.get_text(" ")) for th in ths]
            continue
        cells = [clean(td.get_text(" ")) for td in tr.find_all("td")]
        if len(cells) < 5 or not cells[0].isdigit():
            continue
        if all(c.isdigit() for c in cells if c):          # pager row
            continue
        rows.append(cells)
    for a in t.find_all("a", href=True):
        m = re.search(r"Page\$(\d+)", a["href"])
        if m:
            pages.append(int(m.group(1)))
    return header, rows, max(pages)


def plain_amount(raw):
    """A single plain amount ('63750', '63,750', 'Rs. 63750/-') -> int, else None."""
    nums = re.findall(r"\d[\d,]*", raw or "")
    if len(nums) == 1 and re.fullmatch(r"(Rs\.?\s*)?\d[\d,]*\s*(/-)?", (raw or "").strip(), re.I):
        return int(nums[0].replace(",", ""))
    return None


def fetch_year(year):
    c = AFRC()
    c.start()
    session = f"{year}-{year + 1}"
    text = c.show(year)
    header, rows, last = parse_grid(c.soup)
    pages = [(1, text, rows)]
    for n in range(2, last + 1):
        text = c.page(year, n)
        h2, rows2, last2 = parse_grid(c.soup)
        pages.append((n, text, rows2))
        last = max(last, last2)
    out = []
    for n, text, rows in pages:
        raw = os.path.join(OUTDIR, f"afrc_MCA_{session}__p{n}.txt")
        with open(raw, "w", encoding="utf-8") as fh:
            fh.write(text)
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        for cells in rows:
            out.append({
                "session": session, "sno": int(cells[0]), "afrc_name": cells[1],
                "address": cells[2], "place": cells[3], "raw_fee_text": cells[4],
                "fee_rs": plain_amount(cells[4]), "extra_cells": cells[5:],
                "fee_column_header": header[4] if len(header) > 4 else None,
                "raw_file": os.path.relpath(raw, PROJECT), "raw_sha256": sha, "page": n,
            })
    return header, out


def main():
    years = [int(y) for y in sys.argv[1:]] or [2026, 2025, 2024]
    os.makedirs(OUTDIR, exist_ok=True)
    for year in years:
        header, rows = fetch_year(year)
        session = f"{year}-{year + 1}"
        unparsed = [r for r in rows if r["fee_rs"] is None]
        doc = {"source_url": URL, "stream": "Technical Education", "program": PROGRAM_LABEL,
               "session": session, "fetched": date.today().isoformat(), "grid_header": header,
               "row_count": len(rows), "rows_with_non_plain_fee": len(unparsed), "rows": rows}
        path = os.path.join(OUTDIR, f"afrc_MCA_{session}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=1)
        print(f"{session}: {len(rows)} rows, header {header}")
        for r in unparsed:
            print(f"   ⚠ fee kept as text: {r['afrc_name'][:60]} → {r['raw_fee_text']!r}")
        print(f"   → {os.path.relpath(path, PROJECT)}")


if __name__ == "__main__":
    main()
