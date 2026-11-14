#!/usr/bin/env python3
"""Step 1 (collect): download MCA cut-off PDFs from the MP-DTE portal.

Adapted from fetch_btech_cutoffs.py in i-shivamsoni/mp-dte-btech-predictor (MIT).

The portal is an ASP.NET WebForms page, so we replay the clicks a person makes:
  1. GET the page and keep the hidden __VIEWSTATE / __EVENTVALIDATION fields.
  2. POST back "select year" (a radio button).
  3. POST back "click course" (a link whose postback target is rptCourse$ctlNN$lnkCourse).
  4. Read the PDF paths out of the `src=...pdf` parameter of each returned link.
  5. Download each one and check it really starts with %PDF.

Differences from the B.Tech script:
  * We do NOT hard-code the course index (ctlNN) or the year index (2026 - year).
    Both are looked up on the live page by their visible labels, so they keep
    working when the portal adds a year or a course.
  * TLS certificate checking is ON by default. Pass --insecure only if the
    portal's certificate fails and you understand that the download is then
    not protected against tampering.
  * Files are saved under readable names (2025__cutoff__first-round.pdf).
    _index.txt records: saved name | original name | official title | sha256.

Usage:
  python pipeline/fetch_mca_cutoffs.py --list-courses          # see what the portal offers
  python pipeline/fetch_mca_cutoffs.py --years 2025 2024
  python pipeline/fetch_mca_cutoffs.py --years 2025 --course "MASTER OF COMPUTER APPLICATION"
"""
import argparse
import hashlib
import os
import re
import sys
import unicodedata
import urllib.parse

import requests
from bs4 import BeautifulSoup

BASE = "https://dte.mponline.gov.in/Portal/Services/OnlineCounselling/NW/Utilities/Citizen/CutOffList.aspx"
ROOT = "https://dte.mponline.gov.in"
VIEWER = ROOT + "/Portal/Services/OnlineCounselling/NW/Utilities/InitailizeCommonView.aspx?UserType=C&src="
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(PROJECT, "data", "raw", "mca")
DEFAULT_COURSE = "MASTER OF COMPUTER APPLICATION"
# The portal labels courses in Hindi. Known labels for each English name:
COURSE_ALIASES = {
    "MASTER OF COMPUTER APPLICATION": ["मास्टर ऑफ़ कम्प्यूटर एप्लीकेशन", "मास्टर ऑफ़ कंप्यूटर एप्लीकेशन"],
}


def course_matches(wanted, label):
    """Exact-ish match on the course label (English or a known Hindi alias).
    Uses equality after collapsing spaces, so 'MCA' never matches 'Integrated MCA'."""
    # NFD: the portal writes फ़ as one code point (U+095E); typed text often uses फ + ़.
    norm = lambda x: unicodedata.normalize("NFD", re.sub(r"\s+", " ", x)).strip().upper()
    names = [wanted] + COURSE_ALIASES.get(wanted.upper(), [])
    return any(norm(n) == norm(label) for n in names)
HEAD = {"User-Agent": "Mozilla/5.0 (compatible; mca-college-explorer student project)"}


def hidden_fields(soup):
    return {i["name"]: i.get("value", "") for i in soup.find_all("input")
            if i.get("name", "").startswith("__")}


def year_radios(soup):
    """{year: postback target} read from the radio list, e.g. {2025: 'rblYear$1'}."""
    out = {}
    for inp in soup.find_all("input", {"name": "rblYear"}):
        val, iid = inp.get("value", ""), inp.get("id", "")
        m = re.search(r"_(\d+)$", iid)
        if val.isdigit() and m:
            out[int(val)] = f"rblYear${m.group(1)}"
    return out


def course_links(soup):
    """[(label, postback target)] for every course link on the page."""
    out = []
    for a in soup.find_all("a", href=True):
        m = re.search(r"__doPostBack\('([^']*lnkCourse)'", a["href"])
        if m:
            out.append((a.get_text(" ", strip=True), m.group(1)))
    return out


def post(session, soup, target, year, verify):
    d = hidden_fields(soup)
    d.update({"__EVENTTARGET": target, "__EVENTARGUMENT": "", "rblYear": str(year)})
    r = session.post(BASE, data=d, verify=verify, timeout=60)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def readable_name(year, title):
    """'OPENING AND CLOSING FIRST ROUND 2025' -> '2025__cutoff__first-round.pdf'."""
    t = re.sub(r"\b(OPENING|AND|CLOSING|RANK|CUT ?OFF|LIST|MCA|\d{4})\b", " ", title.upper())
    slug = re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-") or "untitled"
    return f"{year}__cutoff__{slug}.pdf"


def download(session, src, verify):
    for url in (ROOT + urllib.parse.quote(src), VIEWER + urllib.parse.quote(src)):
        try:
            r = session.get(url, verify=verify, timeout=120)
        except requests.RequestException as e:
            print(f"      ! {e}")
            continue
        if r.status_code == 200 and r.content[:4] == b"%PDF":   # an HTML error page is not a PDF
            return r.content
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs="*", default=[])
    ap.add_argument("--course", default=DEFAULT_COURSE,
                    help="course label (English name with a known Hindi alias, or the exact Hindi label)")
    ap.add_argument("--list-courses", action="store_true", help="print years and courses, then stop")
    ap.add_argument("--insecure", action="store_true", help="disable TLS certificate checks (see docstring)")
    args = ap.parse_args()
    verify = not args.insecure
    if args.insecure:
        import warnings
        warnings.filterwarnings("ignore")
        print("WARNING: TLS certificate checking is OFF for this run.")

    s = requests.Session()
    s.headers.update(HEAD)
    first = BeautifulSoup(s.get(BASE, verify=verify, timeout=30).text, "html.parser")
    years = year_radios(first)
    print("Years on portal:", sorted(years, reverse=True) or "(none found: page layout changed?)")

    if args.list_courses:
        y = max(years)
        for label, target in course_links(post(s, first, years[y], y, verify)):
            print(f"  {target:35s} {label}")
        return

    if not args.years:
        sys.exit("Pass --years (or --list-courses first).")

    total = 0
    for year in args.years:
        print(f"\n=== {year} ===")
        if year not in years:
            print("  year not offered on the portal; skipped")
            continue
        sess = requests.Session()
        sess.headers.update(HEAD)
        soup = BeautifulSoup(sess.get(BASE, verify=verify, timeout=30).text, "html.parser")
        soup = post(sess, soup, years[year], year, verify)
        matches = [(l, t) for l, t in course_links(soup) if course_matches(args.course, l)]
        if len(matches) != 1:
            print(f"  expected exactly 1 course matching {args.course!r}, found {matches}")
            print("  run with --list-courses and pass the exact label with --course")
            continue
        label, target = matches[0]
        print(f"  course: {label}  ({target})")
        soup = post(sess, soup, target, year, verify)

        links = []
        for a in soup.find_all("a", href=True):
            m = re.search(r"src=([^&]+\.pdf)", a["href"], re.IGNORECASE)
            if m:
                links.append((a.get_text(" ", strip=True), urllib.parse.unquote(m.group(1)).strip()))
        if not links:
            print("  (no PDFs listed)")
            continue

        ydir = os.path.join(OUTDIR, str(year))
        os.makedirs(ydir, exist_ok=True)
        index, used = [], set()
        for title, src in links:
            name = base = readable_name(year, title)
            n = 2
            while name in used:                         # two docs with the same title slug
                name = base[:-4] + f"__v{n}.pdf"
                n += 1
            used.add(name)
            data = download(sess, src, verify)
            if not data:
                print(f"  ✗ FAILED  {title}  src={src}")
                index.append(f"FAILED | {os.path.basename(src)} | {title} | -")
                continue
            dest = os.path.join(ydir, name)
            with open(dest, "wb") as fh:
                fh.write(data)
            sha = hashlib.sha256(data).hexdigest()
            print(f"  ✓ {name}  ({len(data):,} B)  [{title}]")
            index.append(f"{name} | {os.path.basename(src)} | {title} | {sha}")
            total += 1
        with open(os.path.join(ydir, "_index.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"# {label} cut-off lists {year}\n# saved_name | original_name | official_title | sha256\n")
            fh.write("\n".join(index) + "\n")
    print(f"\nDONE. {total} PDFs under {OUTDIR}")


if __name__ == "__main__":
    main()
