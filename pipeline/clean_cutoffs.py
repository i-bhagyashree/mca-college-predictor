#!/usr/bin/env python3
"""Clean: master cut-off rows -> tidy rows joined to official College IDs.

Inputs
  data/processed/mca_cutoffs_master.json   (merge_cutoffs.py)
  data/raw/intake/intake_MCA_*.json        (fetch_intake.py; official College IDs)
  data/raw/fees/fees_MCA_*.json            (fetch_fees.py; optional, fee per category)
  data/raw/afrc/afrc_MCA_*.json            (fetch_afrc_fees.py; optional, AFRC semester fees,
                                            matched by afrc_match.py + pipeline/afrc_aliases.csv)
Outputs (data/processed/)
  mca_cutoffs_clean.json       one row per cut-off line, with derived fields
  colleges.json                one entry per college (official or provisional ID)
  review_unmatched_names.csv   cut-off names with no official ID: review these by hand

Cleaning rules (cleaning ideas from scripts/preprocess.py in mp-dte-btech-predictor):
  * College names -> College ID. Exact match after norm_name() against the intake
    list, else an explicit NAME_ALIASES entry with written evidence. NO fuzzy
    matching: similar names can be different campuses (two "Technocrats" and two
    "Shri Ram" colleges appear side by side in the same PDF).
    Anything unmatched gets a provisional ID "U-<slug>" and is listed for review.
  * Category "SOCIAL/CLASS/GENDER", e.g. OBC/NCC/F, split into three fields.
    Short forms seen in MCA PDFs: "EWS", "FW", "FW/OP", "FW/F". A trailing
    "(After Conversion)" note becomes after_conversion=true.
  * Domicile: Y -> MP, AI -> All-India, N -> Other state, blank -> null (unspecified).
  * Round: from the file name (first-round -> R1, ...), "result-awaited-candidates"
    becomes a separate flag, not a different round.
Every row keeps source_pdf, page and raw_cells. Unknown codes are kept and reported,
never guessed. The script stops (exit 1) if a sanity rule fails.

Usage: python pipeline/clean_cutoffs.py
"""
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(PROJECT, "data", "processed")
MASTER = os.path.join(PROC, "mca_cutoffs_master.json")

# Cut-off name (after norm_name) -> official College ID, with the evidence for each.
NAME_ALIASES = {
    # Same name, city suffix missing in 2023; 2024+ PDFs and the intake list add ", INDORE".
    "chameli devi group of institutions": "118",
    # Same, 2023 PDF omits ", INDORE".
    "ips academy institute of business management and research sanwer campus": "119",
    # Vikram University was renamed Samrat Vikramaditya Vishwavidyalaya; the 2026 PDF
    # and intake list use the new name for the same Institute of Computer Science.
    "institute of computer science vikram university ujjain": "010",
    # Same college; old PDFs write "(MCA) (Borawan) Khargone", new ones ", BORAWAN, KHARGONE".
    "jawaharlal institute of technology mca borawan khargone": "128",
    # 2021-22 PDFs omit ", UJJAIN"; 2023+ and the intake list include it.
    "mahakal institute of technology and management": "115",
}

MP_CITIES = ["Indore", "Bhopal", "Gwalior", "Jabalpur", "Ujjain", "Dewas", "Ratlam", "Rewa",
             "Satna", "Vidisha", "Khandwa", "Khargone", "Neemuch", "Mandsaur", "Sagar", "Sanwer"]

SOCIAL = {"UR", "OBC", "SC", "ST", "EWS", "FW"}
CLASS = {"X": "ordinary", "NCC": "NCC", "H": "divyang", "D": "defence", "S": "sainik",
         "FF": "freedom fighter"}
GENDER = {"OP": "open", "F": "female", "M": "male"}
DOMICILE = {"Y": "MP", "AI": "All-India", "N": "Other state"}
ROUNDS = [("special-round-2", "SP2", "Special round 2"), ("special-round", "SP1", "Special round"),
          ("second-round", "R2", "Second round"), ("first-round", "R1", "First round")]


def norm_name(s):
    """Name key for joining across years (adapted from preprocess.py)."""
    s = (s or "").lower()
    s = re.sub(r"\(\s*\d{4}\s*\)", " ", s)                 # "(2001)" establishment year
    s = re.sub(r"\(\s*m\.?\s*p\.?\s*\)", " ", s)           # "(M.P.)"
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\bdeemed( to be)? university\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def city_of(name):
    """The MP city named LAST in the name ('Sagar Institute ..., Bhopal' -> Bhopal)."""
    best = None
    for c in MP_CITIES:
        for m in re.finditer(r"\b" + c.upper() + r"\b", (name or "").upper()):
            if best is None or m.start() > best[0]:
                best = (m.start(), c)
    city = best[1] if best else None
    return "Indore" if city == "Sanwer" else city      # Sanwer is in Indore district


def split_category(cat):
    """'OBC/NCC/F' -> dict; unknown parts are kept and returned as problems."""
    if not cat:
        return {"social": None, "seat_class": None, "gender": None, "after_conversion": False}, ["no category"]
    s = cat.strip()
    conv = bool(re.search(r"\(\s*after\s+conversion\s*\)", s, re.I))
    s = re.sub(r"\(\s*after\s+conversion\s*\)", "", s, flags=re.I).strip()
    p = [x.strip().upper() for x in s.split("/")]
    probs = []
    if p[0] == "FW":                       # tuition fee waiver seat: FW, FW/OP, FW/F
        social, cls, gen = "FW", None, (p[1] if len(p) > 1 else "OP")
    elif len(p) == 3:
        social, cls, gen = p
    elif len(p) == 1:                      # "EWS" alone
        social, cls, gen = p[0], None, "OP"
    else:
        return {"social": None, "seat_class": None, "gender": None, "after_conversion": conv}, [f"unknown category shape {cat!r}"]
    if social not in SOCIAL:
        probs.append(f"unknown social category {social!r}")
    if cls is not None and cls not in CLASS:
        probs.append(f"unknown seat class {cls!r}")
    if gen not in GENDER:
        probs.append(f"unknown gender {gen!r}")
    return {"social": social, "seat_class": cls, "gender": gen, "after_conversion": conv}, probs


def round_of(source_pdf):
    base = os.path.basename(source_pdf)
    for key, code, label in ROUNDS:
        if key in base:
            return code, label, "result-awaited" in base
    return None, None, "result-awaited" in base


def load_intake():
    paths = sorted(glob.glob(os.path.join(PROJECT, "data", "raw", "intake", "intake_MCA_*.json")))
    if not paths:
        sys.exit("no intake list: run pipeline/fetch_intake.py first")
    d = json.load(open(paths[-1], encoding="utf-8"))
    return d, {r["college_id"]: r for r in d["rows"]}


def parse_remark(remark):
    """'PREVIOUS YEAR AFFILIATION : YES, CURRENT YEAR AFFILIATION : APPLIED, ...' -> dict.
    Unknown wording is kept in `remark` and the parsed fields stay None (never guessed)."""
    r = remark or ""
    prev = re.search(r"PREVIOUS\s+YEAR\s+AFFILIATION\s*:\s*([A-Z]+)", r, re.I)
    cur = re.search(r"CURRENT\s+YEAR\s+AFFILIATION\s*:\s*([A-Z]+)", r, re.I)
    return {
        "affiliation_previous_year": prev.group(1).upper() if prev else None,
        "affiliation_current_year": cur.group(1).upper() if cur else None,
        "affiliation_required_before_merit": bool(re.search(r"REQUIRED\s+BEFORE\s+MERIT", r, re.I)),
        "remark": remark,
    }


def intake_details(r, session):
    """Every useful column of the intake (seat matrix) row for one college."""
    seats = lambda v: v if isinstance(v, int) else None
    return {
        "type": r.get("institute_type"),
        "university": r.get("university"),
        "intake_2026": r.get("total_intake"),
        "fee_status": r.get("fee_status"),
        "intake_detail": {
            "session": session,
            "total": seats(r.get("total_intake")),
            "tfw_seats": seats(r.get("tfw_seats")),
            "nri_seats": seats(r.get("nri_seats")),
            "ews": r.get("ews_seats"),                 # the portal prints a YES flag, not a count
            "aicte_code": r.get("aicte_code"),
            "minority": (r.get("minority_status") or "").upper() == "YES",
            "fee_fixed": (r.get("fee_status") or "").upper() == "YES",   # "AFRC/Govt. Fee Status"
            **parse_remark(r.get("remark")),
        },
    }


def load_fees():
    """Latest fees_MCA_*.json from fetch_fees.py, or None if not fetched yet."""
    paths = sorted(glob.glob(os.path.join(PROJECT, "data", "raw", "fees", "fees_MCA_*.json")))
    return json.load(open(paths[-1], encoding="utf-8")) if paths else None


def main():
    master = json.load(open(MASTER, encoding="utf-8"))
    intake_doc, intake = load_intake()
    by_norm = {}
    for cid, r in intake.items():
        k = norm_name(r["institute_name"])
        if k in by_norm:
            sys.exit(f"✗ two intake colleges share the key {k!r}")
        by_norm[k] = cid
    for k, cid in NAME_ALIASES.items():
        if cid not in intake:
            sys.exit(f"✗ alias {k!r} points to unknown College ID {cid}")

    clean, problems = [], []
    names_by_id = defaultdict(Counter)
    years_by_id = defaultdict(set)
    unmatched = defaultdict(lambda: {"years": set(), "files": set(), "rows": 0, "raw": set()})
    for r in master["rows"]:
        raw_name = r["institute_name"]
        k = norm_name(raw_name)
        cid = by_norm.get(k) or NAME_ALIASES.get(k)
        how = "exact" if k in by_norm else ("alias" if cid else "provisional")
        if not cid:
            cid = "U-" + k.replace(" ", "-")
            u = unmatched[cid]
            u["years"].add(r["year"]); u["files"].add(os.path.basename(r["source_pdf"]))
            u["rows"] += 1; u["raw"].add(raw_name)
        names_by_id[cid][raw_name] += 1
        years_by_id[cid].add(r["year"])

        cat, cprobs = split_category(r.get("allotted_category"))
        rcode, rlabel, awaited = round_of(r["source_pdf"])
        where = f"{os.path.basename(r['source_pdf'])} p{r['page']} S.No.{r['sno']}"
        for p in cprobs:
            problems.append(f"{where}: {p}")
        if rcode is None:
            problems.append(f"{where}: unknown round")
        dom = r.get("domicile")
        if dom is not None and dom not in DOMICILE:
            problems.append(f"{where}: unknown domicile {dom!r}")
        o, c = r.get("opening_rank"), r.get("closing_rank")
        if not (isinstance(o, int) and isinstance(c, int) and 0 < o <= c):
            problems.append(f"{where}: bad ranks {o!r}, {c!r}")
        if not isinstance(r.get("total_allotted"), int) or r["total_allotted"] <= 0:
            problems.append(f"{where}: bad total_allotted {r.get('total_allotted')!r}")

        clean.append({
            "college_id": cid, "id_match": how, "institute_name_raw": raw_name,
            "year": r["year"], "round": rcode, "round_label": rlabel, "result_awaited": awaited,
            "branch": r.get("branch") or "MCA",
            "institute_type_raw": r.get("institute_type"),
            "fee_waiver_seat": {"Y": True, "N": False}.get(r.get("fw")),
            "opening_rank": o, "closing_rank": c,
            "opening_score": r.get("opening_score"), "closing_score": r.get("closing_score"),
            "category_raw": r.get("allotted_category"), **cat,
            "eligible_category_raw": r.get("eligible_category"),
            "domicile": DOMICILE.get(dom), "domicile_raw": dom,
            "total_allotted": r.get("total_allotted"),
            "source_pdf": r["source_pdf"], "page": r["page"], "sno": r["sno"],
            "raw_cells": r["raw_cells"],
        })

    # Two different raw names inside ONE pdf mapping to the same ID would mean we merged
    # two separate colleges by mistake.
    seen = defaultdict(set)
    for row in clean:
        seen[(row["source_pdf"], row["college_id"])].add(row["institute_name_raw"])
    for (pdf, cid), names in seen.items():
        if len(names) > 1:
            problems.append(f"{os.path.basename(pdf)}: different names joined to {cid}: {sorted(names)}")

    colleges = []
    for cid in sorted(names_by_id, key=lambda x: (x.startswith("U-"), x)):
        official = intake.get(cid)
        latest_raw = names_by_id[cid].most_common(1)[0][0]
        name = official["institute_name"] if official else latest_raw
        colleges.append({
            "college_id": cid, "official": bool(official), "name": name,
            "city": city_of(name) or city_of(latest_raw),
            **(intake_details(official, intake_doc["session"]) if official else
               {"type": None, "university": None, "intake_2026": None, "fee_status": None}),
            "years_in_cutoffs": sorted(years_by_id[cid]),
            "names_seen": sorted(names_by_id[cid]),
        })
    # Colleges in the intake list that never appeared in a cut-off PDF (e.g. new in 2026-27).
    for cid, r in sorted(intake.items()):
        if cid not in names_by_id:
            colleges.append({"college_id": cid, "official": True, "name": r["institute_name"],
                             "city": city_of(r["institute_name"]),
                             **intake_details(r, intake_doc["session"]),
                             "years_in_cutoffs": [], "names_seen": []})

    for c in colleges:
        if c["official"] and not c["city"]:
            problems.append(f"college {c['college_id']}: no city found in {c['name']!r}")
        d = c.get("intake_detail")
        if d and d["remark"] and d["affiliation_current_year"] is None:
            problems.append(f"college {c['college_id']}: unrecognised intake remark {d['remark']!r}")

    # Fees (DTE "Colleges Fee Details" page): join by the same exact-name / alias rule.
    fees_doc = load_fees()
    if fees_doc:
        by_id = {c["college_id"]: c for c in colleges}
        for fr in fees_doc["rows"]:
            k = norm_name(fr["institute_name"])
            cid = by_norm.get(k) or NAME_ALIASES.get(k)
            if not cid or cid not in by_id:
                problems.append(f"fees: no College ID for {fr['institute_name']!r}")
                continue
            if by_id[cid].get("fees"):
                problems.append(f"fees: two fee rows for College ID {cid}")
                continue
            by_id[cid]["fees"] = {
                "st": fr["fee_st"], "sc": fr["fee_sc"], "obc": fr["fee_obc"], "ur": fr["fee_ur"],
                "fetched": fees_doc["fetched"], "raw_html": fr["raw_html"],
                "note": fees_doc["note"],
            }
        print(f"fees: {sum(1 for c in colleges if c.get('fees'))} colleges have a fee row "
              f"(fetched {fees_doc['fetched']})")
    else:
        print("fees: none (run pipeline/fetch_fees.py to add them)")

    # AFRC fees (private colleges; "Semester wise Fees (in Rs)"), all fetched sessions.
    from afrc_match import match_rows
    afrc_files = sorted(glob.glob(os.path.join(PROJECT, "data", "raw", "afrc", "afrc_MCA_*.json")))
    by_id = {c["college_id"]: c for c in colleges}
    review = []
    for f in afrc_files:
        doc = json.load(open(f, encoding="utf-8"))
        unit = doc["grid_header"][4] if len(doc.get("grid_header", [])) > 4 else None
        if unit != "Semester wise Fees (in Rs)":
            problems.append(f"{os.path.basename(f)}: fee column header is {unit!r}; check the unit before using it")
            continue
        seen = {}
        for row, cid, method, note in match_rows(doc["rows"], colleges, MP_CITIES):
            review.append([doc["session"], row["afrc_name"], row["address"], row["place"], row["raw_fee_text"],
                           cid or "", by_id[cid]["name"] if cid in by_id else "", method, note])
            if cid and cid not in by_id:
                problems.append(f"AFRC alias points to unknown College ID {cid!r} ({row['afrc_name']!r})")
                continue
            if not cid:
                continue
            if cid in seen:
                problems.append(f"AFRC {doc['session']}: two rows matched to {cid}: {seen[cid]!r} and {row['afrc_name']!r}")
                continue
            seen[cid] = row["afrc_name"]
            by_id[cid].setdefault("afrc_fees", []).append({
                "session": doc["session"], "semester_fee_rs": row["fee_rs"], "raw_fee_text": row["raw_fee_text"],
                "afrc_name": row["afrc_name"], "address": row["address"], "match": method,
                "raw_file": row["raw_file"], "fetched": doc["fetched"],
            })
    for c in colleges:
        if c.get("afrc_fees"):
            c["afrc_fees"].sort(key=lambda x: x["session"])
    with open(os.path.join(PROC, "review_afrc_matches.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "afrc_name", "address", "place", "raw_fee_text", "college_id", "college_name", "match", "note"])
        w.writerows(sorted(review, key=lambda r: (r[7] != "unmatched", r[0], r[1])))
    if afrc_files:
        counts = Counter(r[7] for r in review)
        print(f"AFRC fees: {dict(counts)} across {len(afrc_files)} sessions; "
              f"{sum(1 for c in colleges if c.get('afrc_fees'))} colleges have an AFRC fee "
              f"→ data/processed/review_afrc_matches.csv")

    os.makedirs(PROC, exist_ok=True)
    json.dump({"intake_session": intake_doc["session"], "row_count": len(clean), "rows": clean},
              open(os.path.join(PROC, "mca_cutoffs_clean.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    json.dump(colleges, open(os.path.join(PROC, "colleges.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(os.path.join(PROC, "review_unmatched_names.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["provisional_id", "raw_names", "years", "rows", "files"])
        for cid, u in sorted(unmatched.items()):
            w.writerow([cid, " | ".join(sorted(u["raw"])), " ".join(map(str, sorted(u["years"]))),
                        u["rows"], " ".join(sorted(u["files"]))])

    match = Counter(r["id_match"] for r in clean)
    print(f"rows={len(clean):,}  colleges={len(colleges)} "
          f"(official {sum(c['official'] for c in colleges)}, provisional {len(unmatched)})")
    print(f"name join: {dict(match)}")
    print(f"categories: {dict(Counter(r['social'] for r in clean))}")
    print(f"rounds: {dict(Counter((r['year'], r['round']) for r in clean))}")
    if unmatched:
        print(f"⚠ {len(unmatched)} names without an official College ID → data/processed/review_unmatched_names.csv")
    if problems:
        print(f"✗ {len(problems)} PROBLEMS:")
        for p in problems[:50]:
            print("   ", p)
        sys.exit(1)
    print("✓ sanity rules passed")


if __name__ == "__main__":
    main()
