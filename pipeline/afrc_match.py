"""Match AFRC fee rows to College IDs. Used by clean_cutoffs.py.

Rules (no fuzzy scoring; similar names can be different colleges):
  1. AUTO: the set of name words is identical after removing filler words
     (and / of / the / &), city names, "BE" and punctuation, AND the city agrees
     (when AFRC's Place is a known city). A first pass keeps the word "MCA" because
     some colleges have two listings that differ only by it (Shri Ram, Jabalpur);
     a second pass drops "MCA", accepted only if exactly one college fits.
  2. ALIAS: pipeline/afrc_aliases.csv, written by hand, one row per AFRC name with
     the College ID and the evidence for the match.
  3. Anything else is UNMATCHED and listed for review, never guessed.
"""
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ALIASES = os.path.join(HERE, "afrc_aliases.csv")
FILLER = {"and", "of", "the", "be", "m", "p", "mp"}


def tokens(name, cities, keep_mca=True):
    s = (name or "").lower().replace("&", " and ")
    s = re.sub(r"\(\s*\d{4}\s*\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    words = [w for w in s.split() if w not in FILLER and w not in cities]
    if not keep_mca:
        words = [w for w in words if w != "mca"]
    return frozenset(words)


def alias_key(name):
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def load_aliases():
    out = {}
    if os.path.exists(ALIASES):
        with open(ALIASES, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("afrc_name") and not r["afrc_name"].startswith("#"):
                    out[alias_key(r["afrc_name"])] = (r["college_id"].strip(), r["evidence"].strip())
    return out


def match_rows(afrc_rows, colleges, city_list):
    """Return [(row, college_id or None, method, note)] for every AFRC row."""
    cities = {c.lower() for c in city_list}
    aliases = load_aliases()
    cands = []
    for c in colleges:
        for n in {c["name"], *c.get("names_seen", [])}:
            cands.append((c["college_id"], (c.get("city") or "").lower(), n))

    def find(row, keep_mca):
        want = tokens(row["afrc_name"], cities, keep_mca)
        place = (row.get("place") or "").strip().lower()
        hits = {cid for cid, city, n in cands
                if tokens(n, cities, keep_mca) == want and (place not in cities or city == place)}
        return hits

    out = []
    for row in afrc_rows:
        a = aliases.get(alias_key(row["afrc_name"]))
        if a:
            out.append((row, a[0], "alias", a[1]))
            continue
        hits = find(row, True)
        if len(hits) != 1:
            hits2 = find(row, False)
            if len(hits2) == 1 and not hits:
                hits = hits2
        if len(hits) == 1:
            out.append((row, next(iter(hits)), "auto", "identical name words + same city"))
        elif hits:
            out.append((row, None, "ambiguous", "several colleges fit: " + ", ".join(sorted(hits))))
        else:
            out.append((row, None, "unmatched", ""))
    return out
