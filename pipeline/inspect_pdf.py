#!/usr/bin/env python3
"""Step 2 (inspect): look at a PDF before writing any extraction rules.

For the first, middle and last page it prints:
  * how much text the page has (0 chars => scanned image, needs OCR)
  * how many tables pdfplumber's 'lines' strategy finds, and their sizes
  * the raw header row and the first 3 data rows of every table
  * doubled-glyph evidence (how many characters dedupe_chars removes)
  * for each header cell, its letters read backwards: if that reads like a
    word (ELICIMOD -> DOMICILE) the header is rotated

Usage:
  python pipeline/inspect_pdf.py data/raw/mca/2025/2025__cutoff__first-round.pdf
  python pipeline/inspect_pdf.py <pdf> --pages 1 7 12      # choose pages (1-based)
  python pipeline/inspect_pdf.py <pdf> --debug-image 1     # writes <pdf>.p1.debug.png with cell borders
"""
import argparse
import re
import warnings

import pdfplumber

warnings.filterwarnings("ignore")
LINES = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}


def show(cells):
    return [c if c is None else c.replace("\n", "⏎") for c in cells]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", type=int, nargs="*")
    ap.add_argument("--debug-image", type=int, metavar="PAGE")
    args = ap.parse_args()

    with pdfplumber.open(args.pdf) as pdf:
        n = len(pdf.pages)
        pages = args.pages or sorted({1, (n + 1) // 2, n})
        print(f"{args.pdf}: {n} pages; inspecting {pages}\n")
        for pno in pages:
            raw = pdf.pages[pno - 1]
            page = raw.dedupe_chars(tolerance=1)
            removed = len(raw.chars) - len(page.chars)
            text = page.extract_text() or ""
            tables = page.extract_tables(LINES)
            print(f"── page {pno}: {len(text)} chars of text, {removed} doubled glyphs removed, "
                  f"{len(tables)} table(s) {[f'{len(t)}x{len(t[0]) if t else 0}' for t in tables]}")
            if not text.strip():
                print("   NO TEXT → scanned page, needs OCR")
            if pno == 1:
                print("   first text lines:", [l for l in text.splitlines()[:5]])
            for ti, t in enumerate(tables):
                print(f"   table {ti}:")
                data_seen = 0
                for row in t:
                    first = (row[0] or "").strip()
                    if first.isdigit():
                        if data_seen < 3:
                            print("     DATA  ", show(row))
                        data_seen += 1
                    else:
                        print("     OTHER ", show(row))
                        rev = [re.sub(r"[^A-Za-z]", "", c or "")[::-1].upper() for c in row]
                        if any(rev):
                            print("       reversed letters:", rev)
                print(f"     ({data_seen} data rows in this table)")
            print()
        if args.debug_image:
            out = f"{args.pdf}.p{args.debug_image}.debug.png"
            pdf.pages[args.debug_image - 1].to_image(resolution=150).debug_tablefinder(LINES).save(out)
            print("wrote", out)


if __name__ == "__main__":
    main()
