# Project context

Short brief for whoever works on this next. Check it against the files before trusting it.

**Goal and users:** help MCA applicants in Madhya Pradesh read their merit rank from their graduation
percentage and see which colleges admitted that rank in past MP-DTE counselling rounds.

**First-release scope:** Rank Predictor (official merit list), college list with ten years of cut-offs,
college details, Choice Planner, ML analysis (Rank Predictor vs ML), About.
**Later features:** none planned. Any new year needs its PDFs added and the pipeline re-run.

**Stack:** static site (HTML, CSS, plain JavaScript, no build step) on GitHub Pages.
Data pipeline in Python 3.14 in `.venv`: pdfplumber 0.11.10, scikit-learn 1.9.1, numpy 2.5.4,
requests 2.34.2, beautifulsoup4 4.15.0. Fonts (Inter, Manrope) are served from `assets/fonts`.
Jekyll and UX4G are **not** used: one static page needed no build tool, and the design is our own CSS.

**Source register and checksums:** `data/raw/mca/<year>/_index.txt`
(saved name, original name, official title, SHA-256, the official page that lists it, download date).
Every extracted record also keeps `source_pdf`, `page` and `raw_cells`.

**Commands:**

```bash
python pipeline/fetch_mca_cutoffs.py --years 2026        # download
python pipeline/mca_pdf_to_json.py                       # extract cut-off PDFs only
python pipeline/fix_overflow_columns.py                  # 2017 special round 2 correction
python pipeline/merge_cutoffs.py && python pipeline/clean_cutoffs.py
python pipeline/merit_lists.py && python pipeline/ml_rank_from_percentage.py
python pipeline/build_site_data.py                       # writes assets/data/mca-data.js
python pipeline/data_audit.py                            # sanity rules and coverage
python -m unittest discover -s tests                     # 44 tests
```

**Dataset:** 20 cut-off PDFs (2017–2026), 3,028 rows, 71 colleges. A full rebuild from the PDFs
gives a byte-identical site data file (checked 2026-11-18).
Merit lists: 2025 R1, 2026 R1, 2026 R2 (3,329 candidates), percentage and rank only.
**Unresolved:** 13 lines in `data/processed/review_unmatched_names.csv`; hand-written answer key not filled yet.

**Rules:** no student names or roll numbers anywhere (merit PDFs and anything made from them are git-ignored);
no "safe" or "% chance" promises, results are labelled against past cut-offs only; nothing typed by the
user is stored or put in the URL; small changes, one commit each.

**Relevant files:** `index.html` (all pages), `assets/js/app.js` (logic), `assets/css/styles.css`,
`assets/data/mca-data.js` (generated, never edit), `pipeline/` (one script per step), `tests/`, `reports/`.

**Current state:** live at https://i-bhagyashree.github.io/mca-college-predictor/. Last verified commit: see `git log -1`.

**What works / checks run:** see `docs/quality-checks.md`.

**Known problems:** see `docs/data-and-methodology.md`, section "Gaps and limits".

**Next small task:** fill `tests/fixtures/answer_key.csv` by hand from the PDFs and run
`python pipeline/check_answer_key.py`.
