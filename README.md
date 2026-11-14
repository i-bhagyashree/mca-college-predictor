# MCA College Predictor (Madhya Pradesh): data pipeline and website

Turns MP-DTE MCA cut-off PDFs into clean data. Approach adapted from
[mp-dte-btech-predictor](https://github.com/i-shivamsoni/mp-dte-btech-predictor) (MIT).
A read-only copy of that repo is cloned in `reference/` (not ours to edit).

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Work in this order

| # | Step | Command | Output |
|---|------|---------|--------|
| 1 | Collect cut-offs | `.venv/bin/python pipeline/fetch_mca_cutoffs.py --years 2026 2025 2024 2023 2022 2021 2020 2019 2018 2017` | `data/raw/mca/<year>/*.pdf` + `_index.txt` |
| 1b | Collect intake list | `.venv/bin/python pipeline/fetch_intake.py` | `data/raw/intake/intake_MCA_<session>.json/.html`: College IDs, total intake, TFW seats, EWS flag, AICTE code, minority status, AFRC fee status, affiliation remark (all shown on the site) |
| 1d | Collect fees | `.venv/bin/python pipeline/fetch_fees.py` | `data/raw/fees/fees_MCA_<date>.json` + raw HTML per institute (16 govt/aided/university colleges; private fees are set by AFRC and aren't on this page) |
| 1e | Collect AFRC fees | `.venv/bin/python pipeline/fetch_afrc_fees.py 2026 2025 2024` | `data/raw/afrc/afrc_MCA_<session>.json` + raw responses. AFRC = fee regulator for private colleges; amounts are **per semester** (the grid header says so) |
| 1c | Rule book | saved by hand from the portal; see `data/raw/rules/_index.txt` (URL + sha256) | `data/raw/rules/2026-27__admission-rules.pdf` |
| 2 | Inspect | `.venv/bin/python pipeline/inspect_pdf.py data/raw/mca/2025/2025__cutoff__first-round.pdf` | printed header + sample rows |
| 3 | Answer key | fill `tests/fixtures/answer_key.csv` **by hand from the PDF** | |
| 4 | Extract (Layer 1) | `.venv/bin/python pipeline/mca_pdf_to_json.py` | `data/extracted/mca/<year>/*.json` |
| 4b | Repair | `.venv/bin/python pipeline/fix_overflow_columns.py` | re-attaches the 2017 special-round-2 DOMICILE column printed on page 2 |
| 5 | Check | `.venv/bin/python pipeline/check_answer_key.py` | every mismatch listed |
| 6 | Merge (Layer 2) | `.venv/bin/python pipeline/merge_cutoffs.py` | `data/processed/mca_cutoffs_master.json` |
| 7 | Clean | `.venv/bin/python pipeline/clean_cutoffs.py` | `data/processed/mca_cutoffs_clean.json`, `colleges.json` (with fees joined by College ID), `review_unmatched_names.csv` |
| 8 | Website data (Layer 3) | `.venv/bin/python pipeline/build_site_data.py` | `assets/data/mca-data.js` |
| 8b | ML forecast | `.venv/bin/python pipeline/predict_next_year.py` (then step 8 again) | `data/processed/predictions_<next year>.json` with backtest report |
| 8d | Percentage → rank | `.venv/bin/python pipeline/rank_from_percentage.py` (then step 8 again) | `data/processed/rank_from_percentage.json`: students enter their graduation %, the site estimates their rank from the latest list that prints both % and rank (2026 Round 2, leave-one-out error ≈1.4 ranks) and, for the ML page, projects it to next year with a forecast of the list size (benchmarked year by year; "same as last year" was the most accurate of 6 methods) |
| 8e | ML hold-out test | `.venv/bin/python pipeline/evaluate_ml.py` | `reports/ml_test/REPORT.md` + `predictions_vs_actual_<year>.csv`: trains on years before each of the last 3 years, predicts that year, and compares with the real cut-offs and with the Historical method (cut-off error, likely-range coverage, and how often a student's seat decision / label would have been right) |
| 8c | Year-by-year benchmark | `.venv/bin/python pipeline/benchmark_forward_chain.py` (then step 8 again) | `reports/benchmark/REPORT.md`, `benchmark_results.json`, `benchmark_table.csv`, `figures/*.png`: forward-chaining test (train 2017…n, predict n+1) of previous-year, historical-average, weighted, random-forest and gradient-boosting predictions, plus a data inspection (`pipeline/data_audit.py`, also runnable alone) and a leak-free tuning check of the GBM; shown on the Forecast page |
| 9 | View | open `index.html` in a browser | rank checker + college explorer |

Website files: `index.html` (markup), `assets/css/styles.css` (styles, light/dark tokens), `assets/js/app.js` (logic). Only `assets/data/` is generated.

Steps 4–8 are safe to re-run in order at any time (each rebuilds its output from scratch).

## What the MCA PDFs look like (found in step 2)
- 20 PDFs, 2017–2026, same template as B.Tech: rotated headers, ruling lines, one table per page, no doubled glyphs, no scanned pages.
- Column changes over time: 2017 has `JK RESIDENTS`/`JK MIGRANTS` and the typo `ALLOTED CATEGOEY`;
  2019 prints `INSTITUTE NAME` over the institute-type column (fixed by `HEADER_OVERRIDES`);
  2025 adds `ELIGIBLE CATEGORY`; 2026 round 2 adds `OPENING/CLOSING MERIT` (decimal scores).
- 2017 special round 2: the DOMICILE column overflows onto page 2 (step 4b).
- 2025 splits one seat pool into several rows by the candidate's own (eligible) category; the website aggregates them.
- Category = `SOCIAL/CLASS/GENDER` (e.g. `OBC/NCC/F`), plus `EWS`, `FW`, `FW/OP`, `FW/F`, and a `(After Conversion)` note.

## Open items for you
- `tests/fixtures/answer_key.csv` is still empty. Fill it from the PDFs yourself, then run step 5.
- `data/processed/review_unmatched_names.csv`: 12 cut-off names with no official College ID. Most are colleges not in
  the 2026-27 intake list. Possible renames to confirm with evidence before adding to `NAME_ALIASES` in
  `pipeline/clean_cutoffs.py`: Shri Vaishnav Institute of Management → SHRI VAISHNAV INSTITUTE OF MANAGEMENT AND SCIENCE (082);
  INDORE INTERNATIONAL COLLEGE SOLSINDA → INDORE INTERNATIONAL COLLEGE SAWER (131).

Unit tests (no PDF needed): `.venv/bin/python -m unittest discover tests -v`

`assets/data/` and `data/processed/` are generated: never edit them by hand.

## Rules
- Raw PDFs are never modified. Generated JSON is never hand-edited: fix the script and re-run.
- Unknown columns are kept (`colN`) and reported, never dropped.
- Ranks are parsed only from plain digits; anything else stays as text and appears in `problems`.
- Every record carries `source_pdf`, `page`, `table_index`, `raw_cells`; every file carries the PDF's `sha256`.
- Merit lists contain student names and roll numbers: keep them out of Git and off the website.

## Admission rules used on the website
The "Admission rules" section and the rank checker's eligibility notes follow the official
MP-DTE admission rules 2026-27 (`data/raw/rules/2026-27__admission-rules.pdf`, Hindi, 58 pages).
Each card on the site cites the rule number and PDF page. The checker applies: §8.1 (SC 16 / ST 20 /
OBC 14%, MP domicile only), §8.2 (female 30%, divyang 3%, NCC 1%), §10.2 (fee waiver: MP domicile,
income ≤ ₹8 lakh), §10.3 (EWS 10%, MP domicile), §12(2) (MCA: 3-year degree, Maths, 50% / 45%),
§16.3 (qualifying-marks rounds may be fully unreserved) and §16.5 (non-MP candidates).
When the rules are revised for a new session, re-read the PDF and update the site text and `eligible()` in `assets/js/app.js`.

## Fees: two sources, kept apart
- **DTE fee page** (`fetch_fees.py`): 16 government / aided / university colleges, per category, period not stated.
- **AFRC** (`fetch_afrc_fees.py`): private colleges, "Semester wise Fees (in Rs)", sessions 2024-25 to 2026-27.
  Rows are matched to College IDs in `pipeline/afrc_match.py`: identical name words + same city, else an
  entry in `pipeline/afrc_aliases.csv` (each with written evidence), else listed as unmatched in
  `data/processed/review_afrc_matches.csv`. One AFRC row is deliberately unmatched:
  "Indore International College, Indore" (Nanod, Kila Road) has no DTE counterpart.
- The site never mixes the two units. Cards show "₹x/sem" for AFRC; the fee sort lists AFRC colleges first.
- Cross-check: Rustamji is in both. AFRC ₹22,800/sem × 2 = DTE ₹45,600, which suggests DTE amounts are yearly (one data point only, not claimed on the site).
- Note: AFRC lists "Indore International College Solsinda" and "…Sawer" separately, but with the same village/PIN (Solsinda, 453551). They're kept as separate colleges; see the open rename question above.

## Decision: ML is tested, not used for predictions (14 Nov 2026)
The website no longer shows ML predictions. The ML page (`#/ml`, "ML Analysis") shows the year-by-year
(forward-chaining) benchmark instead: 9 experiments (train 2017…n, predict n+1), five approaches (last year's
cut-off, average of all years, weighted average, random forest, gradient boosting), all metrics, the answers to the
seven research questions, and why ML does not do better. Result, pooled 2019–2026: gradient boosting 3.4% lower
error than last year's cut-off, clearly better in 3 of 8 years and clearly worse in 1; random forest 2.5% worse; for
the colleges shown to a student, last year's cut-off was as good or better (90% vs 89% correct, top-5 83% vs 80%).
So predictions use last year's real cut-offs (Rank Predictor). The Choice Planner shows only last year's figures.
- Switches near the top of `assets/js/app.js`: `ML_ON` (false: no ML figures in the Rank Predictor or Planner) and
  `ML_DEMO` (true: the ML Analysis page has a second tab, "Try the ML model", that runs the gradient-boosting model
  for a student, clearly marked as a demo that is not used for the site's predictions).
- The page reads everything from `assets/data/mca-data.js` → `benchmark` (built by `build_site_data.py` from
  `reports/benchmark/benchmark_results.json`); `#/forecast` now redirects to `#/ml`.
- The sections below about the ML forecast and the ML Predictor describe the pipeline, which is still run for the
  analysis.

## Machine-learning forecast (next year's Round-1 closing ranks)
`pipeline/predict_next_year.py` (scikit-learn). Unit: seat pool = college × category × gender, ordinary seats, Round 1.
- **Target:** relative position `p = closing rank / highest R1 closing rank that year`. The rank scale moves a lot
  (326 in 2019, 3,354 in 2022, 1,091 in 2026), so predicting raw ranks would mostly mean guessing the applicant count.
  The forecast converts back with the latest year's scale; that assumption is the main risk.
- **Features:** only from years before the target (tested in `tests/test_predict.py`).
- **Training years: all 10 years, 2017–2026** (`ML_FIRST_YEAR` in `pipeline/list_size.py`; project decision, 14 Nov 2026).
  Until then 2017 was left out (2018–2026). Adding it was compared on the same backtest years: gradient boosting on the
  change went from 0.176 to 0.172 average error over 2022–2026 (last year's cut-off: 0.176) and beat last year in 4 of 5
  years instead of 3; the 2024–2026 error stayed 0.151; the hold-out error stayed 208 ranks. The benchmark and the
  next-year rank forecast use the same 10 years.
- **Models compared** in a time-based backtest (train on years < T, predict T, T = 2022–2026):
  last year, 3-year average, Ridge, gradient boosting on the level, gradient boosting on the *change* from last year.
- **Next-year rank and forecast scale** (`pipeline/list_size.py`): the ML Predictor turns a student's position into a
  2027 rank with a **recency-weighted average of all list sizes 2017–2026** (half-life chosen leak-free on earlier years):
  1,157 candidates. The student's position is their rank ÷ the size of the merit list the curve comes from (2026 Round 2, 1,146 candidates), so 72.3% (rank ~546 of 1,146) → rank ~551 in 2027. Dividing by Round 1's last allotted rank (1,091) instead, as an earlier version did, put the bottom of the list above 100% and past next year's list size, so low percentages saw no open seats. Project decision so that the ML
  page uses all years; in the backtest it is less accurate than "same as last year" (48% vs 39% average error), which
  the site states. The forecast college cut-offs use the same list size, so ranks and cut-offs are comparable; the
  college *chances* do not depend on it (they compare positions).
- **Model choice:** the model with the lowest error on the **most recent backtest years, 2024–2026** (`SELECT_YEARS`).
  This rule was decided *after* seeing that, with 2018–2026 training (the setting at the time), the residual gradient-boosting model ties
  "same as last year" over 2022–2026 (0.176 vs 0.176: for 2022–2023 it had only 2018–2021/2022 to learn from) but is
  better in each of 2024, 2025 and 2026 (0.151 vs 0.165). It is used for the forecast; the site states the rule.
- **Likely range:** 10th–90th percentile of the model's held-out errors (split-conformal); 77% of real cut-offs fell
  inside it on unseen years (target 80%).
- **Hold-out check** (`evaluate_ml.py`, 2024–2026): ML cut-off error 307 / 138 / 180 ranks vs 334 / 166 / 182 for
  last year's cut-offs; "would I get this seat" right 84–87% of the time.
Re-run after adding a new year's PDFs: extract → merge → clean → `predict_next_year.py` → `build_site_data.py`.

## Rank Predictor: two methods
- **Historical predictor** (no ML): uses real past cut-offs. "Single year" compares your rank with that year's
  exact closing ranks. "Last 3 / 5 years" scales each older year to the latest year's rank level
  (factor = latest highest closing rank ÷ that year's highest closing rank) and counts in how many years your rank
  was within the cut-off: every year = Safe, at least half = Likely, at least one = Reach, none = Unlikely.
- **ML predictor**: the gradient-boosting forecast from `predict_next_year.py`. Each result shows its working:
  last closing rank ÷ that year's scale → predicted change → × latest scale = predicted rank.

### What only the ML Predictor shows
The two pages take the same inputs but answer different questions, so their results differ:
- **Rank Predictor**: what really happened. Round and year choice, real opening and closing ranks, seats allotted.
- **ML Predictor**: what is forecast for next year, with the uncertainty.
  - *Seat chance %* per college: the chance that next year's closing rank is at or after your rank. It is read from the
    forecast's likely range (10th percentile = 90%, predicted value = 50%, 90th percentile = 10%, interpolated, capped at 1–99%;
    `seatChance()` in `assets/js/app.js`). It is an approximate reading of the range, not a separately tested number, and the
    Safe / Likely / Reach / Unlikely labels are unchanged.
  - *Likely-range bar*: the likely cut-off range, the predicted cut-off and your rank on one line.
  - *Forecast trend* filter (getting harder / easier than last year), an outlook box with those counts, and two extra sort
    orders (highest seat chance, biggest change from last year).

### "Why this forecast can be wrong" (ML Predictor)
Shown with every ML result, built only from the data file (`limitsBox()` in `assets/js/app.js`): the biggest year-to-year
jump in the list size and the error of the next-year size guess; the typical cut-off error and likely-range coverage on
the recent backtest years; the share of the permutation importance taken by last year's cut-off (about 91%); and the
number of training examples.

### Why "Open seats" exists
In MP MCA counselling almost every ranked candidate gets a seat: 2026 Round 1 allotted 1,091 seats and the last
allotted rank was 1,091 (2025 R1: 1,092 / 1,092; 2024 R1: 1,941 / 1,943), while there are 5,350 seats.
So most private colleges admit up to the last rank. A college whose closing rank is within 10% of the round's last
allotted rank is labelled **Open seats** (any participating candidate who chose it got in). Safe / Likely / Reach /
Unlikely are used only for the competitive colleges, and "Best colleges you can get" lists those first.

## Design & extra features
The site uses the same visual design as the React "MCA College Predictor" app (`~/Minor Project`): indigo theme,
DM Sans + Manrope fonts, two-row header, navy counselling-overview card. Extra features:
- **Percentage mode** in the Rank Predictor: compares your graduation % with the lowest admitted merit % per college
  in the latest list that prints merit scores (2026 Round 2). +5 points or more = Safe, 0–5 = Likely, up to −3 = Reach.
- **Choice Planner** (`#/planner`), with three tabs that share one profile (%, category, gender, domicile):
  - *% needed*: the graduation % that reached each college's closing rank. It is shown for the last year (real
    Round 1) and for next year (ML forecast, with a range). It is the percentage → rank table read backwards, and
    never goes below the eligibility minimum (50%, or 45% for MP SC/ST/OBC).
  - *Compare*: up to 3 colleges side by side, covering seats, fee, cut-offs, % needed, your chance and a 5-year trend.
  - *My choice list*: an ordered list. Add colleges with "+ Add to list" on the predictor results, the % needed table
    or a college's details. You can reorder the list or sort it by competitiveness, and print it or save it as a PDF.
    The list is saved in this browser (localStorage `mca-choices-v1`) and survives a refresh; the % and category
    fields are not saved, so every refresh starts with an empty form.
- **Results survive a reload**: the Rank Predictor and ML Predictor keep the last submitted details in this browser (localStorage `mca-pred-hist-v1`, `mca-pred-ml-v1`) and show the same result again after a refresh. They are removed only by the "Clear my details" button, which asks for confirmation. The Planner's profile fields are still not saved. The light/dark theme and the Planner's choice list are remembered as before.
