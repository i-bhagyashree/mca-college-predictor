# Data and methodology

## Source register

| Data | Source | Where it is recorded |
|---|---|---|
| Cut-off PDFs 2017–2026 (opening and closing ranks per round) | MP-DTE portal, CutOffList.aspx | `data/raw/mca/<year>/_index.txt` (URL, SHA-256, download date) |
| Merit lists 2025 R1, 2026 R1, 2026 R2 | MP-DTE portal | Not committed (contain names). Only percentage → rank points are kept in `data/processed/merit_lists.json` |
| Intake (seats per college) | MP-DTE intake page (HTML) | `data/raw/intake` |
| Fees | AFRC fee orders | `data/raw/afrc`, `data/raw/fees` |
| Admission rules 2026-27 | MP-DTE rules PDF | `data/raw/rules/_index.txt` |

## Field meanings (cut-off rows)

| Field | Meaning |
|---|---|
| `year`, `round` | Counselling year and round (R1, R2, special) |
| `institute_name`, `college_id` | Name as printed; official college ID from the intake list |
| `allotted_category` | Seat category as printed, then split into social category, class (e.g. TFW) and gender |
| `domicile` | MP or other state |
| `opening_rank`, `closing_rank` | Digits only; an empty cell stays `null` |
| `source_pdf`, `page`, `table_index`, `raw_cells`, `sha256` | Where the row came from, for checking by hand |

## Extraction settings

- pdfplumber with the `lines` table strategy; doubled characters removed (`dedupe_chars`).
- Rotated headers are recognised by their letters, not their order; an unknown column is kept as `colN`
  and reported; two columns claiming one field stop the run (`HeaderError`).
- Every page gets a status: extracted, no table, or needs OCR.

## Correction policy

- Corrections never edit the PDF output by hand. Each fix is a script (`fix_overflow_columns.py` for the
  2017 special round 2) or a reviewed alias (`NAME_ALIASES` in `clean_cutoffs.py`, with the evidence for each).
- No fuzzy name matching: an unmatched name goes to `review_unmatched_names.csv`.

## Checks and coverage

- `data_audit.py`: opening ≤ closing, every college ID exists, coverage per year and round.
- Round 1 exists every year; Round 2 for 2021–2023, 2025, 2026; special rounds only in 2017.

## How results are worked out

- **Rank from percentage:** read from the official merit list of the same year and round (exact rank),
  otherwise the latest list of that round, scaled to the year's list size.
- **College labels:** your rank divided by the college's closing rank:
  ≤ 0.8 Well within, ≤ 1 Within, ≤ 1.15 Borderline, otherwise Outside. Colleges that admitted up to the last
  rank are "Open seats".
- **ML model:** seven scikit-learn models trained on the three merit lists; isotonic regression was best.
  On last year's lists the Rank Predictor is exact (error 0) and the ML model is off by about 20 ranks,
  so the Rank Predictor is the main tool. Full results: `reports/ml_rank/REPORT.md`.

## Gaps and limits

- Past cut-offs do not guarantee next year's cut-offs; the labels describe the past only.
- Merit lists exist for only three rounds, so the ML model has little to learn from.
- 13 college names are still unmatched (see the review file).
- The hand-written answer key (`tests/fixtures/answer_key.csv`) is not filled yet.
- Individual allotments are not used, because they contain personal data.
