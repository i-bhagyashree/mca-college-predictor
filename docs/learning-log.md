# Learning log

Write each entry yourself, in your own words (Hindi is fine). Keep it short.

## Entry template

```text
Checkpoint / date / commit:
I built:
I can now explain:
Bug: expected → actual → hypothesis → evidence → cause → fix:
Checks I ran and results:
AI help used; what I verified or rejected:
My independent change:
Still unclear / next task:
```

## Three bug stories (to write)

Real bugs from this project. For each, fill in: expected → actual → hypothesis → evidence → cause → fix.

1. **Rotated table headers:** the PDF headers came out in scrambled letter order, so columns were named wrongly.
2. **2017 special round 2:** values overflowed into the wrong column (see `pipeline/fix_overflow_columns.py`).
3. **Merit lists read by the cut-off extractor:** files with student names were created locally
   (never committed). Fixed by reading only `__cutoff__` PDFs and git-ignoring merit files.

## AI suggestions accepted or rejected

- (write here)
