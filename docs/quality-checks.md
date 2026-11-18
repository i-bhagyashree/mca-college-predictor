# Quality checks

Results from 2026-11-18. Re-run them before every release and update the date and results.

| Area | Check | How | Result |
|---|---|---|---|
| Functional | Unit tests (extraction rules, headers, cleaning, rank, ML, audit) | `python -m unittest discover -s tests` | 44 tests, all pass |
| Functional | Full rebuild from PDFs reproduces the published data | run the pipeline, then `git status data assets/data` | No change (3,028 rows, 71 colleges) |
| Functional | Sanity rules | `python pipeline/data_audit.py` | Pass |
| Functional | Hand-written answer key | `python pipeline/check_answer_key.py` | **Not run yet: key is empty** |
| Functional | Site data export refuses bad data | `build_site_data.py` stops on NaN or empty rows; writes a temp file, then replaces | Done |
| Privacy | No names or roll numbers in Git or on the site | merit PDFs and `*__merit__*` files are git-ignored; extractor reads `__cutoff__` PDFs only | Pass |
| Privacy | Nothing the user types is stored or put in the URL | share link removed; no storage of percentage/category | Pass |
| Privacy | No third-party requests | fonts self-hosted in `assets/fonts` | Pass |
| Security | Text from data is escaped before display | `esc()` in `app.js` | Pass |
| Security | Downloads checked | TLS on by default; file must start with `%PDF` | Pass |
| Accessibility | Keyboard: skip link, visible focus | Tab from page top | Pass |
| Accessibility | Labels on every input, `lang`, live region for results | read `index.html` | Pass |
| Release | Cache version bumped on every change | `?v=` in `index.html` | Pass |
| Release | Live site loads all pages | open each page on phone and desktop | Pass |
| Not done | JavaScript unit tests | none | Gap |
