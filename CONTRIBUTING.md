# Contributing to MCA College Predictor

Thanks for helping! This project helps students in Madhya Pradesh choose MCA colleges in MP-DTE counselling.
Bug reports, data corrections and code improvements are all welcome.

Live site: https://i-bhagyashree.github.io/mca-college-predictor/

## Ways to help

- **Report a bug or wrong number.** Open an [issue](https://github.com/i-bhagyashree/mca-college-predictor/issues)
  and say what you saw, what you expected, and (for data) which official PDF or portal page shows the right value.
- **Suggest a feature.** Open an issue first, so we can agree on it before you write code.
- **Fix something.** Send a pull request (steps below).

## How to send a pull request

1. **Fork** this repository (the *Fork* button at the top right of the GitHub page).
2. **Clone** your fork and create a branch:
   ```bash
   git clone https://github.com/<your-username>/mca-college-predictor.git
   cd mca-college-predictor
   git switch -c fix/short-description
   ```
3. **Make your change** (see "Project layout" below).
4. **Test it:**
   - Website: open `index.html` in a browser, or run `python3 -m http.server` and open http://localhost:8000.
     Check the pages you changed in light and dark mode and on a narrow window.
   - Python: run the unit tests:
     ```bash
     python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
     .venv/bin/python -m unittest discover tests -v
     ```
5. **Commit and push** to your fork, then open a **pull request** against `main`.
   Explain what you changed and why, and add a screenshot for visible changes.

## Project layout

| Folder | What it is |
|---|---|
| `index.html`, `assets/css/`, `assets/js/` | The website (plain HTML, CSS, JavaScript; no build step) |
| `assets/data/mca-data.js` | Website data, **generated** by `pipeline/build_site_data.py`. Do not edit by hand |
| `pipeline/` | Python scripts: download, extract, clean, ML, build website data. Run order is in the README |
| `data/raw/` | Official PDFs and portal downloads, kept unchanged |
| `data/processed/` | Generated clean data. Do not edit by hand: fix the script and re-run it |
| `reports/` | ML benchmark and test reports |
| `tests/` | Unit tests |

## Rules for changes

- **Data comes only from official sources** (MP-DTE, AFRC, the admission rules). Never type numbers in by hand.
  If a value is wrong, fix the script that produces it.
- **No personal data.** Merit lists contain student names and roll numbers. Never commit them
  (they are git-ignored) and never show names, roll numbers or categories on the website.
  Only percentage and rank may be kept.
- **Keep it simple.** No frameworks or build tools for the website; it must keep working as static files on GitHub Pages.
- **Keep the tests passing**, and add a test when you fix a bug in the pipeline.

## Licence

By contributing, you agree that your contribution is released under the project's [MIT License](LICENSE).
