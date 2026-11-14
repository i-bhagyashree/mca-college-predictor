"""Data audit: counts splits vs duplicates, impossible ranks and score coverage correctly.
Run: .venv/bin/python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from data_audit import audit, benchmark_row, markdown  # noqa: E402


def row(**kw):
    r = {"college_id": "001", "id_match": "exact", "institute_name_raw": "A", "year": 2025, "round": "R1",
         "round_label": "First round", "result_awaited": False, "branch": "MCA", "institute_type_raw": None,
         "fee_waiver_seat": None, "opening_rank": 1, "closing_rank": 10, "opening_score": None,
         "closing_score": None, "category_raw": "UR/X/OP", "social": "UR", "seat_class": "X", "gender": "OP",
         "after_conversion": False, "eligible_category_raw": None, "domicile": None, "domicile_raw": None,
         "total_allotted": 1, "source_pdf": "x.pdf", "page": 1, "sno": 1, "raw_cells": []}
    r.update(kw)
    return r


class DataAudit(unittest.TestCase):
    def test_domicile_split_is_not_a_duplicate(self):
        a = audit([row(domicile="MP", domicile_raw="Y"), row(domicile="OTHER", domicile_raw="N", closing_rank=12)])
        self.assertEqual(a["exact_duplicate_rows"], 0)
        self.assertEqual((a["split_pools"], a["split_rows"]), (1, 2))

    def test_exact_duplicate_and_bad_ranks(self):
        a = audit([row(), row(), row(sno=3, opening_rank=50, closing_rank=10, social="SC", category_raw="SC/X/OP")])
        self.assertEqual(a["exact_duplicate_rows"], 1)     # differs only in sno, which is ignored
        self.assertEqual(a["opening_gt_closing"], 1)

    def test_benchmark_filter(self):
        self.assertTrue(benchmark_row(row()))
        self.assertFalse(benchmark_row(row(round="R2")))
        self.assertFalse(benchmark_row(row(result_awaited=True)))
        self.assertFalse(benchmark_row(row(seat_class="NCC")))
        self.assertFalse(benchmark_row(row(after_conversion=True)))

    def test_score_years_reported(self):
        a = audit([row(), row(year=2026, round="R2", closing_score=61.5, sno=2)])
        self.assertEqual(a["per_year"][2026]["score_rows"], 1)
        self.assertIn("2026 (1 rows)", markdown(a))


if __name__ == "__main__":
    unittest.main()
