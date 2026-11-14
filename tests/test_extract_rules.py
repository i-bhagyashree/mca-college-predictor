"""Offline tests for the parts of the extractor that don't need a PDF.
Run: .venv/bin/python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from mca_pdf_to_json import HeaderError, classify, find_header, parse_int, parse_score  # noqa: E402


class Classify(unittest.TestCase):
    def test_rotated_headers_from_real_btech_pdf(self):
        # Exact strings from the 2025 B.Tech first-round PDF (see handover §1).
        cases = {
            ".oN\n.S": "sno",
            "EMAN\nETUTITSNI": "institute_name",
            "EPYT\nETUTITSNI": "institute_type",
            "WF": "fw",
            "HCNARB": "branch",
            "NOMMOC\nKNAR\nEEJ\nGNINEPO": "opening_rank",
            "NOMMOC\nKNAR\nEEJ\nGNISOLC": "closing_rank",
            "YROGETAC\nDETTOLLA": "allotted_category",
            "ELICIMOD": "domicile",
            "DETTOLLA\nLATOT": "total_allotted",
        }
        for cell, field in cases.items():
            self.assertEqual(classify(cell), field, cell)

    def test_specific_rules_beat_short_ones(self):
        # NAME's letters are all inside OPENING JEE COMMON RANK.
        self.assertEqual(classify("OPENING JEE COMMON RANK"), "opening_rank")
        self.assertEqual(classify("ELIGIBLE CATEGORY"), "eligible_category")

    def test_scores(self):
        self.assertEqual(classify("EGATNECREP GNINEPO"), "opening_score")
        self.assertEqual(classify("TIREM GNISOLC"), "closing_score")

    def test_unknown(self):
        self.assertIsNone(classify(""))
        self.assertIsNone(classify("XYZ"))


class Header(unittest.TestCase):
    def test_title_line_is_not_header(self):
        self.assertIsNone(find_header(["DIRECTORATE OF TECHNICAL EDUCATION, MADHYA PRADESH", None]))

    def test_duplicate_field_is_refused(self):
        with self.assertRaises(HeaderError):
            find_header(["S.No.", "INSTITUTE NAME", "OPENING RANK", "OPENING RANK"])


class Values(unittest.TestCase):
    def test_ranks(self):
        self.assertEqual(parse_int("921972"), (921972, None))
        self.assertEqual(parse_int("1,23,456")[1] is not None, True)   # Indian grouping: flag, don't guess
        self.assertEqual(parse_int("123,456"), (123456, None))
        self.assertEqual(parse_int(""), (None, None))
        val, prob = parse_int("X 796584")
        self.assertEqual(val, "X 796584")
        self.assertIsNotNone(prob)

    def test_scores(self):
        self.assertEqual(parse_score("78.45"), (78.45, None))
        self.assertEqual(parse_score(None), (None, None))
        self.assertIsNotNone(parse_score("78.45%")[1])


if __name__ == "__main__":
    unittest.main()
