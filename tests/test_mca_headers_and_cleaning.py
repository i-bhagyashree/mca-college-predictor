"""Tests for rules added after inspecting the real MCA PDFs (2017-2026).
Run: .venv/bin/python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from clean_cutoffs import city_of, norm_name, split_category  # noqa: E402
from mca_pdf_to_json import HEADER_OVERRIDES, classify, find_header  # noqa: E402


class RealMcaHeaders(unittest.TestCase):
    def test_2017_headers(self):
        # Exact cells from 2017__cutoff__first-round.pdf (letters one per line, rotated).
        self.assertEqual(classify("Y\nE\nO\nG\nE\nT\nA\nC\n_\nD\nE\nT\nO\nL\nL\nA"), "allotted_category")  # CATEGOEY typo
        self.assertEqual(classify("D\nE\nT\nO\nL\nL\nA\nL\nA\nT\nO\nT"), "total_allotted")
        self.assertEqual(classify("S\nT\nN\nE\nD\nIS\nE\nR\nK\nJ"), "jk_resident")
        self.assertEqual(classify("S\nT\nN\nA\nR\nG\nIM\nK\nJ"), "jk_migrant")   # contains R-A-N-K
        self.assertEqual(classify("O\nN\n.S"), "sno")

    def test_2025_eligible_before_allotted(self):
        self.assertEqual(classify("YROGETAC\nELBIGILE"), "eligible_category")
        self.assertEqual(classify("DETTOLLA YROGETAC"), "allotted_category")

    def test_2026_merit_and_short_names(self):
        self.assertEqual(classify("TIREM\nGNINEPO"), "opening_score")
        self.assertEqual(classify("TIREM\nGNISOLC"), "closing_score")
        self.assertEqual(classify("EMAN_TSNI"), "institute_name")
        self.assertEqual(classify("EPYT_TSNI"), "institute_type")

    def test_2019_override_only_when_header_matches(self):
        ov = HEADER_OVERRIDES["2019__cutoff__first-round.pdf"]
        row = [".oN\n.S", "EMAN\nETUTITSNI", "EMAN\nETUTTISNI", "WF", "HCNARB", "KNAR\nGNINEPO", "KNAR\nGNISOLC"]
        self.assertEqual(find_header(row, ov)[2], "institute_type")
        changed = list(row)
        changed[2] = "EPYT\nETUTITSNI"     # a corrected PDF: override must not fire, rules handle it
        self.assertEqual(find_header(changed, ov)[2], "institute_type")


class Cleaning(unittest.TestCase):
    def test_categories(self):
        c, p = split_category("OBC/NCC/F")
        self.assertEqual((c["social"], c["seat_class"], c["gender"], p), ("OBC", "NCC", "F", []))
        c, p = split_category("UR/X/OP (After\nConversion)")
        self.assertTrue(c["after_conversion"]); self.assertEqual(c["social"], "UR"); self.assertEqual(p, [])
        self.assertEqual(split_category("FW/F")[0]["gender"], "F")
        self.assertEqual(split_category("FW")[0]["gender"], "OP")
        self.assertEqual(split_category("EWS")[0]["social"], "EWS")
        self.assertTrue(split_category("ZZ/X/OP")[1])          # unknown code: reported, not guessed

    def test_names_and_cities(self):
        self.assertEqual(norm_name("Shri Ram Institute of Technology, Jabalpur (2001)"),
                         norm_name("Shri Ram Institute of Technology, Jabalpur"))
        self.assertEqual(norm_name("Madhav Institute of Technology and Science, Gwalior (Deemed University)"),
                         norm_name("Madhav Institute of Technology and Science, Gwalior"))
        self.assertEqual(city_of("Sagar Institute of Research & Technology, Bhopal"), "Bhopal")
        self.assertEqual(city_of("IPS ACADEMY ... SANWER CAMPUS, INDORE"), "Indore")


if __name__ == "__main__":
    unittest.main()
