"""AFRC name matching: exact word sets + same city, never fuzzy.
Run: .venv/bin/python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from afrc_match import match_rows  # noqa: E402

CITIES = ["Indore", "Bhopal", "Jabalpur", "Sagar"]
COLLEGES = [
    {"college_id": "078", "name": "Shri Ram Institute of Technology (MCA) Jabalpur", "city": "Jabalpur"},
    {"college_id": "079", "name": "Shri Ram Institute of Technology, Jabalpur (2001)", "city": "Jabalpur"},
    {"college_id": "031", "name": "Lakshmi Narain College of Technology(MCA), Bhopal", "city": "Bhopal"},
    {"college_id": "066", "name": "Sagar Institute of Research & Technology, Bhopal", "city": "Bhopal"},
    {"college_id": "093", "name": "Technocrats Institute of Technology, Bhopal", "city": "Bhopal"},
]


def m(name, place):
    return match_rows([{"afrc_name": name, "place": place}], COLLEGES, CITIES)[0][1:3]


class AfrcMatch(unittest.TestCase):
    def test_mca_word_separates_two_listings(self):
        self.assertEqual(m("Shri Ram Institute of Technology MCA", "Jabalpur"), ("078", "auto"))
        self.assertEqual(m("Shri Ram Institute of Technology", "Jabalpur"), ("079", "auto"))

    def test_mca_dropped_only_when_unique(self):
        self.assertEqual(m("Lakshmi Narain College of Technology", "Bhopal"), ("031", "auto"))

    def test_ampersand_and_city_word(self):
        self.assertEqual(m("Sagar Institute of Research Technology", "Bhopal"), ("066", "auto"))

    def test_city_must_agree(self):
        self.assertEqual(m("Lakshmi Narain College of Technology", "Indore"), (None, "unmatched"))

    def test_no_fuzzy_plural(self):
        # "Technocrat" vs "Technocrats" is NOT auto-matched; it needs an alias with evidence.
        self.assertEqual(m("Technocrat Institute of Technology Foo", "Bhopal"), (None, "unmatched"))


if __name__ == "__main__":
    unittest.main()
