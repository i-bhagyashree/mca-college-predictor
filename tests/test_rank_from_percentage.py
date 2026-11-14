"""Percentage → rank: interpolation, monotonic check and list-size benchmark.
Run: .venv/bin/python -m unittest discover tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from rank_from_percentage import (leave_one_out, monotonic_violations, rank_for,  # noqa: E402
                                  size_benchmark)

PCT = np.array([50.0, 60.0, 70.0, 80.0, 90.0])
RANK = np.array([1000.0, 800.0, 500.0, 200.0, 10.0])


class RankFromPercentage(unittest.TestCase):
    def test_interpolates_between_points(self):
        self.assertAlmostEqual(rank_for(75.0, PCT, RANK), 350.0)
        self.assertAlmostEqual(rank_for(80.0, PCT, RANK), 200.0)

    def test_outside_range_is_clamped(self):
        self.assertAlmostEqual(rank_for(99.0, PCT, RANK), 10.0)
        self.assertAlmostEqual(rank_for(30.0, PCT, RANK), 1000.0)

    def test_monotonic_check(self):
        self.assertEqual(monotonic_violations(PCT, RANK), 0)
        self.assertEqual(monotonic_violations(PCT, np.array([1000.0, 800.0, 900.0, 200.0, 10.0])), 1)

    def test_leave_one_out_on_a_straight_line_is_exact(self):
        pct = np.arange(50.0, 100.0, 5.0)
        err = leave_one_out(pct, 2000 - 20 * pct)
        self.assertTrue(np.allclose(err, 0))

    def test_size_benchmark_uses_only_earlier_years(self):
        sizes = {2020: 100, 2021: 100, 2022: 100, 2023: 5000}
        b = size_benchmark(sizes)
        last = b["methods"]["last_year"]["tests"]
        self.assertEqual([t["predicted"] for t in last], [100, 100])   # 2023 predicted from 2022, not itself
        self.assertEqual(b["next_year"], 2024)


if __name__ == "__main__":
    unittest.main()
