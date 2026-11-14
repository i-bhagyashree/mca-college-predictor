"""Forward-chaining benchmark: simple predictors must never see the target year.
Run: .venv/bin/python -m unittest discover tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from benchmark_forward_chain import (FEATURE_NAMES, fit_predict, simple_predictions,  # noqa: E402
                                     student_metrics, weighted_mean)

POOLS = {("A", "UR", "OP"): {2022: {"cl": 200, "op": 1, "n": 1}, 2023: {"cl": 400, "op": 1, "n": 1},
                             2024: {"cl": 800, "op": 1, "n": 1}}}
SCALE = {2022: 1000, 2023: 1000, 2024: 1000}
K = [("A", "UR", "OP")]


class Benchmark(unittest.TestCase):
    def test_simple_predictors_ignore_target_year(self):
        p = simple_predictions(POOLS, SCALE, K, 2024, half_life=1.0)
        self.assertAlmostEqual(p["prev_year"][0], 0.4)          # 2023, not 2024's 0.8
        self.assertAlmostEqual(p["hist_avg"][0], 0.3)           # mean of 2022-2023
        changed = {K[0]: {**POOLS[K[0]], 2024: {"cl": 5, "op": 1, "n": 1}}}
        q = simple_predictions(changed, SCALE, K, 2024, half_life=1.0)
        for m in p:
            self.assertAlmostEqual(p[m][0], q[m][0])

    def test_weighted_mean_favours_recent_years(self):
        w = weighted_mean([2022, 2023], [0.2, 0.4], 2024, half_life=1.0)
        self.assertTrue(0.3 < w < 0.4)                          # weights 1/4 and 1/2 -> 0.333
        self.assertAlmostEqual(w, (0.25 * 0.2 + 0.5 * 0.4) / 0.75)

    def test_window_limits_history(self):
        p = simple_predictions(POOLS, SCALE, K, 2024, half_life=1.0, window=1)
        self.assertAlmostEqual(p["hist_avg"][0], 0.4)           # only 2023

    def test_ml_handles_all_missing_column(self):
        rng = np.random.default_rng(0)
        X = rng.random((80, len(FEATURE_NAMES)))
        X[:, 1] = np.nan                                         # e.g. no 'two years back' yet
        pred = fit_predict("gbm_resid", X, X[:, 0], X[:5])
        self.assertEqual(pred.shape, (5,))

    def test_perfect_prediction_scores_perfectly(self):
        a = np.array([100.0, 300.0, 600.0])
        m = student_metrics([("UR", "OP")] * 3, a, a.copy(), 600)
        self.assertEqual(m["decision_accuracy"], 1.0)
        self.assertEqual(m["list_precision"], 1.0)
        self.assertEqual(m["top5_hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
