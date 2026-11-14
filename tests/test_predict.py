"""The forecast's features must only use years BEFORE the target year (no look-ahead).
Run: .venv/bin/python -m unittest discover tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from predict_next_year import FEATURE_NAMES, ResidualGBM, features  # noqa: E402

POOLS = {
    ("A", "UR", "OP"): {2023: {"cl": 500, "op": 10, "n": 5}, 2024: {"cl": 600, "op": 20, "n": 6},
                        2025: {"cl": 700, "op": 30, "n": 7}},
    ("B", "OBC", "F"): {2024: {"cl": 900, "op": 50, "n": 2}},
}
SCALE = {2023: 1000, 2024: 1000, 2025: 1000}


class NoLookAhead(unittest.TestCase):
    def test_target_year_value_is_never_a_feature(self):
        keys, X = features(POOLS, SCALE, {}, 2025)
        row = X[keys.index(("A", "UR", "OP"))]
        self.assertAlmostEqual(row[FEATURE_NAMES.index("last_p")], 0.6)        # 2024, not 2025's 0.7
        self.assertAlmostEqual(row[FEATURE_NAMES.index("mean3_p")], 0.55)      # 2023-2024 only
        changed = {k: dict(v) for k, v in POOLS.items()}
        changed[("A", "UR", "OP")][2025] = {"cl": 50, "op": 1, "n": 1}         # change the answer
        _, X2 = features(changed, SCALE, {}, 2025)
        np.testing.assert_array_equal(np.nan_to_num(X), np.nan_to_num(X2))    # features unchanged

    def test_pool_without_history_is_skipped(self):
        keys, _ = features(POOLS, SCALE, {}, 2024)
        self.assertNotIn(("B", "OBC", "F"), keys)

    def test_residual_model_starts_from_last_year(self):
        X = np.zeros((60, len(FEATURE_NAMES)))
        X[:, 0] = np.linspace(0.2, 0.9, 60)
        m = ResidualGBM().fit(X, X[:, 0])                  # no change to learn
        np.testing.assert_allclose(m.predict(X), X[:, 0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
