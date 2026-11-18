"""Tests for pipeline/ml_rank_from_percentage.py (ML: rank from percentage)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
import ml_rank_from_percentage as M  # noqa: E402


def make_list(year, rnd, pcts):
    """A merit list in the merit_lists.json format from a list of percentages (any order)."""
    pcts = sorted(pcts, reverse=True)
    points = []
    for rank, p in enumerate(pcts, start=1):
        if points and points[-1][0] == p:
            points[-1][2] = rank
        else:
            points.append([p, rank, rank])
    return {"year": year, "round": rnd, "n": len(pcts), "pct_min": min(pcts), "pct_max": max(pcts), "points": points}


class TestMlRank(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.lists = [make_list(2025, "R1", list(np.round(rng.uniform(50, 95, 400), 1))),
                      make_list(2026, "R1", list(np.round(rng.uniform(50, 95, 420), 1))),
                      make_list(2026, "R2", list(np.round(rng.uniform(50, 95, 380), 1)))]

    def test_rank_of_counts_candidates_above(self):
        l = make_list(2026, "R1", [90, 80, 80, 70])
        self.assertEqual(M.rank_of(95, l), 1)
        self.assertEqual(M.rank_of(90, l), 1)
        self.assertEqual(M.rank_of(80, l), 2)        # ties share the first rank of their percentage
        self.assertEqual(M.rank_of(75, l), 4)
        self.assertEqual(M.rank_of(60, l), 5)

    def test_candidates_rebuilds_every_row(self):
        l = make_list(2026, "R1", [90, 80, 80, 70])
        self.assertEqual(M.candidates(l), [(90, 1), (80, 2), (80, 3), (70, 4)])

    def test_isotonic_never_gives_a_worse_rank_for_more_marks(self):
        f = M.fit("Isotonic regression (monotonic)", "isotonic", self.lists)
        pos = f(np.arange(50, 95.01, 0.5))
        self.assertTrue(np.all(np.diff(pos) <= 1e-12))   # position never rises as the percentage rises

    def test_gradient_boosting_is_monotonic(self):
        name = "Gradient boosting (monotonic)"
        f = M.fit(name, M.models()[name], self.lists)
        pos = f(np.arange(50, 95.01, 0.5))
        self.assertTrue(np.all(np.diff(pos) <= 1e-9))

    def test_leave_one_out_never_trains_on_the_target(self):
        seen = []
        def spy(train):
            seen.append([f"{l['year']} {l['round']}" for l in train])
            return M.rank_predictor(train)
        r = M.evaluate(spy, self.lists)
        for test, train in zip(r["tests"], seen):
            self.assertNotIn(test["target"], train)
            self.assertEqual(len(train), 2)

    def test_similar_lists_give_small_errors(self):
        r = M.evaluate(lambda tr: M.fit("Isotonic regression (monotonic)", "isotonic", tr), self.lists)
        self.assertLess(r["mae_ranks"], 25)          # same uniform spread: rank error should be small


if __name__ == "__main__":
    unittest.main()
