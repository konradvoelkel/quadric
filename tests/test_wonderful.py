import unittest

from bbcells import oracles
from bbcells.algebra import IntPoly
from bbcells.core import bb_cells
from bbcells.frontends import toric, wonderful
from tests._util import doctests_for

load_tests = doctests_for(wonderful)

# PLAN.md 4.3, computed independently when the plan was written
EXPECTED = {
    "A1": (1, 1, 1, 1),
    "A2": (1, 2, 4, 7, 8, 7, 4, 2, 1),
    "B2": (1, 2, 4, 8, 11, 12, 11, 8, 4, 2, 1),
    "G2": (1, 2, 4, 8, 12, 16, 19, 20, 19, 16, 12, 8, 4, 2, 1),
}


class TestWonderful(unittest.TestCase):

    def test_plan_values(self):
        for name, counts in EXPECTED.items():
            cells = bb_cells(wonderful.wonderful_compactification(name))
            self.assertEqual(cells.counts, counts, name)
            self.assertEqual(IntPoly.from_counts(counts), oracles.wonderful(name), name)

    def test_point_count_oracle_rank_3_and_4(self):
        for name in ("A3", "B3", "C3", "A4", "D4"):
            X = wonderful.wonderful_compactification(name)
            cells = bb_cells(X)
            self.assertEqual(IntPoly.from_counts(cells.counts), oracles.wonderful(name), name)

    def test_consistency_checks(self):
        for name in ("A2", "B2", "G2", "A3"):
            report = bb_cells(wonderful.wonderful_compactification(name)).check()
            self.assertTrue(report.ok, str(report))

    def test_pgl2_is_p3(self):
        X = wonderful.wonderful_compactification("A1")
        P3 = toric.fixed_point_data(toric.projective_space(3))
        self.assertEqual(bb_cells(X).counts, bb_cells(P3).counts)

    def test_size_limit(self):
        with self.assertRaisesRegex(ValueError, "more than the limit"):
            wonderful.wonderful_compactification("F4")


if __name__ == "__main__":
    unittest.main()
