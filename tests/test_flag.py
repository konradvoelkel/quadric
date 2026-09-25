import itertools
import unittest

from bbcells import oracles
from bbcells.algebra import IntPoly
from bbcells.core import bb_cells, choose_generic_cocharacter
from bbcells.frontends import flag
from tests._util import doctests_for

load_tests = doctests_for(flag)

SMALL_TYPES = ["A1", "A2", "A3", "A4", "B2", "B3", "B4", "C3", "C4", "D4", "G2", "F4"]


class TestFlagOracles(unittest.TestCase):
    """PLAN.md 4.2: |G/P|(q) = prod[d_G] / prod[d_L]"""

    def test_all_parabolics_of_rank_at_most_four(self):
        for name in SMALL_TYPES:
            rank = int(name[1:])
            for size in range(1, rank + 1):
                for crossed in itertools.combinations(range(1, rank + 1), size):
                    data = flag.flag_variety(name, set(crossed))
                    counts = bb_cells(data).counts
                    self.assertEqual(IntPoly.from_counts(counts),
                                     oracles.flag_variety(name, set(crossed)),
                                     (name, crossed))

    def test_exceptional_minuscule_and_adjoint(self):
        e6p1 = (1, 1, 1, 1, 2, 2, 2, 2, 3, 2, 2, 2, 2, 1, 1, 1, 1)
        self.assertEqual(bb_cells(flag.flag_variety("E6", {1})).counts, e6p1)
        f4p4 = (1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1)
        self.assertEqual(bb_cells(flag.flag_variety("F4", {4})).counts, f4p4)
        for name, node, points in (("E7", {7}, 56), ("E8", {8}, 240), ("E6", {2}, 72)):
            data = flag.flag_variety(name, node)
            self.assertEqual(len(data), points)
            self.assertEqual(IntPoly.from_counts(bb_cells(data).counts),
                             oracles.flag_variety(name, node))

    def test_grassmannians_are_gaussian_binomials(self):
        for n in range(2, 8):
            for k in range(1, n):
                counts = bb_cells(flag.grassmannian(k, n)).counts
                self.assertEqual(IntPoly.from_counts(counts), oracles.gaussian_binomial(n, k))

    def test_quadrics_as_flag_varieties(self):
        # Q_{2m-1} = B_m / P_1, Q_{2m-2} = D_m / P_1
        for m in range(2, 6):
            self.assertEqual(IntPoly.from_counts(bb_cells(flag.flag_variety("B%d" % m, {1})).counts),
                             oracles.quadric(2 * m - 1))
        for m in range(3, 6):
            self.assertEqual(IntPoly.from_counts(bb_cells(flag.flag_variety("D%d" % m, {1})).counts),
                             oracles.quadric(2 * m - 2))


class TestFlagConventions(unittest.TestCase):

    def test_schubert_cells_have_dimension_length(self):
        """PLAN.md 1.2: for dominant regular lambda, dim(cell of wP) = l(w)"""
        for name, crossed in (("A3", None), ("B3", {2}), ("G2", None), ("D4", {1, 3}), ("F4", {1})):
            data = flag.flag_variety(name, crossed)
            cells = bb_cells(data)                      # rho^vee
            for label in data.points:
                self.assertEqual(cells.dim_of(label), data.annotation(label)["length"])

    def test_counts_do_not_depend_on_the_cocharacter(self):
        for name, crossed in (("A3", {2}), ("C3", None), ("G2", {1})):
            data = flag.flag_variety(name, crossed)
            dominant = bb_cells(data).counts
            for reverse in (False, True):
                lam = choose_generic_cocharacter(data, reverse=reverse)
                self.assertEqual(bb_cells(data, lam).counts, dominant)
            self.assertTrue(bb_cells(data).check().ok)

    def test_weights_are_roots(self):
        from bbcells.rootsystem import RootSystem
        R = RootSystem("B3")
        roots = set(R.positive_roots) | {tuple(-c for c in b) for b in R.positive_roots}
        data = flag.flag_variety("B3")
        for wts in data.weights:
            self.assertTrue(set(wts) <= roots)
            # at each point exactly one of beta, -beta is a tangent weight (G/B)
            self.assertEqual(len({tuple(sorted((w, tuple(-c for c in w)))) for w in wts}), 9)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            flag.flag_variety("A2", {3})
        with self.assertRaises(ValueError):
            flag.flag_variety("A2", set())
        with self.assertRaisesRegex(ValueError, "more fixed points than the limit"):
            flag.flag_variety("E8")


if __name__ == "__main__":
    unittest.main()
