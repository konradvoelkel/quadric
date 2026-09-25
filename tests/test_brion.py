"""equivariant cohomology beyond GKM (Brion's local conditions)"""

import unittest

from bbcells import brion
from bbcells.core import bb_cells
from bbcells.equivariant import expected_dimension
from bbcells.frontends import flag, spherical, toric
from tests._util import doctests_for

load_tests = doctests_for(brion)


class TestComponents(unittest.TestCase):

    def test_inferred_curves_are_the_gkm_edges(self):
        for X in [flag.flag_variety("A2"), flag.flag_variety("B2"), flag.flag_variety("G2"),
                  flag.flag_variety("A3", {2}), toric.fixed_point_data(toric.hirzebruch(2))]:
            components = brion.fixed_components(X)
            self.assertEqual({frozenset(points) for _, points, _ in components},
                             {frozenset((p, q)) for p, q, _ in X.edges}, X.name)
            self.assertEqual({kind for _, _, kind in components}, {"curve"})

    def test_complete_conics_have_planes(self):
        X = spherical.complete_quadrics(3)
        kinds = [kind for _, _, kind in brion.fixed_components(X)]
        self.assertEqual((kinds.count("curve"), kinds.count("plane")), (12, 6))


class TestRing(unittest.TestCase):

    def test_free_module_with_bb_poincare_series(self):
        for X, top in [(spherical.complete_quadrics(3), 3), (flag.flag_variety("B2"), 3)]:
            components = brion.fixed_components(X)
            counts = bb_cells(X).counts
            for d in range(top + 1):
                self.assertEqual(brion.ring_dimension(X, components, d),
                                 expected_dimension(counts, X.rank, d), (X.name, d))

    def test_second_order_condition_is_needed(self):
        # treating the planes like curves (congruences mod chi only) gives too much
        X = spherical.complete_quadrics(3)
        components = brion.fixed_components(X)
        weak = [(chi, points, "curves") for chi, points, kind in components]
        counts = bb_cells(X).counts
        self.assertGreater(brion.ring_dimension(X, weak, 2), expected_dimension(counts, X.rank, 2))


if __name__ == "__main__":
    unittest.main()
