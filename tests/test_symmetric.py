"""complete symmetric varieties from Satake diagrams (docs/S6d.md, section 6)"""

import unittest

from bbcells.core import FixedPointData, bb_cells
from bbcells.frontends import spherical, symmetric, two_orbit
from tests._util import doctests_for
from tests.test_spherical import epsilon_columns

load_tests = doctests_for(symmetric)


def weight_multisets(X):
    return sorted(tuple(sorted(w)) for w in X.weights)


def converted(X, columns):
    points = spherical.in_coordinates(dict(zip(X.points, X.weights)), columns)
    return FixedPointData(X.dim, len(columns[0]), tuple(points), tuple(points.values()))


def node_map(images, rank):
    """columns sending alpha_i to alpha_{images[i]} (1-based)"""
    return [tuple(int(k == images[i] - 1) for k in range(rank)) for i in range(1, rank + 1)]


class TestKnownFamilies(unittest.TestCase):
    """the Satake front end reproduces the fixed-point data computed before"""

    def assert_same(self, X, Y):
        self.assertEqual(len(X), len(Y))
        self.assertEqual(weight_multisets(X), weight_multisets(Y))

    def test_complete_quadrics_and_skew_forms(self):
        for n in range(2, 6):
            self.assert_same(symmetric.complete_symmetric_variety("AI", n),
                             spherical.complete_quadrics(n))
        for n in range(1, 4):
            self.assert_same(symmetric.complete_symmetric_variety("AII", n),
                             spherical.complete_skew_forms(n))

    def test_rank_one(self):
        for n in (3, 4, 5):
            X = symmetric.complete_symmetric_variety("AIII", 1, n - 1)
            self.assert_same(converted(X, epsilon_columns("A", n)),
                             two_orbit.pgl_mod_gl(n).completion)
        for m in (3, 4):
            X = symmetric.complete_symmetric_variety("CII", 1, m - 1)
            self.assert_same(converted(X, epsilon_columns("C", m)),
                             two_orbit.quaternionic_projective_space(m - 1).completion)
        for k in (2, 3):
            X = symmetric.complete_symmetric_variety("BI", 1, 2 * k)
            self.assert_same(converted(X, epsilon_columns("B", k)),
                             two_orbit.projective_space_minus_quadric(2 * k - 1).completion)
        self.assert_same(symmetric.complete_symmetric_variety("FII"),
                         two_orbit.octonionic_projective_plane().completion)


class TestIsomorphisms(unittest.TestCase):
    """exceptional isomorphisms: two Satake diagrams on different root
    systems must give the same fixed-point data after relabelling nodes"""

    def assert_isomorphic(self, first, second, images):
        X = symmetric.complete_symmetric_variety(*first)
        Y = symmetric.complete_symmetric_variety(*second)
        self.assertEqual(len(X), len(Y), (first, second))
        self.assertEqual(weight_multisets(converted(X, node_map(images, X.rank))),
                         weight_multisets(Y), (first, second))

    def test_b2_c2(self):
        c2_to_b2 = {1: 2, 2: 1}
        self.assert_isomorphic(("CI", 2), ("BI", 2, 3), c2_to_b2)          # sp_4(R) = so(2, 3)
        self.assert_isomorphic(("CII", 1, 1), ("BI", 1, 4), c2_to_b2)      # sp(1, 1) = so(1, 4)

    def test_d3_a3(self):
        d3_to_a3 = {1: 2, 2: 1, 3: 3}
        self.assert_isomorphic(("DI", 3, 3), ("AI", 4), d3_to_a3)          # complete quadrics in P^3
        self.assert_isomorphic(("DI", 1, 5), ("AII", 2), d3_to_a3)         # P^5
        self.assert_isomorphic(("DI", 2, 4), ("AIII", 2, 2), d3_to_a3)
        self.assert_isomorphic(("DIII", 3), ("AIII", 1, 3), d3_to_a3)

    def test_triality(self):
        # so(2, 6) = so*(8): the triality 1 -> 4, 3 -> 1, 4 -> 3 of D4
        self.assert_isomorphic(("DI", 2, 6), ("DIII", 4), {1: 4, 2: 2, 3: 1, 4: 3})


class TestNewCases(unittest.TestCase):

    def test_consistency(self):
        for case in [("AIII", 2, 2), ("AIII", 3, 3), ("BI", 2, 5), ("BI", 3, 4), ("CI", 3),
                     ("CII", 2, 2), ("DI", 4, 4), ("DI", 3, 5), ("DIII", 4), ("G",), ("FI",),
                     ("EIV",)]:
            X = symmetric.complete_symmetric_variety(*case)
            self.assertTrue(bb_cells(X).check().ok, case)
            self.assertFalse(any(a.get("note") for a in X.annotations), case)

    def test_euler_characteristic_of_g2(self):
        # 12 (closed) + 6 + 6 (the two SL_2-satellites) + 12/4 (G_2/SO_4)
        X = symmetric.complete_symmetric_variety("G")
        self.assertEqual((X.dim, len(X)), (8, 27))

    def test_hermitian_satellites_are_marked_and_checked(self):
        for case in [("AIII", 2, 3), ("DIII", 5), ("EIII",)]:
            X = symmetric.complete_symmetric_variety(*case)
            self.assertTrue(any(a.get("note") == spherical.BEYOND_R for a in X.annotations))
            self.assertTrue(bb_cells(X).check().ok, case)
            with self.assertRaises(ValueError):
                symmetric.complete_symmetric_variety(*case, strict=True)

    def test_invalid_diagrams(self):
        with self.assertRaises(ValueError):
            symmetric.SatakeDiagram("A3", black=[2], arrows=[(1, 2)])
        with self.assertRaises(ValueError):
            symmetric.diagram("EV")


if __name__ == "__main__":
    unittest.main()
