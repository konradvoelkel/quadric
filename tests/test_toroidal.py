"""toroidal varieties over wonderful models (docs/S6d.md, section 8)"""

import unittest
from itertools import combinations

from bbcells.algebra import IntPoly
from bbcells.core import bb_cells
from bbcells.frontends import spherical, symmetric, toroidal
from bbcells.operations import blowup
from tests._util import doctests_for

load_tests = doctests_for(toroidal)

R3 = 3
SIGMA = [tuple(2 * int(k == i) for k in range(R3)) for i in range(R3)]     # complete quadrics in P^3


def satellite(I):
    if any(abs(a - b) == 1 for a in I for b in I):
        return None
    return {"component_reflections": [tuple(int(k == i) for k in range(R3)) for i in I]}


def multisets(X):
    return sorted(tuple(sorted(w)) for w in X.weights)


def orbit_point_counts(W, r):
    """|O_I|(q) by Moebius inversion from the orbit closures of a wonderful W"""
    closures = {}
    for size in range(r + 1):
        for I in combinations(range(r), size):
            closures[I] = IntPoly.from_counts(bb_cells(spherical.orbit_closure(W, I)).counts)
    counts = {}
    for I in closures:
        total = IntPoly(())
        for size in range(len(I) + 1):
            for J in combinations(I, size):
                total = total + closures[J] if (len(I) - size) % 2 == 0 else total - closures[J]
        counts[I] = total
    return counts


class TestToroidal(unittest.TestCase):

    def test_orthant_is_the_wonderful_variety(self):
        W = spherical.wonderful_variety("A3", SIGMA, satellite)
        X = toroidal.toroidal_variety("A3", SIGMA, satellite, toroidal.orthant(R3))
        self.assertEqual(multisets(X), multisets(W))

    def test_blowups_along_orbit_closures(self):
        W = spherical.wonderful_variety("A3", SIGMA, satellite)
        rays = toroidal.orthant(R3)[0]
        for K in [(), (0,), (1,), (2,)]:
            face = tuple(v for j, v in enumerate(rays) if j not in K)
            X = toroidal.toroidal_variety("A3", SIGMA, satellite,
                                          toroidal.star_subdivision(toroidal.orthant(R3), face))
            Z = spherical.orbit_closure(W, K)
            B = blowup(W, Z, {p: p for p in Z.points})
            self.assertEqual(multisets(X), multisets(B), K)

    def test_point_count_by_orbits(self):
        # |X| = sum over cones tau of |O_J(tau)| (q - 1)^(dim F_J - dim tau)
        W = spherical.wonderful_variety("A3", SIGMA, satellite)
        orbits = orbit_point_counts(W, R3)
        fan = toroidal.star_subdivision(toroidal.orthant(R3), toroidal.orthant(R3)[0])
        fan = toroidal.star_subdivision(fan, ((-1, -1, -1), (-1, 0, 0)))
        X = toroidal.toroidal_variety("A3", SIGMA, satellite, fan)
        expected = IntPoly(())
        for tau in toroidal._faces(fan):
            J = tuple(j for j in range(R3) if all(v[j] == 0 for v in tau))
            expected = expected + orbits[J] * IntPoly((-1, 1)) ** (R3 - len(J) - len(tau))
        cells = bb_cells(X)
        self.assertEqual(IntPoly.from_counts(cells.counts), expected)
        self.assertTrue(cells.check().ok)

    def test_symmetric_model(self):
        # Sp_4/GL_2 blown up along its closed orbit
        D = symmetric.diagram("CI", 2)
        fan = toroidal.star_subdivision(toroidal.orthant(2), toroidal.orthant(2)[0])
        X = toroidal.toroidal_variety(D.R.name, D.spherical_roots, D.satellite, fan,
                                      parabolic=tuple(sorted(D.black)))
        W = D.fixed_point_data()
        Z = spherical.orbit_closure(W, ())
        B = blowup(W, Z, {p: p for p in Z.points})
        self.assertEqual(multisets(X), multisets(B))

    def test_invalid_fans(self):
        with self.assertRaises(ValueError):                      # not complete in V
            toroidal.check_fan([((-1, 0), (-1, -1))], 2)
        with self.assertRaises(ValueError):                      # leaves V
            toroidal.check_fan([((1, 0), (0, -1))], 2)
        with self.assertRaises(ValueError):                      # not smooth
            toroidal.check_fan([((-1, 0), (-1, -2)), ((-1, -2), (0, -1))], 2)


if __name__ == "__main__":
    unittest.main()
