"""toroidal varieties over wonderful models (docs/spherical.md, section 8)"""

import unittest
from itertools import combinations

from bbcells.algebra import IntPoly
from bbcells.core import bb_cells
from bbcells.frontends import quadric, spherical, symmetric, toroidal
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


def _q():
    return IntPoly((0, 1))


def _order(N, degrees):
    """|G(F_q)| of a split connected reductive group: q^N prod (q^d - 1)"""
    result = _q() ** N
    for d in degrees:
        result = result * (_q() ** d - IntPoly((1,)))
    return result


def _quotient(a, b):
    """exact division of integer polynomials"""
    a, b = list(a.coefficients), list(b.coefficients)
    out = [0] * (len(a) - len(b) + 1)
    for k in range(len(out) - 1, -1, -1):
        c, r = divmod(a[k + len(b) - 1], b[-1])
        if r:
            raise AssertionError("not divisible")
        out[k] = c
        for i, x in enumerate(b):
            a[i + k] -= c * x
    if any(a):
        raise AssertionError("not divisible")
    return IntPoly(tuple(out))


def cover(kind, *parameters, fan=None):
    """the toroidal G/G^theta-embedding (G simply connected) with the given
    fan, by default the orthant resolved in N' = Hom(Lambda, Z)"""
    D = symmetric.diagram(kind, *parameters)
    L = D.lattice()
    rows = toroidal.lattice_in_sigma(D.R.name, D.spherical_roots, L)
    if fan is None:
        fan = toroidal.resolve(toroidal.orthant(len(D.spherical_roots), rows), rows)
    return toroidal.toroidal_variety(D.R.name, D.spherical_roots, D.satellite, fan,
                                     parabolic=tuple(sorted(D.black)), lattice=L)


def chern_number(X, classes, lam):
    """integral over X of prod_k classes[k](p) by localization at lam"""
    from fractions import Fraction
    total = Fraction(0)
    for p, weights in zip(X.points, X.weights):
        values = [sum(a * b for a, b in zip(w, lam)) for w in weights]
        euler = 1
        for v in values:
            euler *= v
        term = Fraction(1)
        for f in classes:
            term *= f(p, values)
        total += term / euler
    return total


def c1(p, values):
    return sum(values)


class TestFinerLattices(unittest.TestCase):
    """toroidal varieties with Lambda > Z Sigma (docs/spherical.md, section 8)"""

    def test_sl2_mod_torus(self):
        # P^1 x P^1 > SL_2/T over P^2 > SL_2/N(T): the open orbit has two fixed points
        X = cover("AI", 2)
        self.assertEqual(bb_cells(X).counts, (1, 2, 1))
        self.assertEqual(sorted(multisets(X)), [((-2,), (-2,)), ((-2,), (2,)), ((-2,), (2,)),
                                                ((2,), (2,))])

    def test_quadrics(self):
        # Spin_{n+1}/Spin_n embeds in the quadric Q^n, a double cover of P^n
        # branched along the closed orbit Q^{n-1}; c_1(Q^n)^n = 2 n^n
        for case, n in [(("BI", 1, 4), 4), (("DI", 1, 5), 5), (("BI", 1, 6), 6),
                        (("DI", 1, 7), 7), (("AII", 2), 5)]:
            X = cover(*case)
            self.assertEqual(bb_cells(X).counts, bb_cells(quadric.quadric(n)).counts, case)
            lam = bb_cells(X).cocharacter
            self.assertEqual(chern_number(X, [c1] * n, lam), 2 * n ** n, case)

    def test_open_orbit_point_counts(self):
        # |G/K|(q) = |G(F_q)| / |K(F_q)| (Lang, K = G^theta connected and split)
        # against the open-orbit lemma applied to the fixed points
        SL = lambda n: _order(n * (n - 1) // 2, range(2, n + 1))
        GL = lambda n: _order(n * (n - 1) // 2, range(1, n + 1))
        Sp = lambda n: _order(n * n, range(2, 2 * n + 1, 2))
        SO4 = _order(2, [2, 2])
        cases = [(("AI", 2), SL(2), GL(1)), (("AI", 3), SL(3), Sp(1)),
                 (("AI", 4), SL(4), SO4), (("AII", 2), SL(4), Sp(2)),
                 (("AIII", 2, 2), SL(4), _quotient(GL(2) * GL(2), GL(1))),
                 (("CI", 2), Sp(2), GL(2)), (("CI", 3), Sp(3), GL(3)),
                 (("DI", 3, 3), SL(4), SO4), (("DIII", 4), _order(12, [2, 4, 4, 6]), GL(4)),
                 (("CII", 2, 2), Sp(4), Sp(2) * Sp(2))]
        for case, G, K in cases:
            X = cover(*case)
            cells = bb_cells(X)
            self.assertTrue(cells.check().ok, case)
            self.assertEqual(spherical.open_orbit_count(X, cells.cocharacter),
                             _quotient(G, K), case)

    def test_isomorphic_pairs(self):
        pairs = [(("AI", 4), ("DI", 3, 3)), (("CI", 2), ("BI", 2, 3)),
                 (("AII", 2), ("DI", 1, 5)), (("AIII", 2, 2), ("DI", 2, 4))]
        for a, b in pairs:
            self.assertEqual(bb_cells(cover(*a)).counts, bb_cells(cover(*b)).counts, (a, b))

    def test_triple_cover_of_complete_conics(self):
        # Y > SL_3/SO_3 with rays (-3,0), (-1,-1), (0,-3) in N' is a triple cover of
        # X = complete conics blown up along G/B, branched along D = D_1 + D_2
        # (rays (-1,0), (0,-1) of N): K_Y = pi^*(K_X + 2/3 D), so
        # c_1(Y)^5 = 3 (c_1(X) - 2/3 D)^5
        from fractions import Fraction
        D = symmetric.diagram("AI", 3)
        Y = cover("AI", 3, fan=[((-3, 0), (-1, -1)), ((-1, -1), (0, -3))])
        X = toroidal.toroidal_variety("A2", D.spherical_roots, D.satellite,
                                      toroidal.star_subdivision(toroidal.orthant(2),
                                                                toroidal.orthant(2)[0]))

        def boundary(p, values):
            rays = X.annotation(p)["orbit"].split("|")[1].split(";")
            k = len(rays)
            return sum(values[len(values) - k + i] for i, ray in enumerate(rays)
                       if ray in ("-1,0", "0,-1"))

        twisted = lambda p, values: c1(p, values) - Fraction(2, 3) * boundary(p, values)
        left = chern_number(Y, [c1] * 5, bb_cells(Y).cocharacter)
        right = 3 * chern_number(X, [twisted] * 5, bb_cells(X).cocharacter)
        self.assertEqual(left, right)
        self.assertEqual(bb_cells(Y).counts, bb_cells(X).counts)     # same E-polynomial

    def test_lattice_checks(self):
        rows = toroidal.lattice_in_sigma("A2", [(2, 0), (0, 2)], [(2, 0), (0, 2)])
        with self.assertRaises(ValueError):                      # singular in N'
            toroidal.check_fan(toroidal.orthant(2, rows), 2, rows)
        with self.assertRaises(ValueError):                      # ray not in N'
            toroidal.check_fan([((-1, 0), (-1, -1)), ((-1, -1), (0, -1))], 2, rows)
        with self.assertRaises(ValueError):                      # misses Z Sigma
            toroidal.lattice_in_sigma("A2", [(2, 0), (0, 2)], [(4, 0), (0, 4)])
        self.assertEqual([toroidal.sheets(rows, J) for J in [(), (0,), (1,), (0, 1)]],
                         [1, 1, 1, 3])


if __name__ == "__main__":
    unittest.main()
