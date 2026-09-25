import unittest

from bbcells import operations, oracles
from bbcells.algebra import IntPoly
from bbcells.core import bb_cells
from bbcells.frontends import complete_conics, linear, toric
from tests._util import doctests_for

load_tests = doctests_for(operations, linear, complete_conics)


def weight_multisets(data):
    return sorted(sorted(w) for w in data.weights)


class TestBlowup(unittest.TestCase):

    def test_blowup_agrees_with_toric_star_subdivision(self):
        """PLAN 1.6 against the toric description, weight for weight"""
        cases = [(toric.projective_space(2), (0, 1)), (toric.projective_space(3), (0, 1)),
                 (toric.projective_space(3), (0, 1, 2)), (toric.hirzebruch(2), (1, 2)),
                 (toric.projective_space(4), (1, 3))]
        for fan, face in cases:
            ambient = toric.fixed_point_data(fan)
            center = toric.orbit_closure(fan, face)
            mapping = {p: p for p in center.points}
            blown_up = operations.blowup(ambient, center, mapping)
            expected = toric.fixed_point_data(toric.star_subdivision(fan, face))
            self.assertEqual(weight_multisets(blown_up), weight_multisets(expected),
                             (fan.name, face))

    def test_blowup_formula(self):
        P4 = toric.projective_space(4)
        for face in ((0, 1), (0, 1, 2), (0, 1, 2, 3)):
            center = toric.orbit_closure(P4, face)
            X = operations.blowup(toric.fixed_point_data(P4), center,
                                  {p: p for p in center.points})
            expected = oracles.blowup(oracles.projective_space(4),
                                      oracles.projective_space(4 - len(face)), len(face))
            self.assertEqual(IntPoly.from_counts(bb_cells(X).counts), expected)
            self.assertTrue(bb_cells(X).check().ok)

    def test_identify_finds_orbit_closures(self):
        fan = toric.projective_space(3)
        center = toric.orbit_closure(fan, (0, 1))
        self.assertEqual(operations.identify(center, toric.fixed_point_data(fan)),
                         {p: p for p in center.points})

    def test_repeated_normal_weights_are_refused(self):
        # P^1 x P^1 x P^1 with the diagonal torus has repeated weights at points
        P1 = toric.fixed_point_data(toric.projective_space(1))
        cube = operations.restrict(operations.product(operations.product(P1, P1), P1),
                                   [[1, 1, 1]])
        point = cube.__class__(0, 1, ("pt",), ((),))
        with self.assertRaisesRegex(ValueError, "not distinct"):
            operations.blowup(cube, point, {"pt": cube.points[0]})


class TestLinear(unittest.TestCase):

    def test_grassmannians_of_the_standard_representation(self):
        unit = lambda i, n: tuple(1 if k == i else 0 for k in range(n))
        for n in range(2, 7):
            for k in range(1, n):
                data = linear.grassmannian_of([unit(i, n) for i in range(n)], k)
                self.assertEqual(IntPoly.from_counts(bb_cells(data).counts),
                                 oracles.gaussian_binomial(n, k))

    def test_isotropic_grassmannians_match_the_flag_oracle(self):
        for m in range(2, 5):
            for k in range(1, m + 1):
                data = linear.isotropic_grassmannian_of(m, k)
                self.assertTrue(bb_cells(data).check().ok)
                self.assertEqual(IntPoly.from_counts(bb_cells(data).counts),
                                 oracles.flag_variety("C%d" % m, {k}), (m, k))


class TestCompleteConics(unittest.TestCase):

    def test_oracle(self):
        """PLAN.md 4.5: [P^5] - [P^2] + [P^2]^2"""
        X = complete_conics.complete_conics()
        cells = bb_cells(X)
        self.assertTrue(cells.check().ok, str(cells.check()))
        self.assertEqual(len(X), 12)
        self.assertEqual(cells.counts, (1, 2, 3, 3, 2, 1))
        self.assertEqual(IntPoly.from_counts(cells.counts),
                         oracles.blowup(oracles.projective_space(5),
                                        oracles.projective_space(2), 3))


class TestCompleteQuadrics(unittest.TestCase):

    def test_complete_quadrics_in_p3(self):
        """two blow-ups of P^9 (Vainsencher) against the blow-up formula with
        [S_2] from counting rank <= 2 symmetric matrices over F_q"""
        X = complete_conics.complete_quadrics_p3()
        cells = bb_cells(X)
        self.assertTrue(cells.check().ok)
        self.assertEqual(len(X), 66)
        self.assertEqual(cells.counts, (1, 3, 6, 10, 13, 13, 10, 6, 3, 1))
        self.assertEqual(IntPoly.from_counts(cells.counts), oracles.complete_quadrics_p3())

    def test_rank_two_locus_counts(self):
        """[S_2] = [P^3] + L^2 (L^2+1)(L^2+L+1): unordered pairs of planes over F_q
        (distinct, conjugate over F_{q^2}, or double), checked for q = 3, 5, 7"""
        L = IntPoly.monomial(1)
        S2 = oracles.projective_space(3) + L ** 2 * (L ** 2 + 1) * (L ** 2 + L + 1)
        for q in (3, 5, 7):
            planes = sum(q ** i for i in range(4))
            planes_q2 = sum(q ** (2 * i) for i in range(4))
            pairs = planes + planes * (planes - 1) // 2 + (planes_q2 - planes) // 2
            self.assertEqual(S2(q), pairs, q)


if __name__ == "__main__":
    unittest.main()
