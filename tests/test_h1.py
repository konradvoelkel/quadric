"""regression tests for the H1 experiments (docs/real.md)"""

import itertools
import unittest

from bbcells import gkm, h1, oracles, realcells
from bbcells.core import bb_cells
from bbcells.frontends import flag, quadric, toric
from tests._util import doctests_for

load_tests = doctests_for(h1)


class TestGKMKocherlakota(unittest.TestCase):

    def test_equals_kocherlakota_on_flag_varieties(self):
        """sigma(x) = sum of lambda-positive tangent weights is Kocherlakota's sigma"""
        for name in ("A2", "A3", "B2", "B3", "C3", "G2", "D4"):
            rank = int(name[1:])
            for size in range(1, rank + 1):
                for crossed in itertools.combinations(range(1, rank + 1), size):
                    data, kocherlakota = realcells.kocherlakota_incidences(name, set(crossed))
                    predicted = realcells.gkm_incidences(bb_cells(data))
                    self.assertEqual({k: v[1] for k, v in predicted.items()}, kocherlakota,
                                     (name, crossed))


class TestRealRealization(unittest.TestCase):

    def test_toric_varieties_against_choi_park(self):
        P2, P3 = toric.projective_space(2), toric.projective_space(3)
        fans = [P2, P3] + [toric.hirzebruch(a) for a in range(5)] + [
            toric.star_subdivision(P2, (0, 1)), toric.star_subdivision(P3, (0, 1)),
            toric.star_subdivision(P3, (0, 1, 2)),
            toric.product(toric.projective_space(1), P2),
            toric.product(toric.hirzebruch(1), toric.projective_space(1))]
        for fan in fans:
            cells = h1.find_graded_cocharacter(toric.fixed_point_data(fan))
            self.assertIsNotNone(cells, fan.name)
            self.assertEqual(h1.real_prediction(cells),
                             {tuple(oracles.real_toric_rational_betti(fan.rays, fan.cones))},
                             fan.name)

    def test_dp6_has_no_graded_decomposition(self):
        """on the hexagon two 1-cells are always joined by a curve"""
        self.assertIsNone(h1.find_graded_cocharacter(toric.fixed_point_data(toric.del_pezzo_6()),
                                                     bound=8))

    def test_non_graded_surfaces_by_deflection(self):
        """the deflection rule (docs/real.md, section 5.1) against Choi-Park"""
        dp6 = toric.del_pezzo_6()
        fans = [dp6, toric.star_subdivision(dp6, (0, 1)),
                toric.star_subdivision(toric.hirzebruch(1), (0, 1)),
                toric.star_subdivision(toric.hirzebruch(3), (1, 2))]
        for fan in fans:
            data = toric.fixed_point_data(fan)
            oracle = tuple(oracles.real_toric_rational_betti(fan.rays, fan.cones))
            for lam in [(-22, 6), (24, 21), (11, -6), (-14, -23)]:
                cells = bb_cells(data, lam)
                self.assertEqual(h1.surface_prediction(cells), {oracle}, (fan.name, lam))


class TestComplexRealization(unittest.TestCase):

    def test_sq2_on_h2(self):
        examples = [toric.fixed_point_data(toric.projective_space(3)),
                    toric.fixed_point_data(toric.hirzebruch(1)),
                    toric.fixed_point_data(toric.hirzebruch(2)),
                    toric.fixed_point_data(toric.star_subdivision(toric.projective_space(3), (0, 1))),
                    flag.grassmannian(2, 4), flag.flag_variety("A2"), flag.flag_variety("A3"),
                    flag.flag_variety("B2"), flag.flag_variety("G2"), quadric.quadric(4),
                    quadric.quadric(5)]
        for data in examples:
            cells = bb_cells(data)
            if not gkm.is_graded(cells):
                cells = h1.find_graded_cocharacter(data)
            self.assertEqual(h1.sq2_mismatches(cells), [], data.name)


if __name__ == "__main__":
    unittest.main()


class TestBeyondGKM(unittest.TestCase):
    """the rule on the curves of the Brion components (docs/real.md, section 4)"""

    def prediction(self, X, lam):
        from bbcells import brion
        return h1.brion_prediction(bb_cells(brion.with_invariant_curves(X), lam))

    def test_grassmannian_with_symplectic_torus(self):
        from bbcells.frontends import symmetric
        X = symmetric.complete_symmetric_variety("CII", 1, 2)          # Gr(2, 6), Sp_6-torus
        expected = tuple(oracles.real_grassmannian_rational_poincare(2, 6).coefficients)
        for lam in [(50, 48, -53), (-5, 21, -10)]:
            self.assertEqual(self.prediction(X, lam), {expected + (0,) * (9 - len(expected))})

    def test_product_of_projective_spaces(self):
        from bbcells.frontends import symmetric
        X = symmetric.complete_symmetric_variety("AIII", 1, 3)         # P^3 x P^3*
        for lam in [(50, 48, -53), (14, 27, -40)]:
            self.assertEqual(self.prediction(X, lam), {(1, 0, 0, 2, 0, 0, 1)})

    def test_cayley_plane_with_f4_torus(self):
        from bbcells.frontends import symmetric
        X = symmetric.complete_symmetric_variety("FII")                # E6/P1, F4-torus
        F = flag.flag_variety("E6", {1})
        cells = bb_cells(F)
        dims = {p: cells.dim_of(p) for p in F.points}
        magnitudes = {k: v[1] for k, v in realcells.gkm_incidences(cells).items()}
        expected = {tuple(realcells.cellular_rational_betti(dims, s))
                    for s in realcells.sign_choices(dims, magnitudes)}
        self.assertEqual(len(expected), 1)
        self.assertEqual(self.prediction(X, (79, -15, 11, 55)), expected)

    def test_complete_quadric_surfaces_direct_rule_fails(self):
        # the oracle is H_*(X(R); Q) = Q_0 + Q_5 (docs/real.md 5.4); the direct
        # rule on the Brion curves admits no sign completion, as for conics
        from bbcells import brion
        from bbcells.frontends import spherical
        X = brion.with_invariant_curves(spherical.complete_quadrics(4))
        cells = bb_cells(X, (-1, 47, 3))
        dims = {p: cells.dim_of(p) for p in X.points}
        self.assertEqual(sum((-1) ** d for d in dims.values()), 1 - 1)
        magnitudes = {(x, y): 2 for x, y, m in h1.curve_parities(cells)
                      if dims[x] == dims[y] + 1 and m is not None and m % 2 == 0}
        with self.assertRaises(ValueError):
            realcells.sign_choices(dims, magnitudes)

    def test_complete_conics_need_one_correction(self):
        # no graded cocharacter; the direct rule admits no sign completion, and
        # removing the one term from the top of the even equal-dimension curve to
        # the bottom of the odd one gives H^*(X(R); Q) = H^*(RP^5; Q)
        from bbcells import brion
        from bbcells.frontends import spherical
        X = brion.with_invariant_curves(spherical.complete_quadrics(3))
        cells = bb_cells(X, (1, 5))
        dims = {p: cells.dim_of(p) for p in X.points}
        parities = h1.curve_parities(cells)
        equal = [(x, y, m) for x, y, m in parities if dims[x] == dims[y]]
        (top,) = [x for x, y, m in equal if m % 2 == 0]
        (bottom,) = [y for x, y, m in equal if m % 2]
        magnitudes = {(x, y): 2 for x, y, m in parities
                      if dims[x] == dims[y] + 1 and m % 2 == 0}
        with self.assertRaises(ValueError):
            realcells.sign_choices(dims, magnitudes)
        del magnitudes[(top, bottom)]
        self.assertEqual({tuple(realcells.cellular_rational_betti(dims, s))
                          for s in realcells.sign_choices(dims, magnitudes)},
                         {(1, 0, 0, 0, 0, 1)})
