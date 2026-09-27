"""exact real BB incidences of toric varieties by discrete Morse theory"""

import unittest

from bbcells import gkm, h1, oracles, realcells, realtoric
from bbcells.core import bb_cells
from bbcells.frontends import toric
from tests._util import doctests_for

load_tests = doctests_for(realtoric)

P2, P3 = toric.projective_space(2), toric.projective_space(3)
FANS = [P2, P3, toric.hirzebruch(1), toric.hirzebruch(2), toric.del_pezzo_6(),
        toric.star_subdivision(toric.hirzebruch(1), (0, 1)),
        toric.star_subdivision(P3, (0, 1)), toric.star_subdivision(P3, (0, 1, 2)),
        toric.product(toric.hirzebruch(1), toric.projective_space(1))]


class TestRealToric(unittest.TestCase):

    def test_betti_numbers_against_choi_park(self):
        for fan in FANS:
            oracle = oracles.real_toric_rational_betti(fan.rays, fan.cones)
            for lam in [(3, -7, 11), (-13, 5, 2), (1, 17, -29)]:
                X = realtoric.RealToricComplex(fan, lam[:fan.dim])
                self.assertEqual(X.fine_betti(), oracle, fan.name)
                self.assertEqual(X.betti(), oracle, (fan.name, lam))

    def test_integral_homology(self):
        # RP^3, and the Klein bottle F_1(R): only 2-torsion
        self.assertEqual(realtoric.RealToricComplex(P3, (3, -7, 11)).integral_homology(),
                         [[0], [2], [], [0]])
        self.assertEqual(realtoric.RealToricComplex(toric.hirzebruch(1), (3, -7))
                         .integral_homology(), [[0], [0, 2], []])

    def test_twisted_poincare_duality(self):
        # the orientation sheaf of X(R) is Z(K), K = -sum D_rho: H_d(X(R); Z(K))
        # is H^{n-d}(X(R); Z), read off the untwisted complex (Ext of H_{n-d-1})
        for fan in FANS:
            lam = (3, -7, 11)[:fan.dim]
            plain = realtoric.RealToricComplex(fan, lam).integral_homology()
            twisted = realtoric.RealToricComplex(fan, lam, twist=[1] * len(fan.rays))
            n = fan.dim
            cohomology = [[0] * plain[n - d].count(0) +
                          sorted(t for t in (plain[n - d - 1] if n - d - 1 >= 0 else []) if t)
                          for d in range(n + 1)]
            self.assertEqual(twisted.integral_homology(), cohomology, fan.name)

    def test_twist_depends_on_the_class_mod_2(self):
        # D and D + div(chi^m) give the same local system; even twists are trivial
        for fan in FANS:
            lam = (3, -7, 11)[:fan.dim]
            base = [1] + [0] * (len(fan.rays) - 1)
            m = [1] + [0] * (fan.dim - 1)
            moved = [a + sum(x * y for x, y in zip(m, r)) for a, r in zip(base, fan.rays)]
            self.assertEqual(
                realtoric.RealToricComplex(fan, lam, twist=base).integral_homology(),
                realtoric.RealToricComplex(fan, lam, twist=moved).integral_homology(), fan.name)
            self.assertEqual(
                realtoric.RealToricComplex(fan, lam, twist=[2] * len(fan.rays)).integral_homology(),
                realtoric.RealToricComplex(fan, lam).integral_homology(), fan.name)

    def test_gkm_rule_entrywise_on_graded_decompositions(self):
        # H1': |[x : y]| = 2 iff x, y are joined by a curve with m even (docs/real.md)
        for fan in FANS:
            cells = h1.find_graded_cocharacter(toric.fixed_point_data(fan))
            if cells is None:
                continue
            exact = {k: abs(v) for k, v in
                     realtoric.RealToricComplex(fan, cells.cocharacter).incidences().items()}
            predicted = {k: mag for k, (m, mag) in realcells.gkm_incidences(cells).items() if mag}
            self.assertEqual(exact, predicted, fan.name)

    def test_dimensions_are_the_bb_dimensions(self):
        fan = toric.star_subdivision(P3, (0, 1))
        cells = bb_cells(toric.fixed_point_data(fan), (3, -7, 11))
        dims = realtoric.RealToricComplex(fan, (3, -7, 11)).dims()
        self.assertEqual(dims, {p: cells.dim_of(p) for p in cells.data.points})


if __name__ == "__main__":
    unittest.main()
