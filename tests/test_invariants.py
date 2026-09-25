import unittest

from bbcells import invariants
from bbcells.algebra import GWClass, IntPoly
from bbcells.core import bb_cells
from tests._util import doctests_for
from tests.test_core import projective_space

load_tests = doctests_for(invariants)


class TestInvariants(unittest.TestCase):

    def test_projective_plane(self):
        inv = bb_cells(projective_space(2)).invariants()
        self.assertEqual(inv.betti, {0: 1, 2: 1, 4: 1})
        self.assertEqual(inv.poincare, IntPoly((1, 0, 1, 0, 1)))
        self.assertEqual(inv.point_count(3), 13)
        self.assertEqual(inv.gw_euler, GWClass(2, 1))
        self.assertEqual(inv.hodge, {(0, 0): 1, (1, 1): 1, (2, 2): 1})
        self.assertEqual(inv.real_euler_characteristic, 1)       # chi(RP^2)
        self.assertEqual(inv.real_mod2_total_betti, 3)

    def test_projective_line_is_hyperbolic(self):
        inv = bb_cells(projective_space(1)).invariants()
        self.assertEqual(inv.gw_euler, GWClass.hyperbolic())

    def test_real_euler_characteristic_of_rp_n(self):
        for n in range(1, 7):
            inv = bb_cells(projective_space(n)).invariants()
            self.assertEqual(inv.real_euler_characteristic, 1 if n % 2 == 0 else 0)

    def test_checks_pass_on_projective_spaces(self):
        for n in range(1, 5):
            report = bb_cells(projective_space(n)).check()
            self.assertTrue(report.ok, str(report))

    def test_motive_latex(self):
        inv = invariants.Invariants(2, (1, 2, 1))
        self.assertEqual(inv.motive_latex(),
                         r"\mathbb{Z} \oplus \mathbb{Z}(1)[2]^{\oplus 2} \oplus \mathbb{Z}(2)[4]")



class TestLefschetz(unittest.TestCase):
    """the holomorphic Lefschetz check sees errors the BB-based checks miss"""

    def test_flipped_normal_weight_is_detected(self):
        from bbcells.core import FixedPointData, bb_cells
        from bbcells.frontends import spherical
        X = spherical.complete_quadrics(3)
        weights = []
        for label, wts in zip(X.points, X.weights):
            if X.annotation(label)["orbit"] != "closed":    # last weight: normal to the orbit
                wts = wts[:-1] + (tuple(-c for c in wts[-1]),)
            weights.append(wts)
        wrong = FixedPointData(X.dim, X.rank, X.points, tuple(weights))
        report = {item.name: item.passed for item in bb_cells(wrong).check().items}
        self.assertTrue(report["cocharacter independence"])
        self.assertTrue(report["Poincare duality"])
        self.assertFalse(report["holomorphic Lefschetz"])
        self.assertTrue(bb_cells(X).check().ok)


if __name__ == "__main__":
    unittest.main()
