"""the Choi-Park / Suciu-Trevisan oracle for real toric varieties (PLAN.md S5.2)"""

import unittest

from bbcells import oracles, realcells
from bbcells.core import bb_cells
from bbcells.frontends import toric


def fans():
    P2, P3 = toric.projective_space(2), toric.projective_space(3)
    yield P2
    yield P3
    yield toric.projective_space(4)
    for a in range(5):
        yield toric.hirzebruch(a)
    yield toric.del_pezzo_6()
    yield toric.star_subdivision(P2, (0, 1))
    yield toric.star_subdivision(toric.del_pezzo_6(), (0, 1))
    yield toric.star_subdivision(P3, (0, 1))
    yield toric.star_subdivision(P3, (0, 1, 2))
    yield toric.product(toric.projective_space(1), P2)
    yield toric.product(toric.hirzebruch(1), toric.projective_space(1))


class TestRealToricOracle(unittest.TestCase):

    def test_real_projective_spaces(self):
        for n in range(1, 6):
            fan = toric.projective_space(n)
            expected = [1] + [0] * (n - 1) + [1 if n % 2 else 0]
            self.assertEqual(oracles.real_toric_rational_betti(fan.rays, fan.cones), expected)

    def test_surfaces(self):
        # F_a(R) is a torus for a even and a Klein bottle for a odd
        for a in range(5):
            fan = toric.hirzebruch(a)
            self.assertEqual(oracles.real_toric_rational_betti(fan.rays, fan.cones),
                             [1, 2, 1] if a % 2 == 0 else [1, 1, 0])

    def test_euler_characteristic_is_the_signature_of_chi_a1(self):
        """two independent routes to chi(X(R)): the Choi-Park formula and the
        signature of chi^{A^1} = sum_d c_d <-1>^d from the BB cells"""
        for fan in fans():
            betti = oracles.real_toric_rational_betti(fan.rays, fan.cones)
            chi = sum((-1) ** p * b for p, b in enumerate(betti))
            signature = bb_cells(toric.fixed_point_data(fan)).invariants().real_euler_characteristic
            self.assertEqual(chi, signature, fan.name)

    def test_agrees_with_real_schubert_cells_for_projective_spaces(self):
        for n in range(1, 7):
            fan = toric.projective_space(n)
            self.assertEqual(oracles.real_toric_rational_betti(fan.rays, fan.cones),
                             realcells.real_flag_variety("A%d" % n, {1}).rational_betti())


if __name__ == "__main__":
    unittest.main()
