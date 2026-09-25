"""regression tests for the H1 experiments (docs/H1.md)"""

import itertools
import unittest

from bbcells import gkm, h1, oracles, realcells
from bbcells.core import bb_cells
from bbcells.frontends import flag, quadric, toric


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
