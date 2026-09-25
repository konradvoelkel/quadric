import json
import unittest

from bbcells import oracles
from bbcells.algebra import IntPoly
from bbcells.core import bb_cells
from bbcells.frontends import toric
from tests._util import doctests_for

load_tests = doctests_for(toric, oracles)


def cell_polynomial(fan, lam=None):
    cells = bb_cells(toric.fixed_point_data(fan), lam)
    report = cells.check()
    assert report.ok, str(report)
    return IntPoly.from_counts(cells.counts)


class TestToricOracles(unittest.TestCase):
    """PLAN.md 4.1"""

    def test_projective_spaces(self):
        for n in range(1, 6):
            self.assertEqual(cell_polynomial(toric.projective_space(n)),
                             oracles.projective_space(n))

    def test_hirzebruch_surfaces(self):
        for a in range(5):
            self.assertEqual(cell_polynomial(toric.hirzebruch(a)), IntPoly((1, 2, 1)))

    def test_del_pezzo_6(self):
        self.assertEqual(cell_polynomial(toric.del_pezzo_6()), IntPoly((1, 4, 1)))

    def test_products(self):
        P1 = toric.projective_space(1)
        cube = toric.product(toric.product(P1, P1), P1)
        self.assertEqual(cell_polynomial(cube), IntPoly((1, 3, 3, 1)))
        P2xF3 = toric.product(toric.projective_space(2), toric.hirzebruch(3))
        self.assertEqual(cell_polynomial(P2xF3),
                         oracles.projective_space(2) * IntPoly((1, 2, 1)))

    def test_blowups_match_the_blowup_formula(self):
        P3 = toric.projective_space(3)
        point = toric.star_subdivision(P3, P3.cones[0])
        self.assertEqual(cell_polynomial(point),
                         oracles.blowup(oracles.projective_space(3), IntPoly((1,)), 3))
        line = toric.star_subdivision(P3, (0, 1))
        self.assertEqual(cell_polynomial(line),
                         oracles.blowup(oracles.projective_space(3),
                                        oracles.projective_space(1), 2))
        self.assertEqual(cell_polynomial(point), IntPoly((1, 2, 2, 1)))

    def test_blowing_up_a_point_of_p2_gives_a_hirzebruch_surface_count(self):
        F1 = toric.star_subdivision(toric.projective_space(2), (0, 1))
        self.assertEqual(cell_polynomial(F1), cell_polynomial(toric.hirzebruch(1)))


class TestToricConventions(unittest.TestCase):

    def test_open_cell_at_the_cone_containing_lambda(self):
        """PLAN.md 1.1: the open cell is at the cone containing lambda"""
        fan = toric.hirzebruch(2)
        data = toric.fixed_point_data(fan)
        for lam in ((1, 1), (-1, 5), (-3, -1), (5, -1)):
            cells = bb_cells(data, lam)
            (open_point,) = cells.cells_of_dimension(fan.dim)
            cone = data.annotation(open_point)["cone"]
            dual = fan.dual_basis(cone)
            self.assertTrue(all(sum(a * b for a, b in zip(m, lam)) > 0 for m in dual))

    def test_edges_are_gkm(self):
        data = toric.fixed_point_data(toric.projective_space(3))
        self.assertEqual(len(data.edges), 6)          # edges of the tetrahedron
        for p, q, chi in data.edges:
            self.assertIn(tuple(-c for c in chi), data.weights_of(q))


class TestFanValidation(unittest.TestCase):

    def test_rejects_singular_cone(self):
        with self.assertRaisesRegex(ValueError, "not smooth"):
            toric.Fan(((1, 0), (1, 2), (-1, -1)), ((0, 1), (1, 2), (0, 2)))

    def test_rejects_non_primitive_ray(self):
        with self.assertRaisesRegex(ValueError, "not primitive"):
            toric.Fan(((2, 0), (0, 1)), ((0, 1),))

    def test_rejects_incomplete_fan(self):
        P2 = toric.projective_space(2)
        with self.assertRaisesRegex(ValueError, "not complete"):
            toric.Fan(P2.rays, P2.cones[:2])

    def test_rejects_overlapping_cones(self):
        # the cones of P^1 x P^1 plus one more cone overlapping two of them
        rays = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1))
        with self.assertRaisesRegex(ValueError, "lies in 3 maximal cones"):
            toric.Fan(rays, ((0, 1), (1, 2), (2, 3), (0, 3), (0, 4)))

    def test_json_round_trip(self):
        fan = toric.del_pezzo_6()
        again = toric.from_dict(json.loads(json.dumps(toric.to_dict(fan))))
        self.assertEqual(again, fan)


if __name__ == "__main__":
    unittest.main()
