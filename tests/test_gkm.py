import itertools
import unittest

from bbcells import gkm
from bbcells.core import bb_cells
from bbcells.frontends import flag, quadric, toric
from bbcells.rootsystem import RootSystem
from tests._util import doctests_for

load_tests = doctests_for(gkm)


def bruhat_below(R, word):
    """{u(rho) : u <= w} by the subword property (independent of GKM data)"""
    rho = (1,) * R.rank
    elements = {rho}
    for i in reversed(word):
        elements |= {R.reflect_weight(i, x) for x in elements}
    return elements


def element_on_rho(R, word):
    x = (1,) * R.rank
    for i in reversed(word):
        x = R.reflect_weight(i, x)
    return x


class TestGKM(unittest.TestCase):

    def test_flag_and_toric_data_are_gkm(self):
        examples = [flag.flag_variety("A3"), flag.flag_variety("B3", {2}),
                    flag.flag_variety("G2"), flag.flag_variety("E6", {1}),
                    quadric.quadric(5), quadric.quadric(6),
                    toric.fixed_point_data(toric.del_pezzo_6()),
                    toric.fixed_point_data(toric.product(toric.projective_space(2),
                                                         toric.hirzebruch(1)))]
        for data in examples:
            self.assertEqual(gkm.check_gkm(data), [], data.name)

    def test_number_of_invariant_curves(self):
        for name in ("A3", "B3", "C3", "G2", "D4"):
            data = flag.flag_variety(name)
            R = RootSystem(name)
            self.assertEqual(len(data.edges), R.weyl_group_order * len(R.positive_roots) // 2)

    def test_bb_order_is_the_bruhat_order(self):
        """PLAN.md S3.2 for all G/P of rank <= 3 and dominant lambda"""
        for name in ("A1", "A2", "A3", "B2", "B3", "C3", "G2"):
            R = RootSystem(name)
            for size in range(1, R.rank + 1):
                for crossed in itertools.combinations(range(1, R.rank + 1), size):
                    data = flag.flag_variety(name, set(crossed))
                    order = gkm.bb_order(bb_cells(data))
                    words = {p: tuple(i - 1 for i in data.annotation(p)["word"])
                             for p in data.points}
                    for p in data.points:
                        below = bruhat_below(R, words[p])
                        expected = {q for q in data.points if q != p
                                    and element_on_rho(R, words[q]) in below}
                        self.assertEqual(order[p], expected, (name, crossed, p))

    def test_projective_space_order_is_a_chain(self):
        cells = bb_cells(toric.fixed_point_data(toric.projective_space(4)))
        covers = gkm.covering_relations(cells)
        self.assertEqual(len(covers), 4)
        self.assertEqual([cells.dim_of(q) for q, p in covers], [0, 1, 2, 3])

    def test_bb_decompositions_are_not_stratifications_in_general(self):
        """on dP_6 with lambda = (1, 3) an invariant curve joins two 1-cells:
        the closure of one cell meets the other in a point only"""
        data = toric.fixed_point_data(toric.del_pezzo_6())
        cells = bb_cells(data, (1, 3))
        self.assertFalse(gkm.is_graded(cells))
        order = gkm.bb_order(cells)             # still acyclic
        for p, lower in order.items():
            self.assertNotIn(p, lower)

    def test_flag_varieties_are_graded_for_dominant_lambda(self):
        for name, crossed in (("A3", None), ("B3", {1}), ("G2", None)):
            self.assertTrue(gkm.is_graded(bb_cells(flag.flag_variety(name, crossed))))

    def test_cycles_are_detected(self):
        from bbcells.core import FixedPointData
        # two curves joining a and b, one going up and one going down for lambda
        bad = FixedPointData(2, 2, ("a", "b"), (((1, 0), (0, -1)), ((-1, 0), (0, 1))),
                             edges=(("a", "b", (1, 0)), ("a", "b", (0, -1))))
        with self.assertRaisesRegex(ValueError, "cycle"):
            gkm.bb_order(bb_cells(bad, (1, 1)))

if __name__ == "__main__":
    unittest.main()
