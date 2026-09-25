import random
import unittest

from bbcells import operations
from bbcells.algebra import IntPoly
from bbcells.core import bb_cells
from bbcells.frontends import toric
from tests._util import doctests_for

load_tests = doctests_for(operations)


def cell_polynomial(data, lam=None):
    return IntPoly.from_counts(bb_cells(data, lam).counts)


class TestOperations(unittest.TestCase):

    def test_product_multiplies_cell_polynomials(self):
        examples = [toric.projective_space(2), toric.hirzebruch(1), toric.del_pezzo_6()]
        for X in examples:
            for Y in examples:
                dx, dy = toric.fixed_point_data(X), toric.fixed_point_data(Y)
                product = operations.product(dx, dy)
                self.assertEqual(cell_polynomial(product),
                                 cell_polynomial(dx) * cell_polynomial(dy))
                self.assertTrue(bb_cells(product).check().ok)

    def test_product_agrees_with_product_fan(self):
        X, Y = toric.hirzebruch(2), toric.projective_space(1)
        a = operations.product(toric.fixed_point_data(X), toric.fixed_point_data(Y))
        b = toric.fixed_point_data(toric.product(X, Y))
        self.assertEqual(sorted(sorted(w) for w in a.weights),
                         sorted(sorted(w) for w in b.weights))
        self.assertEqual(len(a.edges), len(b.edges))

    def test_generic_one_parameter_subgroup_keeps_the_cells(self):
        rng = random.Random(5)
        data = toric.fixed_point_data(toric.del_pezzo_6())
        for _ in range(20):
            lam = (rng.randint(-50, 50), rng.randint(-50, 50))
            try:
                restricted = operations.restrict(data, [lam])
            except ValueError:
                continue            # lam is not generic
            self.assertEqual(bb_cells(restricted, (1,)).counts, bb_cells(data, lam).counts)

    def test_isolatedness_of_restrictions_to_subtori_of_p1xp1(self):
        data = toric.fixed_point_data(toric.product(toric.projective_space(1),
                                                    toric.projective_space(1)))
        # the diagonal torus (e.g. of SL_2 acting diagonally) has 4 isolated fixed points
        self.assertEqual(bb_cells(operations.restrict(data, [[1, 1]]), (1,)).counts, (1, 2, 1))
        # the torus of one factor fixes whole fibres
        with self.assertRaisesRegex(ValueError, "not isolated"):
            operations.restrict(data, [[1, 0]])


if __name__ == "__main__":
    unittest.main()
