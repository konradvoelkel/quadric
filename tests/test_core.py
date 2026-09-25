import random
import unittest

from bbcells import core
from bbcells.core import FixedPointData, bb_cells, choose_generic_cocharacter, is_generic
from tests._util import doctests_for

load_tests = doctests_for(core)


def projective_space(n):
    """fixed-point data of P^n for the torus of GL_{n+1}: at e_i the tangent
    weights are e_j - e_i (hand-written, independent of the toric front end)"""
    rank = n + 1
    unit = lambda i: tuple(1 if k == i else 0 for k in range(rank))
    points, weights = [], []
    for i in range(rank):
        points.append("e%d" % i)
        weights.append(tuple(tuple(a - b for a, b in zip(unit(j), unit(i)))
                             for j in range(rank) if j != i))
    return FixedPointData(n, rank, tuple(points), tuple(weights), name="P^%d" % n)


def random_data(rng, dim, rank, points):
    weights = []
    for _ in range(points):
        wts = []
        while len(wts) < dim:
            w = tuple(rng.randint(-3, 3) for _ in range(rank))
            if any(w):
                wts.append(w)
        weights.append(tuple(wts))
    return FixedPointData(dim, rank, tuple("p%d" % i for i in range(points)), tuple(weights))


class TestCore(unittest.TestCase):

    def test_projective_space_counts(self):
        for n in range(1, 6):
            self.assertEqual(bb_cells(projective_space(n)).counts, (1,) * (n + 1))

    def test_opposite_reverses_counts(self):
        rng = random.Random(3)
        for _ in range(50):
            data = random_data(rng, rng.randint(1, 4), rng.randint(1, 3), rng.randint(1, 6))
            cells = bb_cells(data)
            self.assertEqual(cells.opposite().counts, cells.counts[::-1])

    def test_default_cocharacter_is_generic(self):
        rng = random.Random(4)
        for _ in range(100):
            data = random_data(rng, 3, rng.randint(1, 4), 3)
            self.assertTrue(is_generic(data, choose_generic_cocharacter(data)))
            self.assertTrue(is_generic(data, choose_generic_cocharacter(data, reverse=True)))

    def test_validation_messages(self):
        with self.assertRaisesRegex(ValueError, "expected 2 tangent weights"):
            FixedPointData(2, 1, ("a",), (((1,),),))
        with self.assertRaisesRegex(ValueError, "integer vector of length 2"):
            FixedPointData(1, 2, ("a",), (((1,),),))
        with self.assertRaisesRegex(ValueError, "not distinct"):
            FixedPointData(1, 1, ("a", "a"), (((1,),), ((-1,),)))
        with self.assertRaisesRegex(ValueError, "not generic"):
            bb_cells(projective_space(2), (1, 1, 1))

    def test_edges_are_validated(self):
        P1 = FixedPointData(1, 1, ("0", "oo"), (((1,),), ((-1,),)), edges=(("0", "oo", (1,)),))
        self.assertEqual(P1.edges, (("0", "oo", (1,)),))
        with self.assertRaises(ValueError):
            FixedPointData(1, 1, ("0", "oo"), (((1,),), ((-1,),)), edges=(("0", "oo", (-1,)),))

    def test_preferred_cocharacter_is_used(self):
        data = projective_space(2)
        data = FixedPointData(data.dim, data.rank, data.points, data.weights,
                              preferred_cocharacter=(3, 2, 1))
        cells = bb_cells(data)
        self.assertEqual(cells.cocharacter, (3, 2, 1))
        # open cell at the point whose tangent weights all pair positively
        self.assertEqual(cells.dim_of("e2"), 2)


if __name__ == "__main__":
    unittest.main()
