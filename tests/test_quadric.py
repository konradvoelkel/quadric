"""quadric front end and the cell-by-cell regression against the legacy
script quadric.py (PLAN.md S2.3)"""

import contextlib
import importlib.util
import io
import itertools
import random
import unittest
from pathlib import Path

from bbcells import oracles
from bbcells.algebra import IntPoly
from bbcells.core import bb_cells
from bbcells.frontends import quadric
from tests._util import doctests_for

load_tests = doctests_for(quadric)

ROOT = Path(__file__).resolve().parent.parent


def load_legacy():
    spec = importlib.util.spec_from_file_location("legacy_quadric", ROOT / "quadric.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load_legacy()


def legacy_dims(n, cocharacter):
    """{label: cell dimension} as computed by quadric.py"""
    Q = legacy.projective_quadric(n)
    with contextlib.redirect_stdout(io.StringIO()):
        _, dims = Q.compute_affine_cells(list(cocharacter))
    h = Q.root_system.rank
    labels = ["x_%d" % i for i in range(h)] + ["y_%d" % i for i in range(h)]
    return dict(zip(labels, dims))


def chambers(rank):
    """all signed permutations of 1..rank"""
    for perm in itertools.permutations(range(1, rank + 1)):
        for signs in itertools.product((1, -1), repeat=rank):
            yield tuple(s * p for s, p in zip(signs, perm))


class TestLegacyRegression(unittest.TestCase):

    def check(self, n, c):
        data = quadric.quadric(n)
        ours = bb_cells(data, tuple(-x for x in c))        # legacy c = our -c
        expected = legacy_dims(n, c)
        self.assertEqual({p: ours.dim_of(p) for p in data.points}, expected, (n, c))

    def test_all_chambers_up_to_dimension_5(self):
        for n in range(2, 6):
            rank = quadric.type_of(n)[1]
            for c in chambers(rank):
                self.check(n, c)

    def test_random_cocharacters_up_to_dimension_8(self):
        rng = random.Random(6)
        for n in range(2, 9):
            rank = quadric.type_of(n)[1]
            for _ in range(40):
                values = rng.sample(range(1, 30), rank)
                c = tuple(v * rng.choice((1, -1)) for v in values)
                self.check(n, c)

    def test_same_sign_does_not_match(self):
        """the relation really is c <-> -c (pins the convention of PLAN 1.3)"""
        data = quadric.quadric(4)
        c = (1, 2, 3)
        same = bb_cells(data, c)
        self.assertNotEqual({p: same.dim_of(p) for p in data.points}, legacy_dims(4, c))


class TestQuadricOracles(unittest.TestCase):

    def test_cell_polynomials(self):
        for n in range(1, 11):
            cells = bb_cells(quadric.quadric(n))
            self.assertTrue(cells.check().ok, n)
            self.assertEqual(IntPoly.from_counts(cells.counts), oracles.quadric(n), n)

    def test_fixed_points_are_the_coordinate_points(self):
        for n in range(1, 8):
            data = quadric.quadric(n)
            rank = quadric.type_of(n)[1]
            self.assertEqual(sorted(data.points),
                             sorted(["x_%d" % i for i in range(rank)] +
                                    ["y_%d" % i for i in range(rank)]))

    def test_q2_is_p1_x_p1(self):
        self.assertEqual(bb_cells(quadric.quadric(2)).counts, (1, 2, 1))


if __name__ == "__main__":
    unittest.main()
