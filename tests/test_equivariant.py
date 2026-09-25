"""H_T^* as a GKM ring (PLAN.md S3.3)"""

import unittest
from fractions import Fraction
from math import factorial

from bbcells import equivariant as eq
from bbcells import polynomial
from bbcells.core import bb_cells
from bbcells.frontends import flag, quadric, toric
from bbcells.polynomial import Poly
from tests._util import doctests_for

load_tests = doctests_for(polynomial)

EXAMPLES = [
    toric.fixed_point_data(toric.projective_space(3)),
    toric.fixed_point_data(toric.hirzebruch(1)),
    toric.fixed_point_data(toric.del_pezzo_6()),
    flag.grassmannian(2, 4),
    flag.flag_variety("A2"), flag.flag_variety("B2"), flag.flag_variety("G2", {1}),
    quadric.quadric(3), quadric.quadric(4),
]


def degree_of_grassmannian(k, n):
    d = factorial(k * (n - k))
    for i in range(k):
        d = d * factorial(i) // factorial(n - k + i)
    return d


def divisor(cells):
    """the flow-up class of codimension 1, if unique"""
    (p,) = [q for q in cells.data.points if cells.dim_of(q) == cells.data.dim - 1]
    return eq.flow_up_class(cells, p)


def power_integral(cells, classes, h, k):
    data = cells.data
    values = {q: h.get(q, Poly(data.rank)) ** k for q in data.points}
    return eq.integrate(data, values)


class TestGKMRing(unittest.TestCase):

    def test_hilbert_series(self):
        """dim H_T^{2k} = coefficient of P_X(t) / (1 - t^2)^r"""
        for data in EXAMPLES:
            counts = bb_cells(data).counts
            for k in range(4):
                self.assertEqual(eq.gkm_ring_dimension(data, k),
                                 eq.expected_dimension(counts, data.rank, k), (data.name, k))

    def test_flow_up_classes(self):
        for data in EXAMPLES:
            cells = bb_cells(data)
            classes = eq.flow_up_classes(cells)
            for p, values in classes.items():
                codim = data.dim - cells.dim_of(p)
                for q, f in values.items():
                    self.assertTrue(f.is_homogeneous(codim) or f.is_zero())
                # tau_p restricted to p is the product of the normal weights: nonzero
                self.assertFalse(values[p].is_zero())

    def test_degrees_by_localization(self):
        """int h^n over X by Atiyah-Bott: deg Gr(k, n), deg Q_n = 2, deg P^n = 1"""
        for k, n in ((2, 4), (2, 5), (3, 6), (2, 6)):
            data = flag.grassmannian(k, n)
            cells = bb_cells(data)
            h = divisor(cells)
            self.assertEqual(power_integral(cells, None, h, data.dim),
                             degree_of_grassmannian(k, n), (k, n))
        for n in (3, 4, 5):
            data = quadric.quadric(n)
            cells = bb_cells(data)
            h = divisor(cells)
            self.assertEqual(power_integral(cells, None, h, n), 2, n)
        data = toric.fixed_point_data(toric.projective_space(4))
        cells = bb_cells(data)
        h = divisor(cells)
        self.assertEqual(power_integral(cells, None, h, 4), 1)

    def test_integrals_of_low_degree_classes_vanish(self):
        data = flag.grassmannian(2, 4)
        cells = bb_cells(data)
        h = divisor(cells)
        # int h^k = 0 for k < dim, at two different evaluation points
        for k in range(data.dim):
            for point in ([Fraction(1), Fraction(5), Fraction(-2)],
                          [Fraction(2), Fraction(-7), Fraction(11)]):
                values = {q: h.get(q, Poly(data.rank)) ** k for q in data.points}
                self.assertEqual(eq.integrate(data, values, point), 0)

    def test_pieri_on_gr24(self):
        """sigma_1^2 = sigma_2 + sigma_{1,1} in H^*(Gr(2,4))"""
        data = flag.grassmannian(2, 4)
        cells = bb_cells(data)
        classes = eq.flow_up_classes(cells)
        (s1,) = [p for p in data.points if cells.dim_of(p) == 3]
        constants = eq.structure_constants(cells, classes, s1, s1)
        nonequivariant = {s: c.constant_term() for s, c in constants.items()
                          if c.degree() == 0}
        codim2 = [p for p in data.points if cells.dim_of(p) == 2]
        self.assertEqual(nonequivariant, {p: 1 for p in codim2})

    def test_structure_constants_reproduce_the_product(self):
        data = flag.flag_variety("A2")
        cells = bb_cells(data)
        classes = eq.flow_up_classes(cells)
        for a in data.points:
            for b in data.points:
                c = eq.structure_constants(cells, classes, a, b)
                for q in data.points:
                    lhs = classes[a].get(q, Poly(2)) * classes[b].get(q, Poly(2))
                    rhs = Poly(2)
                    for s, coefficient in c.items():
                        rhs = rhs + coefficient * classes[s].get(q, Poly(2))
                    self.assertEqual(lhs, rhs)

    def test_products_of_divisors_on_full_flags_of_a2(self):
        """divisor times divisor in SL_3/B: the nonequivariant (Chevalley-Monk)
        structure constants are nonnegative integers"""
        data = flag.flag_variety("A2")
        cells = bb_cells(data)
        classes = eq.flow_up_classes(cells)
        divisors = [p for p in data.points if cells.dim_of(p) == data.dim - 1]
        for a in divisors:
            for b in divisors:
                for s, c in eq.structure_constants(cells, classes, a, b).items():
                    if c.degree() == 0:
                        self.assertGreaterEqual(c.constant_term(), 0)


if __name__ == "__main__":
    unittest.main()
