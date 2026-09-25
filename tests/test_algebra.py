import random
import unittest

from bbcells import algebra
from bbcells.algebra import GWClass, IntPoly
from tests._util import doctests_for

load_tests = doctests_for(algebra)


class TestIntPoly(unittest.TestCase):

    def test_ring_axioms_on_random_polynomials(self):
        rng = random.Random(0)
        for _ in range(200):
            a, b, c = (IntPoly(tuple(rng.randint(-5, 5) for _ in range(rng.randint(0, 5))))
                       for _ in range(3))
            self.assertEqual((a + b) * c, a * c + b * c)
            self.assertEqual(a * b, b * a)
            self.assertEqual((a * b) * c, a * (b * c))
            self.assertEqual(a - a, IntPoly(()))
            for x in (-2, 0, 3):
                self.assertEqual((a * b)(x), a(x) * b(x))

    def test_gaussian_binomial_is_palindromic(self):
        # [4 choose 2]_q = 1 + q + 2q^2 + q^3 + q^4
        p = IntPoly((1, 1, 2, 1, 1))
        self.assertTrue(p.is_palindromic())
        self.assertEqual(p(1), 6)

    def test_format_and_latex(self):
        self.assertEqual(IntPoly((1, 0, 3)).format("t"), "1 + 3t^2")
        self.assertEqual(IntPoly((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)).latex("L"), "L^{10}")
        self.assertEqual(IntPoly((-1,)).format(), "-1")


class TestGWClass(unittest.TestCase):

    def test_hyperbolic_absorbs(self):
        rng = random.Random(1)
        H = GWClass.hyperbolic()
        for _ in range(100):
            x = GWClass(rng.randint(-9, 9), rng.randint(-9, 9))
            self.assertEqual(H * x, x.rank * H)

    def test_signature_is_a_ring_map(self):
        rng = random.Random(2)
        for _ in range(100):
            x = GWClass(rng.randint(-9, 9), rng.randint(-9, 9))
            y = GWClass(rng.randint(-9, 9), rng.randint(-9, 9))
            self.assertEqual((x * y).signature, x.signature * y.signature)
            self.assertEqual((x * y).rank, x.rank * y.rank)


if __name__ == "__main__":
    unittest.main()
