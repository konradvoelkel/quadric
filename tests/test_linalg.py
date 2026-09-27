"""exact and modular linear algebra (PLAN E3)"""

import random
import unittest
from fractions import Fraction

from bbcells import linalg


class TestCertifiedRank(unittest.TestCase):

    def test_agrees_with_exact_rank(self):
        rng = random.Random(1)
        for _ in range(60):
            n, m, r = rng.randrange(1, 10), rng.randrange(1, 10), rng.randrange(0, 7)
            A = [[Fraction(rng.randrange(-9, 10), rng.randrange(1, 5)) for _ in range(r)]
                 for _ in range(n)]
            B = [[rng.randrange(-9, 10) for _ in range(m)] for _ in range(r)]
            M = [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A] \
                if r else [[0] * m for _ in range(n)]
            self.assertEqual(linalg.certified_rank(M), linalg.rank(M), M)

    def test_large_entries(self):
        # a rank-deficient matrix whose kernel needs several primes to lift
        big = 10 ** 40 + 7
        M = [[big, 1, 0], [0, big, 1], [big * big, 2 * big, 1]]
        self.assertEqual(linalg.certified_rank(M), linalg.rank(M))


class TestStreamingInParallel(unittest.TestCase):

    def test_same_counts(self):
        from bbcells.frontends import symmetric
        for case in [("CI", 3), ("AIII", 2, 2)]:
            self.assertEqual(symmetric.cell_counts(*case, processes=2),
                             symmetric.cell_counts(*case), case)


if __name__ == "__main__":
    unittest.main()
