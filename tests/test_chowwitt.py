"""Chow-Witt groups of cellular varieties over R (PLAN R6)"""

import unittest

from bbcells import chowwitt, realcells, realtoric
from bbcells.frontends import toric
from tests._util import doctests_for

load_tests = doctests_for(chowwitt)


class TestChowWitt(unittest.TestCase):

    def test_projective_spaces(self):
        # CH~^0 = GW(R) = Z^2; Z in the middle; the top is GW(R) iff n is odd
        for n in range(1, 6):
            fan = toric.projective_space(n)
            groups = chowwitt.chow_witt(realtoric.RealToricComplex(fan, tuple(range(1, n + 1))))
            self.assertEqual(groups, [(2, [])] + [(1, [])] * (n - 1) + [(2 if n % 2 else 1, [])])
            flag = chowwitt.chow_witt(realcells.real_flag_variety("A%d" % n, {1}))
            self.assertEqual(flag, groups, n)

    def test_ranks(self):
        # the rank of CH~^q is rank CH^q plus the rational Betti number b_q(X(R))
        for X in [realcells.real_flag_variety("A3"), realcells.real_flag_variety("B2"),
                  realcells.real_flag_variety("C3", {3}), realcells.real_flag_variety("A3", {2})]:
            dims, _ = chowwitt.cochain_complex(X)
            betti = X.rational_betti()
            self.assertEqual([free for free, _ in chowwitt.chow_witt(X)],
                             [c + b for c, b in zip(dims, betti)], X.name)

    def test_torsion_free_here(self):
        # the 2-torsion of H^q(X(R); Z) is absorbed: CH~^q is free in these cases
        for fan in [toric.hirzebruch(1), toric.del_pezzo_6(),
                    toric.product(toric.projective_space(2), toric.projective_space(1))]:
            X = realtoric.RealToricComplex(fan, tuple(3 ** k + k for k in range(fan.dim)))
            self.assertTrue(all(not torsion for _, torsion in chowwitt.chow_witt(X)), fan.name)


if __name__ == "__main__":
    unittest.main()
