"""real Schubert cells, Level A of SPEC.md 3.5 (PLAN.md S5.1)"""

import itertools
import unittest

from bbcells import oracles, realcells
from bbcells.algebra import IntPoly
from bbcells.core import bb_cells
from bbcells.frontends import flag
from tests._util import doctests_for

load_tests = doctests_for(realcells)


def parabolics(n):
    """all crossed-node sets of A_n"""
    for size in range(1, n + 1):
        for crossed in itertools.combinations(range(1, n + 1), size):
            yield set(crossed)


class TestTypeA(unittest.TestCase):

    def test_coboundary_squares_to_zero(self):
        """the signs of arXiv:1910.11149 Theorem `signs` give a cochain complex"""
        for n in range(1, 5):
            for crossed in parabolics(n):
                X = realcells.real_flag_variety("A%d" % n, crossed)
                self.assertTrue(X.square_zero(), (n, crossed))
        self.assertTrue(realcells.real_flag_variety("A5").square_zero())

    def test_real_projective_spaces(self):
        for n in range(1, 9):
            expected = [(1, [])]
            for c in range(1, n + 1):
                if c % 2 == 0:
                    expected.append((0, [2]))
                else:
                    expected.append((1, []) if c == n else (0, []))
            X = realcells.real_flag_variety("A%d" % n, {1})
            self.assertEqual(X.cohomology(), expected, n)

    def test_complete_flags_of_r4(self):
        """the table of arXiv:1910.11149, section `Fl4`"""
        X = realcells.real_flag_variety("A3")
        self.assertEqual(X.cohomology(), [(1, []), (0, []), (0, [2, 2, 2]), (2, [2, 2]),
                                          (0, [2, 2]), (0, [2, 2, 2]), (1, [])])

    def test_real_grassmannians_rational_cohomology(self):
        """Casian-Kodama, arXiv:1309.5520, Theorem B"""
        for n in range(2, 8):
            for k in range(1, n):
                X = realcells.real_flag_variety("A%d" % (n - 1), {k})
                betti = IntPoly.from_counts(X.rational_betti())
                self.assertEqual(betti, oracles.real_grassmannian_rational_poincare(k, n), (k, n))

    def test_only_two_torsion(self):
        """Hudson-Matszangosz-Wendt, arXiv:2302.11003: all torsion is 2-torsion"""
        for n in range(1, 5):
            for crossed in parabolics(n):
                for free, torsion in realcells.real_flag_variety("A%d" % n, crossed).cohomology():
                    self.assertTrue(all(t == 2 for t in torsion), (n, crossed, torsion))

    def test_euler_characteristic_matches_chi_a1_signature(self):
        for n in range(1, 5):
            for crossed in parabolics(n):
                X = realcells.real_flag_variety("A%d" % n, crossed)
                from_cohomology = sum((-1) ** c * free for c, (free, _) in enumerate(X.cohomology()))
                signature = bb_cells(flag.flag_variety("A%d" % n, crossed)).invariants() \
                    .real_euler_characteristic
                self.assertEqual(from_cohomology, signature, (n, crossed))
                self.assertEqual(X.euler_characteristic(), signature)

    def test_kocherlakota_agrees_with_matszangosz_on_complete_flags(self):
        for n in range(1, 5):
            signed = realcells.type_a_signed("A%d" % n)
            _, unsigned = realcells.kocherlakota_incidences("A%d" % n)
            self.assertEqual({key: abs(v) for key, v in signed.incidences.items()}, unsigned, n)

    def test_cellular_and_cooriented_conventions_differ_on_rp2(self):
        """Kocherlakota: d e^2 = 2 e^1 (cellular); Matszangosz: delta e^1 = 2 e^0
        in the cooriented complex; both compute the (co)homology of RP^2"""
        _, cellular = realcells.kocherlakota_incidences("A2", {1})
        cooriented = realcells.type_a_signed("A2", {1}).incidences
        self.assertEqual(cellular, {("s1", "e"): 0, ("s2.s1", "s1"): 2})
        self.assertEqual({k: abs(v) for k, v in cooriented.items()},
                         {("s1", "e"): 2, ("s2.s1", "s1"): 0})


class TestOtherTypes(unittest.TestCase):

    def test_unsigned_incidences(self):
        for name, crossed in (("B2", None), ("C3", {1}), ("G2", None), ("D4", {1}),
                              ("B3", {3}), ("F4", {4})):
            X = realcells.real_flag_variety(name, crossed)
            self.assertFalse(X.signed)
            self.assertTrue(set(X.incidences.values()) <= {0, 2})
            with self.assertRaises(ValueError):
                X.cohomology()

    def test_even_quadrics_have_no_adjacent_nonzero_incidence_in_the_middle(self):
        # D_m/P_1 = Q_{2m-2}: the two middle cells are not joined to each other
        X = realcells.real_flag_variety("D4", {1})
        middle = [p for p, d in X.dims.items() if d == 3]
        self.assertEqual(len(middle), 2)
        self.assertFalse(any((a, b) in X.incidences for a in middle for b in middle))


if __name__ == "__main__":
    unittest.main()
