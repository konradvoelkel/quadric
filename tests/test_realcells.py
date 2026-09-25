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


def compact_poincare(letter, n):
    """rational Poincare polynomial of the maximal compact subgroup K of the
    split real form: (G/B)(R) = K/M with M finite acting trivially on
    rational cohomology, so P((G/B)(R)) = P(K)"""
    def so(m):
        p = IntPoly((1,))
        for i in range(1, m // 2 + (m % 2)):
            if m % 2 == 1 or i < m // 2:
                p = p * (IntPoly.monomial(4 * i - 1) + 1)
        if m % 2 == 0 and m >= 2:
            p = p * (IntPoly.monomial(m - 1) + 1)
        return p

    def u(m):
        p = IntPoly((1,))
        for i in range(1, m + 1):
            p = p * (IntPoly.monomial(2 * i - 1) + 1)
        return p

    return {"A": lambda: so(n + 1), "B": lambda: so(n) * so(n + 1), "C": lambda: u(n),
            "D": lambda: so(n) * so(n)}[letter]()


class TestOtherTypes(unittest.TestCase):

    def test_full_flags_against_the_maximal_compact_subgroup(self):
        """P((G/B)(R); Q) = P(K) for SO(n+1), SO(n) x SO(n+1), U(n), SO(n) x SO(n)"""
        for name in ("A1", "A2", "A3", "B2", "B3", "C2", "C3", "D3"):
            letter, n = name[0], int(name[1:])
            X = realcells.real_flag_variety(name)
            self.assertTrue(X.signed and X.square_zero(), name)
            self.assertEqual(IntPoly.from_counts(X.rational_betti()), compact_poincare(letter, n),
                             name)

    def test_rabelo_san_martin_agrees_with_matszangosz_in_type_a(self):
        for n in range(1, 4):
            for crossed in parabolics(n):
                cellular = realcells.rabelo_san_martin("A%d" % n, crossed)
                self.assertTrue(cellular.square_zero())
                self.assertEqual(cellular.cohomology(),
                                 realcells.type_a_signed("A%d" % n, crossed).cohomology(),
                                 (n, crossed))

    def test_signs_from_square_zero_agree_with_rabelo_san_martin(self):
        """where d o d = 0 determines the homology, it matches"""
        for name, crossed in (("B2", None), ("B3", None), ("B3", {1}), ("C3", {2})):
            a = realcells.cellular_real_flag_variety(name, crossed)
            b = realcells.rabelo_san_martin(name, crossed)
            self.assertEqual(a.cellular_homology(), b.cellular_homology(), (name, crossed))

    def test_square_zero_does_not_always_determine_the_signs(self):
        with self.assertRaisesRegex(ValueError, "different homology"):
            realcells.cellular_real_flag_variety("C3")

    def test_exceptional_partial_flags(self):
        """signs from d o d = 0; the Euler characteristic must equal the
        signature of chi^{A^1} from the BB cells (an independent count)"""
        for name, crossed in (("G2", {1}), ("G2", {2}), ("F4", {1}), ("F4", {4}), ("E6", {1})):
            X = realcells.real_flag_variety(name, crossed)
            self.assertTrue(X.signed, (name, crossed))
            chi = sum((-1) ** c * free for c, (free, _) in enumerate(X.cohomology()))
            signature = bb_cells(flag.flag_variety(name, crossed)).invariants() \
                .real_euler_characteristic
            self.assertEqual(chi, signature, (name, crossed))
        # G2/P1 = Q_5, whose real points are (S^2 x S^3)/+-1: Q in degrees 0 and 3
        self.assertEqual(realcells.real_flag_variety("G2", {1}).rational_betti(),
                         [1, 0, 0, 1, 0, 0])

    def test_g2(self):
        """G2/B(R) = SO(4)/M: rational Poincare polynomial (1 + t^3)^2"""
        X = realcells.real_flag_variety("G2")
        self.assertTrue(X.signed)
        self.assertEqual(X.rational_betti(), [1, 0, 0, 2, 0, 0, 1])

    def test_partial_flags_of_classical_types_are_complexes(self):
        for name in ("B2", "B3", "C3", "D4"):
            n = int(name[1:])
            for crossed in parabolics(n):
                if name == "D4" and len(crossed) > 2:
                    continue
                X = realcells.real_flag_variety(name, crossed)
                self.assertTrue(X.square_zero(), (name, crossed))
                for free, torsion in X.cohomology():
                    self.assertTrue(all(t == 2 for t in torsion), (name, crossed))

    def test_even_quadrics_have_no_adjacent_nonzero_incidence_in_the_middle(self):
        # D_m/P_1 = Q_{2m-2}: the two middle cells are not joined to each other
        X = realcells.real_flag_variety("D4", {1})
        middle = [p for p, d in X.dims.items() if d == 3]
        self.assertEqual(len(middle), 2)
        self.assertFalse(any((a, b) in X.incidences for a in middle for b in middle))


if __name__ == "__main__":
    unittest.main()
