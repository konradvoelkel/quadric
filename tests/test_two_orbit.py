"""rank-one two-orbit completions against PLAN.md 4.4"""

import unittest

from bbcells.algebra import IntPoly
from bbcells.frontends import two_orbit
from bbcells.operations import restrict
from bbcells.rootsystem import RootSystem
from tests._util import doctests_for

load_tests = doctests_for(two_orbit)

L = IntPoly.monomial(1)


def geometric_sum(terms, step=1):
    """1 + L^step + ... + L^{step (terms - 1)}"""
    return sum((IntPoly.monomial(step * i) for i in range(terms)), IntPoly(()))


class TestTwoOrbit(unittest.TestCase):

    def check_case(self, case, expected):
        self.assertEqual(case.k0_class(), expected, case.name)
        # the open orbit contains exactly [X](1) fixed points
        self.assertEqual(len(case.open_fixed_points), expected(1), case.name)
        self.assertTrue(case.completion_cells().check().ok, case.name)
        self.assertTrue(case.boundary_cells().check().ok, case.name)
        self.assertEqual(set(case.normal), set(case.boundary.points))

    def test_affine_quadrics(self):
        for n in range(2, 10):
            m = n // 2 if n % 2 == 0 else (n + 1) // 2
            if n % 2 == 0:
                expected = L ** m * (L ** m + 1)
            else:
                expected = L ** (m - 1) * (L ** m - 1)
            self.check_case(two_orbit.affine_quadric(n), expected)

    def test_quaternionic_projective_spaces(self):
        for n in range(1, 5):
            self.check_case(two_orbit.quaternionic_projective_space(n),
                            L ** (2 * n) * geometric_sum(n + 1, 2))

    def test_octonionic_projective_plane(self):
        case = two_orbit.octonionic_projective_plane()
        self.check_case(case, L ** 8 * geometric_sum(3, 4))
        self.assertEqual(len(case.completion), 27)
        self.assertEqual(len(case.boundary), 24)

    def test_pgl_mod_gl(self):
        for n in range(2, 7):
            self.check_case(two_orbit.pgl_mod_gl(n), L ** (n - 1) * geometric_sum(n))

    def test_explicit_embeddings_agree_with_weight_matching(self):
        """where weight inclusion is unambiguous it finds the same embedding"""
        from bbcells.operations import identify
        for case in (two_orbit.affine_quadric(7), two_orbit.affine_quadric(8),
                     two_orbit.quaternionic_projective_space(2)):
            self.assertEqual(identify(case.boundary, case.completion), case.mapping, case.name)

    def test_projective_space_minus_quadric(self):
        # [P^{n+1}] - [Q_n] = L^{n+1} (n odd), L^{n+1} - L^{n/2} (n even)
        for n in range(2, 9):
            expected = L ** (n + 1) - (L ** (n // 2) if n % 2 == 0 else IntPoly(()))
            self.check_case(two_orbit.projective_space_minus_quadric(n), expected)

    def test_g2_and_spin7(self):
        self.check_case(two_orbit.g2_on_p6(), L ** 6)
        spin = two_orbit.spin7_on_p7()
        self.check_case(spin, L ** 7 - L ** 3)
        # Spin(7)/G2 = SO(8)/SO(7): the same class as the affine quadric AQ_7
        self.assertEqual(spin.k0_class(), two_orbit.affine_quadric(7).k0_class())

    def test_folding_restricts_e6_roots_to_f4_roots(self):
        E6, F4 = RootSystem("E6"), RootSystem("F4")
        images = set()
        for beta in E6.positive_roots:
            image = tuple(sum(a * b for a, b in zip(row, beta)) for row in two_orbit.E6_TO_F4)
            images.add(image)
        self.assertEqual(images, set(F4.positive_roots))

    def test_boundary_of_hp_n_is_isotropic(self):
        """the closed orbit in Gr(2, 2n+2) is the isotropic Grassmannian (dim 4n - 1)"""
        case = two_orbit.quaternionic_projective_space(2)
        self.assertEqual(case.boundary.dim, 7)
        for label in case.boundary.points:
            e = {part[1:] for part in label.split("+") if part[0] == "e"}
            f = {part[1:] for part in label.split("+") if part[0] == "f"}
            self.assertFalse(e & f, label)

    def test_compactly_supported_euler_characteristic(self):
        # chi_c^{A^1}(HP^n) = sum_{i=0}^n <-1>^{2n+2i} = (n + 1)<1>
        for n in range(1, 4):
            gw = two_orbit.quaternionic_projective_space(n).gw_euler_compact_support()
            self.assertEqual((gw.plus, gw.minus), (n + 1, 0))


if __name__ == "__main__":
    unittest.main()
