"""full-rank homogeneous pieces G/H against the open orbits of S6b.1"""

import unittest

from bbcells.core import bb_cells
from bbcells.frontends import spherical, two_orbit
from bbcells.rootsystem import RootSystem
from tests._util import doctests_for

load_tests = doctests_for(spherical)


def multisets(weight_lists):
    return sorted(tuple(sorted(ws)) for ws in weight_lists)


def epsilon_columns(letter, m):
    """images of the simple roots of A_{m-1}, B_m, C_m in eps-coordinates"""
    unit = lambda i: [int(k == i) for k in range(m)]
    columns = [[a - b for a, b in zip(unit(i), unit(i + 1))] for i in range(m - 1)]
    if letter == "B":
        columns.append(unit(m - 1))
    elif letter == "C":
        columns.append([2 * c for c in unit(m - 1)])
    return columns


class TestHomogeneous(unittest.TestCase):

    def assert_open_orbit(self, case, points):
        expected = multisets(case.completion.weights_of(p) for p in case.open_fixed_points)
        self.assertEqual(multisets(points.values()), expected, case.name)

    def test_weyl_group_quotient_sizes(self):
        # |W / W_H| for G/T, a Levi, and the full group
        self.assertEqual(len(spherical.homogeneous_fixed_points("A2", [])), 6)
        self.assertEqual(len(spherical.homogeneous_fixed_points("B2", [(1, 0)])), 4)
        self.assertEqual(len(spherical.homogeneous_fixed_points("G2", [(1, 0), (0, 1)])), 1)

    def test_root_subsystem(self):
        from bbcells.rootsystem import RootSystem
        # the long roots of G2 form A2
        self.assertEqual(len(spherical.root_subsystem(RootSystem("G2"), [(0, 1), (3, 1)])), 6)

    def test_parabolic_orbits_are_flag_varieties(self):
        from bbcells.frontends.flag import flag_variety
        from bbcells.rootsystem import RootSystem
        for cartan_type, crossed in (("A3", {2}), ("B3", {1, 3}), ("G2", {2})):
            R = RootSystem(cartan_type)
            levi = {i for i in range(R.rank) if i + 1 not in crossed}
            unipotent = [b for b in R.positive_roots if b not in R.positive_roots_of(levi)]
            simple = [tuple(int(k == i) for k in range(R.rank)) for i in sorted(levi)]
            points = spherical.homogeneous_fixed_points(cartan_type, simple,
                                                         unipotent_roots=unipotent)
            G_P = flag_variety(cartan_type, crossed)
            self.assertEqual(multisets(points.values()), multisets(G_P.weights))

    def test_octonionic_projective_plane(self):
        # F4 / Spin(9): Spin(9) has roots -theta, a1, a2, a3 (extended diagram)
        from bbcells.rootsystem import RootSystem
        theta = max(RootSystem("F4").positive_roots, key=sum)
        points = spherical.homogeneous_fixed_points(
            "F4", [tuple(-c for c in theta), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)])
        self.assert_open_orbit(two_orbit.octonionic_projective_plane(), points)

    def test_g2_mod_sl3(self):
        points = spherical.homogeneous_fixed_points("G2", [(0, 1), (3, 1)],
                                                     component_reflections=[(1, 0)])
        self.assert_open_orbit(two_orbit.g2_on_p6(), points)

    def test_pgl_mod_gl(self):
        for n in (2, 3, 4):
            levi = [tuple(int(k == i) for k in range(n - 1)) for i in range(1, n - 1)]
            points = spherical.homogeneous_fixed_points("A%d" % (n - 1), levi)
            points = spherical.in_coordinates(points, epsilon_columns("A", n))
            self.assert_open_orbit(two_orbit.pgl_mod_gl(n), points)

    def test_quaternionic_projective_spaces(self):
        # Sp_{2m} / (Sp_2 x Sp_{2m-2}): 2 eps_1 and the simple roots a_2, ..., a_m
        for m in (2, 3, 4):
            generators = [tuple([2] * (m - 1) + [1])]
            generators += [tuple(int(k == i) for k in range(m)) for i in range(1, m)]
            points = spherical.homogeneous_fixed_points("C%d" % m, generators)
            points = spherical.in_coordinates(points, epsilon_columns("C", m))
            self.assert_open_orbit(two_orbit.quaternionic_projective_space(m - 1), points)

    def test_even_affine_quadrics(self):
        # SO(2k+1) / SO(2k): the long roots, a D_k subsystem
        for k in (1, 2, 3, 4):
            generators = [tuple(int(j == i) for j in range(k)) for i in range(k - 1)]
            if k >= 2:
                generators.append(tuple([0] * (k - 2) + [1, 2]))
            points = spherical.homogeneous_fixed_points("B%d" % k, generators)
            points = spherical.in_coordinates(points, epsilon_columns("B", k))
            self.assert_open_orbit(two_orbit.affine_quadric(2 * k), points)

    def test_projective_space_minus_odd_quadric(self):
        # SO(2k+1) / S(O(1) x O(2k)) = O(2k): D_k and the reflection in eps_k
        for k in (1, 2, 3):
            generators = [tuple(int(j == i) for j in range(k)) for i in range(k - 1)]
            if k >= 2:
                generators.append(tuple([0] * (k - 2) + [1, 2]))
            short = tuple(int(j == k - 1) for j in range(k))
            points = spherical.homogeneous_fixed_points("B%d" % k, generators,
                                                         component_reflections=[short])
            points = spherical.in_coordinates(points, epsilon_columns("B", k))
            case = two_orbit.projective_space_minus_quadric(2 * k - 1)
            self.assert_open_orbit(case, points)

    def test_rejects_non_roots(self):
        with self.assertRaises(ValueError):
            spherical.homogeneous_fixed_points("A2", [(2, 0)])



def ms(weight_lists):
    return multisets(weight_lists)


def unit(i, r):
    return tuple(int(k == i) for k in range(r))


class TestWonderful(unittest.TestCase):
    """wonderful varieties assembled from spherical roots and satellites"""

    def assert_same_data(self, X, completion, columns=None):
        points = dict(zip(X.points, X.weights))
        if columns is not None:
            points = spherical.in_coordinates(points, columns)
        self.assertEqual(len(X), len(completion))
        self.assertEqual(ms(points.values()), ms(completion.weights), X.name)

    def test_p1_x_p1(self):
        from bbcells.frontends.linear import projectivization
        from bbcells.operations import product, restrict
        P1 = projectivization([(1,), (-1,)])
        diagonal = restrict(product(P1, P1), [[1, 1]])
        X = spherical.wonderful_variety("A1", [(1,)], lambda I: {})
        self.assert_same_data(X, diagonal, [(2,)])

    def test_complete_conics_and_quadrics_in_p3(self):
        from bbcells.frontends.complete_conics import complete_conics, complete_quadrics_p3
        for n, front in ((3, complete_conics), (4, complete_quadrics_p3)):
            self.assert_same_data(spherical.complete_quadrics(n), front(),
                                  epsilon_columns("A", n))

    def test_complete_quadrics_against_orbit_decomposition(self):
        from bbcells import oracles
        from bbcells.algebra import IntPoly
        for n in range(2, 7):
            X = spherical.complete_quadrics(n)
            cells = bb_cells(X)
            self.assertEqual(IntPoly.from_counts(cells.counts), oracles.complete_quadrics(n))
            self.assertTrue(cells.check().ok, n)

    def test_complete_skew_forms_against_orbit_decomposition(self):
        from bbcells import oracles
        from bbcells.algebra import IntPoly
        from bbcells.frontends.linear import projectivization
        from bbcells.operations import restrict
        for n in range(1, 5):
            X = spherical.complete_skew_forms(n)
            self.assertEqual(IntPoly.from_counts(bb_cells(X).counts),
                             oracles.complete_skew_forms(n))
        # n = 2: P(Lambda^2 k^4), weights eps_i + eps_j
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        P5 = projectivization([tuple(int(k in pair) for k in range(4)) for pair in pairs])
        self.assert_same_data(spherical.complete_skew_forms(2), P5, epsilon_columns("A", 4))

    def test_rank_one_completions(self):
        # (type, spherical root, S^p, satellite of the open orbit, case, eps-columns)
        theta = max(RootSystem("F4").positive_roots, key=sum)
        cases = [("F4", (1, 2, 3, 2), (0, 1, 2),
                  {"generators": [tuple(-c for c in theta), (1, 0, 0, 0), (0, 1, 0, 0),
                                  (0, 0, 1, 0)]},
                  two_orbit.octonionic_projective_plane(), None),
                 ("G2", (4, 2), (1,),
                  {"generators": [(0, 1), (3, 1)], "component_reflections": [(1, 0)]},
                  two_orbit.g2_on_p6(), None)]
        for n in (2, 3, 4, 5):
            r = n - 1
            cases.append(("A%d" % r, (1,) * r, tuple(range(1, r - 1)),
                          {"generators": [unit(i, r) for i in range(r - 1)]},
                          two_orbit.pgl_mod_gl(n), epsilon_columns("A", n)))
        for m in (2, 3, 4):
            cases.append(("C%d" % m, tuple([1] + [2] * (m - 2) + [1]), (0,) + tuple(range(2, m)),
                          {"generators": [tuple([2] * (m - 1) + [1])] +
                           [unit(i, m) for i in range(1, m)]},
                          two_orbit.quaternionic_projective_space(m - 1), epsilon_columns("C", m)))
        for k in (2, 3, 4):
            d_k = [unit(i, k) for i in range(k - 1)] + [tuple([0] * (k - 2) + [1, 2])]
            cases.append(("B%d" % k, (1,) * k, tuple(range(1, k)), {"generators": d_k},
                          two_orbit.affine_quadric(2 * k), epsilon_columns("B", k)))
            cases.append(("B%d" % k, (2,) * k, tuple(range(1, k)),
                          {"generators": d_k, "component_reflections": [unit(k - 1, k)]},
                          two_orbit.projective_space_minus_quadric(2 * k - 1),
                          epsilon_columns("B", k)))
        for cartan_type, gamma, parabolic, satellite, case, columns in cases:
            X = spherical.wonderful_variety(cartan_type, [gamma], lambda I: satellite,
                                            parabolic=parabolic, name=case.name)
            self.assert_same_data(X, case.completion, columns)

    def test_undetermined_normal_weights_are_refused(self):
        # a satellite H_L = T in L = GL_2 leaves the normal character undetermined
        with self.assertRaises(ValueError):
            spherical.wonderful_variety("A2", [(2, 0), (0, 2)], lambda I: {})

    def test_spherical_roots_orthogonal_to_parabolic(self):
        with self.assertRaises(ValueError):
            spherical.wonderful_variety("A2", [(1, 1)], lambda I: None, parabolic=(0,))


if __name__ == "__main__":
    unittest.main()
