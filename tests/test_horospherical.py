import unittest

from bbcells import oracles
from bbcells.algebra import IntPoly
from bbcells.core import bb_cells
from bbcells.frontends import horospherical, toric
from tests._util import doctests_for

load_tests = doctests_for(horospherical)

CASES = [
    # (type, crossed, basis of M, fan)
    ("A1", {1}, [(1,)], toric.projective_space(1)),
    ("A2", {1}, [(1, 0)], toric.projective_space(1)),
    ("A2", {1, 2}, [(1, 0), (0, 1)], toric.projective_space(2)),
    ("A2", {1, 2}, [(1, 0), (0, 1)], toric.hirzebruch(2)),
    ("C2", {1, 2}, [(1, 1)], toric.projective_space(1)),
    ("G2", {1, 2}, [(1, 0), (0, 1)], toric.hirzebruch(3)),
    ("B3", {1, 3}, [(2, 0, 1), (0, 0, 1)], toric.del_pezzo_6()),
    ("A3", {2}, [(0, 1, 0)], toric.projective_space(1)),
]


class TestHorospherical(unittest.TestCase):

    def test_fibration_formula(self):
        """PLAN.md S6a.1: P_X = P_{G/P} * P_Y (Zariski-locally trivial fibration)"""
        for cartan_type, crossed, basis, fan in CASES:
            X = horospherical.toroidal_horospherical(cartan_type, crossed, basis, fan)
            cells = bb_cells(X)
            self.assertTrue(cells.check().ok, (cartan_type, crossed, str(cells.check())))
            fibre = IntPoly.from_counts(bb_cells(toric.fixed_point_data(fan)).counts)
            self.assertEqual(IntPoly.from_counts(cells.counts),
                             oracles.flag_variety(cartan_type, crossed) * fibre,
                             (cartan_type, crossed, fan.name))

    def test_sl3_over_p2_is_the_blowup_of_p3_in_a_point(self):
        X = horospherical.toroidal_horospherical("A2", {1}, [(1, 0)], toric.projective_space(1))
        self.assertEqual(IntPoly.from_counts(bb_cells(X).counts),
                         oracles.blowup(oracles.projective_space(3), IntPoly((1,)), 3))

    def test_twist_against_an_independent_blowup(self):
        """SL_3 x_{P_1} P^1 is the blow-up of P(k^3 + k) at the SL_3-fixed point.
        The weights agree with operations.blowup exactly, and dropping the
        w-twist of the fibre weights breaks the agreement (the untwisted data
        is G/P x Y, which the consistency checks cannot tell apart)."""
        from bbcells.core import FixedPointData
        from bbcells.frontends import linear
        from bbcells.operations import blowup
        X = horospherical.toroidal_horospherical("A2", {1}, [(1, 0)], toric.projective_space(1))
        # eps_1 = omega_1, eps_2 = omega_2 - omega_1, eps_3 = -omega_2 in weight coordinates
        P3 = linear.projectivization([(1, 0), (-1, 1), (0, -1), (0, 0)],
                                     ["e1", "e2", "e3", "e4"])
        B = blowup(P3, FixedPointData(0, 2, ("pt",), ((),)), {"pt": "e4"})
        multisets = lambda data: sorted(sorted(w) for w in data.weights)
        self.assertEqual(multisets(X), multisets(B))
        fibre_at_base = {X.annotation(p)["cone"]: w[2:] for p, w in zip(X.points, X.weights)
                         if not X.annotation(p)["word"]}
        untwisted = FixedPointData(X.dim, X.rank, X.points, tuple(
            w[:2] + fibre_at_base[X.annotation(p)["cone"]] for p, w in zip(X.points, X.weights)))
        self.assertNotEqual(multisets(untwisted), multisets(B))
        self.assertTrue(bb_cells(untwisted).check().ok)

    def test_bad_input(self):
        P1 = toric.projective_space(1)
        with self.assertRaisesRegex(ValueError, "not a character of P"):
            horospherical.toroidal_horospherical("A2", {1}, [(0, 1)], P1)
        with self.assertRaisesRegex(ValueError, "rank"):
            horospherical.toroidal_horospherical("A2", {1, 2}, [(1, 0), (0, 1)], P1)


if __name__ == "__main__":
    unittest.main()


class TestColoredHorospherical(unittest.TestCase):
    """PLAN.md S6a.2 (Batyrev-Moreau oracle) and S6a.3 (colored fans)"""

    SL3_CONES = [([(1, 0), (0, 1)], {1}), ([(0, 1), (-1, -1)], set()), ([(-1, -1), (1, 0)], {1})]

    @staticmethod
    def multisets(data):
        return sorted(sorted(w) for w in data.weights)

    def test_projective_spaces_as_colored_embeddings(self):
        """SL_n acting on P(k^n + k): exactly the weights of the linear model"""
        from bbcells.frontends import linear
        for n in (2, 3, 4, 5):
            r = n - 1
            X = horospherical.colored_horospherical(
                "A%d" % r, {1}, [tuple(int(k == 0) for k in range(r))],
                [([(1,)], {1}), ([(-1,)], set())])
            eps = []
            for i in range(n):
                v = [0] * r
                if i < r:
                    v[i] += 1
                if i > 0:
                    v[i - 1] -= 1
                eps.append(tuple(v))
            P = linear.projectivization(eps + [tuple([0] * r)])
            self.assertEqual(self.multisets(X), self.multisets(P), n)

    def test_uncolored_fans_agree_with_the_toroidal_front_end(self):
        for cartan_type, crossed, basis, fan in CASES:
            cones = [([fan.rays[i] for i in cone], set()) for cone in fan.cones]
            colored = horospherical.colored_horospherical(cartan_type, crossed, basis, cones)
            toroidal = horospherical.toroidal_horospherical(cartan_type, crossed, basis, fan)
            self.assertEqual(self.multisets(colored), self.multisets(toroidal), cartan_type)

    def test_batyrev_moreau_oracle(self):
        cases = [("A1", {1}, [(1,)], [([(1,)], {1}), ([(-1,)], set())]),
                 ("A3", {1}, [(1, 0, 0)], [([(1,)], {1}), ([(-1,)], set())]),
                 ("C2", {1}, [(1, 0)], [([(1,)], {1}), ([(-1,)], set())]),
                 ("C3", {1}, [(1, 0, 0)], [([(1,)], {1}), ([(-1,)], set())]),
                 ("A2", {1, 2}, [(1, 0), (0, 1)], self.SL3_CONES)]
        for cartan_type, crossed, basis, cones in cases:
            X = horospherical.colored_horospherical(cartan_type, crossed, basis, cones)
            cells = bb_cells(X)
            self.assertTrue(cells.check().ok, cartan_type)
            self.assertEqual(IntPoly.from_counts(cells.counts),
                             oracles.batyrev_moreau(cartan_type, crossed, basis, cones),
                             cartan_type)

    def test_batyrev_moreau_on_toroidal_fans_is_the_fibration_formula(self):
        for cartan_type, crossed, basis, fan in CASES:
            cones = [([fan.rays[i] for i in cone], set()) for cone in fan.cones]
            fibre = IntPoly.from_counts(bb_cells(toric.fixed_point_data(fan)).counts)
            self.assertEqual(oracles.batyrev_moreau(cartan_type, crossed, basis, cones),
                             oracles.flag_variety(cartan_type, crossed) * fibre, cartan_type)

    def test_non_smooth_colored_cones_are_refused(self):
        cones = [([(1, 0), (0, 1)], {1, 2}), ([(0, 1), (-1, -1)], {2}),
                 ([(-1, -1), (1, 0)], {1})]
        with self.assertRaisesRegex(ValueError, "not smooth"):
            horospherical.colored_horospherical("A2", {1, 2}, [(1, 0), (0, 1)], cones)

    def test_inconsistent_colors_are_refused(self):
        cones = [([(1, 0), (0, 1)], {1, 2}), ([(0, 1), (-1, -1)], set()),
                 ([(-1, -1), (1, 0)], {1})]
        with self.assertRaisesRegex(ValueError, "different colors"):
            horospherical.colored_horospherical("A2", {1, 2}, [(1, 0), (0, 1)], cones)
