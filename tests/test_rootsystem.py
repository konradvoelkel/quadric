import unittest

from bbcells import rootsystem
from bbcells.oracles import q_integer
from bbcells.rootsystem import RootSystem
from tests._util import doctests_for

load_tests = doctests_for(rootsystem)

# |Phi^+| and |W| from the standard tables
TABLE = {"A1": (1, 2), "A4": (10, 120), "B2": (4, 8), "B3": (9, 48), "C3": (9, 48),
         "C4": (16, 384), "D4": (12, 192), "D5": (20, 1920), "G2": (6, 12),
         "F4": (24, 1152), "E6": (36, 51840), "E7": (63, 2903040),
         "E8": (120, 696729600), "B1": (1, 2), "D2": (2, 4), "D3": (6, 24)}


class TestRootSystem(unittest.TestCase):

    def test_tables(self):
        for name, (positive, order) in TABLE.items():
            R = RootSystem(name)
            self.assertEqual(len(R.positive_roots), positive, name)
            self.assertEqual(R.weyl_group_order, order, name)
            self.assertEqual(sum(d - 1 for d in R.degrees), positive, name)

    def test_weyl_group_via_orbit_of_rho(self):
        """|W| and sum_w q^l(w) = prod [d_i]_q by enumerating W.rho (rank <= 6)"""
        for name in ("A1", "A3", "A5", "B2", "B4", "C3", "D4", "D5", "G2", "F4", "E6"):
            R = RootSystem(name)
            orbit = R.orbit(set())
            self.assertEqual(len(orbit), R.weyl_group_order, name)
            counts = {}
            for _, word in orbit:
                counts[len(word)] = counts.get(len(word), 0) + 1
            expected = q_integer(1)
            for d in R.degrees:
                expected = expected * q_integer(d)
            self.assertEqual(tuple(counts[k] for k in range(len(counts))),
                             expected.coefficients, name)

    def test_reflections_preserve_the_root_system_and_form(self):
        for name in ("B3", "C3", "F4", "G2", "D4", "E6"):
            R = RootSystem(name)
            roots = set(R.positive_roots) | {tuple(-c for c in b) for b in R.positive_roots}
            for i in range(R.rank):
                for beta in R.positive_roots:
                    image = R.reflect(i, beta)
                    self.assertIn(image, roots)
                    self.assertEqual(R.inner(image, image), R.inner(beta, beta))

    def test_cartan_matrix_convention(self):
        # Bourbaki: in B_n the last root is short, in C_n long; G2: alpha_1 short
        self.assertEqual(RootSystem("B3").cartan[2][1], -2)
        self.assertEqual(RootSystem("C3").cartan[1][2], -2)
        # G2 with alpha_1 short: <alpha_1^vee, alpha_2> = 2(a1, a2)/(a1, a1) = -3
        self.assertEqual(RootSystem("G2").cartan, ((2, -3), (-1, 2)))
        for name in ("B4", "C4", "F4", "G2", "E7"):
            R = RootSystem(name)
            for i in range(R.rank):
                simple = tuple(int(i == j) for j in range(R.rank))
                for j in range(R.rank):
                    other = tuple(int(k == j) for k in range(R.rank))
                    self.assertEqual(R.coroot_pairing(simple, other), R.cartan[i][j])

    def test_highest_root(self):
        self.assertEqual(RootSystem("E8").positive_roots[-1], (2, 3, 4, 6, 5, 4, 3, 2))
        self.assertEqual(RootSystem("G2").positive_roots[-1], (3, 2))


if __name__ == "__main__":
    unittest.main()
