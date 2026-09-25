import unittest

from bbcells import chevalley
from bbcells.rootsystem import RootSystem
from tests._util import doctests_for

load_tests = doctests_for(chevalley)


class TestChevalley(unittest.TestCase):

    def test_jacobi_identity(self):
        for name in ("A3", "D4", "D5", "E6"):
            self.assertTrue(chevalley.SimplyLaced(name).check_jacobi(samples=400), name)

    def test_lifts_satisfy_the_braid_relations(self):
        """including the folded types F4 (m = 4) and G2 (m = 6)"""
        for name in ("A3", "D4", "E6", "F4", "G2"):
            L = chevalley.Lifts(name)
            R = RootSystem(name)
            roots = L.g.roots
            for i in range(R.rank):
                for j in range(i + 1, R.rank):
                    m = {0: 2, 1: 3, 2: 4, 3: 6}[R.cartan[i][j] * R.cartan[j][i]]
                    for root in roots:
                        left = right = {root: 1}
                        for k in range(m):
                            left = L.apply(i if k % 2 == 0 else j, left)
                            right = L.apply(j if k % 2 == 0 else i, right)
                        self.assertEqual(left, right, (name, i, j, root))

    def test_folded_simple_root_vectors_have_the_folded_weights(self):
        from bbcells.frontends.two_orbit import E6_TO_F4
        L = chevalley.Lifts("F4")
        for i in range(4):
            images = {tuple(sum(a * b for a, b in zip(row, r)) for row in E6_TO_F4)
                      for r in L.simple_vector(i)}
            self.assertEqual(images, {tuple(int(k == i) for k in range(4))})


if __name__ == "__main__":
    unittest.main()
