"""wonderful varieties from Luna data with non-symmetric satellites (docs/spherical.md, section 11)"""

import unittest
from itertools import combinations

from bbcells.core import bb_cells
from bbcells.frontends import luna, spherical
from tests._util import doctests_for

load_tests = doctests_for(luna)


def to_epsilon(v, n):
    """simple-root coordinates of C_n -> epsilon coordinates"""
    e = [0] * n
    for i, c in enumerate(v):
        if i < n - 1:
            e[i] += c
            e[i + 1] -= c
        else:
            e[i] += 2 * c
    return tuple(e)


def bundle_model(n, normalizer):
    """the tangent weights of P(S) x_Gr P(S), resp. P(Sym^2 S), over Gr(2, 2n)
    at its fixed points, in epsilon coordinates: independent of Luna data"""
    signed = [(i, s) for i in range(n) for s in (1, -1)]
    weight = lambda x: tuple(x[1] * int(k == x[0]) for k in range(n))
    sub = lambda u, v: tuple(a - b for a, b in zip(u, v))
    add = lambda u, v: tuple(a + b for a, b in zip(u, v))
    points = []
    for a, b in combinations(signed, 2):
        base = [sub(weight(y), weight(x)) for x in (a, b) for y in signed if y not in (a, b)]
        if normalizer:
            monomials = [add(weight(a), weight(a)), add(weight(a), weight(b)),
                         add(weight(b), weight(b))]
            for m in monomials:
                points.append(base + [sub(o, m) for o in monomials if o != m])
        else:
            other = {a: b, b: a}
            for x1 in (a, b):
                for x2 in (a, b):
                    points.append(base + [sub(weight(other[x1]), weight(x1)),
                                          sub(weight(other[x2]), weight(x2))])
    return sorted(tuple(sorted(p)) for p in points)


class TestSymplecticPairs(unittest.TestCase):

    def test_bundle_model(self):
        # every tangent weight agrees with the projective bundle over Gr(2, 2n)
        for n in (2, 3, 4):
            for normalizer in (False, True):
                X = luna.symplectic_pair_variety(n, normalizer)
                ours = sorted(tuple(sorted(to_epsilon(w, n) for w in weights))
                              for weights in X.weights)
                self.assertEqual(ours, bundle_model(n, normalizer), (n, normalizer))
                self.assertEqual(len(X), (3 if normalizer else 4) * n * (2 * n - 1))
                self.assertTrue(bb_cells(X).check().ok)

    def test_unique_choice(self):
        # the unknowns: c = 0 on the SL_2/T satellite, c = -1 (resp. -2) on
        # G/(B x Sp_{2n-2}), where the W_L-average vanishes
        for normalizer, expected in [(False, (0, -1)), (True, (-2,))]:
            orbits = luna.symplectic_pair_orbits(3, normalizer, strict=False)
            admissible = luna.decide_normal_weights("C3", orbits)
            self.assertEqual([c for c, _ in admissible], [expected])

    def test_orbit_counts(self):
        # Brion-Peyre: |G/P| |L/H_L| against the BB cells of the orbit closures
        for n in (2, 3):
            for normalizer in (False, True):
                orbits = luna.symplectic_pair_orbits(n, normalizer)
                X = spherical.assemble("C%d" % n, orbits)
                for name, (bb, expected) in spherical.orbit_count_check(
                        "C%d" % n, orbits, X, 2).items():
                    self.assertEqual(bb, expected, (n, normalizer, name))

    def test_so5_mod_gl2(self):
        # n = 2 is GL_2 < SO_5 (C_2 = B_2 exchanges the root lengths): Gr(2, 4) = Q^4
        X = luna.symplectic_pair_variety(2)
        self.assertEqual(bb_cells(X).counts, (1, 3, 5, 6, 5, 3, 1))


if __name__ == "__main__":
    unittest.main()
