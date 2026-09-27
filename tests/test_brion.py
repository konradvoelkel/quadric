"""equivariant cohomology beyond GKM (Brion's local conditions)"""

import unittest

from bbcells import brion
from bbcells.core import bb_cells
from bbcells.equivariant import expected_dimension
from bbcells.frontends import flag, spherical, toric
from tests._util import doctests_for

load_tests = doctests_for(brion)


class TestComponents(unittest.TestCase):

    def test_inferred_curves_are_the_gkm_edges(self):
        for X in [flag.flag_variety("A2"), flag.flag_variety("B2"), flag.flag_variety("G2"),
                  flag.flag_variety("A3", {2}), toric.fixed_point_data(toric.hirzebruch(2))]:
            components = brion.fixed_components(X)
            self.assertEqual({frozenset(points) for _, points, _ in components},
                             {frozenset((p, q)) for p, q, _ in X.edges}, X.name)
            self.assertEqual({kind for _, _, kind in components}, {"curve"})

    def test_complete_conics_have_planes(self):
        X = spherical.complete_quadrics(3)
        kinds = [kind for _, _, kind in brion.fixed_components(X)]
        self.assertEqual((kinds.count("curve"), kinds.count("plane")), (12, 6))


class TestRing(unittest.TestCase):

    def test_free_module_with_bb_poincare_series(self):
        for X, top in [(spherical.complete_quadrics(3), 3), (flag.flag_variety("B2"), 3)]:
            components = brion.fixed_components(X)
            counts = bb_cells(X).counts
            for d in range(top + 1):
                self.assertEqual(brion.ring_dimension(X, components, d),
                                 expected_dimension(counts, X.rank, d), (X.name, d))

    def test_second_order_condition_is_needed(self):
        # treating the planes like curves (congruences mod chi only) gives too much
        X = spherical.complete_quadrics(3)
        components = brion.fixed_components(X)
        weak = [(chi, points, "curves") for chi, points, kind in components]
        counts = bb_cells(X).counts
        self.assertGreater(brion.ring_dimension(X, weak, 2), expected_dimension(counts, X.rank, 2))


def colours(X, cartan_type):
    """mu_k = pullback of O(1) from P(Sym^2 Lambda^k V) on complete quadrics"""
    from bbcells.rootsystem import RootSystem
    R = RootSystem(cartan_type)
    components = brion.fixed_components(X)
    return [brion.line_bundle_class(X, components, cartan_type,
                                    tuple(-2 * x for x in R.fundamental_weight(k)))
            for k in range(R.rank)]


class TestCharacteristicNumbers(unittest.TestCase):
    """Schubert's characteristic numbers by equivariant localization"""

    def test_conics(self):
        X = spherical.complete_quadrics(3)
        mu, nu = colours(X, "A2")
        for lam in [(3, 7), (11, 2)]:
            self.assertEqual([brion.integrate_monomial(X, [mu, nu], [a, 5 - a], lam)
                              for a in range(5, -1, -1)], [1, 2, 4, 4, 2, 1])
        tangency = {p: tuple(2 * a + 2 * b for a, b in zip(mu[p], nu[p])) for p in X.points}
        self.assertEqual(brion.integrate_monomial(X, [tangency], [5], (5, 13)), 3264)

    def test_quadric_surfaces(self):
        X = spherical.complete_quadrics(4)
        mu, nu, rho = colours(X, "A3")
        lam = (7, 101, 1009)
        self.assertEqual([brion.integrate_monomial(X, [mu, nu], [a, 9 - a], lam)
                          for a in range(9, -1, -1)], [1, 2, 4, 8, 16, 32, 56, 80, 92, 92])
        self.assertEqual(brion.integrate_monomial(X, [rho], [9], lam), 1)
        tangency = {p: tuple(2 * (a + b + c) for a, b, c in zip(mu[p], nu[p], rho[p]))
                    for p in X.points}
        self.assertEqual(brion.integrate_monomial(X, [tangency], [9], lam), 666841088)
        # entries of Schubert's triangle (Brysiewicz-Fevola-Sturmfels, arXiv:2010.10879)
        self.assertEqual(brion.integrate_monomial(X, [mu, nu, rho], [3, 3, 3], lam), 104)
        self.assertEqual(brion.integrate_monomial(X, [mu, nu, rho], [2, 5, 2], lam), 128)

    def test_quadric_threefolds(self):
        # Sturmfels, "3264 questions about symmetric matrices", Question 5
        X = spherical.complete_quadrics(5)
        classes = colours(X, "A4")
        lam = (7, 101, 1009, 10007)
        tangency = {p: tuple(2 * sum(c[p][i] for c in classes) for i in range(4))
                    for p in X.points}
        self.assertEqual(brion.integrate_monomial(X, [tangency], [14], lam), 48942189946470400)
        self.assertEqual(brion.integrate_monomial(X, [classes[1]], [14], lam), 7703)


class TestCharacteristicNumbersBySections(unittest.TestCase):
    """the same numbers without fixed points: De Concini-Procesi's decomposition
    of H^0(X, L^m), Weyl's dimension formula and exact interpolation"""

    def test_agree_with_localization(self):
        from bbcells import oracles
        for n, coefficients in [(3, (1, 0)), (3, (1, 1)), (3, (2, 3)), (4, (1, 2, 1)),
                                (4, (0, 1, 0))]:
            self.assertEqual(oracles.complete_quadrics_degree(n, coefficients),
                             brion.characteristic_number(n, coefficients=coefficients),
                             (n, coefficients))

    def test_quadric_threefolds(self):
        from bbcells import oracles
        self.assertEqual(oracles.complete_quadrics_degree(5, (2, 2, 2, 2)), 48942189946470400)


class TestCanonicalClasses(unittest.TestCase):
    """Goldin-Tolman classes by interpolation over Brion components"""

    def test_agree_with_gkm_flow_up_classes(self):
        from bbcells import equivariant
        from bbcells.polynomial import Poly
        for X in [flag.flag_variety("A2"), flag.flag_variety("B2"), flag.grassmannian(2, 4)]:
            cells = bb_cells(X)
            ours = brion.canonical_classes(X, brion.fixed_components(X), cells.cocharacter)
            gkm = equivariant.flow_up_classes(cells)
            for p in X.points:
                for q in X.points:
                    self.assertTrue((ours[p][q] - gkm[p].get(q, Poly(X.rank))).is_zero(),
                                    (X.name, p, q))

    def test_integral_cohomology_is_unimodular(self):
        for X in [flag.flag_variety("B2"), toric.fixed_point_data(toric.hirzebruch(2))]:
            cells = bb_cells(X)
            order, codim, constants, pairing = brion.integral_cohomology(
                X, brion.fixed_components(X), cells.cocharacter)
            self.assertEqual(sorted(codim.values()).count(1), cells.counts[1])

    def test_beyond_gkm(self):
        # CII(1,2) (ruled surfaces) and BI(1,4) (planes): canonical classes exist
        from bbcells.frontends import symmetric
        for case in [("CII", 1, 2), ("BI", 1, 4)]:
            X = symmetric.complete_symmetric_variety(*case)
            brion.integral_cohomology(X, brion.fixed_components(X), bb_cells(X).cocharacter)

    def test_complete_conics_have_no_canonical_classes(self):
        # in every chamber two plus-cell closures meet cells of their own dimension
        X = spherical.complete_quadrics(3)
        components = brion.fixed_components(X)
        for lam in [(1, 5), (5, 1), (1, -5), (-2, -3)]:
            failures = 0
            for p in X.points:
                try:
                    brion.canonical_classes(X, components, lam, points=[p])
                except ValueError:
                    failures += 1
            self.assertEqual(failures, 2, lam)


class TestVolumeRing(unittest.TestCase):

    def test_complete_quadric_surfaces(self):
        # generated by the colours: Q[mu, nu, rho]/Ann(V), relations in degrees 4, 5, 6
        X = spherical.complete_quadrics(4)
        ring = brion.volume_ring(X, colours(X, "A3"), (7, 101, 1009))
        self.assertEqual(ring["hilbert"], list(bb_cells(X).counts))
        self.assertEqual(ring["generators"], {4: 2, 5: 2, 6: 1})
        self.assertEqual(ring["numbers"][(3, 3, 3)], 104)


if __name__ == "__main__":
    unittest.main()
