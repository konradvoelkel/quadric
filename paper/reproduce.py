#!/usr/bin/env python3
"""
Reproduce the numbers of the companion paper (paper/bbcells.tex, PLAN.md P2).

Every number stated in the paper is a *claim* below: a function that
recomputes it with the package and the value printed in the paper.

    python3 paper/reproduce.py            # fast claims (about half a minute)
    python3 paper/reproduce.py --all      # also the slow ones (hours: E7, E8, P^5)
    python3 paper/reproduce.py --list     # list the claims
    python3 paper/reproduce.py comp:large # claims whose id starts with a prefix

The exit status is nonzero if a claim fails. The fast claims are run by
tests/test_paper.py. The random experiments use fixed seeds, so their
counts are reproducible exactly.
"""

import itertools
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bbcells import brion, gkm, h1, oracles, realcells, realtoric  # noqa: E402
from bbcells.core import bb_cells  # noqa: E402
from bbcells.frontends import flag, spherical, symmetric, toric  # noqa: E402
from bbcells.rootsystem import RootSystem  # noqa: E402

CLAIMS = []


def claim(identifier, expected, slow=False):
    """register a claim: the decorated function must return `expected`"""
    def register(function):
        CLAIMS.append((identifier, expected, slow, function))
        return function
    return register


# -- helpers -------------------------------------------------------------------

def colours(X, cartan_type):
    R = RootSystem(cartan_type)
    components = brion.fixed_components(X)
    return [brion.line_bundle_class(X, components, cartan_type,
                                    tuple(-2 * x for x in R.fundamental_weight(k)))
            for k in range(R.rank)]


def integral(X, classes, exponents, lam):
    return int(brion.integrate_monomial(X, classes, exponents, lam))


def tangency(X, classes, lam):
    """int (2 sum mu_k)^dim"""
    tangent = {p: tuple(2 * sum(c[p][i] for c in classes) for i in range(X.rank))
               for p in X.points}
    return integral(X, [tangent], [X.dim], lam)


def symmetric_size(*case):
    X = symmetric.complete_symmetric_variety(*case)
    return (X.dim, len(X))


# -- Section 3: spherical varieties ----------------------------------------------

@claim("ex:quadrics/chi", [3, 12, 66, 450, 3690])
def _():
    return [len(spherical.complete_quadrics(n)) for n in range(2, 7)]


SYMMETRIC_TABLE = {
    ("AIII", 2, 2): (8, 39), ("AIII", 3, 3): (18, 1180), ("CI", 3): (12, 148),
    ("CII", 2, 2): (16, 123), ("BI", 3, 4): (12, 147), ("DI", 4, 4): (16, 747),
    ("G",): (8, 27), ("EIV",): (26, 270), ("EIII",): (32, 2619),
}
SYMMETRIC_TABLE_SLOW = {
    ("CI", 4): (20, 1624), ("DIII", 6): (30, 4576), ("FI",): (28, 4788),
    ("AIII", 3, 4): (24, 8015), ("DIII", 7): (42, 55168),
}

for _case, _value in SYMMETRIC_TABLE.items():
    claim("comp:symmetric/" + "".join(map(str, _case)), _value)(
        lambda case=_case: symmetric_size(*case))
for _case, _value in SYMMETRIC_TABLE_SLOW.items():
    claim("comp:symmetric/" + "".join(map(str, _case)), _value, slow=True)(
        lambda case=_case: symmetric_size(*case))


# -- Section 4: certificates -------------------------------------------------------

def certified(*case):
    D = symmetric.diagram(*case)
    result = spherical.certify_by_closures(D.R.name, D.orbits(strict=False))
    return all(v == [0] for v in result.values()) and bool(result)


for _case, _slow in [(("AIII", 2, 3), False), (("AIII", 2, 4), False), (("AIII", 2, 5), False),
                     (("DIII", 5), False), (("EIII",), False), (("AIII", 3, 4), False),
                     (("AIII", 3, 5), True), (("AIII", 3, 6), True), (("AIII", 4, 5), True),
                     (("DIII", 7), True)]:
    claim("comp:certified/" + "".join(map(str, _case)), True, slow=_slow)(
        lambda case=_case: certified(*case))


def open_orbits_match(case):
    """Lemma (open orbits): |O_J| from the fixed points = Brion-Peyre, every orbit"""
    D = symmetric.diagram(*case)
    X = D.fixed_point_data()
    rng = random.Random(1)
    for o in D.orbits(strict=False):
        lam = [rng.randrange(1, 10 ** 6) for _ in range(D.R.rank)]
        got = spherical.open_orbit_count(spherical.orbit_closure(X, o.roots), lam)
        expected = (oracles.from_degrees(D.R.degrees, oracles.levi_degrees(D.R.cartan, set(o.levi)))
                    * spherical.satellite_point_count(D.R.name, o))
        if got != expected:
            return False
    return True


for _case in [("AI", 3), ("AI", 4), ("AIII", 2, 2), ("AIII", 2, 3), ("CI", 3), ("G",),
              ("BI", 3, 4), ("DIII", 5), ("EIII",), ("FI",), ("AIII", 3, 4)]:
    claim("lem:openorbit/" + "".join(map(str, _case)), True,
          slow=_case in (("EIII",), ("FI",), ("AIII", 3, 4)))(
        lambda case=_case: open_orbits_match(case))


def opposition_decides_all(case):
    """Theorem (opposition): its hypotheses hold for every unknown"""
    D = symmetric.diagram(*case)
    orbits = D.orbits(strict=False)
    unknowns = {(o.name, g) for o in orbits if o.note == spherical.BEYOND_R for g in o.normal_roots}
    decided = spherical.certify_by_symmetry(D.R.name, orbits, D.spherical_roots)
    return bool(unknowns) and set(decided) == unknowns


_opposition = [("AIII", p, q) for p in range(2, 7) for q in range(p + 1, 11) if p + q <= 12] + \
              [("DIII", 5), ("DIII", 7), ("DIII", 9), ("DIII", 11), ("EIII",)]
for _case in _opposition:
    claim("comp:opposition/" + "".join(map(str, _case)), True,
          slow=sum(_case[1:]) > 9 or _case == ("DIII", 11))(
        lambda case=_case: opposition_decides_all(case))


# -- Section 5: cell counts without listing fixed points ---------------------------

LARGE = {"EII": (40, 110916), "EI": (42, 370170), "EVII": (54, 23464),
         "EVI": (64, 758079), "EV": (70, 28373976), "EIX": (112, 7445880)}


_COUNTS = {}


def cell_counts(kind):
    if kind not in _COUNTS:
        _COUNTS[kind] = symmetric.cell_counts(kind, processes=os.cpu_count() or 1)
    return _COUNTS[kind]


def large(kind):
    counts = cell_counts(kind)
    return (len(counts) - 1, sum(counts))


for _kind, _value in LARGE.items():
    claim("comp:large/" + _kind, _value, slow=_kind != "EVII")(lambda kind=_kind: large(kind))


@claim("comp:eviii", (128, 9297296775, True, [1, 8, 36, 126, 372, 970, 2286, 4952, 9984, 18924],
                      241169479), slow=True)
def _():
    counts = symmetric.split_cell_counts("E8", processes=os.cpu_count() or 1)
    return (len(counts) - 1, sum(counts), counts == counts[::-1], list(counts[:10]),
            counts[len(counts) // 2])


@claim("comp:eviii/orbits", 9297296775, slow=True)
def _():
    # the Euler characteristic as sum |W| / |W_H| over the orbits with fixed points
    D = symmetric.diagram("EVIII")
    R = RootSystem("E8")
    total = 0
    for o in D.orbits(strict=False):
        reflections = list(o.generators) + list(o.component_reflections)
        phi = spherical.root_subsystem(R, reflections) if reflections else set()
        degrees, _ = spherical._subsystem_degrees(R, phi) if phi else ([], 0)
        order = math.prod(degrees)
        if o.component_elements:
            order *= spherical._component_group_order(R, reflections, o.component_elements)
        total += math.prod(R.degrees) // order
    return total


@claim("comp:split/E7", True, slow=True)
def _():
    return symmetric.split_cell_counts("E7") == cell_counts("EV")


@claim("comp:large/EVII-counts", [1, 3, 7, 12, 19, 29, 44, 63, 87, 117, 155, 198, 248, 303, 366,
                                   432, 503, 575, 652, 724, 795, 858, 920, 970, 1013, 1041,
                                   1063, 1068])
def _():
    return list(cell_counts("EVII")[:28])


# -- Section 5: finite covers G/G^theta ----------------------------------------------

def split_order(N, degrees):
    """|G(F_q)| = q^N prod (q^d - 1) of a split connected reductive group"""
    from bbcells.algebra import IntPoly
    result = IntPoly((0, 1)) ** N
    for d in degrees:
        result = result * (IntPoly((0, 1)) ** d - IntPoly((1,)))
    return result


def cover(kind, *parameters):
    """(index, maximal cones, chi, open orbit count) of the toroidal G/G^theta-embedding
    with the refined orthant"""
    from bbcells.frontends import toroidal
    D = symmetric.diagram(kind, *parameters)
    L = D.lattice()
    rows = toroidal.lattice_in_sigma(D.R.name, D.spherical_roots, L)
    r = len(D.spherical_roots)
    fan = toroidal.resolve(toroidal.orthant(r, rows), rows)
    X = toroidal.toroidal_variety(D.R.name, D.spherical_roots, D.satellite, fan,
                                  parabolic=tuple(sorted(D.black)), lattice=L)
    cells = bb_cells(X)
    if not cells.check().ok:
        raise AssertionError("the checks fail")
    return (toroidal.lattice_index(rows, range(r)), len(fan), len(X),
            spherical.open_orbit_count(X, cells.cocharacter))


_SL = lambda n: split_order(n * (n - 1) // 2, range(2, n + 1))
_GL = lambda n: split_order(n * (n - 1) // 2, range(1, n + 1))
_SP = lambda n: split_order(n * n, range(2, 2 * n + 1, 2))
_E6 = split_order(36, [2, 5, 6, 8, 9, 12])
for _case, _G, _K, _index, _cones, _chi, _slow in [
        (("AI", 3), _SL(3), _SP(1), 3, 2, 18, False),
        (("AI", 4), _SL(4), split_order(2, [2, 2]), 4, 4, 180, False),
        (("AI", 5), _SL(5), _SP(2), 5, 12, 3120, False),
        (("AII", 2), _SL(4), _SP(2), 2, 1, 6, False),
        (("AIII", 2, 2), _SL(4), split_order(2, [1, 2, 2]), 2, 1, 54, False),
        (("CI", 2), _SP(2), _GL(2), 2, 1, 24, False),
        (("CI", 3), _SP(3), _GL(3), 2, 1, 200, False),
        (("CII", 2, 2), _SP(4), _SP(2) * _SP(2), 2, 1, 150, False),
        (("DIII", 4), split_order(12, [2, 4, 4, 6]), _GL(4), 2, 1, 104, False),
        (("EIV",), _E6, split_order(24, [2, 6, 8, 12]), 3, 2, 540, False),
        (("EVII",), split_order(63, [2, 6, 8, 10, 12, 14, 18]), _E6 * _GL(1), 2, 1, 31808,
         True)]:
    def _check(case=_case, G=_G, K=_K):
        index, cones, chi, count = cover(*case)
        return index, cones, chi, count * K == G
    claim("comp:covers/" + "".join(map(str, _case)), (_index, _cones, _chi, True),
          slow=_slow)(_check)


# -- Section 4.5: a non-symmetric family --------------------------------------------

def symplectic_pairs(n, normalizer):
    """(the admissible c, chi, the Betti numbers) of the wonderful variety of
    Sp_2n > GL_1 x Sp_2n-2 (or its normalizer) from Luna data"""
    from bbcells.frontends import luna
    orbits = luna.symplectic_pair_orbits(n, normalizer, strict=False)
    admissible = luna.decide_normal_weights("C%d" % n, orbits)
    X = spherical.assemble("C%d" % n, admissible[0][1])
    return [tuple(int(x) for x in c) for c, _ in admissible], len(X), list(bb_cells(X).counts)


for _n, _normalizer, _value in [
        (3, False, ([(0, -1)], 60, [1, 3, 5, 7, 9, 10, 9, 7, 5, 3, 1])),
        (3, True, ([(-2,)], 45, [1, 2, 4, 5, 7, 7, 7, 5, 4, 2, 1])),
        (4, False, ([(0, -1)], 112, [1, 3, 5, 7, 9, 11, 13, 14, 13, 11, 9, 7, 5, 3, 1])),
        (4, True, ([(-2,)], 84, [1, 2, 4, 5, 7, 8, 10, 10, 10, 8, 7, 5, 4, 2, 1]))]:
    claim("comp:nonsymmetric/C%d%s" % (_n, "N" if _normalizer else ""), _value)(
        lambda n=_n, normalizer=_normalizer: symplectic_pairs(n, normalizer))


# -- Section 6: equivariant cohomology beyond GKM ------------------------------------

@claim("lem:components/conics", (12, 6))
def _():
    kinds = [k for _, _, k in brion.fixed_components(spherical.complete_quadrics(3))]
    return (kinds.count("curve"), kinds.count("plane"))


@claim("lem:components/surfaces", (153, 48))
def _():
    kinds = [k for _, _, k in brion.fixed_components(spherical.complete_quadrics(4))]
    return (kinds.count("curve"), kinds.count("plane"))


@claim("comp:characteristic/conics", ([1, 2, 4, 4, 2, 1], 3264))
def _():
    X = spherical.complete_quadrics(3)
    mu, nu = colours(X, "A2")
    row = [integral(X, [mu, nu], [a, 5 - a], (3, 7)) for a in range(5, -1, -1)]
    return (row, tangency(X, [mu, nu], (5, 13)))


@claim("comp:characteristic/surfaces",
       ([1, 2, 4, 8, 16, 32, 56, 80, 92, 92], 1, 666841088, 104, 128))
def _():
    X = spherical.complete_quadrics(4)
    mu, nu, rho = colours(X, "A3")
    lam = (7, 101, 1009)
    row = [integral(X, [mu, nu], [a, 9 - a], lam) for a in range(9, -1, -1)]
    return (row, integral(X, [rho], [9], lam), tangency(X, [mu, nu, rho], lam),
            integral(X, [mu, nu, rho], [3, 3, 3], lam), integral(X, [mu, nu, rho], [2, 5, 2], lam))


@claim("comp:characteristic/P4", (48942189946470400, 7703), slow=True)
def _():
    X = spherical.complete_quadrics(5)
    classes = colours(X, "A4")
    lam = (7, 101, 1009, 10007)
    return (tangency(X, classes, lam), integral(X, [classes[1]], [14], lam))


def ml_degrees(n, top):
    X = spherical.complete_quadrics(n)
    classes = colours(X, "A%d" % (n - 1))
    lam = tuple(7 ** k + 3 * k for k in range(n - 1))
    N = n * (n + 1) // 2
    return [integral(X, [classes[0], classes[-1]], [N - d, d - 1], lam) for d in range(1, top + 1)]


@claim("comp:characteristic/phi4", [1, 3, 9, 17, 21, 21, 17, 9, 3, 1])
def _():
    return ml_degrees(4, 10)


@claim("comp:characteristic/phi5", [1, 4, 16, 44, 86, 137, 188, 212], slow=True)
def _():
    return ml_degrees(5, 8)


@claim("comp:characteristic/phi6", [1, 5, 25, 90, 240, 528, 1016], slow=True)
def _():
    return ml_degrees(6, 7)


@claim("comp:characteristic/P5",
       (1810718299257984458113941504, [1, 803128, 61520094, 803128, 1]), slow=True)
def _():
    X = spherical.complete_quadrics(6)
    classes = colours(X, "A5")
    lam = tuple(7 ** k + 3 * k for k in range(5))
    return (tangency(X, classes, lam), [integral(X, [c], [20], lam) for c in classes])


@claim("comp:characteristic/sections", (3264, 666841088))
def _():
    # the same numbers without fixed points (De Concini-Procesi, Weyl, interpolation)
    return tuple(oracles.complete_quadrics_degree(n, (2,) * (n - 1)) for n in (3, 4))


@claim("comp:characteristic/P4-sections", 48942189946470400, slow=True)
def _():
    return oracles.complete_quadrics_degree(5, (2,) * 4)


@claim("comp:characteristic/P5-sections", 1810718299257984458113941504, slow=True)
def _():
    return oracles.complete_quadrics_degree(6, (2,) * 5, processes=os.cpu_count() or 1)


@claim("comp:nocanonical/conics", (12, {2}))
def _():
    X = spherical.complete_quadrics(3)
    components = brion.fixed_components(X)
    directions = set()
    for weights in X.weights:
        for w in weights:
            g = math.gcd(*w)
            v = tuple(c // g for c in w)
            directions.add(max(v, tuple(-c for c in v)))
    # one cocharacter in each chamber of the line arrangement {<lam, w> = 0}
    angles = sorted({a % (2 * math.pi) for w in directions
                     for a in (math.atan2(-w[0], w[1]), math.atan2(-w[0], w[1]) + math.pi)})
    failures = set()
    chambers = 0
    for a, b in zip(angles, angles[1:] + [angles[0] + 2 * math.pi]):
        t = (a + b) / 2
        lam = (round(1000 * math.cos(t)), round(1000 * math.sin(t)))
        chambers += 1
        count = 0
        for p in X.points:
            try:
                brion.canonical_classes(X, components, lam, points=[p])
            except ValueError:
                count += 1
        failures.add(count)
    return (chambers, failures)


@claim("comp:nocanonical/exist", True)
def _():
    for case in [("AIII", 1, 2), ("BI", 1, 4), ("CII", 1, 2)]:
        X = symmetric.complete_symmetric_variety(*case)
        brion.canonical_classes(X, brion.fixed_components(X), bb_cells(X).cocharacter)
    return True


@claim("comp:divisors/conics", ([1, 2, 3, 3, 2, 1], {3: 1, 4: 1}))
def _():
    X = spherical.complete_quadrics(3)
    ring = brion.volume_ring(X, colours(X, "A2"), (3, 7))
    return (ring["hilbert"], ring["generators"])


@claim("comp:divisors/surfaces", ([1, 3, 6, 10, 13, 13, 10, 6, 3, 1], {4: 2, 5: 2, 6: 1}))
def _():
    X = spherical.complete_quadrics(4)
    ring = brion.volume_ring(X, colours(X, "A3"), (7, 101, 1009))
    return (ring["hilbert"], ring["generators"])


@claim("prop:notgenerated/betti", [1, 4, 10, 21, 36, 53, 65, 70])
def _():
    return list(bb_cells(spherical.complete_quadrics(5)).counts[:8])


@claim("prop:notgenerated/hilbert", [1, 4, 10, 20, 35, 52, 65, 70], slow=True)
def _():
    X = spherical.complete_quadrics(5)
    ring = brion.volume_ring(X, colours(X, "A4"), tuple(7 ** k + 3 * k for k in range(4)))
    return ring["hilbert"][:8]


@claim("comp:generators/P4", ([1, 4, 10, 21, 36, 53, 65, 70, 65, 53, 36, 21, 10, 4, 1],
                               {4: 3, 5: 1, 6: 6}), slow=True)
def _():
    # the colours and c_3(T_X) generate H^*(X; Q) for complete quadrics in P^4
    from fractions import Fraction
    X = spherical.complete_quadrics(5)
    classes = colours(X, "A4")
    lam = tuple(7 ** k + 3 * k for k in range(4))
    pair = lambda w: sum(Fraction(a) * b for a, b in zip(w, lam))

    def e3(xs):
        e = [Fraction(1), Fraction(0), Fraction(0), Fraction(0)]
        for x in xs:
            for j in (3, 2, 1):
                e[j] += e[j - 1] * x
        return e[3]
    values = {p: [pair(c[p]) for c in classes] + [e3([pair(w) for w in ws])]
              for p, ws in zip(X.points, X.weights)}
    ring = brion.subalgebra(X, values, [1, 1, 1, 1, 3], lam)
    return (ring["hilbert"], ring["generators"])


# -- Section 7: real points ---------------------------------------------------------

def random_fan(rng, dim, steps):
    if dim == 2:
        fan = rng.choice([toric.projective_space(2), toric.hirzebruch(rng.randrange(4))])
    elif dim == 3:
        fan = rng.choice([toric.projective_space(3),
                          toric.product(toric.projective_space(1), toric.projective_space(2)),
                          toric.product(toric.hirzebruch(rng.randrange(3)),
                                        toric.projective_space(1))])
    else:
        fan = rng.choice([toric.projective_space(4),
                          toric.product(toric.projective_space(2), toric.projective_space(2)),
                          toric.product(toric.hirzebruch(1), toric.hirzebruch(2)),
                          toric.product(toric.projective_space(1), toric.projective_space(3))])
    for _ in range(steps):
        c = rng.choice(fan.cones)
        k = rng.randrange(2, dim + 1)
        fan = toric.star_subdivision(fan, tuple(rng.sample(list(c), k)))
    return fan


def graded_sweep(seed, dim, trials):
    """(graded decompositions, nonzero incidences, mismatches)"""
    rng = random.Random(seed)
    tested = incidences = mismatches = 0
    for _ in range(trials):
        fan = random_fan(rng, dim, rng.randrange(0, 3))
        data = toric.fixed_point_data(fan)
        for _ in range(40):
            lam = [rng.randrange(-60, 61) for _ in range(dim)]
            try:
                cells = bb_cells(data, lam)
            except ValueError:
                continue
            if not gkm.is_graded(cells):
                continue
            exact = {k: abs(v) for k, v in
                     realtoric.RealToricComplex(fan, lam).incidences().items()}
            predicted = {k: mag for k, (m, mag) in realcells.gkm_incidences(cells).items() if mag}
            tested += 1
            incidences += len(exact)
            mismatches += exact != predicted
            break
    return (tested, incidences, mismatches)


@claim("obs:entrywise/dim2", (14, 9, 0), slow=True)
def _():
    return graded_sweep(5, 2, 25)


@claim("obs:entrywise/dim3", (42, 70, 0), slow=True)
def _():
    a, b = graded_sweep(5, 3, 25), graded_sweep(11, 3, 80)
    return tuple(x + y for x, y in zip(a, b))


@claim("obs:entrywise/dim4", (40, 151, 0), slow=True)
def _():
    a, b = graded_sweep(5, 4, 25), graded_sweep(11, 4, 80)
    return tuple(x + y for x, y in zip(a, b))


def rp_betti(q):
    return [1] + [0] * (q - 1) + ([1] if q % 2 else [0])


def beyond_gkm(case, expected, chambers, seed=2026, bound=90):
    X = brion.with_invariant_curves(symmetric.complete_symmetric_variety(*case))
    expected = tuple(expected) + (0,) * (X.dim + 1 - len(expected))
    rng = random.Random(seed)
    good = 0
    while chambers:
        lam = tuple(rng.randrange(-bound, bound + 1) for _ in range(X.rank))
        try:
            cells = bb_cells(X, lam)
        except ValueError:
            continue
        good += h1.brion_prediction(cells) == {expected}
        chambers -= 1
    return good


def product(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


for _q in (2, 3, 4):
    claim("comp:beyondgkm/AIII1%d" % _q, 12, slow=_q == 4)(
        lambda q=_q: beyond_gkm(("AIII", 1, q), product(rp_betti(q), rp_betti(q)), 12))
for _n, _slow in ((2, False), (3, True)):
    claim("comp:beyondgkm/CII1%d" % _n, 12, slow=_slow)(
        lambda n=_n: beyond_gkm(("CII", 1, n),
                                oracles.real_grassmannian_rational_poincare(2, 2 * n + 2).coefficients,
                                12))


@claim("comp:beyondgkm/FII", 12, slow=True)
def _():
    F = flag.flag_variety("E6", {1})
    cells = bb_cells(F)
    dims = {p: cells.dim_of(p) for p in F.points}
    magnitudes = {k: v[1] for k, v in realcells.gkm_incidences(cells).items()}
    (expected,) = {tuple(realcells.cellular_rational_betti(dims, s))
                   for s in realcells.sign_choices(dims, magnitudes)}
    return beyond_gkm(("FII",), expected, 12)


@claim("obs:deflection/surfaces", (144, 0, 6), slow=True)
def _():
    rng = random.Random(7)
    ok = bad = skipped = 0
    for _ in range(30):
        steps = rng.randrange(1, 6)
        fan = toric.projective_space(2) if rng.random() < 0.5 else toric.hirzebruch(rng.randrange(4))
        for _ in range(steps):
            fan = toric.star_subdivision(fan, rng.choice(fan.cones))
        oracle = tuple(oracles.real_toric_rational_betti(fan.rays, fan.cones))
        data = toric.fixed_point_data(fan)
        for _ in range(5):
            lam = (rng.randrange(-40, 41), rng.randrange(-40, 41))
            try:
                prediction = h1.surface_prediction(bb_cells(data, lam))
            except ValueError:
                skipped += 1
                continue
            if prediction == {oracle}:
                ok += 1
            else:
                bad += 1
    return (ok, bad, skipped)


@claim("sec:nongraded/576", (576, 0), slow=True)
def _():
    tried = graded = 0
    for case in [("G",), ("CI", 2), ("AIII", 2, 2), ("BI", 2, 3), ("AII", 3), ("CII", 2, 2),
                 ("DI", 2, 4), ("AIII", 2, 3), ("CI", 3), ("AI", 4)]:
        X = brion.with_invariant_curves(symmetric.complete_symmetric_variety(*case))
        rng = random.Random(3)
        for _ in range(60):
            lam = tuple(rng.randrange(-200, 201) for _ in range(X.rank))
            try:
                cells = bb_cells(X, lam)
            except ValueError:
                continue
            tried += 1
            dims = {p: cells.dim_of(p) for p in X.points}
            graded += all(dims[x] > dims[y] for x, y, _ in h1.curve_parities(cells))
    return (tried, graded)


def conics_correction(lam):
    """the characterization of Observation 7.7 at one cocharacter"""
    X = brion.with_invariant_curves(spherical.complete_quadrics(3))
    cells = bb_cells(X, lam)
    dims = {p: cells.dim_of(p) for p in X.points}
    parities = h1.curve_parities(cells)
    equal = [(x, y, m) for x, y, m in parities if dims[x] == dims[y]]
    tops = [x for x, y, m in equal if m % 2 == 0]
    bottoms = [y for x, y, m in equal if m % 2]
    if len(equal) != 2 or len(tops) != 1 or len(bottoms) != 1:
        return False
    adjacent = [(x, y, m) for x, y, m in parities if dims[x] == dims[y] + 1]
    predicted = {(x, y) for x, y, m in adjacent if m % 2 == 0}

    def betti(support):
        try:
            signed, _ = realcells.signs_from_square_zero(dims, {k: 2 for k in support})
        except ValueError:
            return None
        return tuple(realcells.cellular_rational_betti(dims, signed))

    target = (1, 0, 0, 0, 0, 1)
    if betti(predicted) is not None:
        return False
    # all supports at distance one from the prediction
    good = [k for k in predicted if betti(predicted - {k}) == target]
    good += [k for x, y, m in adjacent for k in [(x, y)]
             if k not in predicted and betti(predicted | {k}) == target]
    return good == [(tops[0], bottoms[0])]


@claim("obs:conics/one", True)
def _():
    return conics_correction((1, 5))


@claim("obs:conics/21", 21, slow=True)
def _():
    lams = [(1, 5), (5, 1), (-40, -39), (2, 3), (1, -5), (7, -3), (-1, 5), (3, 7), (11, -4), (-9, 2)]
    rng = random.Random(4)
    lams += [(rng.randrange(-50, 51), rng.randrange(-50, 51)) for _ in range(12)]
    X = spherical.complete_quadrics(3)
    count = 0
    for lam in lams:
        try:
            bb_cells(X, lam)
        except ValueError:
            continue
        count += conics_correction(lam)
    return count


# -- driver ------------------------------------------------------------------------

def run(selected):
    failures = 0
    for identifier, expected, slow, function in selected:
        start = time.time()
        try:
            value = function()
        except Exception as error:          # report and continue
            value = "error: %s" % error
        ok = value == expected
        failures += not ok
        print("%-4s %-36s %7.1fs  %s" % ("ok" if ok else "FAIL", identifier, time.time() - start,
                                         "" if ok else "got %r, paper says %r" % (value, expected)),
              flush=True)
    return failures


def main(arguments):
    if "--list" in arguments:
        for identifier, _, slow, _ in CLAIMS:
            print(identifier, "(slow)" if slow else "")
        return 0
    prefixes = [a for a in arguments if not a.startswith("--")]
    everything = "--all" in arguments or bool(prefixes)
    selected = [c for c in CLAIMS if (everything or not c[2])
                and (not prefixes or any(c[0].startswith(p) for p in prefixes))]
    return 1 if run(selected) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
