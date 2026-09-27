"""
Wonderful varieties from Luna data with non-symmetric satellites (PLAN.md R2,
docs/spherical.md, section 11).

Given the spherical roots, S^p and the satellites H_{L,I} > T (as for
spherical.wonderful_orbits), the normal weights beyond condition (R) are
chi = pr(-gamma) + c zeta with c in c0 + Z. For symmetric varieties c = 0
throughout, but not in general: for Sp_{2n} > GL_1 x Sp_{2n-2} the orbit
O_{gamma_2} = G/(B_{Sp_2} x Sp_{2n-2}) has L_I = G, so pr(-gamma_1) = 0 and
chi = c zeta. Then the point counts of section 7 see only the sign of c, and
decide_normal_weights searches all c in a box instead: a choice is admissible
if the assembled data has no zero weight and passes the checks of the engine
(cocharacter independence, Poincare duality, holomorphic Lefschetz).

The family implemented here (Bravi-Pezzini, arXiv:1109.6777, the entries
GL(1) x Sp(2p) < Sp(2p+2) and its normalizer; for n = 2 it is GL(2) < SO(5)):
    Sigma = {alpha_1, eps_1 + eps_2}, resp. {2 alpha_1, eps_1 + eps_2},
    S^p = {alpha_3, ..., alpha_n}.
Its wonderful variety is P(S) x_Gr P(S), resp. P(Sym^2 S), over the
Grassmannian Gr(2, 2n) with S the tautological bundle: a plane with an
ordered, resp. unordered, pair of lines. The boundary divisors are the
isotropic planes and the coinciding lines. The tests compare every tangent
weight with this model.
"""

from dataclasses import replace
from fractions import Fraction
from itertools import product

from bbcells.frontends import spherical
from bbcells.rootsystem import RootSystem


def decide_normal_weights(cartan_type, orbits, bound=6):
    """all admissible choices of c for the orbits marked BEYOND_R (one
    unknown each, the last normal weight), with |c - c0| <= bound: returns
    [(tuple of c, orbits with these normal weights)]. A single admissible
    choice is the true one if the true c lies in the box.
    >>> orbits = symplectic_pair_orbits(2, strict=False)          # GL_2 < SO_5
    >>> [c for c, _ in decide_normal_weights("C2", orbits)]
    [(Fraction(0, 1), Fraction(-1, 1))]
    """
    from bbcells.core import bb_cells
    R = RootSystem(cartan_type)
    unknowns = []
    for o in orbits:
        if o.note != spherical.BEYOND_R:
            continue
        basis = spherical.invariant_basis(
            R, o.levi, list(o.generators) + list(o.component_reflections), o.component_elements)
        if len(basis) != 1 or not o.normal:
            raise ValueError("orbit %s: %d invariant directions" % (o.name, len(basis)))
        c0 = spherical._lattice_shift(o.normal[-1], basis[0])
        if c0 is None:
            raise ValueError("no c puts the normal weight of %s in the root lattice" % o.name)
        unknowns.append((o.name, basis[0], c0))
    admissible = []
    for steps in product(range(-bound, bound + 1), repeat=len(unknowns)):
        values = {name: c0 + k for (name, _, c0), k in zip(unknowns, steps)}
        zetas = {name: zeta for name, zeta, _ in unknowns}
        candidate = []
        for o in orbits:
            if o.name not in values:
                candidate.append(o)
                continue
            chi = tuple(a + values[o.name] * b for a, b in zip(o.normal[-1], zetas[o.name]))
            if not any(chi) or any(Fraction(x).denominator != 1 for x in chi):
                break
            candidate.append(replace(o, normal=o.normal[:-1] + (tuple(int(x) for x in chi),),
                                     note=DECIDED))
        else:
            X = spherical.assemble(cartan_type, candidate)
            if bb_cells(X).check().ok:
                admissible.append((tuple(values[name] for name, _, _ in unknowns), candidate))
    return admissible


DECIDED = ("normal weight beyond (R) decided by decide_normal_weights (a search over c "
           "against the checks of the engine)")


def symplectic_pair_orbits(n, normalizer=False, strict=True):
    """OrbitData of the wonderful variety of Sp_{2n} > GL_1 x Sp_{2n-2}, or of
    its normalizer, n >= 2 (the normal weights beyond (R) decided unless
    strict=False, which leaves them marked)
    >>> [(o.name, o.normal) for o in symplectic_pair_orbits(3, normalizer=True)]
    [('closed', ((-2, 0, 0), (-1, -2, -1))), ('O1', ((-1, -2, -1),)), ('O2', ((-4, -4, -2),)), ('O12', ())]
    """
    if n < 2:
        raise ValueError("need n >= 2")
    cartan_type = "C%d" % n
    unit = lambda i: tuple(int(k == i) for k in range(n))
    gamma1 = tuple(2 * x for x in unit(0)) if normalizer else unit(0)
    gamma2 = (1,) + (2,) * (n - 2) + (1,) if n > 2 else (1, 1)     # eps_1 + eps_2
    two_eps1 = (2,) * (n - 1) + (1,)
    sp = [unit(i) for i in range(1, n)]                    # Sp_{2n-2} on eps_2, ..., eps_n
    parabolic = tuple(range(2, n))

    def satellite(I):
        if I == (0, 1):                                    # GL_1 x Sp_{2n-2} (N: + s_{2 eps_1})
            data = {"generators": sp}
            if normalizer:
                data["component_reflections"] = [two_eps1]
            return data
        if I == (1,):                                      # G/(B_{Sp_2} x Sp_{2n-2})
            return {"generators": sp, "unipotent_roots": [two_eps1]}
        data = {"generators": [unit(i) for i in parabolic]}   # SL_2/T or SL_2/N(T)
        if normalizer:
            data["component_reflections"] = [unit(0)]
        return data

    orbits = spherical.wonderful_orbits(cartan_type, [gamma1, gamma2], satellite, parabolic,
                                        strict=False)
    if not strict:
        return orbits
    admissible = decide_normal_weights(cartan_type, orbits)
    if len(admissible) != 1:
        raise AssertionError("%d admissible choices of normal weights" % len(admissible))
    return admissible[0][1]


def symplectic_pair_variety(n, normalizer=False):
    """FixedPointData of the wonderful variety of Sp_{2n} > GL_1 x Sp_{2n-2}
    (P(S) x P(S) over Gr(2, 2n)), or of its normalizer (P(Sym^2 S))
    >>> X = symplectic_pair_variety(3)
    >>> from bbcells.core import bb_cells
    >>> X.dim, len(X), bb_cells(X).counts
    (10, 60, (1, 3, 5, 7, 9, 10, 9, 7, 5, 3, 1))
    """
    name = "wonderful Sp_%d/%sGL_1 x Sp_%d%s" % (2 * n, "N(" if normalizer else "",
                                                 2 * n - 2, ")" if normalizer else "")
    return spherical.assemble("C%d" % n, symplectic_pair_orbits(n, normalizer), name=name)
