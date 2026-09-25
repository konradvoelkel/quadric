"""
Smooth complete toroidal varieties over a wonderful model (docs/S6d.md, section 8).

Let X_w be the wonderful variety of G/H (H = N_G(H)) with spherical roots
gamma_1, ..., gamma_r, a basis of the weight lattice M = Z Sigma. The valuation
cone is the negative orthant V = {n : <gamma_i, n> <= 0} in N = Hom(M, Z), and
the G-orbit O_J of X_w corresponds to the face F_J = {n in V : <gamma, n> = 0
for gamma in J}. A smooth complete toroidal G/H-embedding X is given by a
smooth fan subdividing V; it maps to X_w, and the orbit O_tau of a cone tau
whose relative interior lies in that of F_J maps onto O_J with fibres tori
of dimension dim F_J - dim tau.

Hence the T-fixed points of X are the pairs (x in O_J^T, tau) with tau a cone
of dimension dim F_J = |Sigma - J| inside F_J. Near x the toroidal morphism is
the toric modification, given by the fan, of the normal slice to O_J. T acts on
that slice through phi_J: M -> X(T), -gamma -> (normal weight of D_gamma at
x_J), i.e. phi_J = the W_L-average (the projection used by wonderful_orbits).
So the tangent weights at (w.x_J, tau) are w(Phi - Phi_H) and w(phi_J(m)) for
the dual basis m of tau, the toric rule of C2 transported by phi_J.

Cones are given by their primitive generators in the coordinates
n = (<gamma_1, n>, ..., <gamma_r, n>).
"""

from fractions import Fraction
from itertools import combinations

from bbcells.frontends import spherical
from bbcells.linalg import determinant, inverse
from bbcells.rootsystem import RootSystem


def orthant(r):
    """the fan of the wonderful variety itself: the negative orthant"""
    return [tuple(tuple(-int(i == j) for i in range(r)) for j in range(r))]


def star_subdivision(cones, face):
    """star subdivision of a simplicial fan (maximal cones as tuples of rays)
    at a cone `face` (a tuple of rays of the fan): every maximal cone
    containing the face is replaced by the cones obtained by swapping one ray
    of the face for the sum of its rays
    >>> star_subdivision(orthant(2), ((-1, 0), (0, -1)))
    [((-1, -1), (0, -1)), ((-1, 0), (-1, -1))]
    """
    face = [tuple(v) for v in face]
    new_ray = tuple(sum(c) for c in zip(*face))
    result = []
    for cone in cones:
        cone = [tuple(v) for v in cone]
        if all(v in cone for v in face):
            for v in face:
                result.append(tuple(new_ray if u == v else u for u in cone))
        else:
            result.append(tuple(cone))
    return result


def _faces(cones):
    faces = set()
    for cone in cones:
        cone = tuple(sorted(tuple(v) for v in cone))
        for k in range(len(cone) + 1):
            faces.update(combinations(cone, k))
    return faces


def check_fan(cones, r):
    """validate a smooth fan with support the negative orthant in Z^r"""
    cones = [tuple(tuple(v) for v in cone) for cone in cones]
    for cone in cones:
        if len(cone) != r:
            raise ValueError("maximal cones must have dimension %d: %r" % (r, cone))
        if any(c > 0 for v in cone for c in v):
            raise ValueError("cone %r leaves the valuation cone" % (cone,))
        if abs(determinant([list(v) for v in cone])) != 1:
            raise ValueError("cone %r is not smooth" % (cone,))
    # pseudomanifold test: every facet lies in a wall of V or in exactly two cones,
    # on opposite sides
    facets = {}
    for cone in cones:
        for i in range(r):
            facet = tuple(sorted(cone[:i] + cone[i + 1:]))
            facets.setdefault(facet, []).append(cone[i])
    for facet, opposite in facets.items():
        wall = [j for j in range(r) if all(v[j] == 0 for v in facet)]
        if wall:
            if len(opposite) != 1:
                raise ValueError("facet %r on the boundary lies in %d cones" % (facet, len(opposite)))
            continue
        if len(opposite) != 2:
            raise ValueError("facet %r lies in %d cones" % (facet, len(opposite)))
        sides = [determinant([list(v) for v in facet] + [list(u)]) for u in opposite]
        if sides[0] * sides[1] >= 0:
            raise ValueError("cones overlap along %r" % (facet,))
    return cones


def toroidal_orbits(cartan_type, spherical_roots, satellite, cones, parabolic=(), strict=True):
    """OrbitData of the toroidal variety with the given fan (maximal cones) over
    the wonderful variety of (spherical_roots, parabolic, satellite)"""
    R = RootSystem(cartan_type)
    r = len(spherical_roots)
    cones = check_fan(cones, r)
    faces = _faces(cones)
    base = spherical.wonderful_orbits(cartan_type, spherical_roots, satellite, parabolic,
                                      strict=strict)
    sigma = [tuple(g) for g in spherical_roots]
    orbits = []
    for orbit in base:
        free = list(orbit.normal_roots)                   # coordinates of F_J
        J = set(orbit.roots)
        levi = set(parabolic).union(*({i for i, c in enumerate(sigma[j]) if c} for j in J))
        for tau in sorted(faces):
            if len(tau) != len(free) or any(v[j] for v in tau for j in J):
                continue
            U = [[Fraction(v[j]) for j in free] for v in tau]
            if not free:
                dual = []
            else:
                Uinv = inverse(U)                          # columns: the dual basis
                dual = [[Uinv[i][k] for i in range(len(free))] for k in range(len(free))]
            normal = []
            for coefficients in dual:
                m = [0] * R.rank
                for c, j in zip(coefficients, free):
                    if c.denominator != 1:
                        raise AssertionError("cone %r is not unimodular in its face" % (tau,))
                    m = [x + int(c) * y for x, y in zip(m, sigma[j])]
                normal.append(spherical.levi_projection(R, levi, tuple(m)))
            label = orbit.name + ("|" + ";".join(",".join(map(str, v)) for v in tau) if tau else "")
            orbits.append(spherical.OrbitDatum(
                label, orbit.generators, orbit.component_reflections, orbit.unipotent_roots,
                tuple(normal), orbit.component_elements, orbit.note, orbit.roots, ()))
    return orbits


def toroidal_variety(cartan_type, spherical_roots, satellite, cones, parabolic=(), name="",
                     strict=True):
    """FixedPointData of a smooth complete toroidal variety over a wonderful model
    >>> from bbcells.frontends.spherical import complete_quadrics
    >>> sigma = [(2, 0), (0, 2)]                       # complete conics, blown up along G/B
    >>> sat = lambda I: None if len(I) > 1 else {"component_reflections": [(1, 0) if I == (0,) else (0, 1)]}
    >>> fan = star_subdivision(orthant(2), orthant(2)[0])
    >>> X = toroidal_variety("A2", sigma, sat, fan)
    >>> X.dim, len(X)
    (5, 18)
    """
    orbits = toroidal_orbits(cartan_type, spherical_roots, satellite, cones, parabolic, strict)
    return spherical.assemble(cartan_type, orbits, name=name or "toroidal variety")
