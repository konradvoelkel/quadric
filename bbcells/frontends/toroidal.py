"""
Smooth complete toroidal varieties over a wonderful model (docs/spherical.md, section 8).

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
the dual basis m of tau, the toric rule of C2 transported by phi_J. The code
uses phi_J(-gamma) = the normal weight of the orbit datum, so certified
weights beyond (R) carry over.

Cones are given by their primitive generators in the coordinates
n = (<gamma_1, n>, ..., <gamma_r, n>).

Finer lattices (docs/spherical.md, section 8). Let G/H' -> G/H be a finite
cover with weight lattice Lambda containing Z Sigma, e.g. SL_3/SO_3 over
SL_3/N(SO_3) with Lambda = <2 omega_1, 2 omega_2> of index 3. The fan now lives
in N' = Hom(Lambda, Z), a sublattice of N, and must be smooth there. The
group A = H/H' acts on X' with quotient the toroidal variety X of G/H with
the same (rational) fan. The orbit O'_tau -> O_tau is a finite cover of degree
[Lambda : Z Sigma] / [pr_F Lambda : Z(Sigma - J)] = [Lambda cap Q J : Z J] =: d_J,
and O_tau = O_J when dim tau = dim F_J. So X'^T has d_J points over each
(x, tau), with the tangent weights of O_J at x (the cover is etale) and the
normal weights pr_J(m) for the dual basis m of tau, now taken in Lambda: the
normal slice is the affine toric variety of tau in N', on which T acts
through pr_J on Lambda / (Lambda cap Q J). The weights leave the root lattice
in general, so the result is in fundamental-weight coordinates.
"""

from fractions import Fraction
from itertools import combinations
from math import lcm, prod

from bbcells.frontends import spherical
from bbcells.linalg import determinant, inverse, smith_invariants, solve
from bbcells.rootsystem import RootSystem


def lattice_in_sigma(cartan_type, spherical_roots, generators):
    """the lattice Lambda spanned by `generators` (fundamental-weight
    coordinates) as vectors in the basis Sigma of Q Sigma; checks that
    Lambda contains Z Sigma and has rank |Sigma|
    >>> lattice_in_sigma("A2", [(2, 0), (0, 2)], [(2, 0), (0, 2)])     # SL_3/SO_3
    [(Fraction(2, 3), Fraction(1, 3)), (Fraction(1, 3), Fraction(2, 3))]
    >>> lattice_in_sigma("A1", [(2,)], [(8,)])                  # 2 gamma only
    Traceback (most recent call last):
    ...
    ValueError: the lattice does not contain the spherical roots
    """
    R = RootSystem(cartan_type)
    sigma = [tuple(g) for g in spherical_roots]
    r = len(sigma)
    omega = [R.fundamental_weight(i) for i in range(R.rank)]
    gram = [[R.inner(a, b) for b in sigma] for a in sigma]
    rows = []
    for g in generators:
        if len(g) != R.rank:
            raise ValueError("lattice generator %r: expected %d coordinates" % (g, R.rank))
        v = tuple(sum(c * w[k] for c, w in zip(g, omega)) for k in range(R.rank))
        c = solve(gram, [R.inner(a, v) for a in sigma])
        if tuple(sum(x * a[k] for x, a in zip(c, sigma)) for k in range(R.rank)) != v:
            raise ValueError("lattice generator %r is not in the span of the spherical roots"
                             % (g,))
        rows.append(tuple(Fraction(x) for x in c))
    units = [tuple(Fraction(int(i == j)) for j in range(r)) for i in range(r)]
    if _covolume(rows, range(r)) != _covolume(rows + units, range(r)):
        raise ValueError("the lattice does not contain the spherical roots")
    return rows


def _covolume(rows, columns):
    """[pr_columns(Lambda) : Z^columns] ** -1 for Lambda spanned by rows, as a
    Fraction; None if the projection has lower rank"""
    columns = list(columns)
    if not columns:
        return Fraction(1)
    D = lcm(*(x.denominator for row in rows for x in (row[j] for j in columns)))
    M = [[int(row[j] * D) for j in columns] for row in rows]
    invariants = smith_invariants(M)
    if len(invariants) < len(columns):
        return None
    return Fraction(prod(invariants), D ** len(columns))


def lattice_index(rows, columns):
    """[pr_columns(Lambda) : Z^columns] for Lambda (given by lattice_in_sigma)
    containing Z Sigma
    >>> rows = lattice_in_sigma("A2", [(2, 0), (0, 2)], [(2, 0), (0, 2)])
    >>> lattice_index(rows, [0, 1]), lattice_index(rows, [1]), lattice_index(rows, [])
    (3, 3, 1)
    """
    r = len(rows[0])
    units = [tuple(Fraction(int(i == j)) for j in range(r)) for i in range(r)]
    index = 1 / _covolume(list(rows) + units, columns)
    if index.denominator != 1:
        raise AssertionError("the lattice does not contain Z Sigma")
    return int(index)


def sheets(rows, J):
    """d_J = [Lambda cap Q J : Z J], the degree of O'_tau -> O_J
    >>> rows = lattice_in_sigma("A1", [(2,)], [(2,)])            # SL_2/T over SL_2/N(T)
    >>> sheets(rows, ()), sheets(rows, (0,))
    (1, 2)
    """
    r = len(rows[0])
    free = [j for j in range(r) if j not in J]
    return lattice_index(rows, range(r)) // lattice_index(rows, free)


def _pairings(rows, v):
    return [sum(c * x for c, x in zip(row, v)) for row in rows]


def orthant(r, lattice=None):
    """the fan of the wonderful variety itself: the negative orthant. With a
    finer lattice (rows from lattice_in_sigma) its rays are the primitive
    vectors of N' = Hom(Lambda, Z), and the orthant may be singular in N'
    >>> orthant(2)
    [((-1, 0), (0, -1))]
    >>> orthant(2, lattice_in_sigma("A2", [(2, 0), (0, 2)], [(2, 0), (0, 2)]))
    [((-3, 0), (0, -3))]
    """
    rays = []
    for j in range(r):
        e = tuple(-int(i == j) for i in range(r))
        k = lcm(*(x.denominator for x in _pairings(lattice, e))) if lattice else 1
        rays.append(tuple(k * x for x in e))
    return [tuple(rays)]


def star_subdivision(cones, face, ray=None):
    """star subdivision of a simplicial fan (maximal cones as tuples of rays)
    at a cone `face` (a tuple of rays of the fan): every maximal cone
    containing the face is replaced by the cones obtained by swapping one ray
    of the face for the new ray, by default the sum of its rays (any ray in
    the relative interior of the face will do)
    >>> star_subdivision(orthant(2), ((-1, 0), (0, -1)))
    [((-1, -1), (0, -1)), ((-1, 0), (-1, -1))]
    """
    face = [tuple(v) for v in face]
    new_ray = tuple(ray) if ray is not None else tuple(sum(c) for c in zip(*face))
    result = []
    for cone in cones:
        cone = [tuple(v) for v in cone]
        if all(v in cone for v in face):
            for v in face:
                result.append(tuple(new_ray if u == v else u for u in cone))
        else:
            result.append(tuple(cone))
    return result


def resolve(cones, lattice=None):
    """a smooth refinement of a simplicial fan (rays in Z^r), smooth for
    N' = Hom(Lambda, Z) if a finer lattice is given: star subdivisions at
    lattice points of the fundamental parallelepipeds of singular cones, as
    in the toric resolution of singularities
    >>> rows = lattice_in_sigma("A2", [(2, 0), (0, 2)], [(2, 0), (0, 2)])   # SL_3/SO_3
    >>> resolve(orthant(2, rows), rows)
    [((-1, -1), (0, -3)), ((-3, 0), (-1, -1))]
    """
    cones = [tuple(tuple(v) for v in cone) for cone in cones]
    r = len(cones[0])
    covolume = lattice_index(lattice, range(r)) if lattice else 1
    while True:
        bad = [c for c in cones if abs(determinant([list(v) for v in c])) != covolume]
        if not bad:
            return cones
        cone = bad[0]
        ray, t = _parallelepiped_point(cone, lattice)
        cones = star_subdivision(cones, tuple(v for v, x in zip(cone, t) if x), ray)


def _parallelepiped_point(cone, lattice):
    """a nonzero point u = sum t_i v_i of N' with 0 <= t_i < 1, sum t_i minimal"""
    from itertools import product as cartesian
    r = len(cone)
    V = inverse([[Fraction(x) for x in v] for v in cone])
    low = [sum(min(0, v[k]) for v in cone) for k in range(r)]
    high = [sum(max(0, v[k]) for v in cone) for k in range(r)]
    best = None
    for x in cartesian(*(range(a, b + 1) for a, b in zip(low, high))):
        if not any(x):
            continue
        t = [sum(x[k] * V[k][i] for k in range(r)) for i in range(r)]
        if not all(0 <= c < 1 for c in t):
            continue
        if lattice and any(p.denominator != 1 for p in _pairings(lattice, x)):
            continue
        if best is None or sum(t) < sum(best[1]):
            best = (tuple(x), t)
    if best is None:
        raise AssertionError("no lattice point in the parallelepiped of %r" % (cone,))
    return best


def _faces(cones):
    faces = set()
    for cone in cones:
        cone = tuple(sorted(tuple(v) for v in cone))
        for k in range(len(cone) + 1):
            faces.update(combinations(cone, k))
    return faces


def check_fan(cones, r, lattice=None):
    """validate a smooth fan with support the negative orthant in Z^r (smooth
    for N' = Hom(Lambda, Z) if the rows of a finer lattice are given); returns
    its maximal cones as tuples of tuples
    >>> check_fan(star_subdivision(orthant(2), orthant(2)[0]), 2)
    [((-1, -1), (0, -1)), ((-1, 0), (-1, -1))]
    >>> check_fan([((-1, 0), (-1, -1))], 2)            # only half of the orthant
    Traceback (most recent call last):
    ...
    ValueError: facet ((-1, -1),) lies in 1 cones
    >>> rows = lattice_in_sigma("A2", [(2, 0), (0, 2)], [(2, 0), (0, 2)])   # SL_3/SO_3
    >>> check_fan(orthant(2, rows), 2, rows)
    Traceback (most recent call last):
    ...
    ValueError: cone ((-3, 0), (0, -3)) is not smooth
    >>> len(check_fan([((-3, 0), (-1, -1)), ((-1, -1), (0, -3))], 2, rows))
    2
    """
    cones = [tuple(tuple(v) for v in cone) for cone in cones]
    covolume = lattice_index(lattice, range(r)) if lattice else 1
    for cone in cones:
        if len(cone) != r:
            raise ValueError("maximal cones must have dimension %d: %r" % (r, cone))
        if any(c > 0 for v in cone for c in v):
            raise ValueError("cone %r leaves the valuation cone" % (cone,))
        if lattice and any(x.denominator != 1 for v in cone for x in _pairings(lattice, v)):
            raise ValueError("a ray of %r is not in Hom(Lambda, Z)" % (cone,))
        if abs(determinant([list(v) for v in cone])) != covolume:
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


def toroidal_orbits(cartan_type, spherical_roots, satellite, cones, parabolic=(), strict=True,
                    lattice=None):
    """OrbitData of the toroidal variety with the given fan (maximal cones) over
    the wonderful variety of (spherical_roots, parabolic, satellite), or over
    its finite cover with weight lattice spanned by `lattice` (fundamental-
    weight coordinates; the normal weights are then Fractions in simple-root
    coordinates). For complete conics blown up along the closed orbit G/B,
    that orbit is replaced by two orbits, one for each new cone:
    >>> sigma = [(2, 0), (0, 2)]                       # complete conics
    >>> reflections = {(0,): [(1, 0)], (1,): [(0, 1)]}
    >>> sat = lambda I: {"component_reflections": reflections[I]} if len(I) == 1 else None
    >>> [o.name for o in toroidal_orbits("A2", sigma, sat, orthant(2))]
    ['closed|-1,0;0,-1', 'O1|0,-1', 'O2|-1,0']
    >>> fan = star_subdivision(orthant(2), orthant(2)[0])
    >>> [o.name for o in toroidal_orbits("A2", sigma, sat, fan)]
    ['closed|-1,-1;-1,0', 'closed|-1,-1;0,-1', 'O1|0,-1', 'O2|-1,0']
    """
    R = RootSystem(cartan_type)
    r = len(spherical_roots)
    rows = lattice_in_sigma(cartan_type, spherical_roots, lattice) if lattice else None
    cones = check_fan(cones, r, rows)
    faces = _faces(cones)
    base = spherical.wonderful_orbits(cartan_type, spherical_roots, satellite, parabolic,
                                      strict=strict)
    sigma = [tuple(g) for g in spherical_roots]
    orbits = []
    for orbit in base:
        free = list(orbit.normal_roots)                   # coordinates of F_J
        J = set(orbit.roots)
        # phi_J(-gamma_j) = the normal weight nu_j of D_{gamma_j} at x_J
        nu = dict(zip(orbit.normal_roots, orbit.normal))
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
            for coefficients in dual:                      # m = sum c_j gamma_j
                if not rows and any(c.denominator != 1 for c in coefficients):
                    raise AssertionError("cone %r is not unimodular in its face" % (tau,))
                weight = [Fraction(0)] * R.rank
                for c, j in zip(coefficients, free):
                    weight = [x - c * y for x, y in zip(weight, nu[j])]
                normal.append(tuple(weight) if rows else tuple(int(x) for x in weight))
            label = orbit.name + ("|" + ";".join(",".join(map(str, v)) for v in tau) if tau else "")
            orbits.append(spherical.OrbitDatum(
                label, orbit.generators, orbit.component_reflections, orbit.unipotent_roots,
                tuple(normal), orbit.component_elements, orbit.note, orbit.roots, (),
                multiplicity=sheets(rows, J) if rows else 1))
    return orbits


def toroidal_variety(cartan_type, spherical_roots, satellite, cones, parabolic=(), name="",
                     strict=True, lattice=None):
    """FixedPointData of a smooth complete toroidal variety over a wonderful
    model, or over a finite cover of its open orbit with weight lattice
    spanned by `lattice` (then in fundamental-weight coordinates)
    >>> from bbcells.frontends.spherical import complete_quadrics
    >>> sigma = [(2, 0), (0, 2)]                       # complete conics, blown up along G/B
    >>> sat = lambda I: None if len(I) > 1 else {"component_reflections": [(1, 0) if I == (0,) else (0, 1)]}
    >>> fan = star_subdivision(orthant(2), orthant(2)[0])
    >>> X = toroidal_variety("A2", sigma, sat, fan)
    >>> X.dim, len(X)
    (5, 18)
    >>> Y = toroidal_variety("A2", sigma, sat, [((-3, 0), (-1, -1)), ((-1, -1), (0, -3))],
    ...                      lattice=[(2, 0), (0, 2)])     # over SL_3/SO_3 instead
    >>> Y.dim, len(Y), Y.weights_of("O1|0,-3:e")[-1]     # the normal weight -omega_2
    (5, 18, (0, -1))
    """
    orbits = toroidal_orbits(cartan_type, spherical_roots, satellite, cones, parabolic, strict,
                             lattice)
    return spherical.assemble(cartan_type, orbits, name=name or "toroidal variety",
                              coordinates="weights" if lattice else "roots")
