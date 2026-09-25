"""
Smooth complete toroidal horospherical varieties X = G x_P Y (PLAN.md S6a.1).

Data: a Cartan type, the crossed nodes J of P = P_J, a basis of the
character lattice M of the torus P/H (weights in fundamental-weight
coordinates, supported on J, i.e. characters of P), and a smooth complete
fan in N = M^vee, written in the dual basis. Y is the toric variety of the
fan; P acts on Y through P -> P/H.

Characters of the maximal torus of the simply connected group are written
in fundamental-weight coordinates; cocharacters in coroot coordinates, so
the pairing is the dot product. The fixed points are (wP, x_sigma) with
tangent weights w(Phi^- - Phi_I^-) and w(m) for the toric weights m of
x_sigma: T acts on the fibre over wP through t -> w^{-1} t w.
"""

from bbcells.core import FixedPointData
from bbcells.frontends import toric
from bbcells.frontends.flag import _crossed_to_levi, word_label
from bbcells.linalg import rank as matrix_rank
from bbcells.rootsystem import RootSystem


def toroidal_horospherical(cartan_type, crossed, lattice_basis, fan, name=None):
    """
    >>> X = toroidal_horospherical("A1", {1}, [(1,)], toric.projective_space(1))
    >>> X.dim, len(X)                      # SL_2 x_B P^1 = F_1
    (2, 4)
    """
    R = RootSystem(cartan_type)
    levi, crossed = _crossed_to_levi(R, crossed)
    basis = [tuple(v) for v in lattice_basis]
    for v in basis:
        if len(v) != R.rank:
            raise ValueError("lattice basis vectors need %d coordinates" % R.rank)
        if any(c != 0 and i in levi for i, c in enumerate(v)):
            raise ValueError("%r is not a character of P: it is supported on a Levi node" % (v,))
    if matrix_rank([list(v) for v in basis]) != len(basis):
        raise ValueError("the lattice basis is not linearly independent")
    if fan.dim != len(basis):
        raise ValueError("the fan lives in a lattice of rank %d, but M has rank %d"
                         % (fan.dim, len(basis)))

    def to_character(coefficients):
        return tuple(sum(c * v[j] for c, v in zip(coefficients, basis)) for j in range(R.rank))

    levi_roots = set(R.positive_roots_of(levi))
    base = [R.root_to_weight(tuple(-c for c in beta)) for beta in R.positive_roots
            if beta not in levi_roots]
    fibre = toric.fixed_point_data(fan)
    fibre_weights = [[to_character(m) for m in wts] for wts in fibre.weights]

    orbit = R.orbit(levi)
    twisted = {orbit[0][0]: (base, fibre_weights)}
    points, weights, annotations = [], [], []
    for mu, word in orbit:
        if word:
            parent = R.reflect_weight(word[0], mu)
            b, f = twisted[parent]
            reflect = lambda x: R.reflect_weight(word[0], x)
            twisted[mu] = ([reflect(x) for x in b], [[reflect(x) for x in wts] for wts in f])
        b, f = twisted[mu]
        for cone_label, wts in zip(fibre.points, f):
            points.append("%s|%s" % (word_label(word), cone_label))
            weights.append(tuple(b) + tuple(wts))
            annotations.append({"word": tuple(i + 1 for i in word), "cone": cone_label})
    return FixedPointData(len(base) + fan.dim, R.rank, tuple(points), tuple(weights),
                          name=name or "%s x_P %s" % (R.name, fan.name or "Y"),
                          annotations=tuple(annotations))


def _orbit_under_levi(R, levi, mu):
    """W_levi-orbit of a weight (fundamental-weight coordinates)"""
    seen, frontier = {tuple(mu)}, [tuple(mu)]
    while frontier:
        new = []
        for x in frontier:
            for i in levi:
                y = R.reflect_weight(i, x)
                if y not in seen:
                    seen.add(y)
                    new.append(y)
        frontier = new
    return sorted(seen)


def color_vectors(R, colors, basis):
    """{alpha (1-based): rho_alpha in N}: rho_alpha(m_k) = <m_k, alpha^vee>, the
    omega_alpha-coordinate of the k-th basis vector of M"""
    return {a: tuple(v[a - 1] for v in basis) for a in colors}


def colored_horospherical(cartan_type, crossed, lattice_basis, cones, name=None):
    """smooth complete horospherical G/H-embedding from a colored fan
    (PLAN.md S6a.3).

    cartan_type, crossed, lattice_basis: as for toroidal_horospherical (the
    colors are the crossed nodes). cones: the maximal colored cones as pairs
    (generators, colors) with generators primitive vectors of N (coordinates
    dual to lattice_basis) forming a basis of N, and colors a set of crossed
    nodes alpha whose rho_alpha is one of the generators.

    Model (local structure theorem): the closed orbit of (sigma, F) is G/Q,
    Q = P_{I + F}; at its base point the normal space is the L_Q-module with
    one summand per generator v_i, of weights W_{I+F}.m_i (m_i the dual basis
    of M); uncolored generators give characters of L_Q. The dimension of this
    module is checked; a mismatch means the colored cone is not smooth.
    >>> P2 = colored_horospherical("A1", {1}, [(1,)], [([(1,)], {1}), ([(-1,)], set())])
    >>> len(P2), P2.dim            # SL_2 acting on P(k^2 + k)
    (3, 2)
    """
    from bbcells.linalg import integer_inverse, transpose
    R = RootSystem(cartan_type)
    levi, crossed = _crossed_to_levi(R, crossed)
    basis = [tuple(v) for v in lattice_basis]
    r = len(basis)
    for v in basis:
        if any(c != 0 and i in levi for i, c in enumerate(v)):
            raise ValueError("%r is not a character of P" % (v,))
    rho = color_vectors(R, crossed, basis)
    # validate the underlying fan (smooth and complete) by building it
    rays = sorted({tuple(g) for gens, _ in cones for g in gens})
    ray_index = {v: i for i, v in enumerate(rays)}
    toric.Fan(tuple(rays), tuple(tuple(ray_index[tuple(g)] for g in gens) for gens, _ in cones))
    colored_rays = {}
    for gens, colors in cones:
        for a in colors:
            if a not in crossed:
                raise ValueError("color %r is not a crossed node" % a)
            if rho[a] not in [tuple(g) for g in gens]:
                raise ValueError("rho_%d = %r is not a generator of its cone" % (a, rho[a]))
        for g in gens:
            present = {a for a in colors if rho[a] == tuple(g)}
            if tuple(g) in colored_rays and colored_rays[tuple(g)] != present:
                raise ValueError("the ray %r has different colors in different cones" % (g,))
            colored_rays[tuple(g)] = present

    def to_character(coefficients):
        return tuple(sum(c * v[j] for c, v in zip(coefficients, basis)) for j in range(R.rank))

    points, weights, annotations = [], [], []
    total_dim = None
    for index, (gens, colors) in enumerate(cones):
        big_levi = set(levi) | {a - 1 for a in colors}
        dual = [tuple(row) for row in integer_inverse(transpose([list(g) for g in gens]))]
        normal = []
        for g, m in zip(gens, dual):
            character = to_character(m)
            if colored_rays[tuple(g)]:
                normal.extend(_orbit_under_levi(R, big_levi, character))
            else:
                if any(character[i] != 0 for i in big_levi):
                    raise ValueError("uncolored generator %r does not give a character of L_Q" % (g,))
                normal.append(character)
        levi_roots = set(R.positive_roots_of(big_levi))
        base = [R.root_to_weight(tuple(-c for c in b)) for b in R.positive_roots
                if b not in levi_roots]
        dim = len(base) + len(normal)
        expected = len([b for b in R.positive_roots if b not in set(R.positive_roots_of(levi))]) + r
        if dim != expected:
            raise ValueError("colored cone %d is not smooth: the normal module has dimension "
                             "%d instead of %d" % (index, len(normal), expected - len(base)))
        total_dim = dim
        orbit = R.orbit(big_levi)
        twisted = {orbit[0][0]: (base, normal)}
        for mu, word in orbit:
            if word:
                parent = R.reflect_weight(word[0], mu)
                b, f = twisted[parent]
                reflect = lambda x: R.reflect_weight(word[0], x)
                twisted[mu] = ([reflect(x) for x in b], [reflect(x) for x in f])
            b, f = twisted[mu]
            points.append("%s|c%d" % (word_label(word), index))
            weights.append(tuple(b) + tuple(f))
            annotations.append({"word": tuple(i + 1 for i in word), "cone": index,
                                "colors": tuple(sorted(colors))})
    return FixedPointData(total_dim, R.rank, tuple(points), tuple(weights),
                          name=name or "%s horospherical" % R.name,
                          annotations=tuple(annotations))
