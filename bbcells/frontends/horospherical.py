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
