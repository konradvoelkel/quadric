"""
Fixed-point data of a torus action and its Bialynicki-Birula decomposition.

Conventions (SPEC.md section 2):
  * weights are *tangent* weights: T acts on T_p X through these characters;
  * characters and cocharacters are integer vectors, paired by the dot product;
  * cells are plus-cells, X_p^+ = {x : lim_{t -> 0} lambda(t) x = p}, of
    dimension #{chi in wt(p) : <lambda, chi> > 0}.
"""

from dataclasses import dataclass, field


def pairing(lam, chi):
    """the pairing <lambda, chi> of a cocharacter with a character
    >>> pairing((1, 2), (3, -1))
    1
    """
    return sum(a * b for a, b in zip(lam, chi))


def _as_vector(vector, rank, where):
    vector = tuple(vector)
    if len(vector) != rank or not all(isinstance(c, int) for c in vector):
        raise ValueError("%s: expected an integer vector of length %d, got %r"
                         % (where, rank, vector))
    return vector


@dataclass(frozen=True)
class FixedPointData(object):
    """the T-fixed points of a smooth projective n-dimensional X with their
    tangent weights, for a split torus T of rank r.

    points      labels of the fixed points (strings are recommended)
    weights     for each point (in the same order) the n tangent weights
    edges       optional GKM data: triples (p, q, chi) for a T-invariant P^1
                joining p and q whose tangent weight at p is chi
    preferred_cocharacter
                a generic cocharacter the front end recommends (for example
                rho^vee for flag varieties, whose cells are Schubert cells)
    annotations optional per-point metadata, e.g. the Weyl group element

    >>> P1 = FixedPointData(1, 1, ("0", "oo"), (((1,),), ((-1,),)), name="P^1")
    >>> P1.weights_of("oo")
    ((-1,),)
    >>> FixedPointData(1, 1, ("0",), (((0,),),))
    Traceback (most recent call last):
    ...
    ValueError: point '0': zero tangent weight, the fixed point is not isolated
    """
    dim: int
    rank: int
    points: tuple
    weights: tuple
    edges: tuple = None
    preferred_cocharacter: tuple = None
    name: str = ""
    annotations: tuple = field(default=None, compare=False)

    def __post_init__(self):
        points = tuple(self.points)
        if len(set(points)) != len(points):
            raise ValueError("fixed point labels are not distinct")
        weights = tuple(self.weights)
        if len(weights) != len(points):
            raise ValueError("got %d weight lists for %d points" % (len(weights), len(points)))
        checked = []
        for label, wts in zip(points, weights):
            wts = tuple(_as_vector(w, self.rank, "point %r" % (label,)) for w in wts)
            if len(wts) != self.dim:
                raise ValueError("point %r: expected %d tangent weights, got %d"
                                 % (label, self.dim, len(wts)))
            if any(not any(w) for w in wts):
                raise ValueError("point %r: zero tangent weight, the fixed point is "
                                 "not isolated" % (label,))
            checked.append(wts)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "weights", tuple(checked))
        object.__setattr__(self, "_index", {p: i for i, p in enumerate(points)})
        if self.preferred_cocharacter is not None:
            object.__setattr__(self, "preferred_cocharacter", _as_vector(
                self.preferred_cocharacter, self.rank, "preferred cocharacter"))
        if self.annotations is not None and len(self.annotations) != len(points):
            raise ValueError("annotations must be aligned with points")
        if self.edges is not None:
            object.__setattr__(self, "edges", tuple(self._checked_edges()))

    def _checked_edges(self):
        for p, q, chi in self.edges:
            chi = _as_vector(chi, self.rank, "edge %r-%r" % (p, q))
            if p not in self._index or q not in self._index:
                raise ValueError("edge %r-%r: unknown fixed point" % (p, q))
            if chi not in self.weights_of(p):
                raise ValueError("edge %r-%r: %r is not a tangent weight at %r" % (p, q, chi, p))
            if tuple(-c for c in chi) not in self.weights_of(q):
                raise ValueError("edge %r-%r: %r is not a tangent weight at %r"
                                 % (p, q, tuple(-c for c in chi), q))
            yield (p, q, chi)

    def index(self, label):
        return self._index[label]

    def weights_of(self, label):
        return self.weights[self._index[label]]

    def annotation(self, label):
        if self.annotations is None:
            return {}
        return self.annotations[self._index[label]]

    def all_weights(self):
        """the set of all tangent weights occurring at some fixed point"""
        return {w for wts in self.weights for w in wts}

    def __len__(self):
        return len(self.points)


def is_generic(data, lam):
    """lam pairs nonzero with every tangent weight"""
    return all(pairing(lam, w) != 0 for w in data.all_weights())


def choose_generic_cocharacter(data, reverse=False):
    """a deterministic generic cocharacter (1, N, N^2, ...) with N larger
    than twice the largest absolute weight coordinate. A nonzero integer
    vector with coordinates in [-M, M] cannot pair to zero with it (base-N
    digits), so it is generic by construction. With reverse=True the powers
    are reversed, which gives a second, independent choice.
    >>> P1xP1 = FixedPointData(2, 2, ("a",), (((1, 0), (0, 1)),))
    >>> choose_generic_cocharacter(P1xP1), choose_generic_cocharacter(P1xP1, reverse=True)
    ((1, 3), (3, 1))
    """
    bound = max((abs(c) for w in data.all_weights() for c in w), default=0)
    base = 2 * bound + 1
    lam = tuple(base ** i for i in range(data.rank))
    return tuple(reversed(lam)) if reverse else lam


@dataclass(frozen=True)
class CellDecomposition(object):
    """the plus-decomposition of a FixedPointData for a generic cocharacter
    >>> P1 = FixedPointData(1, 1, ("0", "oo"), (((1,),), ((-1,),)))
    >>> cells = bb_cells(P1)
    >>> cells.dim_of("0"), cells.dim_of("oo"), cells.counts
    (1, 0, (1, 1))
    """
    data: FixedPointData
    cocharacter: tuple
    dims: tuple

    @property
    def counts(self):
        """(c_0, ..., c_n) with c_d the number of cells of dimension d"""
        counts = [0] * (self.data.dim + 1)
        for d in self.dims:
            counts[d] += 1
        return tuple(counts)

    def dim_of(self, label):
        return self.dims[self.data.index(label)]

    def cells(self):
        """list of (dimension, label), sorted by dimension then input order"""
        return sorted(((d, p) for p, d in zip(self.data.points, self.dims)),
                      key=lambda pair: (pair[0], self.data.index(pair[1])))

    def cells_of_dimension(self, d):
        return [p for p, dim in zip(self.data.points, self.dims) if dim == d]

    def opposite(self):
        """the decomposition for -lambda (the minus-cells for lambda)"""
        return bb_cells(self.data, tuple(-c for c in self.cocharacter))

    # convenience wrappers around bbcells.invariants
    def invariants(self):
        from bbcells import invariants
        return invariants.compute(self)

    def check(self):
        from bbcells import invariants
        return invariants.check(self)


def bb_cells(data, lam=None):
    """plus-cell dimensions for the cocharacter lam (default: the front end's
    preferred cocharacter, else a deterministic generic one)
    >>> P2 = FixedPointData(2, 2, ("p0", "p1", "p2"),
    ...     (((1, 0), (0, 1)), ((-1, 0), (-1, 1)), ((0, -1), (1, -1))))
    >>> bb_cells(P2, (1, 2)).counts
    (1, 1, 1)
    >>> bb_cells(P2, (1, 1))
    Traceback (most recent call last):
    ...
    ValueError: cocharacter (1, 1) is not generic: it pairs to zero with (-1, 1) at point 'p1'
    """
    if lam is None:
        lam = data.preferred_cocharacter or choose_generic_cocharacter(data)
    lam = _as_vector(lam, data.rank, "cocharacter")
    dims = []
    for label, wts in zip(data.points, data.weights):
        values = [pairing(lam, w) for w in wts]
        for w, value in zip(wts, values):
            if value == 0:
                raise ValueError("cocharacter %r is not generic: it pairs to zero with "
                                 "%r at point %r" % (lam, w, label))
        dims.append(sum(1 for value in values if value > 0))
    return CellDecomposition(data, lam, tuple(dims))
