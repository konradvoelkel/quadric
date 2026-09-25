"""
Operations on fixed-point data (PLAN.md S1.6, S3.4): products and
restriction to a subtorus (or any change of torus along a homomorphism).
"""

from bbcells.core import FixedPointData
from bbcells.linalg import mat_vec


def product(first, second, separator="|"):
    """X x Y with the torus T_1 x T_2; labels 'p|q'
    >>> P1 = FixedPointData(1, 1, ("0", "oo"), (((1,),), ((-1,),)), name="P1")
    >>> P1xP1 = product(P1, P1)
    >>> P1xP1.points
    ('0|0', '0|oo', 'oo|0', 'oo|oo')
    >>> P1xP1.weights_of("0|oo")
    ((1, 0), (0, -1))
    """
    r1, r2 = first.rank, second.rank
    points, weights, annotations = [], [], []
    for p, wp in zip(first.points, first.weights):
        for q, wq in zip(second.points, second.weights):
            points.append("%s%s%s" % (p, separator, q))
            weights.append(tuple(w + (0,) * r2 for w in wp) +
                           tuple((0,) * r1 + w for w in wq))
            annotations.append({"factors": (p, q)})
    edges = None
    if first.edges is not None and second.edges is not None:
        edges = []
        for p, p2, chi in first.edges:
            for q in second.points:
                edges.append(("%s%s%s" % (p, separator, q), "%s%s%s" % (p2, separator, q),
                              chi + (0,) * r2))
        for q, q2, chi in second.edges:
            for p in first.points:
                edges.append(("%s%s%s" % (p, separator, q), "%s%s%s" % (p, separator, q2),
                              (0,) * r1 + chi))
    preferred = None
    if first.preferred_cocharacter and second.preferred_cocharacter:
        preferred = first.preferred_cocharacter + second.preferred_cocharacter
    return FixedPointData(first.dim + second.dim, r1 + r2, tuple(points), tuple(weights),
                          edges=None if edges is None else tuple(edges),
                          preferred_cocharacter=preferred,
                          name="%s x %s" % (first.name or "X", second.name or "Y"),
                          annotations=tuple(annotations))


def restrict(data, matrix, name=None):
    """change the torus along a homomorphism T' -> T whose effect on
    characters X^*(T) -> X^*(T') is the integer matrix (rows = coordinates
    of X^*(T')). Typical use: restriction to a subtorus.

    The fixed points stay the same (and isolated) exactly when no tangent
    weight restricts to zero (PLAN.md 1.5); otherwise ValueError.
    >>> P1 = FixedPointData(1, 1, ("0", "oo"), (((1,),), ((-1,),)))
    >>> P1xP1 = product(P1, P1)
    >>> restrict(P1xP1, [[1, 2]]).weights_of("0|oo")
    ((1,), (-2,))
    >>> restrict(P1xP1, [[1, 1]]).weights_of("0|oo")      # the diagonal torus
    ((1,), (-1,))
    >>> restrict(P1xP1, [[1, 0]])                          # the first factor only
    Traceback (most recent call last):
    ...
    ValueError: restriction is not isolated: the weight (0, 1) at '0|0' restricts to 0
    """
    matrix = [list(row) for row in matrix]
    if any(len(row) != data.rank for row in matrix):
        raise ValueError("the restriction matrix needs %d columns" % data.rank)
    new_rank = len(matrix)
    weights = []
    for label, wts in zip(data.points, data.weights):
        restricted = []
        for w in wts:
            image = mat_vec(matrix, w)
            if not any(image):
                raise ValueError("restriction is not isolated: the weight %r at %r "
                                 "restricts to 0" % (w, label))
            restricted.append(image)
        weights.append(tuple(restricted))
    edges = None
    if data.edges is not None:
        edges = tuple((p, q, mat_vec(matrix, chi)) for p, q, chi in data.edges)
    return FixedPointData(data.dim, new_rank, data.points, tuple(weights), edges=edges,
                          name=name if name is not None else data.name,
                          annotations=data.annotations)
