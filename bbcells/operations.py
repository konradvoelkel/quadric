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


def _counter(weights):
    from collections import Counter
    return Counter(weights)


def identify(sub, ambient):
    """match the fixed points of a T-stable smooth subvariety to those of the
    ambient variety (same torus), by multiset inclusion of tangent weights.
    Returns {sub label: ambient label}; raises unless the match is unique.
    >>> P1 = FixedPointData(1, 1, ("0", "oo"), (((1,),), ((-1,),)))
    >>> P1xP1 = product(P1, P1)
    >>> diagonal = FixedPointData(1, 2, ("d0", "doo"), (((1, 0),), ((-1, 0),)))
    >>> identify(diagonal, P1xP1)
    Traceback (most recent call last):
    ...
    ValueError: fixed point 'd0' of the subvariety matches 2 fixed points: ['0|0', '0|oo']
    """
    if sub.rank != ambient.rank:
        raise ValueError("subvariety and ambient variety need the same torus")
    ambient_counters = [(label, _counter(wts)) for label, wts in zip(ambient.points, ambient.weights)]
    mapping, used = {}, set()
    for label, wts in zip(sub.points, sub.weights):
        needed = _counter(wts)
        matches = [a for a, have in ambient_counters
                   if all(have[w] >= c for w, c in needed.items())]
        if len(matches) != 1:
            raise ValueError("fixed point %r of the subvariety matches %d fixed points: %r"
                             % (label, len(matches), matches[:5]))
        if matches[0] in used:
            raise ValueError("two fixed points of the subvariety match %r" % (matches[0],))
        used.add(matches[0])
        mapping[label] = matches[0]
    return mapping


def normal_weights(sub, ambient, mapping):
    """{sub label: tuple of the normal weights} (ambient weights minus sub weights)"""
    result = {}
    for label, wts in zip(sub.points, sub.weights):
        remaining = _counter(ambient.weights_of(mapping[label]))
        remaining.subtract(_counter(wts))
        normal = []
        for w, c in sorted(remaining.items()):
            if c < 0:
                raise ValueError("weights at %r are not contained in the ambient ones" % label)
            normal.extend([w] * c)
        result[label] = tuple(normal)
    return result


def blowup(ambient, center, mapping=None, name=None):
    """blow-up along a T-stable smooth centre (PLAN.md 1.6). Over a fixed point
    p of the centre with pairwise distinct normal weights nu_1..nu_c, the new
    fixed points are the eigenlines [nu_j], labelled 'p~j', with tangent
    weights wt(T_p Z), nu_j, and nu_k - nu_j (k != j). Edges are not kept.
    >>> P2 = FixedPointData(2, 2, ("a", "b", "c"),
    ...     (((1, 0), (0, 1)), ((-1, 0), (-1, 1)), ((0, -1), (1, -1))))
    >>> point = FixedPointData(0, 2, ("a",), ((),))
    >>> F1 = blowup(P2, point, {"a": "a"})     # a point needs an explicit embedding
    >>> F1.points
    ('b', 'c', 'a~0', 'a~1')
    >>> F1.weights_of("a~0")
    ((0, 1), (1, -1))
    """
    if center.dim >= ambient.dim - 1:
        raise ValueError("the centre must have codimension >= 2")
    mapping = identify(center, ambient) if mapping is None else dict(mapping)
    normals = normal_weights(center, ambient, mapping)
    in_center = {mapping[p]: p for p in center.points}
    points, weights, annotations = [], [], []
    for label, wts in zip(ambient.points, ambient.weights):
        if label not in in_center:
            points.append(label)
            weights.append(wts)
            annotations.append(dict(ambient.annotation(label)))
    for label in ambient.points:
        if label not in in_center:
            continue
        z = in_center[label]
        normal = normals[z]
        if len(set(normal)) != len(normal):
            raise ValueError("normal weights at %r are not distinct: %r; the fixed points "
                             "of the exceptional divisor are not isolated" % (label, normal))
        for j, nu in enumerate(normal):
            points.append("%s~%d" % (label, j))
            new = tuple(center.weights_of(z)) + (nu,) + tuple(
                tuple(a - b for a, b in zip(other, nu)) for k, other in enumerate(normal)
                if k != j)
            weights.append(new)
            annotations.append({"over": label, "normal_weight": nu})
    return FixedPointData(ambient.dim, ambient.rank, tuple(points), tuple(weights),
                          name=name or "Bl(%s)" % (ambient.name or "X"),
                          annotations=tuple(annotations))
