"""
Varieties of coordinate subspaces of a representation with diagonal torus
action: projective spaces, Grassmannians and symplectic isotropic
Grassmannians (PLAN.md S6b.1, S6c.1).

A representation is given by the T-weights of a basis. Fixed points are the
coordinate subspaces; for a subspace L with weights S the tangent space
Hom(L, V/L) has the weights c - a (a in S, c not in S).
"""

import itertools

from bbcells.core import FixedPointData


def _vec(w):
    return tuple(w)


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def projectivization(weights, labels=None, name="P(V)"):
    """P(V) for a basis of weight vectors with pairwise distinct weights
    >>> P2 = projectivization([(1, 0, 0), (0, 1, 0), (0, 0, 1)], ["e1", "e2", "e3"])
    >>> P2.weights_of("e1")
    ((-1, 1, 0), (-1, 0, 1))
    """
    return grassmannian_of(weights, 1, labels=labels, name=name)


def grassmannian_of(weights, k, labels=None, name=None):
    """Gr(k, V) for a basis of weight vectors; labels of the fixed points are
    the labels of the spanning basis vectors joined by '+'"""
    weights = [_vec(w) for w in weights]
    rank = len(weights[0])
    labels = list(labels) if labels is not None else ["e%d" % (i + 1) for i in range(len(weights))]
    points, tangent, edges = [], [], []
    subsets = list(itertools.combinations(range(len(weights)), k))
    label_of = {S: "+".join(labels[i] for i in S) for S in subsets}
    for S in subsets:
        outside = [c for c in range(len(weights)) if c not in S]
        points.append(label_of[S])
        tangent.append(tuple(_sub(weights[c], weights[a]) for a in S for c in outside))
        for a in S:
            for c in outside:
                T = tuple(sorted(set(S) - {a} | {c}))
                if S < T:
                    edges.append((label_of[S], label_of[T], _sub(weights[c], weights[a])))
    return FixedPointData(k * (len(weights) - k), rank, tuple(points), tuple(tangent),
                          edges=tuple(edges),
                          name=name or "Gr(%d,%d)" % (k, len(weights)))


def symplectic_weights(m):
    """weights eps_1, ..., eps_m, -eps_1, ..., -eps_m of the standard
    representation of Sp_{2m}, with labels e1..em, f1..fm"""
    unit = lambda i, s: tuple(s if j == i else 0 for j in range(m))
    weights = [unit(i, 1) for i in range(m)] + [unit(i, -1) for i in range(m)]
    labels = ["e%d" % (i + 1) for i in range(m)] + ["f%d" % (i + 1) for i in range(m)]
    return weights, labels


def isotropic_grassmannian_of(m, k):
    """isotropic k-planes in k^{2m} with the torus of Sp_{2m}: coordinate
    planes without a pair e_i, f_i. Tangent space Hom(L, L^perp/L) + Sym^2 L^*,
    with weights c - a (c not in +-S) and -(a + b) for a <= b in S.
    >>> IG = isotropic_grassmannian_of(2, 2)            # Lagrangian Grassmannian LG(2, 4)
    >>> IG.dim, len(IG)
    (3, 4)
    """
    weights, labels = symplectic_weights(m)
    neg = lambda w: tuple(-x for x in w)
    points, tangent = [], []
    for S in itertools.combinations(range(2 * m), k):
        ws = [weights[i] for i in S]
        if any(neg(w) in ws for w in ws):
            continue
        outside = [w for w in weights if w not in ws and neg(w) not in ws]
        t = [_sub(c, a) for a in ws for c in outside]
        t += [neg(tuple(x + y for x, y in zip(ws[i], ws[j])))
              for i in range(k) for j in range(i, k)]
        points.append("+".join(labels[i] for i in S))
        tangent.append(tuple(t))
    return FixedPointData(k * (2 * m - 2 * k) + k * (k + 1) // 2, m, tuple(points),
                          tuple(tangent), name="IG(%d,%d)" % (k, 2 * m))
