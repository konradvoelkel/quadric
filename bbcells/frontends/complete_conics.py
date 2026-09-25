"""
The variety of complete conics: the blow-up of P^5 = P(Sym^2 k^3) along the
Veronese surface (PLAN.md S6c.1), with the torus of GL_3.

The basis x_i x_j of Sym^2 has weight eps_i + eps_j; the Veronese surface is
v_2(P^2) with fixed points x_i^2 and tangent weights eps_j - eps_i (the
Veronese map is equivariant). The normal weights at x_i^2 are 2(eps_j -
eps_i) and eps_j + eps_k - 2 eps_i, pairwise distinct, so PLAN.md 1.6
applies: 3 + 3 * 3 = 12 fixed points.
"""

from bbcells.core import FixedPointData
from bbcells.frontends.linear import projectivization
from bbcells.operations import blowup


def _unit(i):
    return tuple(1 if k == i else 0 for k in range(3))


def complete_conics():
    """
    >>> X = complete_conics()
    >>> X.dim, len(X)
    (5, 12)
    """
    pairs = [(i, j) for i in range(3) for j in range(i, 3)]
    weights = [tuple(a + b for a, b in zip(_unit(i), _unit(j))) for i, j in pairs]
    labels = ["x%dx%d" % (i + 1, j + 1) for i, j in pairs]
    P5 = projectivization(weights, labels, name="P(Sym^2)")
    veronese = projectivization([_unit(i) for i in range(3)],
                                ["v%d" % (i + 1) for i in range(3)], name="v_2(P^2)")
    embedding = {"v%d" % (i + 1): "x%dx%d" % (i + 1, i + 1) for i in range(3)}
    return blowup(P5, veronese, embedding, name="complete conics")


def _eps(i, n):
    return tuple(1 if k == i else 0 for k in range(n))


def _add(*vectors):
    return tuple(sum(c) for c in zip(*vectors))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def complete_quadrics_p3():
    """complete quadrics in P^3 (the wonderful compactification of PGL_4/PO_4),
    by Vainsencher's two blow-ups of P^9 = P(Sym^2 k^4) with the torus of
    GL_4: first along the Veronese V = v_2(P^3), then along the strict
    transform S~_2 of the rank <= 2 locus (PLAN.md S6c.2).

    Fixed points of S~_2: the rank-2 points x_i x_j (i != j), where S_2 is
    smooth with tangent weights eps_k - eps_i (k != i) and eps_l - eps_j
    (l != j); and over each x_i^2 the three points of the normal cone
    v_2(P^2) c P(N), with weights T V (eps_k - eps_i), the normal line
    nu_j = 2 eps_j - 2 eps_i, and eps_k - eps_j (k != i, j).
    >>> X = complete_quadrics_p3()
    >>> X.dim, len(X)
    (9, 66)
    """
    n = 4
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    label = lambda i, j: "x%dx%d" % (i + 1, j + 1)
    P9 = projectivization([_add(_eps(i, n), _eps(j, n)) for i, j in pairs],
                          [label(i, j) for i, j in pairs], name="P(Sym^2 k^4)")
    V = projectivization([_eps(i, n) for i in range(n)], ["v%d" % (i + 1) for i in range(n)])
    X1 = blowup(P9, V, {"v%d" % (i + 1): label(i, i) for i in range(n)}, name="Bl_V P^9")
    # the strict transform of the rank <= 2 locus
    points, weights = [], []
    for i in range(n):
        for j in range(i + 1, n):
            w = [_sub(_eps(k, n), _eps(i, n)) for k in range(n) if k != i]
            w += [_sub(_eps(l, n), _eps(j, n)) for l in range(n) if l != j]
            points.append("S2:" + label(i, j))
            weights.append(tuple(w))
    for i in range(n):
        tangent_v = [_sub(_eps(k, n), _eps(i, n)) for k in range(n) if k != i]
        for j in range(n):
            if j == i:
                continue
            nu = _sub(_add(_eps(j, n), _eps(j, n)), _add(_eps(i, n), _eps(i, n)))
            cone = [_sub(_eps(k, n), _eps(j, n)) for k in range(n) if k not in (i, j)]
            points.append("S2:%s/%d" % (label(i, i), j + 1))
            weights.append(tuple(tangent_v + [nu] + cone))
    S2 = FixedPointData(6, n, tuple(points), tuple(weights), name="S~_2")
    return blowup(X1, S2, name="complete quadrics in P^3")
