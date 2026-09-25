"""
The variety of complete conics: the blow-up of P^5 = P(Sym^2 k^3) along the
Veronese surface (PLAN.md S6c.1), with the torus of GL_3.

The basis x_i x_j of Sym^2 has weight eps_i + eps_j; the Veronese surface is
v_2(P^2) with fixed points x_i^2 and tangent weights eps_j - eps_i (the
Veronese map is equivariant). The normal weights at x_i^2 are 2(eps_j -
eps_i) and eps_j + eps_k - 2 eps_i, pairwise distinct, so PLAN.md 1.6
applies: 3 + 3 * 3 = 12 fixed points.
"""

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
