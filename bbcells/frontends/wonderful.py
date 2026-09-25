"""
Wonderful compactifications of adjoint groups (PLAN.md S4.1, 1.4).

The torus is T x T, acting on G by (s, t) . x = s x t^{-1}; characters are
pairs (chi_1, chi_2) of simple-root coordinate vectors, concatenated. The
fixed points are (u, v) in W x W, all in the closed orbit G/B^- x G/B, with
tangent weights

    (u alpha, 0) and (0, -v alpha) for alpha > 0,  (u alpha_i, -v alpha_i) for simple alpha_i.

The preferred cocharacter (rho^vee, h rho^vee), h the Coxeter number, is
generic: |ht(u alpha_i)| < h <= |h ht(v alpha_i)|.
"""

from bbcells.core import FixedPointData
from bbcells.frontends.flag import word_label
from bbcells.rootsystem import RootSystem

MAX_FIXED_POINTS = 200000


def _images(R, orbit, roots):
    """{w (as its image of rho): tuple of w(beta) for beta in roots}"""
    images = {orbit[0][0]: tuple(roots)}
    for mu, word in orbit[1:]:
        parent = R.reflect_weight(word[0], mu)
        images[mu] = tuple(R.reflect(word[0], beta) for beta in images[parent])
    return images


def wonderful_compactification(cartan_type, max_points=MAX_FIXED_POINTS):
    """
    >>> X = wonderful_compactification("A1")          # P^3 = P(M_2)
    >>> X.dim, X.rank, len(X)
    (3, 2, 4)
    >>> X.weights_of("e|e")
    ((1, 0), (0, -1), (1, -1))
    """
    R = RootSystem(cartan_type)
    if R.weyl_group_order ** 2 > max_points:
        raise ValueError("the wonderful compactification of %s has %d fixed points, more "
                         "than the limit %d" % (R.name, R.weyl_group_order ** 2, max_points))
    orbit = R.orbit(set())
    positive = list(R.positive_roots)
    simple = [tuple(int(i == j) for j in range(R.rank)) for i in range(R.rank)]
    images = _images(R, orbit, positive + simple)
    n_pos, zero = len(positive), (0,) * R.rank
    neg = lambda x: tuple(-c for c in x)
    points, weights, annotations = [], [], []
    for mu_u, word_u in orbit:
        u_pos, u_simple = images[mu_u][:n_pos], images[mu_u][n_pos:]
        for mu_v, word_v in orbit:
            v_pos, v_simple = images[mu_v][:n_pos], images[mu_v][n_pos:]
            wts = ([a + zero for a in u_pos] + [zero + neg(b) for b in v_pos] +
                   [a + neg(b) for a, b in zip(u_simple, v_simple)])
            points.append("%s|%s" % (word_label(word_u), word_label(word_v)))
            weights.append(tuple(wts))
            annotations.append({"u": tuple(i + 1 for i in word_u),
                                "v": tuple(i + 1 for i in word_v)})
    h = max(R.height(b) for b in positive) + 1
    return FixedPointData(n_pos * 2 + R.rank, 2 * R.rank, tuple(points), tuple(weights),
                          preferred_cocharacter=(1,) * R.rank + (h,) * R.rank,
                          name="wonderful(%s)" % R.name, annotations=tuple(annotations))
