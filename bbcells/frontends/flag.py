"""
Flag varieties G/P (PLAN.md S2.2).

P = P_J is the parabolic subgroup of the *crossed* nodes J of the Dynkin
diagram (1-based, Bourbaki numbering): its Levi subgroup has the simple
roots I = (all nodes) - J. So crossed = {1} gives E6/P1, crossed = all
nodes (the default) gives the full flag variety G/B.

Conventions (PLAN.md 1.2): fixed points wP for w in W^I (minimal coset
representatives), tangent weights w(Phi^- - Phi_I^-), characters in
simple-root coordinates (the torus of the adjoint group), cocharacters by
their values on the simple roots. The preferred cocharacter is rho^vee =
(1, ..., 1), which is dominant regular, so the cells are the Schubert cells
BwP/P and have dimension l(w).
"""

from bbcells.core import FixedPointData
from bbcells.rootsystem import RootSystem

MAX_FIXED_POINTS = 200000


def word_label(word):
    """'e' for the identity, else 's1.s3.s2' (1-based)
    >>> word_label(()), word_label((0, 2, 1))
    ('e', 's1.s3.s2')
    """
    return ".".join("s%d" % (i + 1) for i in word) if word else "e"


def _crossed_to_levi(root_system, crossed):
    n = root_system.rank
    crossed = set(range(1, n + 1)) if crossed is None else set(crossed)
    if not crossed or not crossed <= set(range(1, n + 1)):
        raise ValueError("crossed nodes must be a nonempty subset of 1..%d, got %r"
                         % (n, sorted(crossed)))
    return {i - 1 for i in range(1, n + 1) if i not in crossed}, crossed


def flag_variety(cartan_type, crossed=None, name=None):
    """FixedPointData of G/P_J for J = crossed (1-based; default: G/B)
    >>> P2 = flag_variety("A2", crossed={1})
    >>> P2.points, P2.dim
    (('e', 's1', 's2.s1'), 2)
    >>> P2.weights_of("e")
    ((-1, 0), (-1, -1))
    """
    R = RootSystem(cartan_type)
    levi, crossed = _crossed_to_levi(R, crossed)
    levi_roots = set(R.positive_roots_of(levi))
    base_weights = tuple(tuple(-c for c in beta) for beta in R.positive_roots
                         if beta not in levi_roots)
    try:
        orbit = R.orbit(levi, limit=MAX_FIXED_POINTS)
    except ValueError:
        raise ValueError("%s/P has more fixed points than the limit %d"
                         % (R.name, MAX_FIXED_POINTS)) from None
    # every orbit element s_i w is reached from w by one reflection, so its
    # tangent weights are s_i applied to those of w
    weights_by_mu = {orbit[0][0]: base_weights}
    points, weights, annotations = [], [], []
    for mu, word in orbit:
        if word:
            parent = R.reflect_weight(word[0], mu)
            wts = tuple(R.reflect(word[0], beta) for beta in weights_by_mu[parent])
            weights_by_mu[mu] = wts
        points.append(word_label(word))
        weights.append(weights_by_mu[mu])
        annotations.append({"word": tuple(i + 1 for i in word), "length": len(word),
                            "weight": mu})
    if name is None:
        name = "%s/B" % R.name if not levi else "%s/P_{%s}" % (
            R.name, ",".join(str(j) for j in sorted(crossed)))
    return FixedPointData(len(base_weights), R.rank, tuple(points), tuple(weights),
                          edges=tuple(_edges(R, orbit, points, weights)),
                          preferred_cocharacter=(1,) * R.rank, name=name,
                          annotations=tuple(annotations))


def _edges(R, orbit, points, weights):
    """T-invariant curves: through wP with tangent weight beta, joining wP
    and s_beta wP (GKM: the tangent weights at a point are pairwise
    independent roots). Each curve is listed once."""
    label_of = {mu: label for (mu, _), label in zip(orbit, points)}
    seen = set()
    for (mu, _), label, wts in zip(orbit, points, weights):
        for beta in wts:
            other = label_of[R.reflect_weight_by_root(beta, mu)]
            key = frozenset((label, other))
            if key not in seen:
                seen.add(key)
                yield (label, other, beta)


def grassmannian(k, n):
    """Gr(k, n) = SL_n / P_k"""
    if not 0 < k < n:
        raise ValueError("need 0 < k < n")
    return flag_variety("A%d" % (n - 1), crossed={k}, name="Gr(%d,%d)" % (k, n))


def full_flags(cartan_type):
    return flag_variety(cartan_type)


def isotropic_grassmannian(k, n):
    """isotropic k-planes for a symplectic form on k^{2n}: Sp_{2n} / P_k"""
    if not 0 < k <= n:
        raise ValueError("need 0 < k <= n")
    letter = "C%d" % n if n >= 2 else "A1"
    return flag_variety(letter, crossed={k}, name="IG(%d,%d)" % (k, 2 * n))
