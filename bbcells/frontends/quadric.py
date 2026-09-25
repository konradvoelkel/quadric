"""
Smooth split projective quadrics Q_n = SO(n+2)/P_1 (PLAN.md S2.3, 1.3).

Q_n = {x_0 y_0 + ... + x_{m-1} y_{m-1} (- z^2) = 0} in P^{n+1}, with the
maximal torus of SO(n+2) in epsilon-coordinates: the basis vector of x_i has
weight epsilon_{i+1}, that of y_i weight -epsilon_{i+1}, and z weight 0.
Type B_m for n = 2m - 1 and D_m for n = 2m - 2 (Bourbaki: alpha_i =
eps_i - eps_{i+1}, alpha_m = eps_m in B_m, eps_{m-1} + eps_m in D_m).

The fixed points are the coordinate points x_i, y_i (labels as in quadric.py),
where the line has weight nu = +-eps_{i+1}. The tangent weights are mu - nu
for mu the weights of L^perp/L; their sum is -n nu, which is how the labels
are recovered from the flag front end.

Relation to the legacy script: quadric.py's cell dimensions for a
cocharacter c are the plus-cell dimensions here for -c (SPEC C4).
"""

from bbcells.core import FixedPointData
from bbcells.frontends.flag import flag_variety
from bbcells.operations import restrict


def type_of(n):
    """Cartan type and rank of SO(n+2)
    >>> [type_of(n) for n in (1, 2, 3, 4, 5)]
    [('B', 1), ('D', 2), ('B', 2), ('D', 3), ('B', 3)]
    """
    if n < 1:
        raise ValueError("quadrics of dimension >= 1 only")
    m = n // 2 + 1
    return ("B" if n % 2 else "D"), m


def simple_roots_epsilon(letter, m):
    """columns: the simple roots in epsilon-coordinates (an m x m matrix)"""
    columns = []
    for i in range(m - 1):
        columns.append([1 if k == i else -1 if k == i + 1 else 0 for k in range(m)])
    if letter == "B":
        columns.append([1 if k == m - 1 else 0 for k in range(m)])
    else:
        columns.append([1 if k in (m - 2, m - 1) else 0 for k in range(m)])
    return [[columns[j][i] for j in range(m)] for i in range(m)]


def rho_vee_epsilon(letter, m):
    """the cocharacter with value 1 on every simple root, in epsilon-coordinates
    >>> rho_vee_epsilon("B", 3), rho_vee_epsilon("D", 3)
    ((3, 2, 1), (2, 1, 0))
    """
    if letter == "B":
        return tuple(range(m, 0, -1))
    return tuple(range(m - 1, -1, -1))


def coordinate_label(nu):
    """legacy coordinate name of the line with weight nu = +-eps_{i+1}
    >>> coordinate_label((0, 1, 0)), coordinate_label((0, 0, -1))
    ('x_1', 'y_2')
    """
    (i,) = [k for k, c in enumerate(nu) if c]
    return ("x_%d" if nu[i] > 0 else "y_%d") % i


def quadric(n):
    """FixedPointData of Q_n with epsilon-coordinates and labels x_i, y_i
    >>> Q4 = quadric(4)
    >>> Q4.points
    ('x_0', 'x_1', 'x_2', 'y_2', 'y_1', 'y_0')
    >>> Q4.weights_of("x_0")
    ((-1, -1, 0), (-1, 0, -1), (-1, 0, 1), (-1, 1, 0))
    """
    letter, m = type_of(n)
    crossed = {1, 2} if (letter, m) == ("D", 2) else {1}
    data = flag_variety("%s%d" % (letter, m), crossed=crossed)
    data = restrict(data, simple_roots_epsilon(letter, m))
    points, weights, annotations = [], [], []
    for label, wts in zip(data.points, data.weights):
        total = [sum(w[k] for w in wts) for k in range(m)]
        if any(c % n for c in total):
            raise AssertionError("tangent weights at %r do not sum to -n nu" % label)
        nu = tuple(-c // n for c in total)
        points.append(coordinate_label(nu))
        weights.append(tuple(sorted(wts)))
        annotations.append(dict(data.annotation(label), line_weight=nu, word_label=label))
    return FixedPointData(n, m, tuple(points), tuple(weights),
                          preferred_cocharacter=rho_vee_epsilon(letter, m),
                          name="Q_%d" % n, annotations=tuple(annotations))
