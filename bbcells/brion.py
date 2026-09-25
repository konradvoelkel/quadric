"""
Equivariant cohomology beyond GKM (docs/S6d.md, section 10).

Brion (Equivariant cohomology and equivariant intersection theory, 1998,
Thm 6 and the proof of Thm 9): for a smooth complete spherical X with
H_T^*(X) free, the restriction to X^T is injective, and its image is cut out,
for each codimension-one subtorus ker(chi), by conditions on the connected
components Y of X^{ker chi}. Such a Y is a point, a P^1, the plane P(sl_2)
or a rational ruled surface, and the conditions on (f_p) are
  (1) P^1 with fixed points y, z:             f_y = f_z mod chi;
  (2), (3) a surface with fixed points Y^T:  all f_p congruent mod chi and
      sum_{p in Y^T} f_p / e_p a polynomial, e_p the product of the two
      weights of T_p Y (for a P(sl_2) these are {a, -a}, {a, 2a}, {-a, -2a}).
GKM data is the case where every Y is a point or a P^1.

The components are inferred from the fixed-point data: points of one
component have isomorphic ker(chi)-representations T_p X (their weights agree
modulo Z chi), and the weights along chi must form one of the patterns
above. Ambiguous situations raise. The result is validated by comparing the
graded dimensions of the ring with those of a free module with the BB
Poincare series (ring_dimension vs equivariant.expected_dimension).
"""

from fractions import Fraction
from math import gcd

from bbcells.linalg import rank
from bbcells.polynomial import Poly, monomials


def primitive(w):
    """the primitive vector on the line of w, with first nonzero coordinate > 0
    >>> primitive((0, -2, 4))
    (0, 1, -2)
    """
    g = 0
    for c in w:
        g = gcd(g, abs(c))
    v = tuple(c // g for c in w)
    first = next(c for c in v if c)
    return v if first > 0 else tuple(-c for c in v)


def _multiple(w, chi):
    """k with w = k chi, or None"""
    j = next(i for i, c in enumerate(chi) if c)
    if w[j] % chi[j]:
        return None
    k = w[j] // chi[j]
    return k if all(a == k * b for a, b in zip(w, chi)) else None


def _class_mod(w, chi):
    """canonical representative of w in Z^r / Z chi"""
    j = next(i for i, c in enumerate(chi) if c)
    k = (w[j] - w[j] % abs(chi[j])) // chi[j]          # makes coordinate j lie in [0, |chi_j|)
    return tuple(a - k * b for a, b in zip(w, chi))


def fixed_components(data):
    """[(chi, points, kind)] for the components of X^{ker chi} with at least
    two T-fixed points, kind in {"curve", "plane", "ruled"}
    >>> from bbcells.frontends.toric import fixed_point_data, projective_space
    >>> sorted(kind for _, _, kind in fixed_components(fixed_point_data(projective_space(2))))
    ['curve', 'curve', 'curve']
    """
    directions = {primitive(w) for weights in data.weights for w in weights}
    components = []
    for chi in sorted(directions):
        groups = {}
        for p, weights in zip(data.points, data.weights):
            along = [k for k in (_multiple(w, chi) for w in weights) if k is not None]
            if not along:
                continue
            rest = sorted(_class_mod(w, chi) for w in weights if _multiple(w, chi) is None)
            groups.setdefault((len(along), tuple(rest)), []).append((p, tuple(sorted(along))))
        for (m, _), members in groups.items():
            if m == 1:
                by_weight = {}
                for p, (k,) in members:
                    by_weight.setdefault(k, []).append(p)
                if any(-k not in by_weight for k in by_weight):
                    raise ValueError("unpaired T-curve of direction %r" % (chi,))
                for k in sorted(k for k in by_weight if k > 0):
                    ends, opposite = by_weight[k], by_weight[-k]
                    if len(ends) != 1 or len(opposite) != 1:
                        raise ValueError("ambiguous T-curves of weight %d.%r: %r and %r"
                                         % (k, chi, ends, opposite))
                    components.append((chi, (ends[0], opposite[0]), "curve"))
            elif m == 2:
                patterns = sorted(along for _, along in members)
                points = tuple(p for p, _ in members)
                values = sorted({abs(x) for pattern in patterns for x in pattern})
                kind = None
                if len(members) == 3 and len(values) == 2 and values[1] == 2 * values[0]:
                    a = values[0]                  # P(sl_2): {a, -a}, {a, 2a}, {-a, -2a}
                    if patterns == sorted([(-a, a), (a, 2 * a), (-2 * a, -a)]):
                        kind = "plane"
                if len(members) == 4 and len(values) in (1, 2):
                    a, b = (values * 2)[:2] if len(values) == 1 else values
                    expected = sorted(tuple(sorted((s * a, t * b)))
                                      for s in (1, -1) for t in (1, -1))
                    if patterns == expected:
                        kind = "ruled"             # {+-a, +-b}: F_n or P^1 x P^1
                if kind is None:
                    raise ValueError("unrecognized fixed-point surface of direction %r: %r"
                                     % (chi, members))
                components.append((chi, points, kind))
            else:
                raise ValueError("a component of X^{ker %r} of dimension %d" % (chi, m))
    return components


def _derivative(poly, j):
    terms = {}
    for e, c in poly.terms.items():
        if e[j]:
            f = tuple(x - (1 if i == j else 0) for i, x in enumerate(e))
            terms[f] = terms.get(f, 0) + c * e[j]
    return Poly(poly.nvars, terms)


def ring_dimension(data, components, degree):
    """dimension of the degree-`degree` part of the ring of tuples (f_p)
    satisfying Brion's conditions for the given components"""
    nvars = data.rank
    basis = monomials(nvars, degree)
    index = {}
    for p in data.points:
        for m in basis:
            index[(p, m)] = len(index)
    rows = []
    weights_of = dict(zip(data.points, data.weights))

    def add_rows(combination, transform):
        """rows for 'transform(sum_p c_p f_p) = 0' (a linear map to polynomials)"""
        images = {}
        for p, c in combination:
            for m in basis:
                image = transform(Poly(nvars, {m: Fraction(c)}))
                for key, value in image.terms.items():
                    images.setdefault(key, {})
                    images[key][index[(p, m)]] = images[key].get(index[(p, m)], 0) + value
        for key, entries in images.items():
            row = [Fraction(0)] * len(index)
            for column, value in entries.items():
                row[column] += value
            if any(row):
                rows.append(row)

    for chi, points, kind in components:
        restrict = lambda f, chi=chi: f.restrict_to_hyperplane(chi)
        for q in points[1:]:
            add_rows([(points[0], 1), (q, -1)], restrict)
        if kind in ("plane", "ruled"):
            coefficients = []
            for p in points:
                along = [w for w in weights_of[p] if _multiple(w, chi) is not None]
                k = _multiple(along[0], chi) * _multiple(along[1], chi)
                coefficients.append((p, Fraction(1, k)))
            j = next(i for i, c in enumerate(chi) if c)
            add_rows(coefficients, restrict)
            add_rows(coefficients, lambda f, chi=chi, j=j: _derivative(f, j).restrict_to_hyperplane(chi))
    return len(index) - (rank(rows) if rows else 0)
