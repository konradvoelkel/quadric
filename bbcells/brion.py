"""
Equivariant cohomology beyond GKM (docs/cohomology.md).

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



def invariant_curves(data, components=None):
    """the T-invariant curves joining fixed points that the components of
    the X^{ker chi} provide, as edges (p, q, weight at p):
      * every P^1 component;
      * the two lines of a P(sl_2) plane, from the points with weights
        {a, 2a} and {-a, -2a} to the point with {a, -a}, of weight +-a (the
        conics of weight 2a joining the two ends are left out);
      * the four boundary curves of a ruled surface {+-a, +-b}, joining
        points that differ in one sign.
    For GKM data these are the GKM edges. (The surfaces contain infinitely
    many further T-curves, all joining the source and the sink of the
    surface.)
    >>> from bbcells.frontends.spherical import complete_quadrics
    >>> len(invariant_curves(complete_quadrics(3)))      # 12 curves + 6 planes x 2 lines
    24
    """
    components = fixed_components(data) if components is None else components
    weights_of = dict(zip(data.points, data.weights))
    edges = []
    for chi, points, kind in components:
        along = {p: sorted((_multiple(w, chi), w) for w in weights_of[p]
                           if _multiple(w, chi) is not None) for p in points}
        if kind == "curve":
            p, q = points
            (_, w), = along[p]
            edges.append((p, q, w))
        elif kind == "plane":
            pattern = {tuple(k for k, _ in along[p]): p for p in points}
            a = min(abs(k) for p in points for k, _ in along[p])
            middle = pattern[(-a, a)]
            for sign in (1, -1):
                end = pattern[tuple(sorted((sign * a, 2 * sign * a)))]
                edges.append((end, middle, next(w for k, w in along[end] if k == sign * a)))
        else:
            # extremes {a, b}, {-a, -b} (one sign) and saddles {a, -b}, {-a, b}; the
            # boundary curves join each extreme to each saddle (also when a = b)
            same = lambda p: len({k > 0 for k, _ in along[p]}) == 1
            for p in points:
                if not same(p):
                    continue
                for q in points:
                    if same(q):
                        continue
                    kq = [k for k, _ in along[q]]
                    k, w = next((k, w) for k, w in along[p] if -k in kq)
                    edges.append((p, q, w))
    return edges


def with_invariant_curves(data):
    """the fixed-point data with invariant_curves as its edges
    >>> from bbcells.frontends.spherical import complete_quadrics
    >>> X = complete_quadrics(3)                    # complete conics, not GKM
    >>> X.edges is None, len(with_invariant_curves(X).edges)
    (True, 24)
    """
    from dataclasses import replace
    return replace(data, edges=tuple(invariant_curves(data)))


def _derivative(poly, j):
    terms = {}
    for e, c in poly.terms.items():
        if e[j]:
            f = tuple(x - (1 if i == j else 0) for i, x in enumerate(e))
            terms[f] = terms.get(f, 0) + c * e[j]
    return Poly(poly.nvars, terms)


def ring_dimension(data, components, degree):
    """dimension of the degree-`degree` part of the ring of tuples (f_p)
    satisfying Brion's conditions for the given components. For complete
    conics it is that of a free module with the BB Poincare series:
    >>> from bbcells.core import bb_cells
    >>> from bbcells.equivariant import expected_dimension
    >>> from bbcells.frontends.spherical import complete_quadrics
    >>> X = complete_quadrics(3)
    >>> components = fixed_components(X)
    >>> [ring_dimension(X, components, d) for d in range(3)]
    [1, 4, 10]
    >>> [expected_dimension(bb_cells(X).counts, X.rank, d) for d in range(3)]
    [1, 4, 10]
    """
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


def _proportional_rows(chi):
    """rows of the linear conditions 'v is a multiple of chi'"""
    r = len(chi)
    rows = []
    for i in range(r):
        for j in range(i + 1, r):
            v = [0] * r
            v[i], v[j] = chi[j], -chi[i]
            if any(v):
                rows.append(v)
    return rows


def degree_one_class(data, components, prescribed):
    """the degree-one class (f_p) of the ring (a character f_p at every fixed
    point, as a tuple of Fractions) with f_p = prescribed[p] for the given
    points. Conditions: f_p - f_q is a multiple of chi along every component,
    and sum_p f_p / k_p = 0 on every surface (a linear form divisible by chi^2
    vanishes). The values are propagated: an unknown point whose components to
    known points determine it (a small linear system of full rank) is solved
    first. At the end every condition is verified. Raises unless the class
    exists and is determined by the prescribed values. On P^2 = P(V), with
    lines of weights 0, (1, 0), (0, 1) at a, b, c, O(1) has weight -u at the
    line of weight u; its values at a and b determine it:
    >>> from bbcells.core import FixedPointData
    >>> P2 = FixedPointData(2, 2, ("a", "b", "c"),
    ...     (((1, 0), (0, 1)), ((-1, 0), (-1, 1)), ((0, -1), (1, -1))))
    >>> H = degree_one_class(P2, fixed_components(P2), {"a": (0, 0), "b": (-1, 0)})
    >>> [tuple(int(x) for x in H[p]) for p in P2.points]
    [(0, 0), (-1, 0), (0, -1)]
    """
    from bbcells.linalg import solve
    r = data.rank
    weights_of = dict(zip(data.points, data.weights))
    values = {p: tuple(Fraction(x) for x in v) for p, v in prescribed.items()}
    at = {p: [] for p in data.points}
    surface_coefficients = {}
    for index, (chi, points, kind) in enumerate(components):
        for p in points:
            at[p].append(index)
        if kind in ("plane", "ruled"):
            coefficients = {}
            for p in points:
                along = [w for w in weights_of[p] if _multiple(w, chi) is not None]
                coefficients[p] = Fraction(1, _multiple(along[0], chi) * _multiple(along[1], chi))
            surface_coefficients[index] = coefficients
    rows_of = {}

    def rows(chi):
        if chi not in rows_of:
            rows_of[chi] = _proportional_rows(chi)
        return rows_of[chi]

    progress = True
    while progress and len(values) < len(data.points):
        progress = False
        for q in data.points:
            if q in values:
                continue
            A, b = [], []
            for index in at[q]:
                chi, points, kind = components[index]
                for p in points:
                    if p != q and p in values:
                        for row in rows(chi):       # row . (f_q - f_p) = 0
                            A.append(list(row))
                            b.append(sum(x * y for x, y in zip(row, values[p])))
                coefficients = surface_coefficients.get(index)
                if coefficients and all(p in values for p in points if p != q):
                    for i in range(r):              # sum_p c_p f_p = 0, coordinate i
                        A.append([coefficients[q] if j == i else 0 for j in range(r)])
                        b.append(-sum(coefficients[p] * values[p][i]
                                      for p in points if p != q))
            if A and rank(A) == r:
                solution = solve(A, b)
                if solution is None:
                    raise ValueError("no class with these values (at %r)" % (q,))
                values[q] = tuple(Fraction(x) for x in solution)
                progress = True
    if len(values) < len(data.points):
        raise ValueError("the prescribed values do not determine the class")
    for index, (chi, points, kind) in enumerate(components):     # verify everything
        for q in points[1:]:
            difference = [x - y for x, y in zip(values[q], values[points[0]])]
            if any(sum(a * d for a, d in zip(row, difference)) for row in rows(chi)):
                raise ValueError("no class with these values (along %r)" % (chi,))
        coefficients = surface_coefficients.get(index)
        if coefficients:
            for i in range(r):
                if sum(c * values[p][i] for p, c in coefficients.items()):
                    raise ValueError("no class with these values (surface %r)" % (chi,))
    return values


def integrate_monomial(data, classes, exponents, lam):
    """ABBV: the integral of prod_i c_i^{e_i} over X, for degree-one classes
    c_i ({point: character}) with sum e_i = dim X, evaluated at lambda (the
    result does not depend on lambda). The hyperplane class of P^2 (see
    degree_one_class) has H^2 = 1:
    >>> from bbcells.core import FixedPointData
    >>> P2 = FixedPointData(2, 2, ("a", "b", "c"),
    ...     (((1, 0), (0, 1)), ((-1, 0), (-1, 1)), ((0, -1), (1, -1))))
    >>> H = {"a": (0, 0), "b": (-1, 0), "c": (0, -1)}
    >>> integrate_monomial(P2, [H], [2], (1, 2)), integrate_monomial(P2, [H], [2], (5, -3))
    (Fraction(1, 1), Fraction(1, 1))
    """
    pair = lambda w: sum(Fraction(a) * b for a, b in zip(w, lam))
    total = Fraction(0)
    for p, weights in zip(data.points, data.weights):
        numerator = Fraction(1)
        for c, e in zip(classes, exponents):
            numerator *= pair(c[p]) ** e
        denominator = Fraction(1)
        for w in weights:
            denominator *= pair(w)
        total += numerator / denominator
    return total


def line_bundle_class(data, components, cartan_type, weight, orbit="closed"):
    """the equivariant first Chern class of the G-linearized line bundle whose
    fibre at the base point z of the closed orbit (the B-fixed point, labelled
    '<orbit>:e' by spherical.assemble) has T-weight `weight` (simple-root
    coordinates, Fractions allowed): it is w(weight) at w.z, and is extended to
    all fixed points by degree_one_class. For the colour of a wonderful
    variety with B-weight omega_D the weight is -omega_D (O(1) on a projective
    space has weight -lambda at a line of weight lambda); for complete quadrics
    the colours are the pullbacks mu_k of O(1) from P(Sym^2 Lambda^k V), with
    omega_D = 2 omega_k.
    >>> from bbcells.frontends.spherical import complete_quadrics
    >>> from bbcells.rootsystem import RootSystem
    >>> X = complete_quadrics(3)
    >>> components = fixed_components(X)
    >>> R = RootSystem("A2")
    >>> mu, nu = (line_bundle_class(X, components, "A2",
    ...           tuple(-2 * x for x in R.fundamental_weight(k))) for k in (0, 1))
    >>> tangency = {p: tuple(2 * a + 2 * b for a, b in zip(mu[p], nu[p])) for p in X.points}
    >>> integrate_monomial(X, [tangency], [5], (3, 7))           # Chasles: 3264 conics
    Fraction(3264, 1)
    """
    from bbcells.frontends.spherical import _word_from_label
    from bbcells.rootsystem import RootSystem
    R = RootSystem(cartan_type)
    prescribed = {}
    for p in data.points:
        annotation = data.annotation(p)
        if annotation.get("orbit") == orbit:
            word = _word_from_label(annotation["word"])
            prescribed[p] = R.apply_word(word, tuple(weight))
    if not prescribed:
        raise ValueError("no fixed points in the orbit %r" % (orbit,))
    return degree_one_class(data, components, prescribed)


def complete_quadric_colours(n):
    """(X, [mu_1, ..., mu_{n-1}]): the fixed-point data of complete quadrics in
    P^{n-1} and the equivariant classes of the colours mu_k, the pullbacks of
    O(1) from P(Sym^2 Lambda^k V)
    >>> X, (mu, nu) = complete_quadric_colours(3)
    >>> len(X), integrate_monomial(X, [mu, nu], [2, 3], (3, 7))
    (12, Fraction(4, 1))
    """
    from bbcells.frontends.spherical import complete_quadrics
    from bbcells.rootsystem import RootSystem
    X = complete_quadrics(n)
    cartan_type = "A%d" % (n - 1)
    R = RootSystem(cartan_type)
    components = fixed_components(X)
    return X, [line_bundle_class(X, components, cartan_type,
                                 tuple(-2 * x for x in R.fundamental_weight(k)))
               for k in range(R.rank)]


def characteristic_number(n, exponents=None, coefficients=None, lam=None):
    """a characteristic number of complete quadrics in P^{n-1}: the integral
    of prod mu_k^{e_k} (exponents), or of (sum a_k mu_k)^N (coefficients), by
    localization at a generic cocharacter; checked to be an integer.
    >>> characteristic_number(3, coefficients=(2, 2))       # Chasles
    3264
    >>> characteristic_number(4, exponents=(3, 3, 3))       # Schubert's triangle
    104
    """
    X, mu = complete_quadric_colours(n)
    if lam is None:
        lam = tuple(7 ** (k + 1) + 3 * k for k in range(n - 1))
    if exponents is not None:
        if len(exponents) != n - 1 or sum(exponents) != X.dim:
            raise ValueError("need %d exponents with sum %d" % (n - 1, X.dim))
        value = integrate_monomial(X, mu, exponents, lam)
    elif coefficients is not None:
        if len(coefficients) != n - 1:
            raise ValueError("need %d coefficients" % (n - 1))
        L = {p: tuple(sum(a * c[p][i] for a, c in zip(coefficients, mu))
                      for i in range(n - 1)) for p in X.points}
        value = integrate_monomial(X, [L], [X.dim], lam)
    else:
        raise ValueError("give exponents or coefficients")
    if value.denominator != 1:
        raise ValueError("the cocharacter %r is not generic" % (lam,))
    return int(value)


# -- canonical classes and the integral cohomology ring (PLAN.md S7.3b) -------

def _negative_multiplicities(data, components, lam):
    """{(component index, point): number of lambda-negative weights of the
    point along chi}"""
    weights_of = dict(zip(data.points, data.weights))
    pair = lambda w: sum(a * b for a, b in zip(w, lam))
    result = {}
    for index, (chi, points, kind) in enumerate(components):
        for p in points:
            along = [w for w in weights_of[p] if _multiple(w, chi) is not None]
            result[(index, p)] = sum(1 for w in along if pair(w) < 0)
    return result


def morse_order(data, components, lam):
    """the fixed points in an order in which, on every component Y of some
    X^{ker chi}, a point comes after the points of Y with fewer lambda-negative
    weights along chi (the order of a Morse function of an ample class; raises
    on a cycle). On P^2 it runs from the open cell down to the point:
    >>> from bbcells.core import FixedPointData
    >>> P2 = FixedPointData(2, 2, ("a", "b", "c"),
    ...     (((1, 0), (0, 1)), ((-1, 0), (-1, 1)), ((0, -1), (1, -1))))
    >>> components = fixed_components(P2)
    >>> morse_order(P2, components, (1, 2)), morse_order(P2, components, (-1, -2))
    (['a', 'b', 'c'], ['c', 'b', 'a'])
    """
    m = _negative_multiplicities(data, components, lam)
    after = {p: set() for p in data.points}          # after[p]: points that must precede p
    for index, (chi, points, kind) in enumerate(components):
        for p in points:
            for q in points:
                if m[(index, q)] < m[(index, p)]:
                    after[p].add(q)
    position = {p: k for k, p in enumerate(data.points)}
    waiting = {p: len(before) for p, before in after.items()}
    successors = {p: [] for p in data.points}
    for p, before in after.items():
        for q in before:
            successors[q].append(p)
    import heapq
    ready = [(position[p], p) for p, k in waiting.items() if k == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, p = heapq.heappop(ready)
        order.append(p)
        for s in successors[p]:
            waiting[s] -= 1
            if waiting[s] == 0:
                heapq.heappush(ready, (position[s], s))
    if len(order) != len(data.points):
        raise ValueError("the component relations have a cycle")
    return order


def _split(F, ell):
    """(F0, F1) with F = F0 + ell F1 mod ell^2, both free of the variable that
    restrict_to_hyperplane(ell) eliminates"""
    F0 = F.restrict_to_hyperplane(ell)
    rest = F - F0
    F1 = rest.divide_by_linear(ell).restrict_to_hyperplane(ell) if not rest.is_zero() else rest
    return F0, F1


def _on_hyperplane(chi, ell):
    """the linear form chi restricted to {ell = 0} (eliminating the variable
    that restrict_to_hyperplane(ell) eliminates), as a vector"""
    j = max(i for i, c in enumerate(ell) if c)
    k = Fraction(chi[j], ell[j])
    return tuple(Fraction(a) - k * b for a, b in zip(chi, ell))


def _divide(F, factors):
    for chi, e in factors:
        for _ in range(e):
            F = F.divide_by_linear(chi)
    return F


def _interpolate(constraints, degree, nvars):
    """the homogeneous polynomial g of the given degree with g = L mod ell^m
    for every (ell, m, L) (m in {1, 2}, the ell pairwise non-proportional,
    sum m > degree), by Newton's scheme g = A + prod(ell_i^m_i) g'. Raises if
    the constraints are inconsistent."""
    (ell, m, L), rest = constraints[0], constraints[1:]
    A, moduli = L, [(ell, m)]
    for ell, m, L in rest:
        e = degree - sum(k for _, k in moduli)
        D0, D1 = _split(L - A, ell)
        if e < 0:
            if not D0.is_zero() or (m == 2 and not D1.is_zero()):
                raise ValueError("inconsistent constraints")
            moduli.append((ell, m))
            continue
        restricted = [(_on_hyperplane(chi, ell), k) for chi, k in moduli]
        g0 = _divide(D0, restricted)
        correction = g0
        if m == 2:
            P = Poly.product_of_linear([chi for chi, k in moduli for _ in range(k)], nvars)
            P0, P1 = _split(P, ell)
            numerator = D1 - (P1 * g0).restrict_to_hyperplane(ell)
            if e == 0:
                if not numerator.is_zero():
                    raise ValueError("inconsistent constraints")
            else:
                correction = g0 + Poly.linear(ell) * _divide(numerator, restricted)
        product = Poly.product_of_linear([chi for chi, k in moduli for _ in range(k)], nvars)
        A = A + product * correction
        moduli.append((ell, m))
    return A


def canonical_classes(data, components, lam, points=None, verify=True):
    """{p: {q: Poly}}: the classes tau_p in H_T^*(X) with tau_p(p) = the
    product of the lambda-negative weights at p and tau_p(q) = 0 for q != p
    with codim(q) <= codim(p) (Goldin-Tolman canonical classes; the classes of
    the closures of the plus-cells, which they characterize uniquely).
    Points are processed in morse_order. At q, every lambda-negative weight
    lies on a component Y of X^{ker chi} through q; for m negative weights of
    q on Y the ring conditions give g = L mod chi^m, with L the value at an
    earlier point of Y (m = 1) or from the sum condition of the surface
    (m = 2). The product of these moduli is the Euler class of the negative
    part, of degree codim(q) > codim(p), so g is unique; it is found by
    _interpolate. With verify, every ring condition is checked at the end.
    Such classes exist iff no plus-cell closure meets a cell of at least its
    own dimension (then they are the closures' classes); otherwise this
    raises. For complete conics it raises for every generic lambda.
    On P^2 the class of the line (cells a > b > c) as in flow_up_class:
    >>> from bbcells.core import FixedPointData
    >>> P2 = FixedPointData(2, 2, ("a", "b", "c"),
    ...     (((1, 0), (0, 1)), ((-1, 0), (-1, 1)), ((0, -1), (1, -1))))
    >>> classes = canonical_classes(P2, fixed_components(P2), (1, 2))
    >>> {q: str(f) for q, f in classes["b"].items()}
    {'a': '0', 'b': '-t1', 'c': '-t2'}
    >>> from bbcells.frontends.spherical import complete_quadrics
    >>> X = complete_quadrics(3)
    >>> canonical_classes(X, fixed_components(X), (1, 5))    # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: no canonical class for 'closed:s2.s1': the closure of its plus-cell meets ...
    """
    nvars = data.rank
    weights_of = dict(zip(data.points, data.weights))
    pair = lambda w: sum(a * b for a, b in zip(w, lam))
    order = morse_order(data, components, lam)
    rank_in_order = {p: k for k, p in enumerate(order)}
    m = _negative_multiplicities(data, components, lam)
    codim = {p: sum(1 for w in weights_of[p] if pair(w) < 0) for p in data.points}
    at = {p: [] for p in data.points}
    coefficients = {}
    for index, (chi, pts, kind) in enumerate(components):
        for p in pts:
            at[p].append(index)
        if kind in ("plane", "ruled"):
            c = {}
            for p in pts:
                along = [w for w in weights_of[p] if _multiple(w, chi) is not None]
                c[p] = Fraction(1, _multiple(along[0], chi) * _multiple(along[1], chi))
            coefficients[index] = c
    zero = Poly(nvars)
    result = {}
    for p in (data.points if points is None else points):
        d = codim[p]
        values = {}
        for q in order:
            if q == p:
                values[q] = Poly.product_of_linear(
                    [w for w in weights_of[q] if pair(w) < 0], nvars)
                continue
            if codim[q] <= d or rank_in_order[q] < rank_in_order[p]:
                values[q] = zero          # before p: zero by induction (checked below)
                continue
            constraints = []
            for index in at[q]:
                k = m[(index, q)]
                if k == 0:
                    continue
                chi, pts, kind = components[index]
                if k == 1:
                    r = next(x for x in pts if x != q and m[(index, x)] == 0)
                    constraints.append((chi, 1, values[r]))
                else:
                    c = coefficients[index]
                    L = zero
                    for y in pts:
                        if y != q:
                            L = L + values[y] * Poly.constant(-c[y] / c[q], nvars)
                    constraints.append((chi, 2, L))
            if all(L.is_zero() for _, _, L in constraints):
                values[q] = zero
                continue
            try:
                values[q] = _interpolate(constraints, d, nvars)
            except ValueError:
                raise ValueError("no canonical class for %r: the closure of its plus-cell "
                                 "meets a cell of at least its dimension (at %r)" % (p, q))
        if verify:
            _verify_class(values, components, coefficients)
        result[p] = values
    return result


def _verify_class(values, components, coefficients):
    for index, (chi, points, kind) in enumerate(components):
        for q in points[1:]:
            if not (values[q] - values[points[0]]).restrict_to_hyperplane(chi).is_zero():
                raise ValueError("not in the ring: %r along %r" % (q, chi))
        c = coefficients.get(index)
        if c:
            total = Poly(values[points[0]].nvars)
            for p in points:
                total = total + values[p] * Poly.constant(c[p], total.nvars)
            j = next(i for i, x in enumerate(chi) if x)
            if not total.restrict_to_hyperplane(chi).is_zero() or \
                    not _derivative(total, j).restrict_to_hyperplane(chi).is_zero():
                raise ValueError("not in the ring: the surface of direction %r" % (chi,))


def integral_cohomology(data, components, lam, point=None):
    """the ring H^*(X; Z) in the basis of the canonical classes (cell
    closures): returns (order, codim, constants, pairing) with constants[(a, b)]
    = {s: c} for tau_a tau_b = sum_s c tau_s (codim s = codim a + codim b) and
    pairing[(a, b)] = int_X tau_a tau_b (codim a + codim b = dim X). The
    equivariant expansion is evaluated at a rational point, where it is
    triangular in morse_order; degree-zero coefficients are the ordinary
    structure constants. Raises unless they are integers and the pairing is
    unimodular in every degree. For Gr(2, 4): Pieri's rule sigma_1^2 =
    sigma_2 + sigma_11, and the two classes of codimension 2 are self-dual:
    >>> from bbcells.core import bb_cells
    >>> from bbcells.frontends.flag import grassmannian
    >>> G = grassmannian(2, 4)
    >>> order, codim, constants, pairing = integral_cohomology(
    ...     G, fixed_components(G), bb_cells(G).cocharacter)
    >>> (s1,) = [p for p in order if codim[p] == 1]
    >>> a, b = [p for p in order if codim[p] == 2]
    >>> constants[(s1, s1)] == {a: 1, b: 1}
    True
    >>> pairing[(a, a)], pairing[(a, b)], pairing[(b, b)]
    (1, 0, 1)
    """
    from bbcells.linalg import determinant
    classes = canonical_classes(data, components, lam)
    order = morse_order(data, components, lam)
    weights_of = dict(zip(data.points, data.weights))
    pair = lambda w: sum(a * b for a, b in zip(w, lam))
    codim = {p: sum(1 for w in weights_of[p] if pair(w) < 0) for p in data.points}
    if point is None:
        point = [Fraction(3 + 7 * i, 1 + 2 * i) for i in range(data.rank)]
    ev = lambda w: sum(Fraction(a) * b for a, b in zip(w, point))
    euler_minus = {}
    for q in data.points:
        e = Fraction(1)
        for w in weights_of[q]:
            if pair(w) < 0:
                e *= ev(w)
        if e == 0:
            raise ValueError("the evaluation point lies on a weight hyperplane")
        euler_minus[q] = e
    value = {p: {q: classes[p][q].evaluate(point) for q in order if not classes[p][q].is_zero()}
             for p in data.points}
    n = data.dim
    constants, pairing = {}, {}
    top = [p for p in data.points if codim[p] == n]
    for i, a in enumerate(order):
        for b in order[i:]:
            if codim[a] + codim[b] > n:
                continue
            product = {q: value[a][q] * value[b][q] for q in value[a] if q in value[b]}
            coefficient = {}
            for q in order:
                remainder = product.get(q, Fraction(0))
                for s, c in coefficient.items():
                    remainder -= c * value[s].get(q, 0)
                if remainder:
                    coefficient[q] = remainder / euler_minus[q]
            ordinary = {s: c for s, c in coefficient.items() if codim[s] == codim[a] + codim[b]}
            for s, c in ordinary.items():
                if c.denominator != 1:
                    raise ValueError("non-integral structure constant %s for %r, %r" % (c, a, b))
            ordinary = {s: int(c) for s, c in ordinary.items()}
            constants[(a, b)] = constants[(b, a)] = ordinary
            if codim[a] + codim[b] == n:
                pairing[(a, b)] = pairing[(b, a)] = sum(ordinary.get(t, 0) for t in top)
    for k in range(n // 2 + 1):
        rows = [p for p in order if codim[p] == k]
        columns = [p for p in order if codim[p] == n - k]
        matrix = [[pairing[(a, b)] for b in columns] for a in rows]
        if len(rows) != len(columns) or abs(determinant(matrix)) != 1:
            raise ValueError("the pairing of degrees %d and %d is not unimodular" % (k, n - k))
    return order, codim, constants, pairing


def volume_ring(data, classes, lam):
    """Experimental: slow exact linear algebra over Q, and it describes only
    the subalgebra generated by divisors (docs/cohomology.md, section 5).

    The subalgebra of H^*(X; Q) generated by degree-one classes D_1..D_k
    ({point: character} each), from the numbers int D^a (ABBV at lam):
    by Poincare duality a form f of degree d vanishes iff int f g = 0 for
    all g of degree dim X - d, so the algebra is Q[x_1..x_k]/Ann(V) with the
    volume polynomial V = int (sum x_i D_i)^n / n! (Macaulay's inverse
    system). Returns {"numbers": {a: int D^a}, "hilbert": [dim in degree d],
    "relations": {d: basis of the relations of degree d, as {a: coefficient}},
    "generators": {d: number of new relations of degree d}}. If "hilbert"
    equals the Betti numbers, the D_i generate H^*(X; Q) and this is a
    presentation of the ring.
    >>> from bbcells.frontends.spherical import complete_quadrics
    >>> from bbcells.rootsystem import RootSystem
    >>> X = complete_quadrics(3)
    >>> components = fixed_components(X)
    >>> R = RootSystem("A2")
    >>> mu, nu = (line_bundle_class(X, components, "A2",
    ...           tuple(-2 * x for x in R.fundamental_weight(k))) for k in (0, 1))
    >>> ring = volume_ring(X, [mu, nu], (3, 7))
    >>> ring["hilbert"], ring["generators"]
    ([1, 2, 3, 3, 2, 1], {3: 1, 4: 1})
    """
    from itertools import combinations_with_replacement
    from bbcells.linalg import certified_rank, null_space_and_free
    n, k = data.dim, len(classes)
    pair = lambda w: sum(Fraction(a) * b for a, b in zip(w, lam))

    def exponents(d):
        result = []
        for combo in combinations_with_replacement(range(k), d):
            a = [0] * k
            for i in combo:
                a[i] += 1
            result.append(tuple(a))
        return sorted(result, reverse=True)

    values = []
    for p, weights in zip(data.points, data.weights):
        e = Fraction(1)
        for w in weights:
            e *= pair(w)
        if e == 0:
            raise ValueError("lam is not generic")
        values.append(([pair(c[p]) for c in classes], e))
    numbers = {}
    for a in exponents(n):
        total = Fraction(0)
        for x, e in values:
            term = Fraction(1)
            for xi, ai in zip(x, a):
                if ai:
                    term *= xi ** ai
            total += term / e
        if total.denominator != 1:
            raise ValueError("non-integral intersection number %s for %r" % (total, a))
        numbers[a] = int(total)
    add = lambda a, b: tuple(x + y for x, y in zip(a, b))
    hilbert, relations, generators = [], {}, {}
    for d in range(n + 1):
        rows, columns = exponents(d), exponents(n - d)
        matrix = [[numbers[add(a, b)] for a in rows] for b in columns]   # f -> (int f g)_g
        kernel, free = null_space_and_free(matrix, len(rows)) if matrix else ([], [])
        hilbert.append(len(rows) - len(kernel))
        relations[d] = [{a: c for a, c in zip(rows, v) if c} for v in kernel]
        if kernel:
            # the products of the relations of degree d - 1 with the variables,
            # in the coordinates of the kernel basis (their values at the free
            # columns); their rank is certified by modular elimination
            column = {rows[f]: j for j, f in enumerate(free)}
            products = []
            for r in relations.get(d - 1, []):
                for i in range(k):
                    unit = tuple(int(j == i) for j in range(k))
                    v = [Fraction(0)] * len(free)
                    for a, c in r.items():
                        j = column.get(add(a, unit))
                        if j is not None:
                            v[j] += c
                    products.append(v)
            new = len(kernel) - (certified_rank(products) if products else 0)
            if new:
                generators[d] = new
    return {"numbers": numbers, "hilbert": hilbert, "relations": relations,
            "generators": generators}
