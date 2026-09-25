"""
Equivariant cohomology H_T^*(X; Q) as a GKM ring (PLAN.md S3.3).

For an equivariantly formal X with GKM data, H_T^*(X; Q) is the ring of
tuples (f_p) of polynomials in t_1..t_r with f_p - f_q divisible by chi for
every invariant curve (p, q, chi) (Chang-Skjelbred, Goresky-Kottwitz-
MacPherson). Degree k of the polynomial corresponds to H^{2k}.

* gkm_ring_dimension: dim of the degree-k part, by linear algebra;
* flow_up_classes: tau_p = [closure of the plus-cell of p], supported on the
  points below p in the BB order, with tau_p(p) = product of the
  lambda-negative weights at p (the normal weights of the cell);
* structure_constants: tau_a tau_b = sum_s c_ab^s tau_s;
* integrate: Atiyah-Bott-Berline-Vergne localization, sum_p f_p / e_p.
"""

from fractions import Fraction
from math import comb

from bbcells.core import pairing
from bbcells.gkm import bb_order
from bbcells.linalg import rank, solve
from bbcells.polynomial import Poly, monomials


def _restriction_rows(chi, basis, nvars):
    """for each monomial in basis, its restriction to {chi = 0} as a dict"""
    return [Poly(nvars, {m: 1}).restrict_to_hyperplane(chi).terms for m in basis]


def _edge_constraints(data, degree, unknown_index, fixed=None):
    """rows of the linear conditions 'f_p - f_q vanishes on chi = 0' for all
    edges; unknown_index maps (point, monomial) -> column. fixed: {point:
    Poly} values that are known (contribute to the right-hand side)."""
    nvars = data.rank
    basis = monomials(nvars, degree)
    fixed = fixed or {}
    rows, rhs = [], []
    cache = {}
    for p, q, chi in data.edges:
        if not any((x, basis[0]) in unknown_index for x in (p, q)) and not (
                p in fixed or q in fixed):
            continue
        if chi not in cache:
            cache[chi] = _restriction_rows(chi, basis, nvars)
        restricted = cache[chi]
        keys = sorted({k for r in restricted for k in r})
        # contribution of the fixed values
        known = Poly(nvars)
        if p in fixed:
            known = known + fixed[p]
        if q in fixed:
            known = known - fixed[q]
        known_restricted = known.restrict_to_hyperplane(chi).terms if not known.is_zero() else {}
        keys = sorted(set(keys) | set(known_restricted))
        for key in keys:
            row = [Fraction(0)] * len(unknown_index)
            for sign, point in ((1, p), (-1, q)):
                for m, r in zip(basis, restricted):
                    column = unknown_index.get((point, m))
                    if column is not None and key in r:
                        row[column] += sign * r[key]
            value = -known_restricted.get(key, Fraction(0))
            if any(row) or value:
                rows.append(row)
                rhs.append(value)
    return rows, rhs


def gkm_ring_dimension(data, degree):
    """dimension over Q of the degree-`degree` part of the GKM ring"""
    if data.edges is None:
        raise ValueError("GKM data (edges) required")
    basis = monomials(data.rank, degree)
    index = {}
    for p in data.points:
        for m in basis:
            index[(p, m)] = len(index)
    rows, _ = _edge_constraints(data, degree, index)
    return len(index) - (rank(rows) if rows else 0)


def expected_dimension(counts, rank_of_torus, degree):
    """coefficient of t^{2 degree} in P_X(t) / (1 - t^2)^r"""
    return sum(c * comb(degree - j + rank_of_torus - 1, rank_of_torus - 1)
               for j, c in enumerate(counts) if j <= degree)


def _topological_order_down(cells, order):
    """points ordered so that s comes before every point below it"""
    result, seen = [], set()

    def visit(s):
        if s in seen:
            return
        seen.add(s)
        for r in cells.data.points:
            if s in order[r]:
                visit(r)
        result.append(s)

    for s in cells.data.points:
        visit(s)
    return result


def flow_up_class(cells, p, order=None):
    """{q: Poly}: tau_p supported on {q <= p}, tau_p(p) = product of the
    lambda-negative tangent weights at p, homogeneous of degree n - dim(cell).
    Cost grows with (points below p) x (monomials of degree codim in r
    variables); divisors are cheap, top-codimension classes in large rank are not."""
    data = cells.data
    order = bb_order(cells) if order is None else order
    nvars = data.rank
    negative = [w for w in data.weights_of(p) if pairing(cells.cocharacter, w) < 0]
    top = Poly.product_of_linear(negative, nvars)
    degree = len(negative)
    below = sorted(order[p], key=str)
    basis = monomials(nvars, degree)
    index = {}
    for q in below:
        for m in basis:
            index[(q, m)] = len(index)
    values = {p: top}
    if index:
        fixed = {p: top}
        for x in data.points:
            if x != p and x not in order[p]:
                fixed[x] = Poly(nvars)
        rows, rhs = _edge_constraints(data, degree, index, fixed)
        solution = solve(rows, rhs) if rows else [Fraction(0)] * len(index)
        if solution is None:
            raise ValueError("no flow-up class for %r: the curve order does not "
                             "contain the fixed points of the cell closure" % (p,))
        for q in below:
            values[q] = Poly(nvars, {m: solution[index[(q, m)]] for m in basis})
    else:
        for a, b, chi in data.edges:
            if p in (a, b) and not top.restrict_to_hyperplane(chi).is_zero():
                raise ValueError("inconsistent GKM data at %r" % (p,))
    return values


def flow_up_classes(cells, points=None):
    """{p: flow_up_class(cells, p)} for the given points (default: all)"""
    order = bb_order(cells)
    points = cells.data.points if points is None else points
    return {p: flow_up_class(cells, p, order) for p in points}


def value(classes, p, q, nvars):
    return classes[p].get(q, Poly(nvars))


def structure_constants(cells, classes, a, b):
    """{s: c_ab^s} with tau_a tau_b = sum_s c_ab^s tau_s"""
    data = cells.data
    nvars = data.rank
    order = bb_order(cells)
    product = {q: value(classes, a, q, nvars) * value(classes, b, q, nvars)
               for q in data.points}
    coefficients = {}
    for s in _topological_order_down(cells, order):
        remainder = product[s]
        for s2, c in coefficients.items():
            if not c.is_zero():
                remainder = remainder - c * value(classes, s2, s, nvars)
        if remainder.is_zero():
            continue
        for chi in [w for w in data.weights_of(s) if pairing(cells.cocharacter, w) < 0]:
            remainder = remainder.divide_by_linear(chi)
        coefficients[s] = remainder
    return coefficients


def integrate(data, values, point_values=None):
    """sum_p f_p / e_p with e_p = product of all tangent weights at p (ABBV),
    evaluated at the rational point `point_values` of the parameter space"""
    if point_values is None:
        point_values = [Fraction(3 + 7 * i, 1 + 2 * i) for i in range(data.rank)]
    total = Fraction(0)
    for p, wts in zip(data.points, data.weights):
        e = Fraction(1)
        for w in wts:
            e *= sum(Fraction(c) * v for c, v in zip(w, point_values))
        if e == 0:
            raise ValueError("the evaluation point lies on a weight hyperplane")
        total += values[p].evaluate(point_values) / e
    return total
