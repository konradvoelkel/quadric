"""
Experiments on hypothesis H1 (SPEC.md 3.5, docs/H1.md, PLAN.md S5.3).

The GKM form of Kocherlakota's rule (realcells.gkm_incidences) predicts, for
cells x, y of adjacent dimension joined by an invariant curve with weight phi,
whether the eta-component of the attaching map is nonzero: iff m is even,
where sigma(x) - sigma(y) = m phi. Two realizations can be tested
independently:

* real: unique sign completions of the predicted incidences give the
  rational cohomology of X(R), compared with an oracle (Choi-Park for toric
  varieties, Kocherlakota/Matszangosz for flag varieties);
* complex: eta_top is detected by Sq^2, and on H^2 Sq^2(a) = a^2 mod 2, which
  the GKM ring (equivariant.py) computes independently.
"""

from bbcells import equivariant, gkm, realcells
from bbcells.core import bb_cells


def find_graded_cocharacter(data, bound=12):
    """a small generic cocharacter whose decomposition is graded, or None"""
    import itertools
    ranges = [range(1, bound)] + [range(-bound, bound + 1)] * (data.rank - 1)
    for lam in itertools.product(*ranges):
        try:
            cells = bb_cells(data, lam)
        except ValueError:
            continue
        if gkm.is_graded(cells):
            return cells
    return None


def real_prediction(cells):
    """set of rational Betti vectors of X(R) over all sign completions of the
    predicted incidences (a singleton when the prediction is unambiguous)"""
    incidences = realcells.gkm_incidences(cells)
    if any(m is None for m, _ in incidences.values()):
        raise ValueError("sigma(x) - sigma(y) is not a multiple of phi for some curve")
    dims = {p: cells.dim_of(p) for p in cells.data.points}
    return {tuple(realcells.cellular_rational_betti(dims, signed))
            for signed in realcells.signed_completions(cells, incidences)}


def sq2_mismatches(cells):
    """compare (y*)^2 mod 2 with the parity rule, for all cells y of dimension
    1 and x of dimension 2. The dual cochain y* is the class of the closure of
    the minus-cell of y (flow-up class for -lambda). Returns a list of
    (x, y, observed, predicted) disagreements."""
    data = cells.data
    opposite = bb_cells(data, tuple(-c for c in cells.cocharacter))
    ones = [p for p in data.points if cells.dim_of(p) == 1]
    twos = [p for p in data.points if cells.dim_of(p) == 2]
    needed = equivariant.flow_up_classes(
        opposite, [p for p in data.points if cells.dim_of(p) <= 2])
    incidences = realcells.gkm_incidences(cells)
    mismatches = []
    for y in ones:
        constants = equivariant.structure_constants(opposite, needed, y, y)
        for x in twos:
            c = constants.get(x)
            observed = int(c.constant_term()) % 2 if c is not None and c.degree() == 0 else 0
            predicted = 1 if incidences.get((x, y), (None, 0))[1] == 2 else 0
            if observed != predicted:
                mismatches.append((x, y, observed, predicted))
    return mismatches


# -- non-graded decompositions of surfaces (docs/H1.md, section 5) ------------

def _multiple_of(d, phi):
    k = next(i for i, c in enumerate(phi) if c)
    q, r = divmod(d[k], phi[k])
    return q if not r and all(e == q * c for e, c in zip(d, phi)) else None


def pair_normal_weights(at_x, at_y, phi):
    """match the normal weights of a curve at its two ends: chi at x with
    chi - a phi at y. Equal weights are matched first (a = 0), the rest by
    congruence modulo phi with the smallest |a|; None if impossible. (When
    normal weights are congruent modulo phi the splitting of the normal
    bundle is not visible in the weights; this is a choice.)"""
    at_y, pairs, rest = list(at_y), [], []
    for w in at_x:
        if w in at_y:
            at_y.remove(w)
            pairs.append((w, w, 0))
        else:
            rest.append(w)
    for w in rest:
        options = [(abs(a), v, a) for v in at_y
                   for a in [_multiple_of([p - q for p, q in zip(w, v)], phi)] if a is not None]
        if not options:
            return None
        _, v, a = min(options)
        at_y.remove(v)
        pairs.append((w, v, a))
    return pairs


def curve_parities(cells):
    """(x, y, m) for every invariant curve flowing from x down to y (phi, the
    weight at x, is lambda-positive). With the normal weights paired
    (pair_normal_weights), m = 1 + sum of a over the directions positive at
    both ends, defined when dim x = dim y + 1 and no normal direction changes
    sign (the local picture of docs/H1.md, section 2), or when dim x = dim y
    and exactly one direction goes from negative at x to positive at y
    (section 6). Otherwise m is None."""
    from bbcells.core import pairing
    data, lam = cells.data, cells.cocharacter
    result = []
    for p, q, chi in data.edges:
        if pairing(lam, chi) > 0:
            x, y, phi = p, q, chi
        else:
            x, y, phi = q, p, tuple(-c for c in chi)
        at_x, at_y = list(data.weights_of(x)), list(data.weights_of(y))
        at_x.remove(phi)
        at_y.remove(tuple(-c for c in phi))
        pairs = pair_normal_weights(at_x, at_y, phi)
        m = None
        if pairs is not None:
            up = [1 for w, v, a in pairs if pairing(lam, w) < 0 < pairing(lam, v)]
            down = [1 for w, v, a in pairs if pairing(lam, v) < 0 < pairing(lam, w)]
            gap = cells.dim_of(x) - cells.dim_of(y)
            if (gap == 1 and not up and not down) or (gap == 0 and len(up) == 1 and not down):
                m = 1 + sum(a for w, v, a in pairs if pairing(lam, w) > 0 and pairing(lam, v) > 0)
        result.append((x, y, m))
    return result


def deflection_terms(cells):
    """the terms (x, y, 2) of the conjectural boundary of X(R) for a surface
    whose decomposition need not be graded:
      * a curve x -> y with dim x = dim y + 1 and m even (the GKM rule);
      * a curve a -> b between cells of equal dimension with m odd carries
        two real arcs that add up to +-2; after a perturbation each arc from b
        reaching a continues along one ascending arc of a, so every curve
        c -> a with dim c = dim a + 1 gives a term (c, b, 2).
    Each term has its own sign. Raises if some needed m is undefined."""
    dims = {p: cells.dim_of(p) for p in cells.data.points}
    terms, adjacent, slides = [], [], []
    for x, y, m in curve_parities(cells):
        if dims[x] - dims[y] not in (0, 1):
            continue
        if m is None:
            raise ValueError("undetermined parity for the curve %r -> %r" % (x, y))
        if dims[x] == dims[y] + 1:
            adjacent.append((x, y))
            if m % 2 == 0:
                terms.append((x, y, 2))
        elif m % 2:
            slides.append((x, y))
    for a, b in slides:
        terms += [(c, b, 2) for c, a2 in adjacent if a2 == a]
    return terms


def surface_prediction(cells, limit=1 << 18):
    """the set of rational Betti vectors of X(R) over all sign choices of
    deflection_terms with boundary o boundary = 0 (surfaces only)"""
    import itertools
    if cells.data.dim != 2:
        raise ValueError("the deflection rule is only tested for surfaces")
    terms = deflection_terms(cells)
    if 2 ** len(terms) > limit:
        raise ValueError("too many sign choices")
    dims = {p: cells.dim_of(p) for p in cells.data.points}
    result = set()
    for signs in itertools.product((1, -1), repeat=len(terms)):
        boundary = {}
        for (x, y, magnitude), s in zip(terms, signs):
            boundary[(x, y)] = boundary.get((x, y), 0) + s * magnitude
        boundary = {k: v for k, v in boundary.items() if v}
        if realcells._square_zero_cellular(boundary, dims):
            result.add(tuple(realcells.cellular_rational_betti(dims, boundary)))
    return result


# -- beyond GKM: the curves of the Brion components (docs/H1.md, section 8) ----

def brion_prediction(cells):
    """the rational Betti vectors of X(R) over all sign completions of the
    rule of section 2 applied to the invariant curves of the Brion components
    (brion.invariant_curves; the cells must be built on
    brion.with_invariant_curves(data)). Only for graded decompositions and
    curves whose parity is defined; raises otherwise."""
    dims = {p: cells.dim_of(p) for p in cells.data.points}
    parities = curve_parities(cells)
    if any(dims[x] <= dims[y] for x, y, _ in parities):
        raise ValueError("the decomposition is not graded along the invariant curves")
    magnitudes = {}
    for x, y, m in parities:
        if dims[x] == dims[y] + 1:
            if m is None:
                raise ValueError("undetermined parity for the curve %r -> %r" % (x, y))
            magnitudes[(x, y)] = 0 if m % 2 else 2
    return {tuple(realcells.cellular_rational_betti(dims, signed))
            for signed in realcells.sign_choices(dims, magnitudes)}
