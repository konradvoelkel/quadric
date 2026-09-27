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

def curve_parities(cells):
    """(x, y, m) for every invariant curve flowing from x down to y (phi, the
    weight at x, is lambda-positive): m = (sigma(x) - sigma(y)) / phi if
    dim x = dim y + 1; if dim x = dim y, one normal direction psi is negative
    at x and positive at y, and m = (sigma(x) - sigma(y) + psi_y) / phi. m is
    None if the quotient is not an integer; curves two or more steps down
    get m = None too (their flow lines come in families)."""
    from bbcells.core import pairing
    data, lam = cells.data, cells.cocharacter
    result = []
    for p, q, chi in data.edges:
        if pairing(lam, chi) > 0:
            x, y, phi = p, q, chi
        else:
            x, y, phi = q, p, tuple(-c for c in chi)
        dx, dy = cells.dim_of(x), cells.dim_of(y)
        diff = [a - b for a, b in zip(realcells.positive_weight_sum(cells, x),
                                      realcells.positive_weight_sum(cells, y))]
        m = None
        if dx == dy:
            k = next(i for i, c in enumerate(phi) if c)
            flipped = []
            for w in data.weights_of(x):
                if w == phi or pairing(lam, w) >= 0:
                    continue
                for v in data.weights_of(y):
                    d = [a - b for a, b in zip(w, v)]
                    if d[k] % phi[k] == 0 and all(e == d[k] // phi[k] * c for e, c in zip(d, phi)) \
                            and pairing(lam, v) > 0 and v != tuple(-c for c in phi):
                        flipped.append(v)
            if len(flipped) == 1:
                diff = [a + b for a, b in zip(diff, flipped[0])]
                dx += 1                      # now handled like an adjacent pair
        if dx == dy + 1:
            k = next(i for i, c in enumerate(phi) if c)
            q, r = divmod(diff[k], phi[k])
            if not r and all(d == q * c for d, c in zip(diff, phi)):
                m = q
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
