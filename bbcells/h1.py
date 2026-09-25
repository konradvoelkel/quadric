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
