"""
GKM data: the graph of T-invariant curves and the order it induces on the
cells (PLAN.md S3.1, S3.2).

If C is a T-invariant P^1 joining p and q with tangent weight chi at p and
<lambda, chi> > 0, then C - {q} lies in the plus-cell of p, so q lies in the
closure of that cell. The BB order is the transitive closure of these
relations; for G/P and dominant lambda it is the Bruhat order.
"""

from bbcells.core import pairing


def check_gkm(data):
    """list of problems (empty if the GKM conditions hold): edges present, the
    tangent weights at each point pairwise linearly independent, and one
    edge per tangent weight"""
    problems = []
    if data.edges is None:
        return ["no edges"]
    for label, wts in zip(data.points, data.weights):
        for i in range(len(wts)):
            for j in range(i + 1, len(wts)):
                a, b = wts[i], wts[j]
                if all(a[k] * b[l] == a[l] * b[k] for k in range(len(a)) for l in range(len(a))):
                    problems.append("weights %r and %r at %r are proportional" % (a, b, label))
    incident = {p: [] for p in data.points}
    for p, q, chi in data.edges:
        incident[p].append(chi)
        incident[q].append(tuple(-c for c in chi))
    for label, wts in zip(data.points, data.weights):
        if sorted(incident[label]) != sorted(wts):
            problems.append("edges at %r do not match its tangent weights" % (label,))
    return problems


def cell_relations(cells):
    """pairs (q, p): q lies in the closure of the plus-cell of p, one pair
    per invariant curve"""
    data = cells.data
    if data.edges is None:
        raise ValueError("the fixed-point data has no GKM edges")
    relations = []
    for p, q, chi in data.edges:
        if pairing(cells.cocharacter, chi) > 0:
            relations.append((q, p))
        else:
            relations.append((p, q))
    return relations


def bb_order(cells):
    """{p: set of fixed points strictly below p}: the transitive closure of
    "q lies in the closure of the plus-cell of p" along invariant curves.

    The relation is acyclic (the plus-decomposition of a projective variety
    is filtrable), but it need not go down in dimension: BB decompositions
    are not stratifications in general (see tests/test_gkm.py for dP_6).
    For G/P and dominant lambda it is the Bruhat order.
    >>> from bbcells.core import FixedPointData, bb_cells
    >>> P2 = FixedPointData(2, 2, ("a", "b", "c"),
    ...     (((1, 0), (0, 1)), ((-1, 0), (-1, 1)), ((0, -1), (1, -1))),
    ...     edges=(("a", "b", (1, 0)), ("a", "c", (0, 1)), ("b", "c", (-1, 1))))
    >>> order = bb_order(bb_cells(P2, (1, 2)))
    >>> sorted(order["a"]), sorted(order["c"]), sorted(order["b"])
    (['b', 'c'], [], ['c'])
    """
    direct = {p: set() for p in cells.data.points}
    for q, p in cell_relations(cells):
        direct[p].add(q)
    below, state = {}, {}

    def visit(p):
        if state.get(p) == "done":
            return below[p]
        if state.get(p) == "active":
            raise ValueError("the curve relation has a cycle through %r; the data "
                             "cannot come from a projective variety" % (p,))
        state[p] = "active"
        result = set()
        for q in direct[p]:
            result.add(q)
            result |= visit(q)
        below[p] = result
        state[p] = "done"
        return result

    for p in cells.data.points:
        visit(p)
    return below


def is_graded(cells):
    """whether every invariant curve goes down in cell dimension (a necessary
    condition for the decomposition to be a stratification)"""
    return all(cells.dim_of(q) < cells.dim_of(p) for q, p in cell_relations(cells))


def covering_relations(cells):
    """the Hasse diagram of the BB order: pairs (q, p) with q < p and nothing between"""
    below = bb_order(cells)
    result = []
    for p, lower in below.items():
        for q in lower:
            if not any(q in below[r] for r in lower if r != q):
                result.append((q, p))
    return sorted(result, key=lambda pair: (cells.dim_of(pair[1]), str(pair)))
