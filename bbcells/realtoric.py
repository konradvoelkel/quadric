"""
Exact real BB incidences of smooth complete toric varieties (PLAN.md S7.5,
docs/H1.md section 7).

The real points X(R) = P x (Z/2)^n / ~ carry a regular CW structure: a face
G of P, i.e. a cone tau of the fan, contributes one cell of dimension
n - |tau| for each class of (Z/2)^n modulo Lambda_tau, the span of the rays of
tau mod 2 (Davis-Januszkiewicz). Every cell is oriented by its face; the
attaching maps are the identity on P, so

    d(tau, e) = sum over cones tau + r of (-1)^(n - |tau| + #{i in tau : i < r}) (tau + r, e).

For a generic cocharacter lambda, the real BB cell of the fixed point v (a
maximal cone) is the union of the cells (tau, e) whose face has v as its
lambda-minimal vertex; in the ray coordinates of v it is {+, -, 0}^d with
d = dim of the cell. The lexicographic matching (first coordinate that is 0
or -: pair 0 with -) leaves one critical cell per fixed point, the positive
orthant, and is acyclic (paths leave a BB cell only towards larger lambda).
Algebraic Morse theory then gives an integral chain complex with one
generator per BB cell: the exact incidences of the real BB decomposition,
graded or not. Its homology is H_*(X(R); Z).
"""

from fractions import Fraction
from itertools import combinations

from bbcells.linalg import rank as matrix_rank, smith_invariants


def _bits(vector):
    return sum(1 << i for i, c in enumerate(vector) if c % 2)


def _reduce(e, basis):
    for b in basis:
        if e & (1 << (b.bit_length() - 1)):
            e ^= b
    return e


def _echelon(vectors):
    """a reduced basis over F_2 (bit masks), sorted by decreasing pivot"""
    basis = []
    for v in vectors:
        v = _reduce(v, basis)
        if v:
            pivot = 1 << (v.bit_length() - 1)
            basis = [b ^ v if b & pivot else b for b in basis] + [v]
            basis.sort(key=lambda b: -b.bit_length())
    return basis


def _label(cone):
    return "s" + ",".join(str(i) for i in sorted(cone))


class RealToricComplex(object):
    """the fine CW structure of X(R) and the Morse complex of the real BB cells
    >>> from bbcells.frontends import toric
    >>> X = RealToricComplex(toric.projective_space(2), (3, -7))
    >>> X.fine_betti(), X.betti()                       # RP^2
    ([1, 0, 0], [1, 0, 0])
    >>> X.integral_homology()
    [[0], [2], []]
    """

    def __init__(self, fan, cocharacter):
        self.fan, self.cocharacter, self.n = fan, tuple(cocharacter), fan.dim
        self._rays = [_bits(r) for r in fan.rays]
        self.cones = {frozenset(sub) for c in fan.cones for k in range(self.n + 1)
                      for sub in combinations(c, k)}
        self._basis, self._coordinates = {}, {}
        self.up = {}
        for c in fan.cones:
            c = tuple(sorted(c))
            values = [sum(a * b for a, b in zip(m, self.cocharacter)) for m in fan.dual_basis(c)]
            if any(v == 0 for v in values):
                raise ValueError("cocharacter %r is not generic" % (self.cocharacter,))
            self.up[c] = frozenset(c[k] for k in range(self.n) if values[k] > 0)
        self.minimal_vertex = {}
        for tau in self.cones:
            (v,) = [c for c in self.up if tau <= set(c) and set(c) - tau <= self.up[c]]
            self.minimal_vertex[tau] = v

    # -- the fine complex ------------------------------------------------------
    def _lambda(self, tau):
        if tau not in self._basis:
            self._basis[tau] = _echelon([self._rays[i] for i in tau])
        return self._basis[tau]

    def cell(self, tau, e):
        tau = frozenset(tau)
        return (tau, _reduce(e, self._lambda(tau)))

    def dim(self, cell):
        return self.n - len(cell[0])

    def boundary(self, cell):
        tau, e = cell
        result = {}
        for r in range(len(self.fan.rays)):
            if r in tau or tau | {r} not in self.cones:
                continue
            sign = -1 if (self.n - len(tau) + sum(1 for i in tau if i < r)) % 2 else 1
            face = self.cell(tau | {r}, e)
            result[face] = result.get(face, 0) + sign
        return result

    def fine_cells(self):
        cells = []
        for tau in self.cones:
            pivots = {b.bit_length() - 1 for b in self._lambda(tau)}
            free = [i for i in range(self.n) if i not in pivots]
            for k in range(1 << len(free)):
                cells.append(self.cell(tau, sum(1 << free[j] for j in range(len(free))
                                                if k >> j & 1)))
        return cells

    def fine_betti(self):
        by_dim = {}
        for c in self.fine_cells():
            by_dim.setdefault(self.dim(c), []).append(c)
        return _betti(by_dim, self.boundary, self.n)

    # -- the BB Morse complex ---------------------------------------------------
    def coordinates(self, cell):
        """(v, S, c): the fixed point whose BB cell contains the cell, the rays
        of v not in tau, and the bits c_i (i in S) of e = sum c_i r_i mod Lambda_tau"""
        if cell not in self._coordinates:
            tau, e = cell
            v = self.minimal_vertex[tau]
            S = [i for i in v if i not in tau]
            found = None
            for k in range(1 << len(S)):
                s = 0
                for j, i in enumerate(S):
                    if k >> j & 1:
                        s ^= self._rays[i]
                if _reduce(s ^ e, self._lambda(tau)) == 0:
                    found = {i: k >> j & 1 for j, i in enumerate(S)}
                    break
            self._coordinates[cell] = (v, S, found)
        return self._coordinates[cell]

    def partner(self, cell):
        """("up", cell') or ("down", cell') in the lexicographic matching, or
        ("critical", None)"""
        v, S, c = self.coordinates(cell)
        tau, _ = cell
        lift = 0
        for i in S:
            if c[i]:
                lift ^= self._rays[i]
        for j in sorted(self.up[v]):
            if j not in S:
                return "up", self.cell(tau - {j}, lift ^ self._rays[j])
            if c[j]:
                return "down", self.cell(tau | {j}, lift)
        return "critical", None

    def critical_cells(self):
        """{fixed point label: its critical cell}"""
        return {_label(v): self.cell(set(v) - self.up[v], 0) for v in self.up}

    def morse_boundary(self, cell):
        chain = {k: Fraction(v) for k, v in self.boundary(cell).items()}
        result = {}
        while chain:
            sigma, coefficient = chain.popitem()
            if not coefficient:
                continue
            kind, tau = self.partner(sigma)
            if kind == "critical":
                result[sigma] = result.get(sigma, 0) + coefficient
            elif kind == "up":
                boundary = self.boundary(tau)
                factor = coefficient / boundary[sigma]
                for face, value in boundary.items():
                    if face != sigma:
                        chain[face] = chain.get(face, 0) - factor * value
        return {k: v for k, v in result.items() if v}

    def incidences(self):
        """{(x, y): [x : y]} for the real BB cells (labels as in
        toric.fixed_point_data), dim x = dim y + 1, integers"""
        critical = self.critical_cells()
        name = {c: p for p, c in critical.items()}
        result = {}
        for p, c in critical.items():
            for face, value in self.morse_boundary(c).items():
                if value.denominator != 1:
                    raise AssertionError("non-integral Morse incidence")
                result[(p, name[face])] = int(value)
        return result

    def dims(self):
        return {p: self.dim(c) for p, c in self.critical_cells().items()}

    def betti(self):
        """rational Betti numbers of X(R) from the BB Morse complex"""
        dims, incidences = self.dims(), self.incidences()
        by_dim = {}
        for p, d in dims.items():
            by_dim.setdefault(d, []).append(p)
        boundary = lambda x: {y: v for (a, y), v in incidences.items() if a == x}
        return _betti(by_dim, boundary, self.n)

    def integral_homology(self):
        """[torsion coefficients and zeros for free summands] per degree:
        H_d = Z^(number of 0) + sum Z/k"""
        dims, incidences = self.dims(), self.incidences()
        by_dim = {}
        for p, d in dims.items():
            by_dim.setdefault(d, []).append(p)
        result = []
        for d in range(self.n + 1):
            here, below, above = by_dim.get(d, []), by_dim.get(d - 1, []), by_dim.get(d + 1, [])
            out_rank = 0
            if here and below:
                M = [[incidences.get((x, y), 0) for x in here] for y in below]
                out_rank = matrix_rank(M)
            torsion = []
            if above and here:
                M = [[incidences.get((x, y), 0) for x in above] for y in here]
                torsion = [k for k in smith_invariants(M) if k not in (0, 1)]
                in_rank = matrix_rank(M)
            else:
                in_rank = 0
            free = len(here) - out_rank - in_rank
            result.append([0] * free + sorted(torsion))
        return result


def _betti(by_dim, boundary, n):
    ranks = {}
    for d in range(1, n + 1):
        index = {c: i for i, c in enumerate(by_dim.get(d - 1, []))}
        rows = []
        for c in by_dim.get(d, []):
            row = [0] * len(index)
            for face, value in boundary(c).items():
                row[index[face]] += value
            rows.append(row)
        ranks[d] = matrix_rank(rows) if rows and rows[0] else 0
    return [len(by_dim.get(d, [])) - ranks.get(d, 0) - ranks.get(d + 1, 0) for d in range(n + 1)]
