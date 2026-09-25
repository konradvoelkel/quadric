"""
Real points of flag varieties: the cellular chain complex of the real
Schubert cells and H_*(X(R); Z) (PLAN.md S5.1).

For a split real form, the real points of the Schubert cells BwP/P form a
CW structure on X(R) whose incidence numbers are 0 or +-2 (Ehresmann).

* Kocherlakota (Adv. Math. 110, 1995; restated as Theorem `Kocherlakota` in
  arXiv:1910.11149), any type, up to sign: for x -> y = r_phi(x) with
  l(y) = l(x) - 1 and sigma(x) = sum{phi > 0 : phi(x) < 0}, write
  sigma(x) - sigma(y) = m phi; then [x, y] = 0 if m is odd, +-2 if m is even.
* Matszangosz (arXiv:1910.11149, Theorems `incidencecoeffs` and `signs`),
  type A with signs, for lexicographic coorientations: [I, J] = 0 if N_I(a, b)
  is even, (-1)^{s(I,J)} 2 otherwise.

The two rules belong to different complexes. Kocherlakota's incidences are
those of the cellular chain complex (graded by dimension); Matszangosz's are
those of the cochain complex of cooriented cells (graded by codimension).
They agree up to sign for complete flags, where sigma(I) - sigma(J) =
(N_I(a, b) + 1) e_ba holds exactly (checked for all complete flags with
N <= 5), but not for partial flags when X(R) is not orientable: on RP^2
Kocherlakota puts the 2 between the 2-cell and the 1-cell (d e^2 = 2 e^1),
Matszangosz between the 1-cell and the point. Each gives the correct
(co)homology in its own convention; the relation m = N + 1 used in
arXiv:1910.11149 to rederive Kocherlakota's theorem holds only for complete
flags.

In the stable motivic picture (SPEC.md 3.5) the incidence +-2 is the real
realization of the eta-component of the attaching map; under hypothesis H1
that component is +-eta with a coefficient in W(Z) = Z.
"""

from dataclasses import dataclass

from bbcells.frontends.flag import flag_variety
from bbcells.linalg import smith_invariants
from bbcells.rootsystem import RootSystem


# -- the chain complex ------------------------------------------------------

@dataclass(frozen=True)
class RealCellComplex(object):
    """real Schubert cells with incidence numbers [x, y] for dim x = dim y + 1.

    As in arXiv:1910.11149 the cells are cooriented and the incidences form
    Vassiliev's cochain complex, graded by codimension: the coboundary of a
    cell of codimension c is sum_y [x, y] y over cells y of codimension c + 1.
    Its cohomology is H^*(X(R); Z); homology follows by universal coefficients.
    """
    name: str
    dims: dict
    incidences: dict
    signed: bool
    convention: str = "cooriented"     # or "cellular" (boundary, graded by dimension)

    @property
    def dim(self):
        return max(self.dims.values())

    def cells_of_codimension(self, c):
        return sorted((p for p, e in self.dims.items() if self.dim - e == c), key=str)

    def coboundary_matrix(self, c):
        """matrix of delta: C^c -> C^{c+1}; rows = cells of codimension c + 1"""
        rows, cols = self.cells_of_codimension(c + 1), self.cells_of_codimension(c)
        return [[self.incidences.get((x, y), 0) for x in cols] for y in rows]

    def square_zero(self):
        """whether delta o delta = 0 (a strong check of the signs)"""
        for c in range(0, self.dim - 1):
            a, b = self.coboundary_matrix(c + 1), self.coboundary_matrix(c)
            for i in range(len(a)):
                for j in range(len(b[0]) if b else 0):
                    if sum(a[i][k] * b[k][j] for k in range(len(b))):
                        return False
        return True

    def cohomology(self):
        """[(free rank, [torsion orders]) for c = 0..dim] of H^c(X(R); Z)"""
        if not self.signed:
            raise ValueError("integral cohomology needs signed incidences")
        if self.convention != "cooriented":
            raise ValueError("cohomology is implemented for the cooriented convention")
        ranks, torsion = {}, {}
        for c in range(-1, self.dim + 1):
            matrix = self.coboundary_matrix(c) if 0 <= c < self.dim else []
            invariants = smith_invariants(matrix) if matrix and matrix[0] else []
            ranks[c] = len(invariants)
            torsion[c] = [x for x in invariants if x > 1]
        return [(len(self.cells_of_codimension(c)) - ranks[c] - ranks[c - 1], torsion[c - 1])
                for c in range(0, self.dim + 1)]

    def homology(self):
        """H_d: free part of H^d, torsion of H^{d+1} (universal coefficients)"""
        h = self.cohomology()
        return [(h[d][0], h[d + 1][1] if d + 1 < len(h) else []) for d in range(len(h))]

    def rational_betti(self):
        return [free for free, _ in self.cohomology()]

    def euler_characteristic(self):
        return sum((-1) ** d for d in self.dims.values())


# -- Kocherlakota, any type, unsigned ---------------------------------------

def _sigma(R, mu):
    total = [0] * R.rank
    for phi in R.positive_roots:
        if R.coroot_on_weight(phi, mu) < 0:
            total = [t + c for t, c in zip(total, phi)]
    return tuple(total)


def _adjacent_pairs(data):
    """(x, y) joined by an invariant curve with l(y) = l(x) - 1"""
    length = {p: data.annotation(p)["length"] for p in data.points}
    for p, q, beta in data.edges:
        if length[p] == length[q] + 1:
            yield p, q, beta
        elif length[q] == length[p] + 1:
            yield q, p, tuple(-c for c in beta)


def kocherlakota_incidences(cartan_type, crossed=None):
    """{(x, y): 0 or 2} for adjacent real Schubert cells of G/P (unsigned)"""
    R = RootSystem(cartan_type)
    data = flag_variety(cartan_type, crossed)
    sigma = {p: _sigma(R, data.annotation(p)["weight"]) for p in data.points}
    result = {}
    for x, y, beta in _adjacent_pairs(data):
        phi = beta if all(c >= 0 for c in beta) else tuple(-c for c in beta)
        difference = [a - b for a, b in zip(sigma[x], sigma[y])]
        k = next(i for i, c in enumerate(phi) if c)
        m, remainder = divmod(difference[k], phi[k])
        if remainder or any(d != m * c for d, c in zip(difference, phi)):
            raise AssertionError("sigma(x) - sigma(y) is not a multiple of phi")
        result[(x, y)] = 0 if m % 2 else 2
    return data, result


def real_flag_variety(cartan_type, crossed=None):
    """the real cell complex of G/P; signed (type A) or unsigned (other types)
    >>> X = real_flag_variety("A2", {1})            # RP^2
    >>> X.cohomology()                              # Z, 0, Z/2
    [(1, []), (0, []), (0, [2])]
    >>> X.homology()                                # Z, Z/2, 0
    [(1, []), (0, [2]), (0, [])]
    """
    letter = cartan_type.strip().upper()[0]
    if letter == "A":
        return type_a_signed(cartan_type, crossed)
    data, unsigned = kocherlakota_incidences(cartan_type, crossed)
    dims = {p: data.annotation(p)["length"] for p in data.points}
    return RealCellComplex(data.name + "(R)", dims, unsigned, signed=False,
                           convention="cellular")


# -- Matszangosz, type A, signed ---------------------------------------------

def _permutation(word, n):
    result = {}
    for j in range(1, n + 1):
        x = j
        for i in reversed(word):
            x = i + 1 if x == i else i if x == i + 1 else x
        result[j] = x
    return result


def ordered_set_partition(word, n, blocks):
    """I as a function [n] -> [m]: I(w(k)) = block of k, for the coordinate flag
    w E_. (blocks: the partial sums s_1 < ... < s_m = n)"""
    w = _permutation(word, n)
    block_of = {}
    start = 0
    for j, s in enumerate(blocks, start=1):
        for k in range(start + 1, s + 1):
            block_of[k] = j
        start = s
    return {w[k]: block_of[k] for k in range(1, n + 1)}


def osp_length(I):
    """l(I) = #{(a, b): a > b, I(a) < I(b)}"""
    return sum(1 for a in I for b in I if a > b and I[a] < I[b])


def _G(I, c, low, high):
    return sum(1 for d in I if d > c and low < I[d] <= high)


def _L(I, c, low, high):
    return sum(1 for d in I if d < c and low < I[d] <= high)


def _interchange(I, J):
    moved = [l for l in I if I[l] != J[l]]
    if len(moved) != 2:
        raise ValueError("I and J differ by more than one interchange")
    a, b = max(moved), min(moved)
    alpha, beta = I[a], I[b]
    if not alpha < beta:
        raise ValueError("expected a in I_alpha, b in I_beta with alpha < beta")
    return a, b, alpha, beta


def matszangosz_N(I, J):
    """N_I(a, b) = G_I(a, alpha, beta) + L_I(b, alpha - 1, beta - 1); the
    incidence is nonzero iff it is odd (Theorem `incidencecoeffs`).
    The paper's example in Fl_{2,2,2}: I = [36, 14, 25], J = [26, 14, 35]:
    >>> I = {3: 1, 6: 1, 1: 2, 4: 2, 2: 3, 5: 3}
    >>> J = {2: 1, 6: 1, 1: 2, 4: 2, 3: 3, 5: 3}
    >>> matszangosz_N(I, J)
    3
    """
    a, b, alpha, beta = _interchange(I, J)
    return _G(I, a, alpha, beta) + _L(I, b, alpha - 1, beta - 1)


def matszangosz_sign_exponent(I, J):
    """s(I, J) = c_1 + c_2 + c_3 + c_4 of Theorem `signs`. The paper's example
    I = (4,5,6,1,2,3), J = (4,2,6,1,5,3) has c = (3, 2, 1, 2), s = 8 (its
    incidence is nevertheless 0: N_I(a, b) = 2 is even, and independently
    sigma(I) - sigma(J) = 3 e_25 has odd coefficient in Kocherlakota's rule).
    >>> one_line = lambda seq: {v: k for k, v in enumerate(seq, start=1)}
    >>> I, J = one_line((4, 5, 6, 1, 2, 3)), one_line((4, 2, 6, 1, 5, 3))
    >>> matszangosz_sign_exponent(I, J), matszangosz_N(I, J)
    (8, 2)
    """
    a, b, alpha, beta = _interchange(I, J)
    m = max(I.values())
    G = lambda c, low: _G(I, c, low, m)             # G_I(c, delta) = G_I(c, delta, m)
    G_own = lambda c: G(c, I[c])                     # G_I(c) = G_I(c, I(c))
    c1 = G(b, alpha) - G_own(a) + sum(G_own(c) for c in I if c < b)
    c2 = (G_own(a) - G(a, beta)) * (sum(G_own(c) for c in I if b < c < a) + G(a, beta))
    c3 = sum(G(b, I[c]) - G(a, I[c]) for c in I if c < b and alpha <= I[c] < beta)
    c4 = _L(I, b, alpha - 1, beta - 1) + 1
    return c1 + c2 + c3 + c4


def matszangosz_incidence(I, J):
    """[Omega_I, Omega_J] for adjacent I, J (l(J) = l(I) - 1) in type A with
    lexicographic coorientations: 0 if N_I(a, b) is even, else (-1)^s 2"""
    if matszangosz_N(I, J) % 2 == 0:
        return 0
    return 2 * (-1) ** matszangosz_sign_exponent(I, J)


def type_a_signed(cartan_type, crossed=None):
    """signed real cell complex of a type A partial flag variety"""
    R = RootSystem(cartan_type)
    if R.letter != "A":
        raise ValueError("signed incidences are implemented for type A only")
    n = R.rank + 1
    data = flag_variety(cartan_type, crossed)
    crossed_nodes = set(range(1, n)) if crossed is None else set(crossed)
    blocks = sorted(crossed_nodes) + [n]
    osp = {p: ordered_set_partition(data.annotation(p)["word"], n, blocks) for p in data.points}
    dims = {}
    for p in data.points:
        dims[p] = osp_length(osp[p])
        if dims[p] != data.annotation(p)["length"]:
            raise AssertionError("l(I) differs from the Weyl group length at %r" % p)
    incidences = {(x, y): matszangosz_incidence(osp[x], osp[y])
                  for x, y, _ in _adjacent_pairs(data)}
    return RealCellComplex(data.name + "(R)", dims, incidences, signed=True)
