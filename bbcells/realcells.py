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
        """whether delta o delta = 0, resp. d o d = 0 (a strong check of the signs)"""
        if self.convention == "cellular":
            return _square_zero_cellular(self.incidences, self.dims)
        for c in range(0, self.dim - 1):
            a, b = self.coboundary_matrix(c + 1), self.coboundary_matrix(c)
            for i in range(len(a)):
                for j in range(len(b[0]) if b else 0):
                    if sum(a[i][k] * b[k][j] for k in range(len(b))):
                        return False
        return True

    def cellular_homology(self):
        """H_d from the boundary C_d -> C_{d-1} with coefficients [x, y]"""
        cells_of = {d: sorted((p for p, e in self.dims.items() if e == d), key=str)
                    for d in range(self.dim + 1)}
        ranks, torsion = {}, {}
        for d in range(0, self.dim + 2):
            if 1 <= d <= self.dim:
                matrix = [[self.incidences.get((x, y), 0) for x in cells_of[d]]
                          for y in cells_of[d - 1]]
            else:
                matrix = []
            invariants = smith_invariants(matrix) if matrix and matrix[0] else []
            ranks[d] = len(invariants)
            torsion[d] = [x for x in invariants if x > 1]
        return [(len(cells_of[d]) - ranks[d] - ranks[d + 1], torsion[d + 1])
                for d in range(self.dim + 1)]

    def cohomology(self):
        """[(free rank, [torsion orders]) for c = 0..dim] of H^c(X(R); Z)"""
        if not self.signed:
            raise ValueError("integral cohomology needs signed incidences")
        if self.convention == "cellular":
            h = self.cellular_homology()
            return [(h[c][0], h[c - 1][1] if c > 0 else []) for c in range(len(h))]
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
        if self.signed and self.convention == "cellular":
            return self.cellular_homology()
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
    """the signed real cell complex of G/P: type A: Matszangosz (cooriented
    complex); other types: Rabelo-San Martin (cellular complex), with
    orientations from exact matrix realizations (B, C, D) or from Chevalley
    bases (E via Frenkel-Kac, F4 and G2 by folding)
    >>> X = real_flag_variety("A2", {1})            # RP^2
    >>> X.cohomology()                              # Z, 0, Z/2
    [(1, []), (0, []), (0, [2])]
    >>> X.homology()                                # Z, Z/2, 0
    [(1, []), (0, [2]), (0, [])]
    """
    letter = RootSystem(cartan_type).letter
    if letter == "A":
        return type_a_signed(cartan_type, crossed)
    return rabelo_san_martin(cartan_type, crossed)


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


# -- the GKM form of Kocherlakota's rule (PLAN.md S5.3, docs/H1.md) ----------

def positive_weight_sum(cells, p):
    """sigma(p): the sum of the lambda-positive tangent weights at p, i.e. the
    weight of det T_p(plus-cell). For G/P and dominant lambda it is
    Kocherlakota's sigma(x)."""
    from bbcells.core import pairing
    rank = cells.data.rank
    total = [0] * rank
    for w in cells.data.weights_of(p):
        if pairing(cells.cocharacter, w) > 0:
            total = [t + c for t, c in zip(total, w)]
    return tuple(total)


def gkm_incidences(cells):
    """{(x, y): (m, magnitude)} for invariant curves joining cells of adjacent
    dimension, dim x = dim y + 1, where sigma(x) - sigma(y) = m phi with phi
    the weight of the curve at x; magnitude 2 if m is even, 0 if odd, None if
    sigma(x) - sigma(y) is not a multiple of phi. Conjecturally these are the
    unsigned incidences of the cellular chain complex of X(R) (docs/H1.md)."""
    from bbcells.core import pairing
    data = cells.data
    result = {}
    for p, q, chi in data.edges:
        dp, dq = cells.dim_of(p), cells.dim_of(q)
        if abs(dp - dq) != 1:
            continue
        x, y, phi = (p, q, chi) if dp > dq else (q, p, tuple(-c for c in chi))
        if pairing(cells.cocharacter, phi) <= 0:
            continue            # the curve does not flow from x to y
        diff = [a - b for a, b in zip(positive_weight_sum(cells, x), positive_weight_sum(cells, y))]
        k = next(i for i, c in enumerate(phi) if c)
        m, remainder = divmod(diff[k], phi[k])
        if remainder or any(d != m * c for d, c in zip(diff, phi)):
            result[(x, y)] = (None, None)
        else:
            result[(x, y)] = (m, 0 if m % 2 else 2)
    return result


def signed_completions(cells, incidences, limit=4096):
    """sign choices for the nonzero unsigned incidences making a chain
    complex (boundary o boundary = 0), up to the gauge of flipping cell
    orientations. Yields dicts {(x, y): +-2}. Exhaustive, so only for small
    examples."""
    import itertools
    nonzero = sorted(k for k, (m, mag) in incidences.items() if mag)
    # fix a spanning forest's signs to + (gauge), enumerate the rest
    parent = {}

    def find(a):
        while parent.get(a, a) != a:
            a = parent[a]
        return a

    tree, rest = [], []
    for x, y in nonzero:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
            tree.append((x, y))
        else:
            rest.append((x, y))
    if 2 ** len(rest) > limit:
        raise ValueError("too many sign choices (%d free edges)" % len(rest))
    dims = {p: cells.dim_of(p) for p in cells.data.points}
    for signs in itertools.product((1, -1), repeat=len(rest)):
        signed = {e: 2 for e in tree}
        signed.update({e: 2 * s for e, s in zip(rest, signs)})
        if _square_zero_cellular(signed, dims):
            yield signed


def _square_zero_cellular(signed, dims):
    by_source = {}
    for (x, y), v in signed.items():
        by_source.setdefault(x, {})[y] = v
    for x, targets in by_source.items():
        total = {}
        for y, v in targets.items():
            for z, w in by_source.get(y, {}).items():
                total[z] = total.get(z, 0) + v * w
        if any(total.values()):
            return False
    return True


def cellular_rational_betti(dims, signed):
    """rational Betti numbers of the cellular chain complex with boundary
    coefficients signed[(x, y)] (dim x = dim y + 1)"""
    from bbcells.linalg import rank as matrix_rank
    top = max(dims.values())
    cells_of = {d: sorted((p for p, e in dims.items() if e == d), key=str) for d in range(top + 1)}
    ranks = {}
    for d in range(1, top + 1):
        rows = [[signed.get((x, y), 0) for x in cells_of[d]] for y in cells_of[d - 1]]
        ranks[d] = matrix_rank(rows) if rows and rows[0] else 0
    return [len(cells_of[d]) - ranks.get(d, 0) - ranks.get(d + 1, 0) for d in range(top + 1)]


# -- signs from d o d = 0 (PLAN.md S5.4) --------------------------------------

def _f2_solve(equations, nvars):
    """solve A e = b over F_2; equations: list of (bitmask, bit). Returns
    (particular solution as bitmask, list of null space basis bitmasks) or None"""
    pivots = []          # (pivot bit, row mask, rhs)
    for mask, rhs in equations:
        for bit, row, value in pivots:
            if mask >> bit & 1:
                mask ^= row
                rhs ^= value
        if mask == 0:
            if rhs:
                return None
            continue
        bit = mask.bit_length() - 1
        # reduce existing pivot rows
        new_pivots = []
        for b2, row, value in pivots:
            if row >> bit & 1:
                row ^= mask
                value ^= rhs
            new_pivots.append((b2, row, value))
        pivots = new_pivots + [(bit, mask, rhs)]
    solution = 0
    pivot_bits = {bit for bit, _, _ in pivots}
    for bit, row, value in pivots:
        if value:
            solution |= 1 << bit
    null = []
    for free in range(nvars):
        if free in pivot_bits:
            continue
        vector = 1 << free
        for bit, row, value in pivots:
            if row >> free & 1:
                vector |= 1 << bit
        null.append(vector)
    return solution, null


def _f2_rank(vectors):
    basis = []
    for v in vectors:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
    return len(basis)


def signs_from_square_zero(dims, magnitudes):
    """sign the nonzero incidences {(x, y): 2} of a cellular complex (dim x =
    dim y + 1) so that the boundary squares to zero. Pairs (x, z) two apart
    with exactly two nonzero paths give linear equations over F_2 for the
    sign exponents. Returns (signed incidences, unique_up_to_reorientation).
    Raises ValueError if the conditions are inconsistent or not linear
    (more than two paths)."""
    edges = sorted(k for k, v in magnitudes.items() if v)
    index = {e: i for i, e in enumerate(edges)}
    down = {}
    for x, y in edges:
        down.setdefault(x, []).append(y)
    equations = []
    for x, ys in down.items():
        paths = {}
        for y in ys:
            for z in down.get(y, []):
                paths.setdefault(z, []).append(y)
        for z, middle in paths.items():
            if len(middle) == 1:
                raise ValueError("boundary cannot square to zero: one path %r -> %r -> %r"
                                 % (x, middle[0], z))
            if len(middle) > 2:
                raise ValueError("more than two paths from %r to %r: nonlinear sign "
                                 "conditions" % (x, z))
            y1, y2 = middle
            mask = (1 << index[(x, y1)]) ^ (1 << index[(y1, z)]) ^ \
                   (1 << index[(x, y2)]) ^ (1 << index[(y2, z)])
            equations.append((mask, 1))
    result = _f2_solve(equations, len(edges))
    if result is None:
        raise ValueError("no sign assignment makes the boundary square to zero")
    solution, null = result
    # reorienting a cell flips the signs of all incidences at it
    cells = sorted({c for e in edges for c in e}, key=str)
    gauge = []
    for c in cells:
        v = 0
        for e, i in index.items():
            if c in e:
                v |= 1 << i
        gauge.append(v)
    unique = _f2_rank(null) == _f2_rank(gauge) and _f2_rank(null + gauge) == _f2_rank(gauge)
    signed = {e: (-2 if solution >> i & 1 else 2) for e, i in index.items()}
    return signed, unique


def sign_choices(dims, magnitudes):
    """all sign assignments with d o d = 0, one per class modulo
    reorientation of cells: a list of 2^k signed incidence dicts"""
    edges = sorted(k for k, v in magnitudes.items() if v)
    index = {e: i for i, e in enumerate(edges)}
    signed, unique = signs_from_square_zero(dims, magnitudes)
    base = 0
    for e, i in index.items():
        if signed[e] < 0:
            base |= 1 << i
    # recompute the null space and the gauge to find the extra directions
    down = {}
    for x, y in edges:
        down.setdefault(x, []).append(y)
    equations = []
    for x, ys in down.items():
        paths = {}
        for y in ys:
            for z in down.get(y, []):
                paths.setdefault(z, []).append(y)
        for z, (y1, y2) in paths.items():
            equations.append(((1 << index[(x, y1)]) ^ (1 << index[(y1, z)]) ^
                              (1 << index[(x, y2)]) ^ (1 << index[(y2, z)]), 1))
    _, null = _f2_solve(equations, len(edges))
    cells = sorted({c for e in edges for c in e}, key=str)
    span = []
    for c in cells:
        v = 0
        for e, i in index.items():
            if c in e:
                v |= 1 << i
        span.append(v)
    extra = []
    for v in null:
        if _f2_rank(span + [v]) > _f2_rank(span):
            span.append(v)
            extra.append(v)
    choices = []
    for k in range(2 ** len(extra)):
        mask = base
        for j, v in enumerate(extra):
            if k >> j & 1:
                mask ^= v
        choices.append({e: (-2 if mask >> i & 1 else 2) for e, i in index.items()})
    return choices


def cellular_real_flag_variety(cartan_type, crossed=None, max_choices=256):
    """the cellular chain complex of G/P(R) for any type: Kocherlakota's
    magnitudes, with signs from d o d = 0. When d o d = 0 leaves k free signs
    beyond reorienting cells, all 2^k choices are tried; the result is
    returned only if their integral homology agrees (then it is the homology
    of G/P(R), since the true signs are among the choices)."""
    data, magnitudes = kocherlakota_incidences(cartan_type, crossed)
    dims = {p: data.annotation(p)["length"] for p in data.points}
    choices = sign_choices(dims, magnitudes)
    if len(choices) > max_choices:
        raise ValueError("%d sign choices, more than %d" % (len(choices), max_choices))
    complexes = [RealCellComplex(data.name + "(R)", dims, signed, signed=True,
                                 convention="cellular") for signed in choices]
    homologies = {tuple((f, tuple(t)) for f, t in X.cellular_homology()) for X in complexes}
    if len(homologies) != 1:
        raise ValueError("the %d sign choices allowed by d o d = 0 give different homology"
                         % len(choices))
    return complexes[0]


# -- Rabelo-San Martin signs for classical types (PLAN.md S5.4) --------------

def rabelo_san_martin(cartan_type, crossed=None, realization=None):
    """the cellular chain complex of G/P(R) for classical G with signs from
    arXiv:1810.00934 (Theorem `teoforcw1`): for w = r_1 ... r_n (our fixed
    reduced words) and w' = r_1 ... ^r_i ... r_n,
        c(w, w') = (-1)^i deg(Phi_{w'}^{-1} o Psi_{w'}) (1 + (-1)^kappa),
    where Psi_{w'} is the parametrization by the deleted word. The degree is
    the orientation of the deleted word's tangent frame at the common point
    n_{w'} b_0 relative to the frame of w's fixed word (Tits lifts satisfy the
    braid relations, and the composite is a diffeomorphism of the open cube)."""
    from bbcells.chevalley import Lifts
    from bbcells.liealgebra import ClassicalRealization
    R = RootSystem(cartan_type)
    if realization is None:
        realization = "matrix" if R.letter in "ABCD" else "chevalley"
    if realization == "matrix":
        realization = ClassicalRealization(R.letter, R.rank)
    elif realization == "chevalley":
        realization = Lifts(cartan_type)
    else:
        raise ValueError("realization must be 'matrix' or 'chevalley'")
    data, magnitudes = kocherlakota_incidences(cartan_type, crossed)
    word = {p: tuple(i - 1 for i in data.annotation(p)["word"]) for p in data.points}
    mu = {p: data.annotation(p)["weight"] for p in data.points}
    dims = {p: len(word[p]) for p in data.points}
    signed = {}
    for (x, y), magnitude in magnitudes.items():
        if not magnitude:
            continue
        wx = word[x]
        start = mu[data.points[0]]
        candidates = []
        for i in range(len(wx)):
            deleted = wx[:i] + wx[i + 1:]
            image = start
            for letter in reversed(deleted):
                image = R.reflect_weight(letter, image)
            if image == mu[y]:
                candidates.append(i)
        if len(candidates) != 1:
            raise AssertionError("expected a unique deletion index, got %r" % candidates)
        i = candidates[0]
        deleted = wx[:i] + wx[i + 1:]
        degree = realization.relative_orientation(deleted, word[y])
        signed[(x, y)] = (-1) ** (i + 1) * degree * magnitude
    return RealCellComplex(data.name + "(R)", dims, signed, signed=True, convention="cellular")
