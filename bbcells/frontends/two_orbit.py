"""
Rank-one two-orbit completions X c Xbar = X + D (PLAN.md S6b.1).

By Akhiezer's classification, as used in arXiv:1805.04338, the completion
Xbar is a flag variety G'/P' of a larger group and the boundary D = G/Q is
a flag variety of G. Everything is computed with the torus T of G:
  * the fixed-point data of Xbar is restricted along T -> T' (PLAN.md 1.5),
  * the fixed points of D are identified inside Xbar by weight inclusion,
  * [X] = [Xbar] - [D] in K_0(Var) and chi_c^{A^1}(X) = chi(Xbar) - chi(D).

Cases, cross-referenced with Knop's table of rank-one spherical varieties
(arXiv:1303.2466, section sec:TABLE; G/H with H reductive, characteristic 0):

    Knop's row                      completion       boundary        function
    SO(n+1)/SO(n) (double covers)   Q_n              Q_{n-1}         affine_quadric
    SO(2n+1)/<s>SO(2n),
    PSO(2n)/SO(2n-1), PGL4/PSp4     P^{n+1}          Q_n             projective_space_minus_quadric
    PGL(n)/GL(n-1)                  P^{n-1} x P^{n-1}v  incidence    pgl_mod_gl
    PSp(2n)/Sp(2).Sp(2n-2)          Gr(2, 2n)        IG(2, 2n)       quaternionic_projective_space
    F4/Spin(9)                      E6/P1            F4/P4           octonionic_projective_plane
    G2/<s>SL(3)                     P^6              Q_5 = G2/P1     g2_on_p6
    SO(7)/G2                        P^7 (spinors)    Q_6 = B3/P3     spin7_on_p7
    (PGL2 x PGL2)/PGL2              P^3              P^1 x P^1       frontends.wonderful ("A1")

Not covered: the rows with non-reductive H (P_n(SO(2n)), P_1(Sp(2)).Sp(2n-2),
GL(2)U... in G2), whose completions are not of this homogeneous form, and
the rows that exist only in characteristic 2.
"""

from dataclasses import dataclass

from bbcells.algebra import IntPoly
from bbcells.core import FixedPointData, bb_cells
from bbcells.frontends import flag, linear, quadric
from bbcells.operations import identify, normal_weights, product, restrict


@dataclass(frozen=True)
class TwoOrbitCompletion(object):
    name: str
    completion: FixedPointData
    boundary: FixedPointData
    mapping: dict          # boundary label -> completion label
    normal: dict           # boundary label -> normal weight of D in Xbar

    @property
    def open_fixed_points(self):
        """fixed points of Xbar that lie in the open orbit X"""
        taken = set(self.mapping.values())
        return tuple(p for p in self.completion.points if p not in taken)

    def completion_cells(self, lam=None):
        return bb_cells(self.completion, lam)

    def boundary_cells(self, lam=None):
        return bb_cells(self.boundary, lam)

    def k0_class(self):
        """[X] = [Xbar] - [D] in K_0(Var), as a polynomial in L"""
        return (IntPoly.from_counts(self.completion_cells().counts) -
                IntPoly.from_counts(self.boundary_cells().counts))

    def gw_euler_compact_support(self):
        """chi_c^{A^1}(X) = chi^{A^1}(Xbar) - chi^{A^1}(D)"""
        return (self.completion_cells().invariants().gw_euler -
                self.boundary_cells().invariants().gw_euler)

    def summary(self):
        k0 = self.k0_class()
        gw = self.gw_euler_compact_support()
        return "\n".join([
            "%s = %s - %s" % (self.name, self.completion.name, self.boundary.name),
            "  dim %d, torus rank %d; %d fixed points of the completion, %d in the open orbit"
            % (self.completion.dim, self.completion.rank, len(self.completion),
               len(self.open_fixed_points)),
            "  [X] = %s" % k0.format("L"),
            "  chi_c^A1(X) = %s  (rank %d, signature %d)" % (gw, gw.rank, gw.signature),
        ])


def two_orbit(completion, boundary, name, mapping=None):
    """assemble and validate a two-orbit completion on a common torus"""
    if boundary.dim != completion.dim - 1:
        raise ValueError("the boundary must be a divisor: dim %d vs %d"
                         % (boundary.dim, completion.dim))
    mapping = identify(boundary, completion) if mapping is None else dict(mapping)
    normals = normal_weights(boundary, completion, mapping)
    normal = {}
    for label, weights in normals.items():
        (nu,) = weights
        normal[label] = nu
    return TwoOrbitCompletion(name, completion, boundary, mapping, normal)


def _truncation(rows, columns):
    """the character restriction dropping the last columns - rows coordinates"""
    return [[1 if i == j else 0 for j in range(columns)] for i in range(rows)]


def affine_quadric(n):
    """the affine quadric AQ_n = SO(n+1)/SO(n) = {q = 1}, i.e. Q_n minus its
    hyperplane section Q_{n-1}, with the torus of SO(n+1)
    >>> print(affine_quadric(4).k0_class().format("L"))
    L^2 + L^4
    """
    if n < 2:
        raise ValueError("need n >= 2")
    big, small = quadric.quadric(n), quadric.quadric(n - 1)
    completion = restrict(big, _truncation(small.rank, big.rank), name=big.name)
    # Q_{n-1} = Q_n cap H contains the coordinate points x_i, y_i of Q_{n-1}
    # (same labels); in low dimension weight inclusion alone is ambiguous
    return two_orbit(completion, small, "AQ_%d" % n, {p: p for p in small.points})


def quaternionic_projective_space(n):
    """HP^n = Sp_{2n+2}/(Sp_2 x Sp_{2n}): the non-degenerate planes in
    Gr(2, 2n+2), whose complement is the isotropic Grassmannian IG(2, 2n+2)
    (note: the isotropic, not the symplectic planes; cf. PLAN.md section 6)
    >>> print(quaternionic_projective_space(1).k0_class().format("L"))
    L^2 + L^4
    """
    if n < 1:
        raise ValueError("need n >= 1")
    m = n + 1
    weights, labels = linear.symplectic_weights(m)
    completion = linear.grassmannian_of(weights, 2, labels, name="Gr(2,%d)" % (2 * m))
    boundary = linear.isotropic_grassmannian_of(m, 2)
    return two_orbit(completion, boundary, "HP^%d" % n, {p: p for p in boundary.points})


# restriction of characters from the maximal torus of E6 to that of F4 = E6^sigma,
# in simple-root coordinates: alpha_1, alpha_6 -> beta_4; alpha_3, alpha_5 -> beta_3;
# alpha_4 -> beta_2; alpha_2 -> beta_1 (Bourbaki numbering on both sides)
E6_TO_F4 = [[0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [1, 0, 0, 0, 0, 1]]


def octonionic_projective_plane():
    """OP^2 = F4/Spin9: the Cayley plane E6/P1 minus F4/P4
    >>> print(octonionic_projective_plane().k0_class().format("L"))
    L^8 + L^12 + L^16
    """
    completion = restrict(flag.flag_variety("E6", {1}), E6_TO_F4, name="E6/P1")
    boundary = flag.flag_variety("F4", {4}, name="F4/P4")
    return two_orbit(completion, boundary, "OP^2")


def pgl_mod_gl(n):
    """PGL_n/GL_{n-1}: pairs (point, hyperplane) in P^{n-1} x (P^{n-1})^vee
    not incident, with the diagonal torus of GL_n; the boundary is the
    incidence variety SL_n/P_{1,n-1}
    >>> print(pgl_mod_gl(3).k0_class().format("L"))
    L^2 + L^3 + L^4
    """
    if n < 2:
        raise ValueError("need n >= 2")
    unit = lambda i, s: tuple(s if k == i else 0 for k in range(n))
    points = linear.projectivization([unit(i, 1) for i in range(n)],
                                     ["p%d" % (i + 1) for i in range(n)], name="P^%d" % (n - 1))
    hyperplanes = linear.projectivization([unit(i, -1) for i in range(n)],
                                          ["H%d" % (i + 1) for i in range(n)],
                                          name="(P^%d)^vee" % (n - 1))
    diagonal = [[1 if j in (i, i + n) else 0 for j in range(2 * n)] for i in range(n)]
    completion = restrict(product(points, hyperplanes), diagonal,
                          name="P^%d x (P^%d)^vee" % (n - 1, n - 1))
    crossed = {1} if n == 2 else {1, n - 1}
    incidence = flag.flag_variety("A%d" % (n - 1), crossed, name="incidence")
    simple_roots = [[1 if i == j else -1 if i == j + 1 else 0 for j in range(n - 1)]
                    for i in range(n)]
    boundary = restrict(incidence, simple_roots, name="incidence")
    # the flag w(<e_1> c <e_1, ..., e_{n-1}>) is (p_{w(1)}, H_{w(n)}); weight
    # inclusion is ambiguous here (eps_j - eps_i occurs twice at (p_i, H_j))
    mapping = {}
    for label in boundary.points:
        w = _permutation(boundary.annotation(label)["word"], n)
        mapping[label] = "p%d|H%d" % (w[1], w[n])
    return two_orbit(completion, boundary, "PGL_%d/GL_%d" % (n, n - 1), mapping)


def _permutation(word, n):
    """{j: w(j)} for w = s_{i_1} ... s_{i_k} (1-based word), s_i = (i, i+1)
    >>> _permutation((1, 2), 3)
    {1: 2, 2: 3, 3: 1}
    """
    result = {}
    for j in range(1, n + 1):
        x = j
        for i in reversed(word):
            x = i + 1 if x == i else i if x == i + 1 else x
        result[j] = x
    return result


def projective_space_minus_quadric(n):
    """P^{n+1} minus the quadric Q_n: SO(n+2)/S(O(1) x O(n+1)) (non-isotropic
    lines), with the torus of SO(n+2). Covers Knop's rows SO(2n+1)/<s>SO(2n),
    PSO(2n)/SO(2n-1) (up to the double cover by AQ) and PGL(4)/PSp(4) = n = 4.
    >>> print(projective_space_minus_quadric(4).k0_class().format("L"))
    -L^2 + L^5
    """
    Q = quadric.quadric(n)
    m = Q.rank
    weights, labels = [], []
    for i in range(m):
        weights.append(tuple(int(k == i) for k in range(m)))
        labels.append("x_%d" % i)
    for i in range(m):
        weights.append(tuple(-int(k == i) for k in range(m)))
        labels.append("y_%d" % i)
    if n % 2 == 1:                     # SO(2m+1): the zero weight
        weights.append((0,) * m)
        labels.append("z")
    P = linear.projectivization(weights, labels, name="P^%d" % (n + 1))
    return two_orbit(P, Q, "P^%d - Q_%d" % (n + 1, n), {p: p for p in Q.points})


def g2_on_p6():
    """P^6 = P(V_7) minus Q_5 = G2/P1 for G2: G2/(<s> SL3) (Knop's row
    G2/<s_{a_1}>.SL(3)). The weights of V_7 are the short roots and 0.
    >>> print(g2_on_p6().k0_class().format("L"))
    L^6
    """
    from bbcells.rootsystem import RootSystem
    R = RootSystem("G2")
    short = [b for b in R.positive_roots if R.inner(b, b) < 2]
    weights = short + [tuple(-c for c in b) for b in short] + [(0, 0)]
    P = linear.projectivization(weights, name="P^6")
    return two_orbit(P, flag.flag_variety("G2", {1}, name="G2/P1"), "G2/(<s>SL3)")


def spin7_on_p7():
    """P^7 = P(spin representation) minus Q_6 = B3/P3 (pure spinors) for
    Spin(7): SO(7)/G2 (Knop's row SO(7)/G_2). Characters in fundamental-weight
    coordinates: the spin weights are the W-orbit of omega_3.
    >>> print(spin7_on_p7().k0_class().format("L"))
    -L^3 + L^7
    """
    from bbcells.rootsystem import RootSystem
    R = RootSystem("B3")
    spin = [mu for mu, _ in R.orbit({0, 1})]            # W . omega_3
    P = linear.projectivization(spin, name="P^7")
    boundary = flag.flag_variety("B3", {3}, name="B3/P3")
    to_weights = [[R.root_to_weight(tuple(int(k == j) for k in range(3)))[i] for j in range(3)]
                  for i in range(3)]
    boundary = restrict(boundary, to_weights, name="B3/P3")
    return two_orbit(P, boundary, "SO(7)/G2")
