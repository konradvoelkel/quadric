"""
Independent formulas used as test oracles (PLAN.md section 4).

Nothing here uses the BB engine: the values come from point counts,
product and blow-up formulas and Weyl group degrees. All polynomials are
cell polynomials sum_d c_d x^d (x = q = L = t^2).
"""

from bbcells.algebra import IntPoly


def q_integer(n):
    """[n]_q = 1 + q + ... + q^{n-1}
    >>> q_integer(3)
    IntPoly((1, 1, 1))
    """
    return IntPoly((1,) * n)


def q_factorial(n):
    result = IntPoly((1,))
    for k in range(1, n + 1):
        result = result * q_integer(k)
    return result


def exact_division(numerator, denominator):
    """polynomial division that must be exact
    >>> exact_division(IntPoly((1, 0, -1)), IntPoly((1, 1)))
    IntPoly((1, -1))
    """
    a = list(numerator.coefficients)
    b = denominator.coefficients
    if not b:
        raise ZeroDivisionError("division by the zero polynomial")
    if len(a) < len(b):
        if any(a):
            raise ValueError("division is not exact")
        return IntPoly(())
    quotient = [0] * (len(a) - len(b) + 1)
    for i in range(len(quotient) - 1, -1, -1):
        coefficient, remainder = divmod(a[i + len(b) - 1], b[-1])
        if remainder:
            raise ValueError("division is not exact")
        quotient[i] = coefficient
        for j, c in enumerate(b):
            a[i + j] -= coefficient * c
    if any(a):
        raise ValueError("division is not exact")
    return IntPoly(tuple(quotient))


def gaussian_binomial(n, k):
    """number of k-planes in F_q^n
    >>> gaussian_binomial(4, 2)
    IntPoly((1, 1, 2, 1, 1))
    """
    return exact_division(q_factorial(n), q_factorial(k) * q_factorial(n - k))


def projective_space(n):
    return q_integer(n + 1)


def blowup(ambient, center, codimension):
    """[Bl_Z X] = [X] - [Z] + [Z][P^{c-1}] for a smooth centre of codimension c
    >>> blowup(projective_space(2), IntPoly((1,)), 2)
    IntPoly((1, 2, 1))
    """
    return ambient - center + center * projective_space(codimension - 1)


def quadric(n):
    """smooth split quadric of dimension n
    >>> quadric(4)
    IntPoly((1, 1, 2, 1, 1))
    """
    result = q_integer(n + 1)
    if n % 2 == 0:
        result = result + IntPoly.monomial(n // 2)
    return result


def from_degrees(group_degrees, levi_degrees):
    """|G/P|(q) = prod_{d in deg W} [d]_q / prod_{d in deg W_I} [d]_q.
    Degrees of a torus factor are 1 ([1]_q = 1), so they may be omitted.
    >>> from_degrees([2, 3], [2])    # P^2 = SL_3 / P_1
    IntPoly((1, 1, 1))
    """
    numerator, denominator = IntPoly((1,)), IntPoly((1,))
    for d in group_degrees:
        numerator = numerator * q_integer(d)
    for d in levi_degrees:
        denominator = denominator * q_integer(d)
    return exact_division(numerator, denominator)


def _classify_component(cartan, nodes):
    """Cartan type letter and rank of a connected Dynkin diagram"""
    k = len(nodes)
    if k == 1:
        return "A", 1
    bonds = {}
    degree = {i: 0 for i in nodes}
    for a in nodes:
        for b in nodes:
            if a < b and cartan[a][b] != 0:
                bonds[(a, b)] = cartan[a][b] * cartan[b][a]
                degree[a] += 1
                degree[b] += 1
    multiplicities = set(bonds.values())
    if 3 in multiplicities:
        return "G", 2
    if 2 in multiplicities:
        (a, b), = [e for e, m in bonds.items() if m == 2]
        if k == 4 and degree[a] == 2 and degree[b] == 2:
            return "F", 4
        return "B", k                      # B_k and C_k have the same degrees
    branch = [i for i in nodes if degree[i] == 3]
    if not branch:
        return "A", k
    (centre,) = branch
    arms = []
    for start in [b for (a, b) in bonds if a == centre] + [a for (a, b) in bonds if b == centre]:
        length, previous, current = 1, centre, start
        while True:
            nxt = [j for j in nodes if j not in (previous, current) and
                   ((current, j) in bonds or (j, current) in bonds)]
            if not nxt:
                break
            previous, current = current, nxt[0]
            length += 1
        arms.append(length)
    arms.sort()
    if arms[:2] == [1, 1]:
        return "D", k
    return {(1, 2, 2): ("E", 6), (1, 2, 3): ("E", 7), (1, 2, 4): ("E", 8)}[tuple(arms)]


def levi_degrees(cartan, levi):
    """degrees of the Weyl group W_I of the Levi with simple roots levi
    (0-based), by classifying the connected components of the sub-diagram
    >>> levi_degrees(((2, -1, 0), (-1, 2, -1), (0, -1, 2)), {0, 1})   # A2 in A3
    [2, 3]
    """
    from bbcells.rootsystem import DEGREES
    remaining, degrees = set(levi), []
    while remaining:
        start = min(remaining)
        component, stack = {start}, [start]
        while stack:
            i = stack.pop()
            for j in list(remaining):
                if j not in component and cartan[i][j] != 0:
                    component.add(j)
                    stack.append(j)
        remaining -= component
        letter, rank = _classify_component(cartan, sorted(component))
        degrees.extend(DEGREES[letter](rank))
    return sorted(degrees)


def flag_variety(cartan_type, crossed=None):
    """|G/P_J|(q) from Weyl group degrees (J = crossed nodes, 1-based)
    >>> flag_variety("E6", {1})(1)
    27
    """
    from bbcells.rootsystem import RootSystem
    R = RootSystem(cartan_type)
    crossed = set(range(1, R.rank + 1)) if crossed is None else set(crossed)
    levi = {i - 1 for i in range(1, R.rank + 1) if i not in crossed}
    return from_degrees(R.degrees, levi_degrees(R.cartan, levi))


def wonderful(cartan_type):
    """|X(F_q)| for the wonderful compactification of the adjoint group, from
    the G x G-orbit decomposition (PLAN.md 4.3):
    sum_I |G/P_I|^2 * q^{N_I} (q - 1)^{|I|} prod_{d in deg W_I} [d]_q
    >>> wonderful("A1")
    IntPoly((1, 1, 1, 1))
    """
    import itertools
    from bbcells.rootsystem import RootSystem
    R = RootSystem(cartan_type)
    nodes = list(range(R.rank))
    total = IntPoly(())
    for size in range(R.rank + 1):
        for levi in itertools.combinations(nodes, size):
            crossed = {i + 1 for i in nodes if i not in levi}
            degrees = levi_degrees(R.cartan, set(levi))
            levi_group = IntPoly.monomial(sum(d - 1 for d in degrees)) * \
                IntPoly((-1, 1)) ** len(levi)
            for d in degrees:
                levi_group = levi_group * q_integer(d)
            base = flag_variety(cartan_type, crossed)
            total = total + base * base * levi_group
    return total


def real_grassmannian_rational_poincare(k, n):
    """rational Poincare polynomial (in t) of the real Grassmannian Gr_k(R^n),
    Casian-Kodama, arXiv:1309.5520, Theorem B:
      [m choose j]_{t^4}                      for (k, n) = (2j, 2m), (2j, 2m+1), (2j+1, 2m+1),
      (t^{2m-1} + 1) [m-1 choose j]_{t^4}     for (k, n) = (2j+1, 2m)
    >>> real_grassmannian_rational_poincare(1, 4)        # RP^3
    IntPoly((1, 0, 0, 1))
    """
    j, m = k // 2, n // 2
    if k % 2 == 1 and n % 2 == 0:
        return (IntPoly.monomial(2 * m - 1) + 1) * gaussian_binomial(m - 1, j).substitute_power(4)
    return gaussian_binomial(m, j).substitute_power(4)


def reduced_rational_betti(faces, top):
    """reduced rational Betti numbers b~_0..b~_top of a simplicial complex
    given by its faces (frozensets, closed under subsets, containing the
    empty face); b~_{-1} = 1 for the empty complex is reported at index -1
    via the returned dict"""
    from bbcells.linalg import rank as matrix_rank
    by_dim = {}
    for f in faces:
        by_dim.setdefault(len(f) - 1, []).append(tuple(sorted(f)))
    for d in by_dim:
        by_dim[d].sort()
    index = {d: {f: i for i, f in enumerate(fs)} for d, fs in by_dim.items()}

    def boundary_rank(d):
        """rank of the boundary C_d -> C_{d-1} (augmented: C_{-1} = empty face)"""
        if d not in by_dim or d - 1 not in by_dim:
            return 0
        rows = []
        for f in by_dim[d]:
            row = [0] * len(by_dim[d - 1])
            for k in range(len(f)):
                row[index[d - 1][f[:k] + f[k + 1:]]] = (-1) ** k
            rows.append(row)
        return matrix_rank(rows)

    betti = {}
    for d in range(-1, top + 1):
        n = len(by_dim.get(d, []))
        betti[d] = n - boundary_rank(d) - boundary_rank(d + 1)
    return betti


def real_toric_rational_betti(rays, cones):
    """rational Betti numbers of the real points X(R) of the smooth complete
    toric variety of a fan, by Choi-Park (arXiv:1311.7056, Theorem
    `cohomofsmallcover`) = Suciu-Trevisan:
        H^p(X(R); Q) = sum over omega in the row space of Lambda (mod 2) of
                       H~^{p-1}(K_omega; Q),
    with Lambda the n x m matrix of the rays mod 2 and K_omega the full
    subcomplex of the fan's simplicial complex on the rays in omega.
    >>> real_toric_rational_betti([(1,), (-1,)], [(0,), (1,)])          # RP^1 = circle
    [1, 1]
    """
    import itertools
    n, m = len(rays[0]), len(rays)
    faces = set()
    for cone in cones:
        for size in range(len(cone) + 1):
            for sub in itertools.combinations(sorted(cone), size):
                faces.add(frozenset(sub))
    rows = [[r[i] % 2 for r in rays] for i in range(n)]
    row_space = set()
    for coefficients in itertools.product((0, 1), repeat=n):
        vector = tuple(sum(c * row[j] for c, row in zip(coefficients, rows)) % 2
                       for j in range(m))
        row_space.add(vector)
    betti = [0] * (n + 1)
    for vector in row_space:
        omega = {j for j in range(m) if vector[j]}
        sub = [f for f in faces if f <= omega]
        reduced = reduced_rational_betti(sub, n - 1)
        for p in range(n + 1):
            betti[p] += reduced.get(p - 1, 0)
    return betti


def batyrev_moreau(cartan_type, crossed, lattice_basis, cones):
    """[X] for a smooth complete horospherical embedding from its colored fan,
    by Batyrev-Moreau (arXiv:1203.0671, Theorem t:main) for smooth cones:
      [X] = [G/P] (L - 1)^r sum_{faces sigma} prod_{v in sigma} 1/(L^{a_v} - 1),
    a_v = 1 for uncolored rays, a_alpha = 2 <rho_S - rho_I, alpha^vee> for colors.
    Same input format as frontends.horospherical.colored_horospherical.
    >>> batyrev_moreau("A1", {1}, [(1,)], [([(1,)], {1}), ([(-1,)], set())])   # P^2
    IntPoly((1, 1, 1))
    """
    import itertools
    from bbcells.rootsystem import RootSystem
    R = RootSystem(cartan_type)
    levi = {i - 1 for i in range(1, R.rank + 1) if i not in set(crossed)}
    r = len(lattice_basis)
    a = {}
    for alpha in crossed:
        simple = tuple(int(k == alpha - 1) for k in range(R.rank))
        a[alpha] = 2 - sum(R.coroot_pairing(simple, gamma) for gamma in R.positive_roots_of(levi))
    rho = {alpha: tuple(v[alpha - 1] for v in lattice_basis) for alpha in crossed}
    faces = {}
    for gens, colors in cones:
        weight_of = {}
        for g in gens:
            colored = [alpha for alpha in colors if rho[alpha] == tuple(g)]
            values = {a[alpha] for alpha in colored}
            if len(values) > 1:
                raise ValueError("inconsistent colors on %r" % (g,))
            weight_of[tuple(g)] = values.pop() if values else 1
        for size in range(len(gens) + 1):
            for sub in itertools.combinations(sorted(weight_of), size):
                faces[frozenset(sub)] = tuple(sorted(weight_of[v] for v in sub))
    # common denominator D = product over faces' denominators via lcm of q-integers
    denominators = [IntPoly((1,))]
    for exponents in faces.values():
        d = IntPoly((1,))
        for e in exponents:
            d = d * q_integer(e)
        denominators.append(d)
    D = IntPoly((1,))
    for d in denominators:
        # multiply D by the part of d not yet dividing D (simple lcm by trial)
        try:
            exact_division(D, d)
        except ValueError:
            D = D * d
    numerator = IntPoly(())
    for exponents in faces.values():
        term = (IntPoly((-1, 1)) ** (r - len(exponents)))
        d = IntPoly((1,))
        for e in exponents:
            d = d * q_integer(e)
        numerator = numerator + term * exact_division(D, d)
    return exact_division(flag_variety(cartan_type, crossed) * numerator, D)


def complete_quadrics_p3():
    """[complete quadrics in P^3] by the blow-up formula along Vainsencher's
    centres: X_1 = Bl_V P^9 (V = v_2(P^3), codim 6), X = Bl_{S~_2} X_1
    (codim 3), [S~_2] = [S_2] - [V] + [V][P^2], and [S_2] (quadrics of rank
    <= 2, i.e. pairs of planes) = [P^3] + L^2 (L^2 + 1)(L^2 + L + 1) from
    counting symmetric 4 x 4 matrices of rank <= 2 over F_q
    >>> complete_quadrics_p3()(1)
    66
    """
    L = IntPoly.monomial(1)
    P = projective_space
    S2 = P(3) + L ** 2 * (L ** 2 + 1) * (L ** 2 + L + 1)
    S2_tilde = S2 - P(3) + P(3) * P(2)
    X1 = blowup(P(9), P(3), 6)
    return blowup(X1, S2_tilde, 3)


def compositions(n):
    """the compositions of n (ordered tuples of positive parts)
    >>> list(compositions(3))
    [(1, 1, 1), (1, 2), (2, 1), (3,)]
    """
    if n == 0:
        yield ()
        return
    for first in range(1, n + 1):
        for rest in compositions(n - first):
            yield (first,) + rest


def gaussian_multinomial(parts):
    """|GL_n/P| for the parabolic P of block sizes `parts`"""
    result = q_factorial(sum(parts))
    for k in parts:
        result = exact_division(result, q_factorial(k))
    return result


def _nondegenerate_symmetric(k):
    """nondegenerate symmetric k x k matrices over F_q (q odd, MacWilliams):
    q^{k(k+1)/2 - m^2} prod_{i<=m} (q^{2i-1} - 1), m = ceil(k/2)"""
    m = (k + 1) // 2
    result = IntPoly.monomial(k * (k + 1) // 2 - m * m)
    for i in range(1, m + 1):
        result = result * (IntPoly.monomial(2 * i - 1) - IntPoly((1,)))
    return result


def _nondegenerate_alternating(k):
    """nondegenerate alternating 2k x 2k matrices, |GL_{2k}|/|Sp_{2k}| =
    q^{k(k-1)} prod_{i<=k} (q^{2i-1} - 1)"""
    result = IntPoly.monomial(k * (k - 1))
    for i in range(1, k + 1):
        result = result * (IntPoly.monomial(2 * i - 1) - IntPoly((1,)))
    return result


def complete_quadrics(n):
    """[complete quadrics in P^{n-1}] from the orbit decomposition: the orbits
    are indexed by compositions (k_1, ..., k_r) of n and fibre over the partial
    flag variety of that type with fibre prod_j PGL_{k_j}/PO_{k_j} (the smooth
    quadrics in P^{k_j - 1}, counted as nondegenerate forms up to scalar)
    >>> complete_quadrics(3) == complete_conics()
    True
    >>> complete_quadrics(4) == complete_quadrics_p3()
    True
    """
    q_minus_1 = IntPoly((-1, 1))
    total = IntPoly(())
    for parts in compositions(n):
        term = gaussian_multinomial(parts)
        for k in parts:
            term = term * exact_division(_nondegenerate_symmetric(k), q_minus_1)
        total = total + term
    return total


def complete_skew_forms(n):
    """[complete skew forms on k^{2n}], likewise: compositions of n, flags of
    type (2k_1, ..., 2k_r), fibres prod_j PGL_{2k_j}/PSp_{2k_j}
    >>> complete_skew_forms(2) == projective_space(5)
    True
    """
    q_minus_1 = IntPoly((-1, 1))
    total = IntPoly(())
    for parts in compositions(n):
        term = gaussian_multinomial([2 * k for k in parts])
        for k in parts:
            term = term * exact_division(_nondegenerate_alternating(k), q_minus_1)
        total = total + term
    return total


def complete_conics():
    """[P^5] - [P^2] + [P^2][P^2] (PLAN.md 4.5)"""
    return blowup(projective_space(5), projective_space(2), 3)
