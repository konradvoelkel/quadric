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
