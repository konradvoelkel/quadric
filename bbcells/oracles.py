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
