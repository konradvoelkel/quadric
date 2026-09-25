"""
Sparse multivariate polynomials over Q in the equivariant parameters
t_1, ..., t_r (PLAN.md S3.3). A character chi in Z^r is the linear form
sum_i chi_i t_i.
"""

from fractions import Fraction
from itertools import combinations_with_replacement


class Poly(object):
    """immutable polynomial: {exponent tuple: Fraction}
    >>> t1, t2 = Poly.variable(0, 2), Poly.variable(1, 2)
    >>> p = (t1 + t2) * (t1 - t2)
    >>> p == t1 * t1 - t2 * t2
    True
    >>> p.divide_by_linear((1, -1)) == t1 + t2
    True
    >>> print(p)
    t1^2 - t2^2
    """
    __slots__ = ("nvars", "terms")

    def __init__(self, nvars, terms=None):
        self.nvars = nvars
        clean = {}
        for exponent, c in (terms or {}).items():
            if c:
                clean[tuple(exponent)] = Fraction(c)
        self.terms = clean

    @classmethod
    def constant(cls, value, nvars):
        return cls(nvars, {(0,) * nvars: value})

    @classmethod
    def variable(cls, i, nvars):
        return cls(nvars, {tuple(int(k == i) for k in range(nvars)): 1})

    @classmethod
    def linear(cls, chi):
        """the linear form sum chi_i t_i"""
        n = len(chi)
        return cls(n, {tuple(int(k == i) for k in range(n)): c for i, c in enumerate(chi) if c})

    @classmethod
    def product_of_linear(cls, characters, nvars):
        result = cls.constant(1, nvars)
        for chi in characters:
            result = result * cls.linear(chi)
        return result

    def is_zero(self):
        return not self.terms

    def degree(self):
        return max((sum(e) for e in self.terms), default=-1)

    def is_homogeneous(self, degree):
        return all(sum(e) == degree for e in self.terms)

    def __eq__(self, other):
        if isinstance(other, (int, Fraction)):
            other = Poly.constant(other, self.nvars)
        return self.nvars == other.nvars and self.terms == other.terms

    def __hash__(self):
        return hash((self.nvars, frozenset(self.terms.items())))

    def __add__(self, other):
        if isinstance(other, (int, Fraction)):
            other = Poly.constant(other, self.nvars)
        terms = dict(self.terms)
        for e, c in other.terms.items():
            terms[e] = terms.get(e, 0) + c
        return Poly(self.nvars, terms)

    __radd__ = __add__

    def __neg__(self):
        return Poly(self.nvars, {e: -c for e, c in self.terms.items()})

    def __sub__(self, other):
        return self + (-other if isinstance(other, Poly) else -Fraction(other))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, (int, Fraction)):
            return Poly(self.nvars, {e: c * other for e, c in self.terms.items()})
        terms = {}
        for e1, c1 in self.terms.items():
            for e2, c2 in other.terms.items():
                e = tuple(a + b for a, b in zip(e1, e2))
                terms[e] = terms.get(e, 0) + c1 * c2
        return Poly(self.nvars, terms)

    __rmul__ = __mul__

    def __pow__(self, k):
        result = Poly.constant(1, self.nvars)
        for _ in range(k):
            result = result * self
        return result

    def restrict_to_hyperplane(self, chi):
        """the restriction to {chi = 0}: substitute t_j = -(sum_{i != j} chi_i t_i)/chi_j
        for the last j with chi_j != 0; zero iff chi divides self"""
        j = max(i for i, c in enumerate(chi) if c)
        replacement = Poly(self.nvars, {tuple(int(k == i) for k in range(self.nvars)):
                                        Fraction(-c, chi[j]) for i, c in enumerate(chi)
                                        if c and i != j})
        result = Poly(self.nvars)
        powers = {0: Poly.constant(1, self.nvars)}
        for e, c in self.terms.items():
            k = e[j]
            if k not in powers:
                powers[k] = replacement ** k
            rest = tuple(0 if i == j else x for i, x in enumerate(e))
            result = result + Poly(self.nvars, {rest: c}) * powers[k]
        return result

    def divide_by_linear(self, chi):
        """exact division by the linear form chi (raises if not divisible)"""
        j = max(i for i, c in enumerate(chi) if c)
        remainder = Poly(self.nvars, dict(self.terms))
        quotient = Poly(self.nvars)
        divisor = Poly.linear(chi)
        # eliminate the highest power of t_j repeatedly
        while not remainder.is_zero():
            e = max(remainder.terms, key=lambda x: (x[j], x))
            if e[j] == 0:
                raise ValueError("not divisible by %r" % (chi,))
            monomial = Poly(self.nvars, {tuple(x - (1 if i == j else 0) for i, x in enumerate(e)):
                                         remainder.terms[e] / chi[j]})
            quotient = quotient + monomial
            remainder = remainder - monomial * divisor
        return quotient

    def constant_term(self):
        return self.terms.get((0,) * self.nvars, Fraction(0))

    def evaluate(self, values):
        total = Fraction(0)
        for e, c in self.terms.items():
            term = c
            for v, k in zip(values, e):
                term *= Fraction(v) ** k
            total += term
        return total

    def coefficients(self, monomials):
        return [self.terms.get(m, Fraction(0)) for m in monomials]

    def __repr__(self):
        return "Poly(%d, %r)" % (self.nvars, self.terms)

    def __str__(self):
        if not self.terms:
            return "0"
        parts = []
        for e in sorted(self.terms, reverse=True):
            c = self.terms[e]
            monomial = "*".join(("t%d" % (i + 1)) + ("^%d" % k if k > 1 else "")
                                for i, k in enumerate(e) if k)
            if not monomial:
                body = str(abs(c))
            elif abs(c) == 1:
                body = monomial
            else:
                body = "%s*%s" % (abs(c), monomial)
            parts.append(("-" if c < 0 else "+", body))
        text = ("-" if parts[0][0] == "-" else "") + parts[0][1]
        for sign, body in parts[1:]:
            text += " %s %s" % (sign, body)
        return text


def monomials(nvars, degree):
    """all exponent tuples of the given total degree
    >>> monomials(2, 2)
    [(2, 0), (1, 1), (0, 2)]
    """
    result = []
    for combo in combinations_with_replacement(range(nvars), degree):
        e = [0] * nvars
        for i in combo:
            e[i] += 1
        result.append(tuple(e))
    return result
