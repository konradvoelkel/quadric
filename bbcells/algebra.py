"""
Small exact algebra: integer polynomials and Grothendieck-Witt classes.

Polynomials are used for Poincare polynomials (variable t), classes in
K_0(Var) (variable L) and point counts (variable q); the variable name is
only a matter of printing.
"""

from dataclasses import dataclass


def _strip(coefficients):
    coefficients = list(coefficients)
    while coefficients and coefficients[-1] == 0:
        coefficients.pop()
    return tuple(coefficients)


@dataclass(frozen=True)
class IntPoly(object):
    """dense univariate polynomial with integer coefficients, lowest degree first
    >>> p = IntPoly((1, 1))
    >>> p * p
    IntPoly((1, 2, 1))
    >>> print((p * p).format("q"))
    1 + 2q + q^2
    >>> (p * p)(2)
    9
    >>> IntPoly((1, 0, 0)) == IntPoly((1,))
    True
    """
    coefficients: tuple = ()

    def __post_init__(self):
        if not all(isinstance(c, int) for c in self.coefficients):
            raise TypeError("coefficients must be integers: %r" % (self.coefficients,))
        object.__setattr__(self, "coefficients", _strip(self.coefficients))

    @classmethod
    def monomial(cls, degree, coefficient=1):
        """
        >>> IntPoly.monomial(2, 3)
        IntPoly((0, 0, 3))
        """
        return cls((0,) * degree + (coefficient,))

    @classmethod
    def from_counts(cls, counts):
        """polynomial sum_d counts[d] x^d from a dict or sequence"""
        if isinstance(counts, dict):
            top = max(counts, default=-1)
            return cls(tuple(counts.get(d, 0) for d in range(top + 1)))
        return cls(tuple(counts))

    @property
    def degree(self):
        """degree, -1 for the zero polynomial
        >>> IntPoly(()).degree, IntPoly((0, 5)).degree
        (-1, 1)
        """
        return len(self.coefficients) - 1

    def __getitem__(self, degree):
        if 0 <= degree < len(self.coefficients):
            return self.coefficients[degree]
        return 0

    def __iter__(self):
        return iter(self.coefficients)

    def __len__(self):
        return len(self.coefficients)

    def __add__(self, other):
        other = _as_poly(other)
        n = max(len(self), len(other))
        return IntPoly(tuple(self[i] + other[i] for i in range(n)))

    __radd__ = __add__

    def __neg__(self):
        return IntPoly(tuple(-c for c in self.coefficients))

    def __sub__(self, other):
        return self + (-_as_poly(other))

    def __rsub__(self, other):
        return _as_poly(other) - self

    def __mul__(self, other):
        other = _as_poly(other)
        if not self.coefficients or not other.coefficients:
            return IntPoly(())
        result = [0] * (len(self) + len(other) - 1)
        for i, a in enumerate(self.coefficients):
            if a:
                for j, b in enumerate(other.coefficients):
                    result[i + j] += a * b
        return IntPoly(tuple(result))

    __rmul__ = __mul__

    def __pow__(self, exponent):
        result = IntPoly((1,))
        for _ in range(exponent):
            result = result * self
        return result

    def __call__(self, value):
        result = 0
        for c in reversed(self.coefficients):
            result = result * value + c
        return result

    def substitute_power(self, k):
        """p(x) -> p(x^k), e.g. from q = t^2 to t
        >>> IntPoly((1, 1)).substitute_power(2)
        IntPoly((1, 0, 1))
        """
        result = [0] * (k * self.degree + 1) if self.coefficients else []
        for i, c in enumerate(self.coefficients):
            result[k * i] = c
        return IntPoly(tuple(result))

    def is_palindromic(self, degree=None):
        """palindromic with respect to the given degree (default: own degree)
        >>> IntPoly((1, 2, 1)).is_palindromic(), IntPoly((0, 1, 2)).is_palindromic(3)
        (True, False)
        """
        n = self.degree if degree is None else degree
        return all(self[i] == self[n - i] for i in range(n + 1)) and self.degree <= n

    def format(self, var="x"):
        """human readable, lowest degree first
        >>> print(IntPoly((0, -1, 0, 2)).format("L"))
        -L + 2L^3
        >>> print(IntPoly(()).format())
        0
        """
        terms = []
        for i, c in enumerate(self.coefficients):
            if c == 0:
                continue
            if i == 0:
                body = str(abs(c))
            else:
                power = var if i == 1 else "%s^%d" % (var, i)
                body = power if abs(c) == 1 else "%d%s" % (abs(c), power)
            terms.append(("-" if c < 0 else "+", body))
        if not terms:
            return "0"
        text = ("-" if terms[0][0] == "-" else "") + terms[0][1]
        for sign, body in terms[1:]:
            text += " %s %s" % (sign, body)
        return text

    def latex(self, var="x"):
        """
        >>> IntPoly((1, 0, 2)).latex("t")
        '1 + 2t^{2}'
        """
        return _latex_terms(self, var)

    def __repr__(self):
        return "IntPoly(%r)" % (self.coefficients,)

    def __str__(self):
        return self.format()


def _latex_terms(poly, var):
    text = poly.format(var)
    out, i = [], 0
    while i < len(text):
        if text[i] == "^":
            j = i + 1
            while j < len(text) and text[j].isdigit():
                j += 1
            out.append("^{%s}" % text[i + 1:j])
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _as_poly(value):
    if isinstance(value, IntPoly):
        return value
    if isinstance(value, int):
        return IntPoly((value,))
    raise TypeError("cannot combine IntPoly with %r" % (value,))


@dataclass(frozen=True)
class GWClass(object):
    """the element plus*<1> + minus*<-1> of the subring of GW(k) generated
    by <-1>, i.e. an element of Z[Z/2] mapped to GW(k).

    Rank and signature are well defined for every field of characteristic
    not 2; the signature is the one for an ordering in which -1 < 0 (for
    k = R it is the usual signature).
    >>> H = GWClass.hyperbolic()
    >>> H, H.rank, H.signature
    (GWClass(plus=1, minus=1), 2, 0)
    >>> print(GWClass(2, 1))
    2<1> + <-1>
    >>> GWClass(0, 1) * GWClass(0, 1) == GWClass.one()
    True
    >>> GWClass(3, -1) * H == GWClass(2, 2)
    True
    """
    plus: int = 0
    minus: int = 0

    @classmethod
    def one(cls):
        return cls(1, 0)

    @classmethod
    def minus_one(cls):
        """the class <-1>"""
        return cls(0, 1)

    @classmethod
    def hyperbolic(cls):
        """the hyperbolic form H = <1> + <-1>"""
        return cls(1, 1)

    @classmethod
    def power_of_minus_one(cls, exponent):
        """<-1>^exponent"""
        return cls(0, 1) if exponent % 2 else cls(1, 0)

    @property
    def rank(self):
        return self.plus + self.minus

    @property
    def signature(self):
        return self.plus - self.minus

    def __add__(self, other):
        return GWClass(self.plus + other.plus, self.minus + other.minus)

    def __neg__(self):
        return GWClass(-self.plus, -self.minus)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, int):
            return GWClass(other * self.plus, other * self.minus)
        return GWClass(self.plus * other.plus + self.minus * other.minus,
                       self.plus * other.minus + self.minus * other.plus)

    __rmul__ = __mul__

    def __str__(self):
        terms = []
        for coefficient, name in ((self.plus, "<1>"), (self.minus, "<-1>")):
            if coefficient:
                prefix = "" if abs(coefficient) == 1 else str(abs(coefficient))
                terms.append(("-" if coefficient < 0 else "+", prefix + name))
        if not terms:
            return "0"
        text = ("-" if terms[0][0] == "-" else "") + terms[0][1]
        for sign, body in terms[1:]:
            text += " %s %s" % (sign, body)
        return text

    def latex(self):
        r"""
        >>> GWClass(2, 1).latex()
        '2\\langle 1\\rangle + \\langle -1\\rangle'
        """
        return str(self).replace("<", "\\langle ").replace(">", "\\rangle")
