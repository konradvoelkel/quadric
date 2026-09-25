"""
Root systems of all Cartan types from their Dynkin diagrams (PLAN.md S2.1).

Conventions (Bourbaki numbering, nodes 1..r in the user interface, 0..r-1
internally):
  * cartan[i][j] = <alpha_i^vee, alpha_j>, so s_i(beta) = beta - <alpha_i^vee, beta> alpha_i;
  * roots are integer vectors in the basis of simple roots ("adjoint
    realization"); weights are integer vectors in the basis of fundamental
    weights; a cocharacter lambda is given by its values <lambda, alpha_i>;
  * the Weyl group is never enumerated for G/P: the orbit W.omega_I is
    enumerated by breadth-first search (PLAN.md 1.8).
"""

import re
from fractions import Fraction

_TYPE = re.compile(r"([A-G])(\d+)")

DEGREES = {
    "A": lambda n: list(range(2, n + 2)),
    "B": lambda n: [2 * k for k in range(1, n + 1)],
    "C": lambda n: [2 * k for k in range(1, n + 1)],
    "D": lambda n: sorted([2 * k for k in range(1, n)] + [n]),
    "E": lambda n: {6: [2, 5, 6, 8, 9, 12], 7: [2, 6, 8, 10, 12, 14, 18],
                    8: [2, 8, 12, 14, 18, 20, 24, 30]}[n],
    "F": lambda n: [2, 6, 8, 12],
    "G": lambda n: [2, 6],
}


def parse_cartan_type(text):
    """
    >>> parse_cartan_type("E6"), parse_cartan_type(" b3 ")
    (('E', 6), ('B', 3))
    >>> parse_cartan_type("E5")
    Traceback (most recent call last):
    ...
    ValueError: unsupported Cartan type 'E5'
    """
    match = _TYPE.fullmatch(text.strip().upper())
    if not match:
        raise ValueError("cannot parse Cartan type %r" % text)
    letter, n = match.group(1), int(match.group(2))
    valid = {"A": n >= 1, "B": n >= 1, "C": n >= 2, "D": n >= 2,
             "E": n in (6, 7, 8), "F": n == 4, "G": n == 2}[letter]
    if not valid:
        raise ValueError("unsupported Cartan type %r" % text.strip())
    return letter, n


def _bonds(letter, n):
    """Dynkin diagram as (i, j, m), 0-based. For m > 1, i is the long root."""
    if letter == "A":
        return [(i, i + 1, 1) for i in range(n - 1)]
    if letter == "B":
        return [(i, i + 1, 1) for i in range(n - 2)] + ([(n - 2, n - 1, 2)] if n >= 2 else [])
    if letter == "C":
        return [(i, i + 1, 1) for i in range(n - 2)] + [(n - 1, n - 2, 2)]
    if letter == "D":
        if n == 2:
            return []
        return [(i, i + 1, 1) for i in range(n - 2)] + [(n - 3, n - 1, 1)]
    if letter == "E":
        chain = [(0, 2, 1), (2, 3, 1), (1, 3, 1)]
        return chain + [(k, k + 1, 1) for k in range(3, n - 1)]
    if letter == "F":
        return [(0, 1, 1), (1, 2, 2), (2, 3, 1)]
    if letter == "G":
        return [(1, 0, 3)]
    raise ValueError(letter)


class RootSystem(object):
    """an irreducible root system
    >>> R = RootSystem("B2")
    >>> R.cartan
    ((2, -1), (-2, 2))
    >>> R.positive_roots
    ((1, 0), (0, 1), (1, 1), (1, 2))
    >>> R.coroot_pairing((1, 2), (1, 0))       # <(a1 + 2 a2)^vee, a1>
    0
    >>> len(RootSystem("E8").positive_roots)
    120
    """

    def __init__(self, cartan_type):
        self.letter, self.rank = parse_cartan_type(cartan_type)
        self.name = "%s%d" % (self.letter, self.rank)
        n = self.rank
        cartan = [[2 if i == j else 0 for j in range(n)] for i in range(n)]
        for i, j, m in _bonds(self.letter, n):
            # i long, j short when m > 1: <a_i^vee, a_j> = -1, <a_j^vee, a_i> = -m
            cartan[i][j] = -1
            cartan[j][i] = -m
        self.cartan = tuple(tuple(row) for row in cartan)
        self._lengths = self._squared_lengths()
        self.positive_roots = self._positive_roots()
        self.degrees = DEGREES[self.letter](n)

    def __repr__(self):
        return "RootSystem(%r)" % self.name

    def _squared_lengths(self):
        """(alpha_i, alpha_i) with the long roots of each component of length 2
        (for B1, the single root is short: length 1)"""
        n = self.rank
        lengths = [None] * n
        for start in range(n):
            if lengths[start] is not None:
                continue
            lengths[start] = Fraction(1)
            stack = [start]
            component = [start]
            while stack:
                i = stack.pop()
                for j in range(n):
                    if j != i and self.cartan[i][j] != 0 and lengths[j] is None:
                        # d_i a_ij = d_j a_ji  (symmetrizability)
                        lengths[j] = lengths[i] * self.cartan[i][j] / self.cartan[j][i]
                        stack.append(j)
                        component.append(j)
            longest = max(lengths[i] for i in component)
            scale = (Fraction(1) if self.letter == "B" and n == 1 else 2 / longest)
            for i in component:
                lengths[i] *= scale
        return tuple(lengths)

    # -- roots in simple-root coordinates ---------------------------------

    def inner(self, beta, gamma):
        """the W-invariant form, with long roots of squared length 2"""
        total = Fraction(0)
        for i, b in enumerate(beta):
            if b:
                for j, c in enumerate(gamma):
                    if c:
                        total += b * c * self.cartan[i][j] * self._lengths[i] / 2
        return total

    def coroot_pairing(self, beta, gamma):
        """<beta^vee, gamma> = 2 (beta, gamma) / (beta, beta), an integer for roots"""
        value = 2 * self.inner(beta, gamma) / self.inner(beta, beta)
        if value.denominator != 1:
            raise ValueError("non-integral pairing: %r, %r" % (beta, gamma))
        return int(value)

    def reflect(self, i, beta):
        """s_i(beta) for beta in simple-root coordinates"""
        c = sum(b * self.cartan[i][j] for j, b in enumerate(beta))
        if c == 0:
            return tuple(beta)
        return tuple(b - c if j == i else b for j, b in enumerate(beta))

    def _positive_roots(self):
        simple = [tuple(int(i == j) for j in range(self.rank)) for i in range(self.rank)]
        roots, frontier = set(simple), list(simple)
        while frontier:
            new = []
            for beta in frontier:
                for i in range(self.rank):
                    gamma = self.reflect(i, beta)
                    if gamma not in roots and all(c >= 0 for c in gamma):
                        roots.add(gamma)
                        new.append(gamma)
            frontier = new
        return tuple(sorted(roots, key=lambda b: (sum(b), tuple(-c for c in b))))

    def height(self, beta):
        return sum(beta)

    def positive_roots_of(self, levi):
        """positive roots supported on the simple roots with indices in levi (0-based)"""
        levi = set(levi)
        return tuple(b for b in self.positive_roots
                     if all(c == 0 or i in levi for i, c in enumerate(b)))

    @property
    def weyl_group_order(self):
        order = 1
        for d in self.degrees:
            order *= d
        return order

    # -- weights in fundamental-weight coordinates ------------------------

    def reflect_weight(self, i, mu):
        """s_i(mu) for mu in fundamental-weight coordinates:
        s_i(mu) = mu - mu_i alpha_i, and alpha_i has coordinates cartan[j][i]"""
        m = mu[i]
        if m == 0:
            return tuple(mu)
        return tuple(x - m * self.cartan[j][i] for j, x in enumerate(mu))

    def root_to_weight(self, beta):
        """a root (simple-root coordinates) in fundamental-weight coordinates"""
        return tuple(sum(b * self.cartan[j][i] for i, b in enumerate(beta))
                     for j in range(self.rank))

    def orbit(self, levi, limit=None):
        """the orbit W.omega_I with omega_I = sum of the fundamental weights not
        in levi (0-based). Returns a list of (mu, word) sorted by length, where
        word = (i_1, ..., i_k) is a reduced word of the minimal coset
        representative w = s_{i_1} ... s_{i_k} with w(omega_I) = mu.
        >>> [(mu, w) for mu, w in RootSystem("A2").orbit({1})]
        [((1, 0), ()), ((-1, 1), (0,)), ((0, -1), (1, 0))]
        >>> RootSystem("E8").orbit(set(), limit=1000)
        Traceback (most recent call last):
        ...
        ValueError: the orbit has more than 1000 elements
        """
        levi = set(levi)
        start = tuple(0 if i in levi else 1 for i in range(self.rank))
        result, seen, level = [(start, ())], {start}, [(start, ())]
        while level:
            nxt = []
            for mu, word in level:
                for i in range(self.rank):
                    if mu[i] > 0:          # length goes up by one
                        nu = self.reflect_weight(i, mu)
                        if nu not in seen:
                            seen.add(nu)
                            nxt.append((nu, (i,) + word))
                            if limit is not None and len(seen) > limit:
                                raise ValueError("the orbit has more than %d elements" % limit)
            result.extend(nxt)
            level = nxt
        return result

    def coroot_on_weight(self, beta, mu):
        """<beta^vee, mu> for a root beta (simple-root coordinates) and a weight
        mu (fundamental-weight coordinates): beta^vee = sum_k b_k (a_k, a_k)/(beta, beta) a_k^vee
        >>> RootSystem("B2").coroot_on_weight((1, 1), (0, 1))
        1
        """
        norm = self.inner(beta, beta)
        value = sum(b * self._lengths[k] / norm * mu[k] for k, b in enumerate(beta))
        if value.denominator != 1:
            raise ValueError("non-integral pairing")
        return int(value)

    def reflect_weight_by_root(self, beta, mu):
        """s_beta(mu) = mu - <beta^vee, mu> beta, all in fundamental-weight coordinates"""
        c = self.coroot_on_weight(beta, mu)
        if c == 0:
            return tuple(mu)
        root = self.root_to_weight(beta)
        return tuple(m - c * r for m, r in zip(mu, root))

    def apply_word(self, word, beta):
        """w(beta) for w = s_{word[0]} ... s_{word[-1]}"""
        for i in reversed(word):
            beta = self.reflect(i, beta)
        return beta
