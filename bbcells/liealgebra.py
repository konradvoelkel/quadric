"""
Exact matrix realizations of the classical Lie algebras (PLAN.md S5.4), used
to orient real Schubert cells.

Types A_{n-1} (sl_n), B_n (so_{2n+1}), C_n (sp_{2n}) and D_n (so_{2n}), the
latter three for the antidiagonal forms, so that the diagonal torus has
weight eps_i at position i and -eps_i at position N + 1 - i. The Chevalley
generators E_i, F_i of the simple roots (Bourbaki numbering) are normalized
by [[E, F], E] = 2E, and n_i = exp(E_i) exp(-F_i) exp(E_i) = exp(pi/2 (E_i - F_i))
is the Tits lift of s_i; these lifts satisfy the braid relations.
"""

from fractions import Fraction


def _zeros(n):
    return [[Fraction(0)] * n for _ in range(n)]


def _unit(n, i, j):
    m = _zeros(n)
    m[i][j] = Fraction(1)
    return m


def mat_mul(a, b):
    n = len(a)
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def mat_add(a, b, scale=1):
    return [[x + scale * y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def mat_scale(a, c):
    return [[c * x for x in row] for row in a]


def bracket(a, b):
    return mat_add(mat_mul(a, b), mat_mul(b, a), -1)


def identity(n):
    return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]


def exp_nilpotent(x):
    """exp of a nilpotent matrix (finite sum)"""
    n = len(x)
    result, term = identity(n), identity(n)
    for k in range(1, n + 1):
        term = mat_scale(mat_mul(term, x), Fraction(1, k))
        if all(v == 0 for row in term for v in row):
            break
        result = mat_add(result, term)
    return result


def transpose(a):
    return [list(col) for col in zip(*a)]


class ClassicalRealization(object):
    """
    >>> R = ClassicalRealization("C", 2)
    >>> R.check_braid_relations()
    True
    """

    def __init__(self, letter, n):
        self.letter, self.n = letter, n
        if letter == "A":
            N = n + 1
            self.form = None
        elif letter in "BCD":
            N = {"B": 2 * n + 1, "C": 2 * n, "D": 2 * n}[letter]
            J = _zeros(N)
            for i in range(N):
                J[i][N - 1 - i] = Fraction(-1 if letter == "C" and i >= n else 1)
            self.form = J
        else:
            raise ValueError("classical types only")
        self.N = N
        self.E, self.F = [], []
        for i in range(n):
            e = self._project(_unit(N, *self._simple_position(i)))
            f = self._project(transpose(e))
            h = bracket(e, f)
            he = bracket(h, e)
            ratio = self._ratio(he, e)
            f = mat_scale(f, Fraction(2) / ratio)
            self.E.append(e)
            self.F.append(f)
        self.lift = [self._tits_lift(i) for i in range(n)]
        self.lift_inverse = [self._tits_lift(i, inverse=True) for i in range(n)]

    def _simple_position(self, i):
        n = self.n
        if self.letter == "A" or i < n - 1:
            return (i, i + 1)
        return {"B": (n - 1, n), "C": (n - 1, n), "D": (n - 2, n)}[self.letter]

    def _project(self, x):
        """projection onto the Lie algebra (sl_n: identity on off-diagonal units)"""
        if self.form is None:
            return x
        J = self.form
        Jinv = _inverse_permutation_matrix(J)
        corrected = mat_mul(Jinv, mat_mul(transpose(x), J))
        return mat_scale(mat_add(x, corrected, -1), Fraction(1, 2))

    @staticmethod
    def _ratio(a, b):
        for ra, rb in zip(a, b):
            for x, y in zip(ra, rb):
                if y != 0:
                    return x / y
        raise ValueError("zero matrix")

    def _tits_lift(self, i, inverse=False):
        e, f = self.E[i], self.F[i]
        if inverse:
            return mat_mul(exp_nilpotent(mat_scale(e, -1)),
                           mat_mul(exp_nilpotent(f), exp_nilpotent(mat_scale(e, -1))))
        return mat_mul(exp_nilpotent(e), mat_mul(exp_nilpotent(mat_scale(f, -1)),
                                                 exp_nilpotent(e)))

    def lift_of_word(self, word):
        m, minv = identity(self.N), identity(self.N)
        for i in word:
            m = mat_mul(m, self.lift[i])
            minv = mat_mul(self.lift_inverse[i], minv)
        return m, minv

    def check_braid_relations(self):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                a, b = self.lift[i], self.lift[j]
                ab = mat_mul(a, b)
                ba = mat_mul(b, a)
                # order of s_i s_j from the Cartan data of the realization
                for m in (2, 3, 4, 6):
                    left = identity(self.N)
                    right = identity(self.N)
                    for k in range(m):
                        left = mat_mul(left, a if k % 2 == 0 else b)
                        right = mat_mul(right, b if k % 2 == 0 else a)
                    if left == right:
                        break
                else:
                    return False
        return True

    def weight_of_position(self, k):
        """epsilon-coordinates of the torus weight of basis vector k"""
        n = self.n
        v = [0] * (n + 1 if self.letter == "A" else n)
        if self.letter == "A":
            v[k] = 1
            return tuple(v)
        if k < n:
            v[k] = 1
        elif self.letter == "B" and k == n:
            pass
        else:
            v[self.N - 1 - k] = -1
        return tuple(v)

    def root_of(self, x):
        """the weight of a root vector (difference of basis weights), or None"""
        roots = set()
        for i, row in enumerate(x):
            for j, v in enumerate(row):
                if v != 0:
                    a, b = self.weight_of_position(i), self.weight_of_position(j)
                    roots.add(tuple(p - q for p, q in zip(a, b)))
        if len(roots) != 1:
            raise ValueError("not a root vector: weights %r" % sorted(roots))
        return roots.pop()

    def frame(self, word):
        """the tangent frame at n_word b_0 of the parametrization by the word:
        [(root, vector)] with vector = Ad(n_{r_1} ... n_{r_{j-1}}) E_{r_j}"""
        result = []
        m, minv = identity(self.N), identity(self.N)
        for i in word:
            v = mat_mul(m, mat_mul(self.E[i], minv))
            result.append((self.root_of(v), v))
            m = mat_mul(m, self.lift[i])
            minv = mat_mul(self.lift_inverse[i], minv)
        return result

    def relative_orientation(self, word1, word2):
        """sign of the frame of word1 relative to the frame of word2; both words
        must be reduced words of the same Weyl group element"""
        f1, f2 = self.frame(word1), self.frame(word2)
        roots2 = [r for r, _ in f2]
        if sorted(r for r, _ in f1) != sorted(roots2):
            raise ValueError("the words do not have the same inversion set")
        permutation = [roots2.index(r) for r, _ in f1]
        sign = _permutation_sign(permutation)
        for (r, v1) in f1:
            v2 = f2[roots2.index(r)][1]
            c = self._ratio(v1, v2)
            if c not in (1, -1):
                raise AssertionError("root vectors differ by %r" % c)
            sign *= int(c)
        return sign


def _inverse_permutation_matrix(J):
    """inverse of a signed permutation matrix"""
    n = len(J)
    inv = _zeros(n)
    for i in range(n):
        for j in range(n):
            if J[i][j] != 0:
                inv[j][i] = 1 / J[i][j]
    return inv


def _permutation_sign(p):
    sign, seen = 1, [False] * len(p)
    for i in range(len(p)):
        if not seen[i]:
            j, length = i, 0
            while not seen[j]:
                seen[j] = True
                j = p[j]
                length += 1
            if length % 2 == 0:
                sign = -sign
    return sign
