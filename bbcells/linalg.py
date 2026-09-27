"""
Exact linear algebra over Z and Q for small matrices (lists of rows).
"""

from fractions import Fraction


def determinant(matrix):
    """exact determinant by fraction-free Gaussian elimination (Bareiss)
    >>> determinant([[2, 1], [1, 1]]), determinant([[1, 2], [2, 4]])
    (1, 0)
    >>> determinant([])
    1
    """
    a = [list(row) for row in matrix]
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("determinant of a non-square matrix")
    sign, previous = 1, 1
    for k in range(n - 1):
        if a[k][k] == 0:
            swap = next((i for i in range(k + 1, n) if a[i][k] != 0), None)
            if swap is None:
                return 0
            a[k], a[swap] = a[swap], a[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) // previous
        previous = a[k][k]
    return sign * a[n - 1][n - 1] if n else 1


def inverse(matrix):
    """exact inverse over Q; raises ValueError if singular
    >>> inverse([[2, 1], [1, 1]])
    [[Fraction(1, 1), Fraction(-1, 1)], [Fraction(-1, 1), Fraction(2, 1)]]
    """
    n = len(matrix)
    a = [[Fraction(x) for x in row] + [Fraction(int(i == j)) for j in range(n)]
         for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col] != 0), None)
        if pivot is None:
            raise ValueError("matrix is singular")
        a[col], a[pivot] = a[pivot], a[col]
        p = a[col][col]
        a[col] = [x / p for x in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0:
                factor = a[r][col]
                a[r] = [x - factor * y for x, y in zip(a[r], a[col])]
    return [row[n:] for row in a]


def integer_inverse(matrix):
    """inverse of a unimodular integer matrix, as integers
    >>> integer_inverse([[2, 1], [1, 1]])
    [[1, -1], [-1, 2]]
    >>> integer_inverse([[2, 0], [0, 1]])
    Traceback (most recent call last):
    ...
    ValueError: matrix is not unimodular (determinant 2)
    """
    d = determinant(matrix)
    if abs(d) != 1:
        raise ValueError("matrix is not unimodular (determinant %d)" % d)
    return [[int(x) for x in row] for row in inverse(matrix)]


def transpose(matrix):
    """the transpose of a matrix (a list of rows)
    >>> transpose([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]
    """
    return [list(col) for col in zip(*matrix)]


def mat_vec(matrix, vector):
    """matrix times column vector
    >>> mat_vec([[1, 2], [0, 1]], (3, 4))
    (11, 4)
    """
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def mat_mul(a, b):
    """the matrix product a b; here the braid relation s1 s2 s1 = s2 s1 s2 of
    the simple reflections of A_2 on the simple-root coordinates
    >>> s1, s2 = [[-1, 1], [0, 1]], [[1, 0], [1, -1]]
    >>> mat_mul(s1, mat_mul(s2, s1)) == mat_mul(s2, mat_mul(s1, s2))
    True
    """
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def rank(matrix):
    """rank over Q
    >>> rank([[1, 2], [2, 4]]), rank([[1, 0], [0, 1]]), rank([])
    (1, 2, 0)
    """
    a = [[Fraction(x) for x in row] for row in matrix]
    r = 0
    cols = len(a[0]) if a else 0
    for col in range(cols):
        pivot = next((i for i in range(r, len(a)) if a[i][col] != 0), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        for i in range(len(a)):
            if i != r and a[i][col] != 0:
                factor = a[i][col] / a[r][col]
                a[i] = [x - factor * y for x, y in zip(a[i], a[r])]
        r += 1
    return r


def gcd_of(vector):
    """
    >>> gcd_of((4, -6, 0))
    2
    """
    from math import gcd
    g = 0
    for x in vector:
        g = gcd(g, x)
    return g


def smith_invariants(matrix):
    """the nonzero invariant factors d_1 | d_2 | ... of an integer matrix
    (Smith normal form diagonal), by row and column operations over Z
    >>> smith_invariants([[2, 4, 4], [-6, 6, 12], [10, -4, -16]])
    [2, 6, 12]
    >>> smith_invariants([[0, 0], [0, 0]]), smith_invariants([])
    ([], [])
    """
    a = [list(row) for row in matrix]
    rows = len(a)
    cols = len(a[0]) if rows else 0
    invariants = []
    t = 0
    while t < min(rows, cols):
        # choose a pivot of smallest absolute value in the remaining block
        entries = [(abs(a[i][j]), i, j) for i in range(t, rows) for j in range(t, cols) if a[i][j]]
        if not entries:
            break
        _, i, j = min(entries)
        a[t], a[i] = a[i], a[t]
        for row in a:
            row[t], row[j] = row[j], row[t]
        done = False
        while not done:
            done = True
            p = a[t][t]
            for i in range(t + 1, rows):          # clear the column
                if a[i][t]:
                    q = a[i][t] // p
                    a[i] = [x - q * y for x, y in zip(a[i], a[t])]
                    if a[i][t]:
                        a[t], a[i] = a[i], a[t]
                        done = False
                        break
            if not done:
                continue
            p = a[t][t]
            for j in range(t + 1, cols):          # clear the row
                if a[t][j]:
                    q = a[t][j] // p
                    for row in a:
                        row[j] -= q * row[t]
                    if a[t][j]:
                        for row in a:
                            row[t], row[j] = row[j], row[t]
                        done = False
                        break
            if not done:
                continue
            p = a[t][t]                            # divisibility of the rest
            for i in range(t + 1, rows):
                for j in range(t + 1, cols):
                    if a[i][j] % p:
                        a[t] = [x + y for x, y in zip(a[t], a[i])]
                        done = False
                        break
                if not done:
                    break
        invariants.append(abs(a[t][t]))
        t += 1
    return invariants


def null_space(matrix, ncols=None):
    """a basis of {x : A x = 0} over Q (list of Fraction vectors)
    >>> null_space([[1, 1, 0], [0, 0, 1]])
    [[Fraction(-1, 1), Fraction(1, 1), Fraction(0, 1)]]
    """
    return null_space_and_free(matrix, ncols)[0]


def null_space_and_free(matrix, ncols=None):
    """(basis, free): the basis of null_space, whose k-th vector is 1 at the
    column free[k] and 0 at the other free columns, so that the coordinates
    of a vector x of the null space are x[free[0]], x[free[1]], ...
    >>> null_space_and_free([[1, 1, 0], [0, 0, 1]])[1]
    [1]
    """
    ncols = len(matrix[0]) if matrix else (ncols or 0)
    a = [[Fraction(x) for x in row] for row in matrix]
    pivots, r = [], 0
    for col in range(ncols):
        pivot = next((i for i in range(r, len(a)) if a[i][col] != 0), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        p = a[r][col]
        a[r] = [x / p for x in a[r]]
        for i in range(len(a)):
            if i != r and a[i][col] != 0:
                factor = a[i][col]
                a[i] = [x - factor * y for x, y in zip(a[i], a[r])]
        pivots.append(col)
        r += 1
    free = [c for c in range(ncols) if c not in pivots]
    basis = []
    for f in free:
        v = [Fraction(0)] * ncols
        v[f] = Fraction(1)
        for row, pc in zip(a, pivots):
            v[pc] = -row[f]
        basis.append(v)
    return basis, free


def solve(matrix, rhs):
    """one solution x of A x = b over Q, or None
    >>> solve([[1, 1], [1, -1]], [2, 0])
    [Fraction(1, 1), Fraction(1, 1)]
    >>> solve([[1, 1], [2, 2]], [1, 3]) is None
    True
    """
    ncols = len(matrix[0]) if matrix else 0
    a = [[Fraction(x) for x in row] + [Fraction(b)] for row, b in zip(matrix, rhs)]
    pivots, r = [], 0
    for col in range(ncols):
        pivot = next((i for i in range(r, len(a)) if a[i][col] != 0), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        p = a[r][col]
        a[r] = [x / p for x in a[r]]
        for i in range(len(a)):
            if i != r and a[i][col] != 0:
                factor = a[i][col]
                a[i] = [x - factor * y for x, y in zip(a[i], a[r])]
        pivots.append(col)
        r += 1
    if any(all(x == 0 for x in row[:-1]) and row[-1] != 0 for row in a):
        return None
    x = [Fraction(0)] * ncols
    for row, pc in zip(a, pivots):
        x[pc] = row[-1]
    return x


def _is_prime(n):
    """deterministic Miller-Rabin for n < 3.3e24"""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for q in small:
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _primes(start=(1 << 61) - 1):
    """the primes below start, descending"""
    n = start
    while True:
        if _is_prime(n):
            yield n
        n -= 1


def _rref_mod(rows, ncols, p):
    """(pivot columns, reduced rows) of an integer matrix modulo the prime p"""
    a = [[x % p for x in row] for row in rows]
    pivots, r = [], 0
    for col in range(ncols):
        pivot = next((i for i in range(r, len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        inverse = pow(a[r][col], p - 2, p)
        a[r] = [x * inverse % p for x in a[r]]
        pivot_row = a[r]
        for i in range(len(a)):
            f = a[i][col]
            if i != r and f:
                a[i] = [(x - f * y) % p for x, y in zip(a[i], pivot_row)]
        pivots.append(col)
        r += 1
    return pivots, a[:r]


def _rational_reconstruction(a, m):
    """the fraction r/s with r = a s mod m and |r|, s <= sqrt(m/2), or None"""
    from math import isqrt
    bound = isqrt(m // 2)
    r0, r1, s0, s1 = m, a % m, 0, 1
    while r1 > bound:
        q = r0 // r1
        r0, r1, s0, s1 = r1, r0 - q * r1, s1, s0 - q * s1
    if s1 == 0 or abs(s1) > bound:
        return None
    return Fraction(r1, s1)


def certified_rank(matrix, max_primes=400):
    """rank over Q of a matrix of rationals, by elimination modulo primes:
    the rank r modulo a prime is a lower bound, and a right kernel of
    dimension ncols - r, lifted by Chinese remaindering and rational
    reconstruction and checked over Q, proves that r is also an upper bound
    (falls back to exact elimination if the lift does not converge)
    >>> certified_rank([[1, 2], [2, 4]]), certified_rank([[1, 0], [0, 1]])
    (1, 2)
    >>> certified_rank([[Fraction(1, 3), 1, 0], [0, 1, Fraction(5, 7)], [1, 6, Fraction(15, 7)]])
    2
    """
    from math import lcm
    rows = []
    for row in matrix:
        row = [Fraction(x) for x in row]
        scale = lcm(*[x.denominator for x in row]) if row else 1
        if any(row):
            rows.append([int(x * scale) for x in row])
    if not rows:
        return 0
    ncols = len(rows[0])
    primes = _primes()
    reference, residues, modulus = None, None, 1
    for _ in range(max_primes):
        p = next(primes)
        pivots, reduced = _rref_mod(rows, ncols, p)
        if reference is None or len(pivots) > len(reference):
            reference, residues, modulus = pivots, None, 1   # a better prime
        elif pivots != reference:
            continue                                         # an unlucky prime
        if len(pivots) == ncols:
            return ncols
        free = [c for c in range(ncols) if c not in pivots]
        kernel = []
        for f in free:
            v = [0] * ncols
            v[f] = 1
            for row, pc in zip(reduced, pivots):
                v[pc] = -row[f] % p
            kernel.append(v)
        if residues is None:
            residues, modulus = kernel, p
        else:
            # Chinese remaindering: x = a mod modulus, x = b mod p
            inverse = pow(modulus, -1, p)
            residues = [[a + modulus * ((b - a) * inverse % p) for a, b in zip(u, v)]
                        for u, v in zip(residues, kernel)]
            modulus *= p
        lifted = []
        for v in residues:
            w = [_rational_reconstruction(x, modulus) for x in v]
            if any(x is None for x in w):
                break
            lifted.append(w)
        else:
            if all(sum(a * b for a, b in zip(row, w)) == 0 for row in rows for w in lifted):
                return len(reference)
    return rank(matrix)
