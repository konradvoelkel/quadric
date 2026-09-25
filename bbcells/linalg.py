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
    return [list(col) for col in zip(*matrix)]


def mat_vec(matrix, vector):
    """matrix times column vector
    >>> mat_vec([[1, 2], [0, 1]], (3, 4))
    (11, 4)
    """
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def mat_mul(a, b):
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
