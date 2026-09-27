"""
Chow-Witt groups of cellular varieties over R, without twist (PLAN R6).

For a smooth cellular X over R the Chow-Witt group is the fibre product

    CH~^q(X) = CH^q(X) x_{Ch^q(X)} H^q(X(R); Z)

(Hornbostel-Wendt, "Chow-Witt rings of classifying spaces...", Prop. 2.11,
as stated in Hudson-Matszangosz-Wendt, arXiv:2302.11003, section 2: the
Chow ring has no 2-torsion; and for cellular X the real cycle class map of
Hornbostel-Wendt-Xie-Zibrowius, arXiv:1911.04150, identifies I-cohomology
with H^*(X(R); Z) and its reduction with the Borel-Haefliger map
Ch^q(X) = H^q(X(R); Z/2)). Both sides have bases indexed by the cells,
the cellular cochain complex of X(R) has even coboundaries, and the
isomorphism type of the fibre product does not depend on how the two bases
correspond modulo 2 (any matrix in GL(F_2) lifts to GL(Z)). So

    CH~^q = (Z^q_cocycles + Z^{c_q}) / {(delta w, -delta w / 2)},

computed here by Smith normal form. docs/real.md, section 6.
"""

from fractions import Fraction

from bbcells.linalg import smith_invariants, solve


def integer_kernel(matrix, ncols):
    """a Z-basis of {x in Z^ncols : A x = 0}, by column operations
    >>> integer_kernel([[2, 4, 6]], 3)
    [[-2, 1, 0], [-3, 0, 1]]
    """
    a = [list(row) for row in matrix]
    u = [[int(i == j) for j in range(ncols)] for i in range(ncols)]   # columns of U
    rows = len(a)
    pivot_col = 0
    for r in range(rows):
        if pivot_col >= ncols:
            break
        while True:
            nonzero = [j for j in range(pivot_col, ncols) if a[r][j]]
            if not nonzero:
                break
            j = min(nonzero, key=lambda j: abs(a[r][j]))
            for row in a:                               # swap columns j and pivot_col
                row[j], row[pivot_col] = row[pivot_col], row[j]
            u[j], u[pivot_col] = u[pivot_col], u[j]
            p = a[r][pivot_col]
            done = True
            for k in range(pivot_col + 1, ncols):
                q = a[r][k] // p
                if q:
                    for row in a:
                        row[k] -= q * row[pivot_col]
                    u[k] = [x - q * y for x, y in zip(u[k], u[pivot_col])]
                if a[r][k]:
                    done = False
            if done:
                pivot_col += 1
                break
    return [list(u[k]) for k in range(pivot_col, ncols)]


def chow_witt_groups(dims, coboundaries):
    """[(free rank, [torsion orders])] of CH~^q for q = 0..n, from the ranks
    dims[q] of the cellular cochain groups C^q of X(R) and the integer
    coboundary matrices coboundaries[q]: C^q -> C^{q+1} (rows index C^{q+1}),
    whose entries must be even. RP^2 (P^2 over R): GW(R) = Z^2, Z, Z:
    >>> chow_witt_groups([1, 1, 1], [[[0]], [[2]]])
    [(2, []), (1, []), (1, [])]
    """
    n = len(dims) - 1
    result = []
    for q in range(n + 1):
        c = dims[q]
        out = coboundaries[q] if q < n else []
        if any(x % 2 for row in out for x in row) or \
                (q > 0 and any(x % 2 for row in coboundaries[q - 1] for x in row)):
            raise ValueError("the coboundaries must be even (cellular X over R)")
        cocycles = integer_kernel(out, c) if out and out[0] else \
            [[int(i == j) for j in range(c)] for i in range(c)]
        k = len(cocycles)
        relations = []
        if q > 0 and dims[q - 1]:
            incoming = coboundaries[q - 1]
            transposed = [list(col) for col in zip(*cocycles)] if cocycles else []
            for w in range(dims[q - 1]):
                image = [incoming[i][w] for i in range(c)]     # delta of the w-th cell
                if not any(image):
                    continue
                coordinates = solve(transposed, image)
                if coordinates is None or any(Fraction(x).denominator != 1 for x in coordinates):
                    raise AssertionError("a coboundary is not an integral cocycle")
                relations.append([int(x) for x in coordinates] + [-x // 2 for x in image])
        invariants = smith_invariants(relations) if relations else []
        free = k + c - len(invariants)
        result.append((free, [d for d in invariants if d > 1]))
    return result


def cochain_complex(complex_):
    """(dims, coboundaries) of the integral cellular cochain complex of a
    realcells.RealCellComplex (cooriented or cellular) or a
    realtoric.RealToricComplex, graded by cohomological degree
    >>> from bbcells import realcells
    >>> cochain_complex(realcells.real_flag_variety("A2", {1}))
    ([1, 1, 1], [[[0]], [[-2]]])
    """
    if hasattr(complex_, "convention") and complex_.convention == "cooriented":
        n = complex_.dim
        dims = [len(complex_.cells_of_codimension(c)) for c in range(n + 1)]
        return dims, [complex_.coboundary_matrix(c) for c in range(n)]
    if hasattr(complex_, "convention"):
        cell_dims, incidences = complex_.dims, complex_.incidences
    else:                                           # a RealToricComplex
        cell_dims, incidences = complex_.dims(), complex_.incidences()
    n = max(cell_dims.values())
    cells = {d: sorted((p for p, e in cell_dims.items() if e == d), key=str)
             for d in range(n + 1)}
    dims = [len(cells[d]) for d in range(n + 1)]
    # the coboundary C^d -> C^{d+1} is the transpose of the boundary C_{d+1} -> C_d
    coboundaries = [[[incidences.get((x, y), 0) for y in cells[d]] for x in cells[d + 1]]
                    for d in range(n)]
    return dims, coboundaries


def chow_witt(complex_):
    """CH~^q(X) for q = 0..dim X, as [(free rank, [torsion orders])], from the
    real cell complex of a cellular X. Real projective 3-space and the flag
    variety of R^3:
    >>> from bbcells import realcells
    >>> chow_witt(realcells.real_flag_variety("A3", {1}))
    [(2, []), (1, []), (1, []), (2, [])]
    >>> chow_witt(realcells.real_flag_variety("A2"))
    [(2, []), (2, []), (2, []), (2, [])]
    """
    return chow_witt_groups(*cochain_complex(complex_))
