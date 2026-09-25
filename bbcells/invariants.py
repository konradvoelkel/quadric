"""
Invariants that depend only on the cell counts c_d (SPEC.md section 3.3),
and the consistency checks every computation should pass.

For a smooth projective X with a BB decomposition into affine cells:
  b_{2d} = c_d,  b_odd = 0,  M(X) = sum_d Z(d)[2d]^{c_d},  h^{d,d} = c_d,
  [X] = sum_d c_d L^d in K_0(Var_k),  |X(F_q)| = sum_d c_d q^d,
  chi^{A^1}(X) = sum_d c_d <-1>^d in GW(k).
"""

from dataclasses import dataclass

from bbcells.algebra import GWClass, IntPoly
from bbcells.core import bb_cells, choose_generic_cocharacter, is_generic


@dataclass(frozen=True)
class Invariants(object):
    """everything computable from the cell counts
    >>> from bbcells.core import FixedPointData
    >>> P2 = FixedPointData(2, 2, ("p0", "p1", "p2"),
    ...     (((1, 0), (0, 1)), ((-1, 0), (-1, 1)), ((0, -1), (1, -1))))
    >>> inv = bb_cells(P2, (1, 2)).invariants()
    >>> print(inv.poincare.format("t"))
    1 + t^2 + t^4
    >>> print(inv.motive_text())
    Z + Z(1)[2] + Z(2)[4]
    >>> print(inv.gw_euler, "|", inv.real_euler_characteristic)
    2<1> + <-1> | 1
    """
    dim: int
    counts: tuple

    @property
    def betti(self):
        """{i: b_i} for the nonzero Betti numbers (all in even degree)"""
        return {2 * d: c for d, c in enumerate(self.counts) if c}

    @property
    def cell_polynomial(self):
        """sum_d c_d x^d; equals [X] in K_0 (x = L) and |X(F_q)| (x = q)"""
        return IntPoly.from_counts(self.counts)

    @property
    def poincare(self):
        """Poincare polynomial in t"""
        return self.cell_polynomial.substitute_power(2)

    @property
    def k0_class(self):
        return self.cell_polynomial

    def point_count(self, q):
        """|X(F_q)|"""
        return self.cell_polynomial(q)

    @property
    def euler_characteristic(self):
        return sum(self.counts)

    @property
    def gw_euler(self):
        """chi^{A^1}(X) = sum_d c_d <-1>^d"""
        result = GWClass()
        for d, c in enumerate(self.counts):
            result = result + c * GWClass.power_of_minus_one(d)
        return result

    @property
    def real_euler_characteristic(self):
        """chi(X(R)) for the split real form: the signature of chi^{A^1}"""
        return self.gw_euler.signature

    @property
    def real_mod2_total_betti(self):
        """dim H^*(X(R); F_2): one real cell per complex cell"""
        return self.euler_characteristic

    @property
    def motive(self):
        """list of (d, multiplicity) meaning Z(d)[2d]^multiplicity"""
        return [(d, c) for d, c in enumerate(self.counts) if c]

    @property
    def hodge(self):
        """{(p, q): h^{p,q}} for the nonzero Hodge numbers"""
        return {(d, d): c for d, c in enumerate(self.counts) if c}

    def motive_text(self):
        parts = []
        for d, c in self.motive:
            term = "Z" if d == 0 else "Z(%d)[%d]" % (d, 2 * d)
            parts.append(term if c == 1 else "%s^%d" % (term, c))
        return " + ".join(parts) or "0"

    def motive_latex(self):
        parts = []
        for d, c in self.motive:
            term = r"\mathbb{Z}" if d == 0 else r"\mathbb{Z}(%d)[%d]" % (d, 2 * d)
            parts.append(term if c == 1 else r"%s^{\oplus %d}" % (term, c))
        return r" \oplus ".join(parts) or "0"

    def hodge_diamond(self):
        """the Hodge diamond as text, h^{n,n} at the top
        >>> print(Invariants(2, (1, 2, 1)).hodge_diamond())
            1
          0   0
        0   2   0
          0   0
            1
        """
        n = self.dim
        lines = []
        for s in range(2 * n, -1, -1):          # s = p + q
            entries = []
            for p in range(max(0, s - n), min(n, s) + 1):
                q = s - p
                entries.append(self.counts[p] if p == q else 0)
            width = len(entries)
            lines.append("  " * (n + 1 - width) + "   ".join(str(e) for e in entries))
        return "\n".join(line.rstrip() for line in lines)


def compute(cells):
    """the Invariants of a CellDecomposition"""
    return Invariants(cells.data.dim, cells.counts)


@dataclass(frozen=True)
class CheckItem(object):
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class CheckReport(object):
    items: tuple

    @property
    def ok(self):
        return all(item.passed for item in self.items)

    def __str__(self):
        return "\n".join("[%s] %s: %s" % ("ok" if i.passed else "FAIL", i.name, i.detail)
                         for i in self.items)


def _alternative_cocharacters(data, lam):
    """a few generic cocharacters different from lam"""
    base = choose_generic_cocharacter(data)
    candidates = [choose_generic_cocharacter(data, reverse=True),
                  tuple(c if i % 2 == 0 else -c for i, c in enumerate(base)),
                  tuple(-c if i % 2 == 0 else c for i, c in enumerate(reversed(base)))]
    seen, result = {tuple(lam), tuple(-c for c in lam)}, []
    for candidate in candidates:
        if candidate not in seen and is_generic(data, candidate):
            seen.add(candidate)
            seen.add(tuple(-c for c in candidate))
            result.append(candidate)
    return result


def check(cells):
    """consistency checks: independence of the cocharacter, Poincare duality,
    connectedness (c_0 = c_n = 1). A failure means the input data is not the
    fixed-point data of a smooth projective connected variety.
    >>> from bbcells.core import FixedPointData
    >>> bad = FixedPointData(1, 2, ("a", "b"), (((1, 0),), ((0, 1),)))
    >>> report = bb_cells(bad, (1, 1)).check()
    >>> report.ok
    False
    >>> print(report)
    [FAIL] cocharacter independence: counts (0, 2) for (1, 1), but (1, 1) for (1, -3)
    [FAIL] Poincare duality: counts (0, 2) are not symmetric
    [FAIL] connected: c_0 = 0, c_n = 2 (expected 1 and 1)
    """
    data, counts = cells.data, cells.counts
    items = []
    alternatives = _alternative_cocharacters(data, cells.cocharacter)
    if not alternatives:
        items.append(CheckItem("cocharacter independence", True,
                               "only one generic cocharacter up to sign (rank 1); skipped"))
    else:
        mismatch = None
        for lam in alternatives:
            other = bb_cells(data, lam).counts
            if other != counts:
                mismatch = (lam, other)
                break
        if mismatch:
            items.append(CheckItem("cocharacter independence", False,
                                   "counts %r for %r, but %r for %r"
                                   % (counts, cells.cocharacter, mismatch[1], mismatch[0])))
        else:
            items.append(CheckItem("cocharacter independence", True,
                                   "same counts for %d other cocharacters" % len(alternatives)))
    symmetric = counts == counts[::-1]
    items.append(CheckItem("Poincare duality", symmetric,
                           "counts %r are %ssymmetric" % (counts, "" if symmetric else "not ")))
    connected = counts[0] == 1 and counts[-1] == 1
    items.append(CheckItem("connected", connected,
                           "c_0 = %d, c_n = %d (expected 1 and 1)" % (counts[0], counts[-1])))
    return CheckReport(tuple(items))
