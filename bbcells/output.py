"""
Renderers for a CellDecomposition: text, LaTeX and JSON (PLAN.md S1.7).
"""

import json

from bbcells.core import pairing


def _summary(cells):
    data = cells.data
    return "%s (dimension %d, torus rank %d, %d fixed points)" % (
        data.name or "X", data.dim, data.rank, len(data))


def to_text(cells, show_weights=False, checks=True):
    """plain text report
    >>> from bbcells.core import FixedPointData, bb_cells
    >>> P1 = FixedPointData(1, 1, ("0", "oo"), (((1,),), ((-1,),)), name="P^1")
    >>> print(to_text(bb_cells(P1)))
    P^1 (dimension 1, torus rank 1, 2 fixed points)
    cocharacter: (1,)
    cells (plus-cells, dimension: fixed point):
      0: oo
      1: 0
    Betti numbers: b0 = 1, b2 = 1
    Poincare polynomial: 1 + t^2
    class in K_0(Var): 1 + L
    Chow motive: Z + Z(1)[2]
    chi^A1 in GW: <1> + <-1>  (rank 2, signature 0 = chi(X(R)))
    Hodge diamond:
        1
      0   0
        1
    checks:
      [ok] cocharacter independence: only one generic cocharacter up to sign (rank 1); skipped
      [ok] holomorphic Lefschetz: sum_p prod (1 + y e^-w)/(1 - e^-w) = chi_y(X)
      [ok] Poincare duality: counts (1, 1) are symmetric
      [ok] connected: c_0 = 1, c_n = 1 (expected 1 and 1)
    """
    inv = cells.invariants()
    lines = [_summary(cells), "cocharacter: %r" % (tuple(cells.cocharacter),),
             "cells (plus-cells, dimension: fixed point):"]
    for d, label in cells.cells():
        line = "  %d: %s" % (d, label)
        if show_weights:
            wts = cells.data.weights_of(label)
            line += "   weights " + ", ".join(
                "%r%s" % (w, "+" if pairing(cells.cocharacter, w) > 0 else "-") for w in wts)
        lines.append(line)
    lines.append("Betti numbers: " + ", ".join("b%d = %d" % (i, b)
                                               for i, b in sorted(inv.betti.items())))
    lines.append("Poincare polynomial: " + inv.poincare.format("t"))
    lines.append("class in K_0(Var): " + inv.k0_class.format("L"))
    lines.append("Chow motive: " + inv.motive_text())
    gw = inv.gw_euler
    lines.append("chi^A1 in GW: %s  (rank %d, signature %d = chi(X(R)))"
                 % (gw, gw.rank, gw.signature))
    lines.append("Hodge diamond:")
    lines.extend("  " + line for line in inv.hodge_diamond().splitlines())
    if checks:
        lines.append("checks:")
        lines.extend("  " + line for line in str(cells.check()).splitlines())
    return "\n".join(lines)


def _latex_label(label):
    return str(label).replace("_", r"\_").replace("|", r"\mid ")


def to_latex(cells):
    r"""LaTeX fragment in the style of quadric.py's output
    >>> from bbcells.core import FixedPointData, bb_cells
    >>> P1 = FixedPointData(1, 1, ("0", "oo"), (((1,),), ((-1,),)), name="P^1")
    >>> print(to_latex(bb_cells(P1)).splitlines()[0])
    \subsection*{$P^1$, cocharacter $(1)$}
    """
    inv = cells.invariants()
    lam = ", ".join(str(c) for c in cells.cocharacter)
    lines = [r"\subsection*{$%s$, cocharacter $(%s)$}" % (cells.data.name or "X", lam),
             r"\begin{itemize}"]
    for d, label in cells.cells():
        lines.append(r"  \item[%d:] $\mathrm{%s}$" % (d, _latex_label(label)))
    lines.append(r"\end{itemize}")
    lines.append(r"\[ P_X(t) = %s, \qquad [X] = %s \in K_0(\mathrm{Var}_k), \]"
                 % (inv.poincare.latex("t"), inv.k0_class.latex(r"\mathbb{L}")))
    lines.append(r"\[ M(X) \cong %s, \qquad \chi^{\mathbb{A}^1}(X) = %s. \]"
                 % (inv.motive_latex(), inv.gw_euler.latex()))
    return "\n".join(lines)


def to_json_dict(cells, checks=True):
    inv = cells.invariants()
    document = {
        "name": cells.data.name,
        "dim": cells.data.dim,
        "rank": cells.data.rank,
        "cocharacter": list(cells.cocharacter),
        "cells": [{"label": str(label), "dim": d} for d, label in cells.cells()],
        "counts": list(cells.counts),
        "poincare_t": list(inv.poincare.coefficients),
        "k0_class_L": list(inv.k0_class.coefficients),
        "gw_euler": {"plus": inv.gw_euler.plus, "minus": inv.gw_euler.minus,
                     "rank": inv.gw_euler.rank, "signature": inv.gw_euler.signature},
        "motive": [{"twist": d, "multiplicity": c} for d, c in inv.motive],
    }
    if checks:
        report = cells.check()
        document["checks"] = [{"name": i.name, "passed": i.passed, "detail": i.detail}
                              for i in report.items]
    return document


def to_json(cells, checks=True, indent=1):
    return json.dumps(to_json_dict(cells, checks), indent=indent)


def render(cells, fmt="text", show_weights=False, checks=True):
    if fmt == "text":
        return to_text(cells, show_weights=show_weights, checks=checks)
    if fmt == "latex":
        return to_latex(cells)
    if fmt == "json":
        return to_json(cells, checks=checks)
    raise ValueError("unknown format %r" % fmt)
