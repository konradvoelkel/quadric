"""
bbcells: Bialynicki-Birula cell structures, motives and quadratic invariants
of smooth projective varieties with torus actions, computed from
combinatorial data (fans, root data, ...). See SPEC.md and PLAN.md.

    >>> from bbcells import FixedPointData, bb_cells
    >>> P1 = FixedPointData(1, 1, ("0", "oo"), (((1,),), ((-1,),)))
    >>> print(bb_cells(P1).invariants().gw_euler)
    <1> + <-1>
"""

from bbcells.algebra import GWClass, IntPoly
from bbcells.core import (CellDecomposition, FixedPointData, bb_cells,
                          choose_generic_cocharacter, is_generic, pairing)
from bbcells.invariants import CheckReport, Invariants, check, compute

__version__ = "0.1.0.dev0"

__all__ = ["GWClass", "IntPoly", "CellDecomposition", "FixedPointData", "bb_cells",
           "choose_generic_cocharacter", "is_generic", "pairing", "CheckReport",
           "Invariants", "check", "compute"]
