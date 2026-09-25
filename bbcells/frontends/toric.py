"""
Smooth complete toric varieties from their fans (PLAN.md S1.5).

Conventions (PLAN.md 1.1): for a maximal cone sigma with ray generators
u_1, ..., u_n (a Z-basis of N), the tangent weights at the fixed point
x_sigma are the dual basis m_1, ..., m_n of M, <m_i, u_j> = delta_ij. The
plus-cell of x_sigma for lambda in N has dimension #{i : <lambda, m_i> > 0};
the open cell sits at the cone containing lambda in its interior.

A fan is given by its rays (primitive integer vectors) and its maximal cones
(tuples of ray indices). JSON: {"name": ..., "rays": [[...]], "cones": [[...]]}.
"""

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from bbcells.core import FixedPointData, bb_cells, choose_generic_cocharacter
from bbcells.linalg import determinant, gcd_of, integer_inverse, transpose
from bbcells.schema import load_schema, validate


@dataclass(frozen=True)
class Fan(object):
    """a smooth complete fan, validated on construction
    >>> P2 = projective_space(2)
    >>> P2.dim, len(P2.rays), P2.cones
    (2, 3, ((0, 1), (0, 2), (1, 2)))
    >>> Fan(((1, 0), (0, 1), (-1, -1)), ((0, 1), (1, 2)))
    Traceback (most recent call last):
    ...
    ValueError: fan is not complete: the wall (0,) lies in only 1 maximal cone
    """
    rays: tuple
    cones: tuple
    name: str = ""

    def __post_init__(self):
        rays = tuple(tuple(r) for r in self.rays)
        if not rays:
            raise ValueError("a fan needs at least one ray")
        n = len(rays[0])
        for i, r in enumerate(rays):
            if len(r) != n or not all(isinstance(c, int) for c in r):
                raise ValueError("ray %d: expected an integer vector of length %d" % (i, n))
            if gcd_of(r) != 1:
                raise ValueError("ray %d = %r is not primitive" % (i, r))
        if len(set(rays)) != len(rays):
            raise ValueError("duplicate rays")
        cones = tuple(sorted(tuple(sorted(c)) for c in self.cones))
        if len(set(cones)) != len(cones):
            raise ValueError("duplicate cones")
        for c in cones:
            if len(c) != n or not all(0 <= i < len(rays) for i in c):
                raise ValueError("cone %r: a maximal cone needs %d valid ray indices" % (c, n))
            d = determinant([list(rays[i]) for i in c])
            if abs(d) != 1:
                raise ValueError("cone %r is not smooth (determinant %d)" % (c, d))
        object.__setattr__(self, "rays", rays)
        object.__setattr__(self, "cones", cones)
        self._check_complete()

    @property
    def dim(self):
        return len(self.rays[0])

    def dual_basis(self, cone):
        """the dual basis m_1..m_n of the cone's generators, in the cone's order"""
        # columns u_i; rows of the inverse are the dual basis
        matrix = transpose([list(self.rays[i]) for i in cone])
        return [tuple(row) for row in integer_inverse(matrix)]

    def _walls(self):
        walls = {}
        for c in self.cones:
            for k in range(len(c)):
                walls.setdefault(c[:k] + c[k + 1:], []).append((c, c[k], k))
        return walls

    def _check_complete(self):
        """walls lie in exactly two cones, on opposite sides, and a generic
        vector lies in exactly one cone (checked for two vectors)"""
        for wall, cones in self._walls().items():
            if len(cones) != 2:
                raise ValueError("fan is not complete: the wall %r lies in %s maximal cone%s"
                                 % (wall, "only 1" if len(cones) == 1 else len(cones),
                                    "" if len(cones) == 1 else "s"))
            (c1, _, k1), (_, extra2, _) = cones
            m = self.dual_basis(c1)[k1]
            if sum(a * b for a, b in zip(m, self.rays[extra2])) >= 0:
                raise ValueError("cones %r and %r lie on the same side of their wall %r"
                                 % (cones[0][0], cones[1][0], wall))
        data = _fixed_point_data(self, with_edges=False)
        for reverse in (False, True):
            lam = choose_generic_cocharacter(data, reverse=reverse)
            covering = bb_cells(data, lam).counts[-1]
            if covering != 1:
                raise ValueError("fan is not a complete fan: a generic vector %r lies in %d "
                                 "maximal cones" % (lam, covering))


def _label(cone):
    return "s" + ",".join(str(i) for i in cone)


def _fixed_point_data(fan, with_edges=True):
    points, weights = [], []
    for c in fan.cones:
        points.append(_label(c))
        weights.append(tuple(fan.dual_basis(c)))
    edges = None
    if with_edges:
        edges = []
        for wall, cones in fan._walls().items():
            (c1, _, k1), (c2, _, _) = cones
            edges.append((_label(c1), _label(c2), fan.dual_basis(c1)[k1]))
    annotations = tuple({"cone": c} for c in fan.cones)
    return FixedPointData(fan.dim, fan.dim, tuple(points), tuple(weights),
                          edges=None if edges is None else tuple(edges),
                          name=fan.name, annotations=annotations)


def fixed_point_data(fan):
    """FixedPointData with GKM edges (one per wall); labels 's<ray indices>'
    >>> data = fixed_point_data(projective_space(1))
    >>> data.points, data.weights, data.edges
    (('s0', 's1'), (((1,),), ((-1,),)), (('s0', 's1', (1,)),))
    """
    return _fixed_point_data(fan)


def orbit_closure(fan, face):
    """FixedPointData of the orbit closure V(tau) for the cone tau spanned by
    the given rays, with the torus of the ambient variety: fixed points are
    the maximal cones containing tau (same labels as fixed_point_data), and
    the tangent weights are the dual basis elements of the rays not in tau.
    >>> line = orbit_closure(projective_space(3), (0, 1))
    >>> line.dim, line.points
    (1, ('s0,1,2', 's0,1,3'))
    """
    face = set(face)
    points, weights = [], []
    for c in fan.cones:
        if face <= set(c):
            dual = fan.dual_basis(c)
            points.append(_label(c))
            weights.append(tuple(m for i, m in zip(c, dual) if i not in face))
    if not points:
        raise ValueError("%r is not a face of the fan" % (sorted(face),))
    return FixedPointData(fan.dim - len(face), fan.dim, tuple(points), tuple(weights),
                          name="V(%s)" % ",".join(str(i) for i in sorted(face)))


# -- constructions ---------------------------------------------------------

def projective_space(n):
    """P^n: rays e_1, ..., e_n, -(e_1 + ... + e_n)"""
    rays = [tuple(1 if k == i else 0 for k in range(n)) for i in range(n)]
    rays.append(tuple(-1 for _ in range(n)))
    return Fan(tuple(rays), tuple(itertools.combinations(range(n + 1), n)),
               name="P^%d" % n)


def hirzebruch(a):
    """the Hirzebruch surface F_a = P(O + O(a)) over P^1"""
    return Fan(((1, 0), (0, 1), (-1, a), (0, -1)), ((0, 1), (1, 2), (2, 3), (0, 3)),
               name="F_%d" % a)


def del_pezzo_6():
    """the toric del Pezzo surface of degree 6 (P^2 blown up in 3 points)"""
    rays = ((1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1))
    return Fan(rays, tuple((i, (i + 1) % 6) for i in range(6)), name="dP_6")


def product(fan1, fan2):
    """the fan of the product variety"""
    n1, n2 = fan1.dim, fan2.dim
    rays = [r + (0,) * n2 for r in fan1.rays] + [(0,) * n1 + r for r in fan2.rays]
    shift = len(fan1.rays)
    cones = [c1 + tuple(i + shift for i in c2) for c1 in fan1.cones for c2 in fan2.cones]
    name = "%s x %s" % (fan1.name or "X", fan2.name or "Y")
    return Fan(tuple(rays), tuple(cones), name=name)


def star_subdivision(fan, face, name=None):
    """blow-up of the orbit closure V(tau) for the cone tau spanned by the
    rays with the given indices (tau = a maximal cone: blow-up of a point)
    >>> F1 = star_subdivision(projective_space(2), (0, 1))
    >>> len(F1.rays), len(F1.cones)
    (4, 4)
    """
    face = tuple(sorted(face))
    if len(face) < 2:
        raise ValueError("star subdivision along a ray does not change the fan")
    if not any(set(face) <= set(c) for c in fan.cones):
        raise ValueError("%r is not a face of the fan" % (face,))
    new_ray = tuple(sum(fan.rays[i][k] for i in face) for k in range(fan.dim))
    new_index = len(fan.rays)
    cones = []
    for c in fan.cones:
        if set(face) <= set(c):
            for i in face:
                cones.append(tuple(sorted([j for j in c if j != i] + [new_index])))
        else:
            cones.append(c)
    return Fan(fan.rays + (new_ray,), tuple(cones),
               name=name or "Bl_%r %s" % (face, fan.name))


# -- JSON --------------------------------------------------------------------

def from_dict(document):
    validate(document, load_schema("fan"))
    return Fan(tuple(tuple(r) for r in document["rays"]),
               tuple(tuple(c) for c in document["cones"]), name=document.get("name", ""))


def to_dict(fan):
    return {"name": fan.name, "rays": [list(r) for r in fan.rays],
            "cones": [list(c) for c in fan.cones]}


def load(path):
    return from_dict(json.loads(Path(path).read_text()))
