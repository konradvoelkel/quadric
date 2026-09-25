"""
Chevalley bases and Tits lifts for all types, used to orient real Schubert
cells (PLAN.md S5.4).

Simply-laced types (A, D, E): Frenkel-Kac construction (Kac, Infinite
dimensional Lie algebras, 7.8): basis E_alpha (roots) and h = root lattice
(x) Q, with
    [h, E_b] = (h|b) E_b,   [E_a, E_{-a}] = -a,   [E_a, E_b] = eps(a, b) E_{a+b},
eps bimultiplicative with eps(a_i, a_i) = -1 and eps(a_i, a_j) eps(a_j, a_i) =
(-1)^{(a_i|a_j)}, from an orientation of the Dynkin diagram.

Non-simply-laced types by folding along a diagram automorphism sigma for
which the orientation is sigma-invariant (so sigma permutes the E_alpha):
F4 from E6, G2 from D4. The simple root vectors of the folded algebra are
sums of E_alpha over sigma-orbits of orthogonal simple roots, and its Tits
lifts are products of the commuting Tits lifts of the orbit.

Tits lifts n_i = exp(ad e_i) exp(-ad f_i) exp(ad e_i) act on root vectors by
signed permutations, which is all the orientation computation needs.
"""

from fractions import Fraction

from bbcells.rootsystem import RootSystem

# orientations i -> j (0-based Bourbaki nodes), invariant under the foldings
_ORIENTATION = {
    "E6": [(0, 2), (2, 3), (5, 4), (4, 3), (1, 3)],     # invariant under 1<->6, 3<->5
    "D4": [(0, 1), (2, 1), (3, 1)],                       # invariant under triality
}

# folding data: (simply-laced type, sigma-orbits of simple roots in the order of
# the folded type's Bourbaki nodes)
_FOLDINGS = {
    "F4": ("E6", [(1,), (3,), (2, 4), (0, 5)]),
    "G2": ("D4", [(0, 2, 3), (1,)]),
}


class SimplyLaced(object):
    """Frenkel-Kac Lie algebra of a simply-laced type
    >>> g = SimplyLaced("A2")
    >>> g.check_jacobi(samples=200)
    True
    """

    def __init__(self, cartan_type, orientation=None):
        self.R = RootSystem(cartan_type)
        if self.R.letter not in "ADE":
            raise ValueError("simply-laced types only")
        n = self.R.rank
        if orientation is None:
            orientation = _ORIENTATION.get(self.R.name)
        if orientation is None:
            orientation = [(i, j) for i in range(n) for j in range(i + 1, n)
                           if self.R.cartan[i][j] != 0]
        self.minus = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        for i, j in orientation:
            self.minus[i][j] = 1
        positive = list(self.R.positive_roots)
        self.roots = positive + [tuple(-c for c in b) for b in positive]
        self.root_set = set(self.roots)

    def form(self, a, b):
        return self.R.inner(a, b)

    def eps(self, a, b):
        exponent = sum(a[i] * b[j] * self.minus[i][j]
                       for i in range(len(a)) for j in range(len(b)))
        return -1 if exponent % 2 else 1

    # elements: dict with keys ('E', root) or ('h', i) -> Fraction
    def bracket(self, x, y):
        result = {}

        def add(key, value):
            if value:
                result[key] = result.get(key, 0) + value
                if result[key] == 0:
                    del result[key]

        for (kx, ax), cx in x.items():
            for (ky, ay), cy in y.items():
                c = cx * cy
                if kx == "h" and ky == "h":
                    continue
                if kx == "h":             # [h_i, E_b] = (a_i | b) E_b
                    simple = tuple(int(k == ax) for k in range(self.R.rank))
                    add(("E", ay), c * self.form(simple, ay))
                elif ky == "h":
                    simple = tuple(int(k == ay) for k in range(self.R.rank))
                    add(("E", ax), -c * self.form(simple, ax))
                else:
                    total = tuple(p + q for p, q in zip(ax, ay))
                    if not any(total):           # [E_a, E_{-a}] = -a
                        for i, coefficient in enumerate(ax):
                            add(("h", i), -c * coefficient)
                    elif total in self.root_set:
                        add(("E", total), c * self.eps(ax, ay))
        return result

    def check_jacobi(self, samples=500):
        import random
        rng = random.Random(0)
        basis = [("E", r) for r in self.roots] + [("h", i) for i in range(self.R.rank)]
        for _ in range(samples):
            a, b, c = ({k: Fraction(1)} for k in rng.sample(basis, 3))
            total = {}
            for x, y, z in ((a, b, c), (b, c, a), (c, a, b)):
                for key, value in self.bracket(x, self.bracket(y, z)).items():
                    total[key] = total.get(key, 0) + value
            if any(total.values()):
                return False
        return True

    def exp_ad(self, x, v, sign=1):
        """exp(sign * ad x) applied to v"""
        result, term, k = dict(v), dict(v), 1
        while True:
            term = self.bracket(x, term)
            term = {key: value * sign / k for key, value in term.items()}
            if not term:
                return result
            for key, value in term.items():
                result[key] = result.get(key, 0) + value
                if result[key] == 0:
                    del result[key]
            k += 1


class Lifts(object):
    """Tits lifts of the simple reflections of a (possibly folded) type acting
    on the root vectors of the ambient simply-laced algebra"""

    def __init__(self, cartan_type):
        name = RootSystem(cartan_type).name
        if name in _FOLDINGS:
            ambient, orbits = _FOLDINGS[name]
        else:
            ambient = name
            orbits = [(i,) for i in range(RootSystem(name).rank)]
        self.g = SimplyLaced(ambient)
        self.orbits = orbits
        rank = self.g.R.rank
        self.action = []        # per ambient simple root: {root: (sign, image root)}
        for i in range(rank):
            simple = tuple(int(k == i) for k in range(rank))
            e = {("E", simple): Fraction(1)}
            f = {("E", tuple(-c for c in simple)): Fraction(-1)}
            table = {}
            for root in self.g.roots:
                v = {("E", root): Fraction(1)}
                v = self.g.exp_ad(e, v)
                v = self.g.exp_ad(f, v, sign=-1)
                v = self.g.exp_ad(e, v)
                (key, coefficient), = v.items()
                if key[0] != "E" or coefficient not in (1, -1):
                    raise AssertionError("the Tits lift does not permute root vectors")
                table[root] = (int(coefficient), key[1])
            self.action.append(table)

    def apply(self, i, vector):
        """Ad(n_i) on a vector {ambient root: coefficient}, n_i of the folded node i"""
        for k in self.orbits[i]:
            table = self.action[k]
            new = {}
            for root, c in vector.items():
                s, image = table[root]
                new[image] = new.get(image, 0) + s * c
            vector = {r: c for r, c in new.items() if c}
        return vector

    def simple_vector(self, i):
        rank = self.g.R.rank
        return {tuple(int(k == j) for k in range(rank)): 1 for j in self.orbits[i]}

    def frame(self, word):
        """[Ad(n_{r_1} ... n_{r_{j-1}}) e_{r_j}] as vectors in the ambient algebra"""
        frames = []
        for j, i in enumerate(word):
            v = self.simple_vector(i)
            for k in reversed(word[:j]):
                v = self.apply(k, v)
            frames.append(v)
        return frames

    def relative_orientation(self, word1, word2):
        """sign of the frame of word1 relative to that of word2 (reduced words of
        the same element): each frame vector spans the root space it lies in"""
        f1, f2 = self.frame(word1), self.frame(word2)
        key = lambda v: frozenset(v)
        position = {key(v): k for k, v in enumerate(f2)}
        if len(position) != len(f2) or any(key(v) not in position for v in f1):
            raise ValueError("the words do not have the same inversion set")
        permutation = [position[key(v)] for v in f1]
        from bbcells.liealgebra import _permutation_sign
        sign = _permutation_sign(permutation)
        for v in f1:
            w = f2[position[key(v)]]
            ratios = {Fraction(v[r]) / w[r] for r in v}
            if len(ratios) != 1 or next(iter(ratios)) not in (1, -1):
                raise AssertionError("frame vectors are not proportional by +-1")
            sign *= int(next(iter(ratios)))
        return sign
