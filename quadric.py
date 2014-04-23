#! /usr/bin/python3

"""
This program computes cell decompositions (resulting from the Bialynicki-Birula theorem) of the projective split quadric PQ_n of dimension n inside P^{n+1}. For this purpose, it takes as input the dimension n and optionally a cocharacter of SO(n,n) or SO(n,n+1) (depending on the dimension) described as vector in the euclidean basis of the coroot lattice.

Note that it would be sensible to use SAGEmath instead of implementing a root_system class here; it just grew naturally out of an attempt to use as little 3rd party as possible (and python3).

(c) 2014 Konrad Voelkel, University of Freiburg
http://blog.konradvoelkel.de
"""

# features that would be nice to have:
# * let the user specify a hyperplane and compute open complement of cells
# * check whether the given hyperplane cuts out a quadric
# * inferencing more info about the affine cells is possible:
#   if for each i except j either x_i or y_i is 0 (and z=0) then also x_j or y_j is 0
#   this should actually give a complete description of the affine cells.
# * so one could directly compute the motivic cell structure for AQ_n here


import sys
import doctest
import itertools
from random import randint


def fold(prefixes, suffixes):
    """
    >>> list(fold(([2], [-2]), ([3], [-3])))
    [[2, 3], [2, -3], [-2, 3], [-2, -3]]
    >>> list(fold(([1], [-1]), ([2, 3], [2, -3], [-2, 3], [-2, -3])))
    [[1, 2, 3], [1, 2, -3], [1, -2, 3], [1, -2, -3], [-1, 2, 3], [-1, 2, -3], [-1, -2, 3], [-1, -2, -3]]
    """
    return (prefix + suffix for prefix in prefixes for suffix in suffixes)


def get_all_signs(unsigned_list):
    """
    >>> list(get_all_signs([1, 2]))
    [[1, 2], [1, -2], [-1, 2], [-1, -2]]
    >>> list(get_all_signs([1, 2, 3]))
    [[1, 2, 3], [1, 2, -3], [1, -2, 3], [1, -2, -3], [-1, 2, 3], [-1, 2, -3], [-1, -2, 3], [-1, -2, -3]]
    """
    if(len(unsigned_list) == 1):
        return (unsigned_list, [-unsigned_list[0]], )
    else:
        # would be better without call to list(),
        # but then the generator gets consumed somehow ...
        return fold(([unsigned_list[0]], [-unsigned_list[0]], ),
                    list(get_all_signs(unsigned_list[1:])))


class projective_quadric(object):

    def __init__(self, dimension):
        """construct a (split) projective quadric by specifying its dimension
        >>> print(projective_quadric(5))
        PQ_{5}
        """
        self.dimension = dimension
        self.root_system = self.get_root_system_from_quadric_dim(dimension)

    def __repr__(self):
        return "%s(%d)" % (self.__class__.__name__, self.dimension)

    def __str__(self):
        return "PQ_{%d}" % (self.dimension)

    def _assert_valid_action(self, weights):
        """narrowly assumes that the action on z coordinate is trivial"""
        if not self.dimension + 2 == len(weights):
            raise AssertionError(
                "expected %d weights, got %d, cannot act on %s" %
                (self.dimension + 2, len(weights), self))
        if not len(set(weights)) == len(weights):
            raise AssertionError("weights are not all distinct: %s" % weights)
        half_dim = len(weights) // 2
        for i in range(half_dim):
            if not weights[i] == - weights[i + half_dim]:
                raise AssertionError(
                    "weights %s don't yield an action on %s: %s" %
                    (weights, self))
        if len(weights) % 2 == 1:
            if not weights[-1] == 0:
                raise AssertionError(
                    "weights %s don't yield an action on %s: %s" %
                    (weights, self))

    @staticmethod
    def get_root_system_from_quadric_dim(quadric_dim):
        """returns type and rank of the root system of the group
        under which PQ of dimension quadric_dim is homogeneous.
        Note that D1 doesn't exist, D2 = A1 x A1,
        A1 = B1 = C1 and A2 = B2 = C2 and A3 = B3.
        >>> projective_quadric.get_root_system_from_quadric_dim(4)
        abstract_root_system('D', 3)
        >>> projective_quadric.get_root_system_from_quadric_dim(5)
        abstract_root_system('B', 3)
        """
        return abstract_root_system(
            'B' if quadric_dim %
            2 == 1 else 'D',
            quadric_dim //
            2 +
            1)

    def _get_weights_on_projective_space_from_cocharacter(self, cocharacter):
        """cocharacter is assumed to be a list of coefficients in the euclidean basis
        >>> projective_quadric(4)._get_weights_on_projective_space_from_cocharacter([1, 2, 3])
        [1, 2, 3, -1, -2, -3]
        >>> projective_quadric(5)._get_weights_on_projective_space_from_cocharacter([3, 2, 1])
        [3, 2, 1, -3, -2, -1, 0]
        """
        if not self.root_system.rank == len(cocharacter):
            raise AssertionError("cocharacter of rank %s expected, got %s"
                                 % (self.root_system.rank, len(cocharacter)))
        weights = (list(cocharacter)
                   + [-w for w in cocharacter]
                   + ([0] if self.root_system.cartan_type == 'B' else []))
        self._assert_valid_action(weights)
        return weights

    @staticmethod
    def _get_weights_at_fp(weights):
        """ returns a list of weights on coordinates in charts {x_i=1}
        >>> projective_quadric._get_weights_at_fp([1,-1])
        [[0, -2], [2, 0]]
        >>> projective_quadric._get_weights_at_fp([1,-1,0])
        [[0, -2, -1], [2, 0, 1]]
        """
        return [[wt_j - wt_i for wt_j in weights]
                for wt_i in weights[:(len(weights) // 2) * 2]]

    def _count_cell_dimensions(self, nonneg_weights_at_fp):
        return [self.dimension - len(nonneg_at_wt)
                for nonneg_at_wt in nonneg_weights_at_fp]

    def get_beautiful_coord_name(self, coord_index):
        assert 0 <= coord_index <= (self.dimension + 1)
        if coord_index == self.dimension + 1 and self.root_system.cartan_type == 'B':
            return "z"
        else:
            if coord_index < self.root_system.rank:
                return "x_{%d}" % (coord_index)
            else:
                return "y_{%d}" % (coord_index - self.root_system.rank)

    def _get_nonnegative_weight_coords(self, weights_at_fp):
        """takes a list of weights on coordinates in charts,
        and a total number of coordinates,
        returns a list of nonnegative-part coords
        >>> projective_quadric(0)._get_nonnegative_weight_coords([[0, -2], [2, 0]])
        [[], []]
        >>> projective_quadric(1)._get_nonnegative_weight_coords([[0, -2, -1], [2, 0, 1]])
        [[], ['z']]
        """
        assert len(weights_at_fp[0]) == self.dimension + 2
        b = lambda i: self.get_beautiful_coord_name(i)
        h = self.root_system.rank
        return [[b(i)
                 for i, wt in enumerate(weights_at_fp_i)
                 if wt >= 0           # nonnegative
                 and (i == self.dimension + 1 == h * 2  # z always ok
                      # now remove x_i in chart y_i=1 and reverse,
                      # to get only the tangent space of the quadric
                      or (j < h and i >= h and i != j + h)
                      or (i < h and j >= h and i != j - h)
                      or (j < h and i < h and i != j)
                      or (i >= h and j >= h and i != j))]
                for j, weights_at_fp_i in enumerate(weights_at_fp)]

    def _get_affine_spaces(self, nonneg_weights_at_fp):
        beauty = lambda i: self.get_beautiful_coord_name(i)
        return ["\\{%s\\}"
                % (", ".join([beauty(i) + "\\neq 0"]
                             + [j + "=0" for j in nonneg_at_wt]))
                for i, nonneg_at_wt in enumerate(nonneg_weights_at_fp)]

    def compute_affine_cells(self, cocharacter):
        weights = self._get_weights_on_projective_space_from_cocharacter(
            cocharacter)
        nonnegative = self._get_nonnegative_weight_coords(
            self._get_weights_at_fp(weights))
        cell_dimensions = self._count_cell_dimensions(nonnegative)
        affine_spaces = self._get_affine_spaces(nonnegative)
        dim_sorted_affine_spaces = sorted(
            zip(cell_dimensions, affine_spaces), key=lambda x: x[0])
        return (dim_sorted_affine_spaces, cell_dimensions)

    def get_quadratic_poly(self):
        beauty = lambda i: self.get_beautiful_coord_name(i)
        return (" + ".join(["%s%s" %
                            (beauty(i), beauty(i +
                                               self.root_system.rank)) for i in range(0, self.root_system.rank)]) +
                (" - %s^2" %
                 (beauty(self.dimension +
                         1)) if self.root_system.cartan_type == 'B' else ""))


class abstract_root_system(object):

    """
    mathematical constants used in the code, taken from Bourbaki Lie 7

     we use the Witt basis [x_0:...:x_n:y_0:...:y_n:z]
       and the notation y_k = x_{-k}
     for the quadratic form q = \sum x_iy_i - z^2

     list of roots in B_{n+1}:
      \alpha_i = \epsilon_i - \epsilon_{i+1} for 0 \leq i < n
      \alpha_n = 2\epsilon_n
     list of corresponding coroots:
      H_{\alpha_i} = E_{i,i} - E_{-i,-i} + E_{-(i+1),-(i+1)} - E_{i+1,i+1}
      H_{\alpha_n} = 2 E_{n,n} - 2 E_{-n,-n}

     list of roots in D_{n+1}:
      \alpha_i = \epsilon_i - \epsilon_{i+1} for 0 \leq i < n
      \alpha_n = \epsilon_{n-1} + \epsilon_n
     list of corresponding coroots:
      H_{\alpha_i} = E_{i,i} - E_{-i,-i} + E_{-{i+1},-{i+1}} - E_{i+1,i+1}
      H_{\alpha_n} = E_{n-1,n-1} - E_{-(n-1),-(n-1)} + E_{n,n} - E_{-n,-n}
    """

    def __init__(self, cartan_type, rank):
        """
        >>> print(abstract_root_system('F', 4))
        F_{4}
        >>> abstract_root_system('E', 6)
        abstract_root_system('E', 6)
        """
        assert cartan_type in ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        if(cartan_type == 'G'):
            assert rank == 2
        if(cartan_type == 'F'):
            assert rank == 4
        if(cartan_type == 'E'):
            assert rank in [6, 7, 8]
        self.cartan_type = cartan_type
        self.rank = rank

    def __repr__(self):
        return "%s('%s', %d)" % (self.__class__.__name__,
                                 self.cartan_type,
                                 self.rank)

    def __str__(self):
        return "%s_{%s}" % (self.cartan_type, self.rank)

    def _test_basis_conversion(self, repeat=0):
        """randomized test of basis conversion back and forth
        >>> abstract_root_system('B', 2)._test_basis_conversion()
        >>> abstract_root_system('D', 3)._test_basis_conversion()
        """
        random_in_root_basis = [randint(-9, 9) for k in range(self.rank)]
        in_standard_basis = self.simple_roots_to_standard_basis(
            random_in_root_basis)
        back_in_root_basis = self.standard_basis_to_simple_roots(
            in_standard_basis)
        assert random_in_root_basis == back_in_root_basis
        if(repeat > 0):
            self._test_basis_conversion(repeat - 1)

    def simple_roots_to_standard_basis(self, coefficients_of_simple_roots):
        """given a sequence [a_0,...,a_n]
        returns a sequence [b_0,...,b_n]
        where the b_i are related to the a_i by
        sum a_i \alpha_i^* = sum b_i \epsilon_i^*
        with \alpha_i the roots of root_system
        and \epsilon_i the standard basis in the euclidean space
        which is spanned by the root lattice.
        >>> abstract_root_system('B', 2).simple_roots_to_standard_basis([1,0])
        [1, -1]
        >>> abstract_root_system('B', 2).simple_roots_to_standard_basis([1,1])
        [1, 1]
        >>> abstract_root_system('D', 3).simple_roots_to_standard_basis([1,1,1])
        [1, 1, 0]
        """
        a = coefficients_of_simple_roots
        n = len(a) - 1
        assert self.rank == n + 1
        if(self.cartan_type == 'B'):
            return ([a[0]]
                    + [a[i] - a[i - 1] for i in range(2, n + 1)]
                    + [2 * a[n] - a[n - 1]])
        elif(self.cartan_type == 'D'):
            if(self.rank <= 2):
                raise NotImplementedError("root systems D1, D2 not supported")
            return ([a[0]]
                    + [a[i] - a[i - 1] for i in range(2, n - 1)]
                    + [a[n - 1] - a[n - 2] + a[n]]
                    + [a[n] - a[n - 1]])
        else:
            raise NotImplementedError("root system not yet supported")

    def standard_basis_to_simple_roots(self, coefficients_of_standard_basis):
        """given a sequence [b_0,...,b_n]
        returns a sequence [a_0,...,a_n]
        where the a_i are related to the b_i by
        sum a_i \alpha_i^* = sum b_i \epsilon_i^*
        with \alpha_i the roots of root_system
        and \epsilon_i the standard basis in the euclidean space
        which is spanned by the root lattice.
        >>> abstract_root_system('B', 2).standard_basis_to_simple_roots([2, 0])
        [2, 1.0]
        >>> abstract_root_system('B', 2).standard_basis_to_simple_roots([1, 3])
        [1, 2.0]
        """
        b = coefficients_of_standard_basis
        n = len(b) - 1
        assert self.rank == n + 1
        if(self.cartan_type == 'B'):
            return ([sum(b[j] for j in range(0, i + 1)) for i in range(0, n)]
                    + [1 / 2 * sum(b[i] for i in range(0, n + 1))])
        elif(self.cartan_type == 'D'):
            return ([sum(b[j] for j in range(0,
                                             i + 1)) for i in range(0,
                                                                    n - 1)] + [1 / 2 * sum(b[i] for i in range(0,
                                                                                                               n)) - 1 / 2 * b[n]] + [1 / 2 * sum(b[i] for i in range(0,
                                                                                                                                                                      n + 1))])
        else:
            raise NotImplementedError("root system not yet supported")

    def get_all_chambers(self):
        """returns representatives for all chambers with one dominant coroot
        in the euclidean basis
        >>> set((tuple(chamber) for chamber in abstract_root_system('B', 2).get_all_chambers())) == {(1, 2), (2, 1), (-1, -2), (-2, -1), (-1, 2), (2, -1), (1, -2), (-2, 1)}
        True
        """
        all_signs = sum(
            (list(
                get_all_signs(
                    list(permutation))) for permutation in itertools.permutations(
                list(
                    range(
                        1,
                        self.rank +
                        1)))),
            [])
        return all_signs


# -----------------------------------------------------
def print_affine_cells(quadric, cocharacter):
    affine_spaces, cell_dims = quadric.compute_affine_cells(cocharacter)
    print(" using cocharacter\n    %s\n  = %s" %
          (" + ".join(["%s\\epsilon_%s" %
                       (c, i) for i, c in enumerate(cocharacter)]), " + ".join(["%s\\alpha_%s" %
                                                                                (c, i) for i, c in enumerate(root_system.standard_basis_to_simple_roots(cocharacter))])))
    print("  cell signature: ", cell_dims)
    print("\n".join(["   %s: %s" % (x) for x in affine_spaces]))


def print_affine_cells_latex(quadric, cocharacter):
    affine_spaces, cell_dims = quadric.compute_affine_cells(cocharacter)
    print("\subsection*{using cocharacter    $%s$}\n$= %s$\\\\" %
          (" + ".join(["%s\\epsilon_%s" %
                       (c, i) for i, c in enumerate(cocharacter)]), " + ".join(["%s\\alpha_%s" %
                                                                                (c, i) for i, c in enumerate(root_system.standard_basis_to_simple_roots(cocharacter))])))
    print("  cell signature: $", cell_dims, "$\n\\begin{itemize}")
    print("\n".join(["   \item[%s]: $%s$" % (x) for x in affine_spaces]))
    print("\end{itemize}")


if __name__ == "__main__":
    doctest.testmod()

    if(len(sys.argv) < 2):
        print(
            "USAGE: quadric.py quadric_dim [coroots]\n example: $ quadric.py 6 1,2,3,4")
        sys.exit()
    quadric_dim = int(sys.argv[1])

    quadric = projective_quadric(quadric_dim)
    root_system = quadric.root_system
    quadratic_poly = quadric.get_quadratic_poly()
    print(
        "computing cells for the type $%s$ space\n $PQ_%s = \{%s = 0\} \subset \mathbb{P}^%s$" %
        (root_system, quadric_dim, quadratic_poly, quadric_dim + 1))

    if(len(sys.argv) > 2):
        # if explicit cocharacter is supplied, show cells for this one
        cocharacter = eval("[" + sys.argv[2] + "]")
        # not assuming that it is given in the simple root basis
        # cocharacter_std_basis = root_system.simple_roots_to_standard_basis(cocharacter)
        print_affine_cells(quadric, cocharacter)
    else:
        # if no cocharacter is supplied, show all possible cell decompositions,
        # use latex output because of overwhelmingness
        all_chambers = root_system.get_all_chambers()
        for cocharacter in all_chambers:
            print_affine_cells_latex(quadric, cocharacter)
