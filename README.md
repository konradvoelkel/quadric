computes cell structures for projective quadrics via the Bialynicki-Birula Theorem

### sample output ###


➜ ./quadric.py 4 1,2,3
computing cells for the type $D_{3}$ space
 $PQ_4 = \{x_{0}y_{0} + x_{1}y_{1} + x_{2}y_{2} = 0\} \subset \mathbb{P}^5$
 using cocharacter
    1\epsilon_0 + 2\epsilon_1 + 3\epsilon_2
  = 1\alpha_0 + 0.0\alpha_1 + 3.0\alpha_2
  cell signature:  [2, 3, 4, 2, 1, 0]
   0: \{y_{2}\neq 0, x_{0}=0, x_{1}=0, y_{0}=0, y_{1}=0\}
   1: \{y_{1}\neq 0, x_{0}=0, x_{2}=0, y_{0}=0\}
   2: \{x_{0}\neq 0, x_{1}=0, x_{2}=0\}
   2: \{y_{0}\neq 0, x_{1}=0, x_{2}=0\}
   3: \{x_{1}\neq 0, x_{2}=0\}
   4: \{x_{2}\neq 0\}

## bbcells

This script is the seed of `bbcells`, a pure-Python tool computing
Bialynicki-Birula cell structures from combinatorial data and, from them,
the Chow motive, Betti and Hodge numbers, classes in K_0(Var), point counts,
the quadratic Euler characteristic in GW(k), and, for real flag varieties,
H^*(X(R); Z). See [SPEC.md](SPEC.md) (what), [PLAN.md](PLAN.md) (how, with
the status of each step) and [LITERATURE.md](LITERATURE.md) (sources).

    python3 -m bbcells toric --named P1xF2          # smooth complete fans (or a JSON file)
    python3 -m bbcells flag E6 --parabolic 1        # G/P for all Cartan types
    python3 -m bbcells quadric 4 --cocharacter=-1,-2,-3   # same cells as ./quadric.py 4 1,2,3
    python3 -m bbcells wonderful A2                 # wonderful compactifications
    python3 -m bbcells two-orbit OP2                # [OP^2] = L^8 + L^12 + L^16
    python3 -m bbcells example complete-conics      # blow-up of P^5 along the Veronese
    python3 -m bbcells spherical complete-quadrics 5   # assembled orbit by orbit, 450 cells
    python3 -m bbcells real A3                      # H^*(Fl(R^4); Z)

Every front end is tested against an independent oracle (point counts,
Weyl group degrees, blow-up and fibration formulas, published tables, and
this script for quadrics).

## Development

The package `bbcells/` needs only Python ≥ 3.10 and its standard library.

    python3 -m unittest            # all tests, including doctests
    python3 -m bbcells --help      # command line interface

`quadric.py` is kept unchanged as the original script and serves as a
regression oracle for the quadric front end.
