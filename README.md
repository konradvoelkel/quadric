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
