# History of the implementation steps

This file archives the steps of `PLAN.md` v1 (S0–S7.6) with their
done-criteria and results, as they stood when PLAN.md v2 (2026-09-27)
replaced them by a status table and a forward plan. Step names (S1.5, S6d,
S7.2, …) are still used in docstrings and commit messages. The mathematics
behind the later steps is in `docs/spherical.md`, `docs/cohomology.md` and
`docs/real.md`.

## Steps

Legend: **dep** = prerequisites; **done when** = acceptance criteria;
size is S (< 200 lines), M (200–600) or L (> 600), tests included.

### S0 — Scaffolding (S)

`pyproject.toml`, the package skeleton with empty modules, a CI workflow
running unittest on Python 3.10–3.13, and a "Development" section in the
README (how to run the tests). The existing fetcher tests move under
the same runner.
**Done when:** CI is green on the branch, and `python3 -m bbcells --help`
prints usage.

### M1 — Engine, invariants, toric (SPEC §3.1–3.3)

- [x] **S1.1 `algebra.py`** (S). `IntPoly`: dense integer polynomials with
  `+ - *`, evaluation, `is_palindromic`, and pretty printing in a chosen
  variable ($t$, $q$ or $\mathbb{L}$). `GWClass(a, b)` meaning
  $a\langle1\rangle + b\langle-1\rangle$: rank, signature, addition,
  multiplication ($\langle-1\rangle^2 = 1$), and $\mathbb{H} = (1,1)$.
  **Done when:** there are doctests for all operations and
  $(\mathbb{H}\cdot x = \mathrm{rank}(x)\,\mathbb{H})$ holds for random $x$.
- [x] **S1.2 `core.py`** (M). `FixedPointData` with the validation of
  SPEC §3.1. `choose_generic_cocharacter` as in SPEC §3.2 (deterministic,
  with a retry loop). `bb_cells(data, lam=None) -> CellDecomposition`, which
  stores $\lambda$, `dim_of[p]` and the counts $c_d$. `with_opposite()`
  returns the decomposition for $-\lambda$.
  **Done when:** $c_d(-\lambda) = c_{n-d}(\lambda)$ holds for random data,
  and invalid data (a zero weight, the wrong number of weights, a
  non-generic $\lambda$) raises `ValueError` with a message naming the
  offending point.
- [x] **S1.3 `invariants.py`** (S). Every row of the SPEC §3.3 table, plus the
  automatic checks: a second cocharacter, Poincaré duality, $c_0 = c_n = 1$.
  Checks return a `CheckReport` rather than raising, so that inputs can be
  explored.
  **Done when:** hand-written fixed-point data for $\mathbb{P}^n$ gives every
  invariant, including $\chi^{\mathbb{A}^1}(\mathbb{P}^2) = 2\langle1\rangle+\langle-1\rangle$.
- [x] **S1.4 `frontends/raw.py`** (S). Load and dump the JSON format
  `{"dim", "rank", "points": [{"label", "weights": [[...]]}], "edges"?}`. A
  JSON Schema lives in `bbcells/schema/fixed_points.json`, checked by a
  small hand-written validator (no `jsonschema` dependency).
- [x] **S1.5 `frontends/toric.py`** (M). Input: rays and maximal cones.
  Checks: every maximal cone is unimodular; completeness via the
  wall-crossing test (every codimension-one face lies in exactly two
  maximal cones), which is then confirmed by Euler characteristic equals
  number of maximal cones and Poincaré duality. Weights as in 1.1. Named
  constructors: `projective_space(n)`, `hirzebruch(a)`, `product_fan`,
  `blowup_fan` (star subdivision of a smooth cone).
  **Done when:** oracles §4.1 pass: $\mathbb{P}^n$, $F_a$ for
  $a = 0,\dots,4$, the degree-6 del Pezzo (hexagon, $1 + 4t^2 + t^4$), and
  $\mathbb{P}^1\times\mathbb{P}^1\times\mathbb{P}^1$.
- [x] **S1.6 `operations.py`, first part** (S). `product` (tangent weights
  concatenated over $T_1 \times T_2$) and `restrict(data, matrix)` with the
  check from 1.5.
  **Done when:** $P_{X\times Y} = P_X P_Y$ holds on toric examples, and
  restricting $\mathbb{P}^1\times\mathbb{P}^1$ to the torus of one factor
  raises (whole fibres are fixed), while the diagonal torus and generic
  rank-1 subtori keep 4 isolated fixed points.
- [x] **S1.7 `output.py`, `cli.py`** (M). Text, LaTeX (the same shape as
  `quadric.py`'s output) and JSON. `python -m bbcells toric fan.json`,
  `python -m bbcells raw data.json`.
  **Done when:** there are golden-file tests for all three formats on
  $\mathbb{P}^2$.

### M2 — Root systems, flag varieties, quadrics

- [x] **S2.1 `rootsystem.py`** (L). Cartan matrices for $A_n, B_n, C_n, D_n,
  E_{6,7,8}, F_4, G_2$ in Bourbaki numbering, and products. Positive roots
  by closure under simple reflections, with coroots. Fundamental weights in
  the adjoint realization. Orbit enumeration and lengths as in 1.8. Degrees
  of $W$ from a table, checked against $\prod_i [d_i]_q = \sum_w q^{\ell(w)}$
  for rank $\le 6$.
  **Done when:** $|\Phi^+|$, $|W|$ and the degrees match the standard
  tables for every type of rank $\le 8$ (with $|W|$ checked through the
  orbit of $\rho$ only up to rank 6), and the ambient and adjoint
  realizations agree on the pairings $\langle\alpha_i^\vee, \alpha_j\rangle$.
- [x] **S2.2 `frontends/flag.py`** (M). `flag_variety("B3", parabolic={1})`
  with the weights of 1.2. Named helpers: `grassmannian(k, n)`,
  `full_flags(type)`, `isotropic_grassmannian(k, 2n)`.
  **Done when:** oracles §4.2 pass, i.e. $P_{G/P}(t) = \prod[d_G]/\prod[d_L]$
  in $q = t^2$ for all $(G, P)$ of rank $\le 4$ and for $E_6/P_1$;
  Grassmannians give Gaussian binomials; and with dominant $\lambda$,
  `dim_of[wP] == length(w)`.
- [x] **S2.3 `frontends/quadric.py` and legacy regression** (S). $Q_n$ via
  flag, with the fixed-point labels of 1.3. The test imports `quadric.py`
  as a module and compares cell by cell for $n \le 8$ and all sign/permutation
  chambers where $n \le 5$ (legacy with $\lambda$ = ours with $-\lambda$).
  **Done when:** the regression passes; note that `quadric.py` itself is not
  modified. *(Implemented: $Q_2 = \mathbb{P}^1\times\mathbb{P}^1$ needs both nodes of
  $D_2$ crossed, because the Levi $GL_1\times SO(2)$ of the line stabilizer has
  no roots.)*

### M3 — GKM data and derived operations (SPEC §3.4)

- [x] **S3.1 `gkm.py`: edges** (M). Toric: one edge per wall
  $\tau = \sigma\cap\sigma'$. With $\sigma = \mathrm{cone}(u_1,\dots,u_n)$ and
  $\tau = \mathrm{cone}(u_1,\dots,u_{n-1})$, the invariant curve
  $V(\tau) \cong \mathbb{P}^1$ has weight $m_n$ (dual to $u_n$) at
  $x_\sigma$ and $-m_n$ at $x_{\sigma'}$. Flag: edges $wP \to s_\beta wP$ for
  $\beta \in w(\Phi^-\smallsetminus\Phi_I^-)$, labelled by $\beta$. Validation
  of the GKM conditions: pairwise linear independence of the weights at
  each vertex, and matching labels at both ends up to sign.
- [x] **S3.2 BB order** (S). The transitive closure of $\lambda$-increasing
  edges. **Done when:** it equals the Bruhat order on $W^I$ (computed
  independently by the subword criterion) for all $G/P$ of rank $\le 3$.
  *(Found while implementing: the relation need not go down in dimension.
  On $dP_6$ with $\lambda = (1,3)$ an invariant curve joins two 1-cells, so
  that BB decomposition is not a stratification. The order is still acyclic,
  and `gkm.is_graded` reports which case applies.)*
- [x] **S3.3 $H_T^*(X;\mathbb{Q})$ as a GKM ring** (M). Tuples of polynomials
  with divisibility along edges; a module basis from the BB order (the
  equivariant Schubert classes are the unique flow-up classes). Poincaré
  series check $P_X(t)/(1-t^2)^r$. **Done when:** $H_T^*(\mathbb{P}^n)$ and
  $H_T^*(\mathrm{Gr}(2,4))$ reproduce the known ring structure after
  setting the equivariant parameters to $0$. *(Implemented in `equivariant.py`
  with Atiyah–Bott localization; degrees of Grassmannians, quadrics and
  $\mathbb{P}^n$ and Pieri on $\mathrm{Gr}(2,4)$ are tested. Flow-up classes of
  high codimension in large rank are expensive: they are computed on demand.)*
- [x] **S3.4 `operations.blowup`** (M). Implements 1.6. Needs `Z` as its own
  `FixedPointData` and an embedding map of fixed points; the embedding is
  checked by multiset inclusion $\mathrm{wt}_Z(p) \subset \mathrm{wt}_X(p)$.
  **Done when:** blowing up a point in $\mathbb{P}^2$ gives $F_1$ (compare
  with toric S1.5), and blowing up $\mathbb{P}^3$ along a coordinate line
  gives the known Betti numbers $1, 2, 2, 1$.

### M4 — Wonderful compactifications of adjoint groups

- [x] **S4.1 `frontends/wonderful.py`** (M). Implements 1.4 on the torus
  $T\times T$ of rank $2r$.
  **Done when:** the point-count oracle §4.3 passes for $A_1$–$A_4$,
  $B_2$, $B_3$, $C_3$, $D_4$ and $G_2$ (at most $192^2$ fixed points; $F_4$
  and $E_n$ are too large for tests and allowed only by explicit CLI
  request); $A_1$ equals $\mathbb{P}^3$ cell for cell.

### M5 — Real incidences and Level A (SPEC §3.5)

- [x] **S5.1 `realcells.py`, type A partial flags** (L). Incidence
  coefficients $[\Omega_I, \Omega_J] \in \{0, \pm 2\}$ from Matszangosz
  (arXiv:1910.11149, Theorems `incidencecoeffs`, `Kocherlakota`, `signs`,
  in terms of ordered set partitions, $N_I(a,b)$ and $s(I,J)$). Build the
  cellular chain complex and compute $H^*(X(\mathbb{R});\mathbb{Z})$ by
  Smith normal form (exact, stdlib).
  **Done when:** $H^*(\mathbb{RP}^n;\mathbb{Z})$ is right; real
  Grassmannians match Casian–Kodama (arXiv:1309.5520) for $n \le 6$; and
  for every type A flag variety with $n \le 5$ all torsion is 2-torsion
  (Hudson–Matszangosz–Wendt, arXiv:2302.11003).
  *(Implemented, with Kocherlakota's unsigned rule for all types. Findings:
  (1) the incidences form the cochain complex of cooriented cells, graded by
  codimension, which only coincides with the cellular grading for orientable
  $X(\mathbb{R})$; (2) Kocherlakota's and Matszangosz's rules agree up to
  sign on complete flags, where $m = N_I(a,b)+1$ holds exactly, but not on
  partial flags such as $\mathbb{RP}^2$, where they describe the cellular and
  the cooriented complex respectively; (3) the sign example in
  arXiv:1910.11149 has $N_I(a,b) = 2$, so its incidence is 0 although the
  paper states $+2$. Its $c_1,\dots,c_4 = 3,2,1,2$ are reproduced.)*
- [x] **S5.2 Real toric oracle** (M). The rational Betti numbers of $X(\mathbb{R})$
  for smooth toric $X$ via the Suciu–Trevisan / Choi–Park formula
  (arXiv:1311.7056), used only as an oracle for later work on toric
  incidences.
- [x] **S5.3 H1 experiments** (research). Look for a rule giving
  $E_d \in W(\mathbb{Z})$ from GKM data plus orientation signs, testing it
  against S5.1 on all type A cases with $n \le 5$. Write up the results in
  `docs/real.md`, whether positive or negative. This step never blocks M6.
- [x] **S5.4 Other types** (later). Rabelo–San Martin (Indag. Math. 30 (2019),
  745–772) give signs for general real flag manifolds; not on arXiv, so
  obtain a copy first.

### M6 — Spherical classes (SPEC D3)

- [x] **S6a.1 Toroidal horospherical $X = G\times_P Y$** (M). Input: Cartan
  type, $I$, a sublattice $M \subset \bigoplus_{j\notin I}\mathbb{Z}\omega_j$
  given by an integer basis, and a smooth complete fan in
  $N = M^\vee$. Fixed points $(wP, x_\sigma)$ with weights
  $w(\Phi^-\smallsetminus\Phi_I^-) \cup w(m_{\sigma,1..n})$: the torus acts on
  the fibre over $wP$ through $t \mapsto w^{-1}tw$, hence the $w$-twist.
  **Done when:** $P_X = P_{G/P}\cdot P_Y$ (Zariski-locally trivial
  fibration) for several $(G,P,\Sigma)$. *(Implemented. The fibration
  formula cannot detect the $w$-twist, since the untwisted data is
  $G/P\times Y$; the twist is pinned by comparing
  $SL_3\times_{P_1}\mathbb{P}^1$ with the blow-up of $\mathbb{P}(k^3\oplus k)$
  computed by `operations.blowup`.)*
- [x] **S6a.2 Batyrev–Moreau oracle** (M). Implement
  $E(X) = E(G/H)\sum_{n\in|\Sigma|\cap N}(uv)^{\omega_X(n)}$ (arXiv:1203.0671)
  straight from a colored fan, including $\omega_X$ and the weighted
  Stanley–Reisner series. Read the definition of $\omega_X$ in the paper
  first (the step starts with a note in `docs/`). Check against S6a.1 in the
  toroidal case.
- [x] **S6a.3 Smooth colored horospherical** (research). BB data from the
  local structure theorem and Pasquier's smoothness criterion; S6a.2 is the
  oracle.
- [x] **S6b.1 `frontends/two_orbit.py`, rank-one two-orbit completions** (M).
  $\overline{X} = G'/P'$ with boundary $D = G/Q$, via flag, restriction (1.5)
  and an explicit torus embedding $T_G \subset T_{G'}$ per case; $D^T$ is
  identified inside $\overline{X}^T$ by weight-multiset inclusion. Outputs
  the cells of $\overline{X}$ and of $D$, $[X] = [\overline{X}] - [D]$, and
  $\chi^{\mathbb{A}^1}_c(X)$. First cases:
  - affine quadrics: $Q_n \supset Q_{n-1}$ with $SO(n+1) \subset SO(n+2)$;
  - $\mathbb{HP}^n$: $\mathrm{Gr}(2,2n+2) \supset \mathrm{IG}(2,2n+2)$ with
    $Sp_{2n+2} \subset SL_{2n+2}$;
  - $\mathbb{OP}^2$: $E_6/P_1 \supset F_4/P_4$, with the torus embedding
    from folding $E_6 \to F_4$ written as an explicit matrix;
  - $PGL_n/GL_{n-1}$: $\mathbb{P}^{n-1}\times\check{\mathbb{P}}^{n-1} \supset$
    incidence variety, with diagonal $PGL_n \subset PGL_n \times PGL_n$.

  *(Implemented: in low dimension, and for the incidence variety where
  $\varepsilon_j - \varepsilon_i$ occurs twice at $(p_i, H_j)$, weight inclusion is
  ambiguous, so these cases use explicit geometric embeddings that are then
  checked by weight inclusion.)*
  **Done when:** oracles §4.4 pass. Afterwards, cross-reference the rest of
  Akhiezer's list with Knop's table of cuspidal rank-one spherical
  varieties (arXiv:1303.2466, §`sec:TABLE`) and add the remaining cases. *(Done: the table in
  `two_orbit.py` maps every characteristic-0 row of Knop's table with
  reductive $H$ to a case: added $\mathbb{P}^{n+1}\smallsetminus Q_n$,
  $\mathbb{P}^6\smallsetminus Q_5$ for $G_2$, and $\mathbb{P}^7\smallsetminus Q_6$ for
  $Spin_7$; rows with non-reductive $H$ or only in characteristic 2 are
  listed as not covered.)*
- [x] **S6c.1 Complete conics** (S). $\mathrm{Bl}_{v_2(\mathbb{P}^2)}\mathbb{P}^5$
  via `blowup` (S3.4), with the $SL_3$-torus on $\mathbb{P}(\mathrm{Sym}^2)$.
  The normal weights at each $x_i^2$ are pairwise distinct, so 1.6 applies.
  **Done when:** $1 + 2q + 3q^2 + 3q^3 + 2q^4 + q^5$ and exactly 12 fixed
  points.
- [x] **S6c.2 Complete symmetric varieties** (research). De Concini–Springer
  (not on arXiv; obtain a copy) and Brion–Joshua (arXiv:0705.1035) for the
  minimal-rank case. Complete quadrics via Vainsencher's iterated blow-ups
  as a second route through `blowup`. *(Implemented: complete quadrics in
  $\mathbb{P}^3$ by two blow-ups, $1,3,6,10,13,13,10,6,3,1$ with 66 fixed points,
  against the blow-up formula; together with complete conics, the rank-one
  complete symmetric varieties (S6b.1) and the group case (M4). Arbitrary
  symmetric pairs need fixed points in non-closed orbits and belong to S6d.)*
- [x] **S6d General toroidal spherical** (research, SPEC M6d). Requires
  $T$-fixed points in non-closed $G$-orbits, which already occur for
  $\mathbb{P}^1\times\mathbb{P}^1 \supset SL_2/T$. Plan the approach only
  after S6a–S6c. *(Done as far as it goes without a Luna-data-to-subgroup
  step; see `docs/spherical.md`. `frontends/spherical.py` assembles $X^T$ orbit by
  orbit: $O^T = W/W_H$ with weights $w(\Phi\smallsetminus\Phi_H) + w(N)$.
  For wonderful $X$ the base point data comes from the spherical roots,
  $S^p$ and the satellites; the normal weights are the $W_L$-averages of
  $-\gamma$ under a checked spanning condition. Complete quadrics in
  $\mathbb{P}^{n-1}$ and complete skew forms on $k^{2n}$ work in every
  dimension. They match the S6c front ends exactly, and the
  orbit-decomposition point counts for $n\le6$ and $n\le4$. Open: satellites
  from Luna data, toroidal non-wonderful $X$, and GKM edges across orbits.)*
- [x] **S6e Complete symmetric varieties from Satake diagrams** (research,
  finishes SPEC M6c). `frontends/symmetric.py` computes the spherical roots
  $\alpha-\theta\alpha$, $S^p$ = black nodes, and the satellites. The
  satellites use the equal-rank test $\varepsilon=-w_0$ on subdiagrams and
  Borel–de Siebenthal nodes, with $W_H = \mathrm{Stab}_{W_L}(t)$ possibly
  containing non-reflections. Real forms of types A–D, $E_6$, $F_4$, $G_2$
  are covered. **Done when:** it reproduces the earlier families exactly;
  the exceptional isomorphisms $B_2=C_2$, $D_3=A_3$ and $D_4$ triality agree;
  and every new case passes the checks. *(Done; Hermitian satellites beyond
  condition (R) are marked and validated by the checks, see `docs/spherical.md`
  §6.)*
- [x] **S6f Holomorphic Lefschetz check** (S). `invariants.check` also tests
  the Atiyah–Bott formula for $\chi_y$ modulo a large prime. **Done when:**
  every front end passes it and it detects a sign error that the BB-based
  checks miss. *(Done.)*
- [x] **S6g Toroidal varieties over a wonderful model** (M).
  `frontends/toroidal.py`: fixed points are (orbit $O_J$, full-dimensional
  cone in the face $F_J$); the weights come from $\mathrm{pr}_J$ of the dual
  basis (`docs/spherical.md` §8). **Done when:** it agrees with blow-ups along
  orbit closures and with the orbit point count. *(Done.)*
- [x] **S6h Cosets without W** (S). A search by Deodhar's lemma for reflection
  subgroups; $\mathrm{Stab}_{W_L}(t)$ by orbit-stabilizer with Schreier
  generators; $E_7$/$E_8$ Satake diagrams. **Done when:** EVII builds.
  *(Done: 23,464 fixed points, all checks pass.)*
- [x] **S6i Hermitian certificate and orbit counts** (M). ABBV certificate
  for one unknown; Brion–Peyre point counts orbit by orbit (`docs/spherical.md`
  §§7, 9). *(Done: c = 0 is certified for AIII(2,3), AIII(2,4), AIII(2,5),
  DIII(5) and EIII; the orbit counts agree everywhere.)*
- [x] **S6j Equivariant cohomology beyond GKM** (M). `brion.py`: components of
  $X^{\ker\chi}$ from fixed-point data; Brion's congruences mod $\chi$ and
  $\chi^2$. **Done when:** it reproduces the GKM edges of flag and toric
  varieties and gives a free module with the BB Poincaré series for
  complete quadrics. *(Done.)*

## Research steps S7

- [ ] **S7.1 General Luna data**: colour calculus, Levi-part step, a table of
  full-rank wonderful reductive subgroups; normal weights beyond (R) from
  colour coefficients or the certificate.
- [x] **S7.2 Joint certificate** (`docs/spherical.md` §7). Point counts of orbit
  closures against Brion–Peyre, for one unknown or two on different orbits,
  by induction over closures. *(Done: $c=0$ is certified for every Hermitian
  case tested, including AIII(3,4) and DIII(7) with their joint steps. Still
  open: two unknowns on one orbit, closures whose open orbit has no fixed
  points, and a full proof of the proposition's Hermitian case.)*
- [ ] **S7.3 Schubert calculus beyond GKM** (a, b done; c open).
  - [x] S7.3a Characteristic numbers: degree-one classes on Brion
    components, colours of complete quadrics, ABBV integration
    (`docs/cohomology.md`). *(Done: 3264, 666841088, 48942189946470400 and the
    ML degrees $\varphi(n,d)$ are reproduced.)*
  - [x] S7.3b Canonical (Goldin–Tolman) classes by Newton interpolation on
    Brion components, the integral ring where they exist, and the divisor
    subalgebra $\mathbb Q[x]/\operatorname{Ann}(V)$ (`docs/cohomology.md`).
    *(Done. Complete conics have no canonical classes in any chamber, and
    the colours generate $H^*$ for $n\le4$ but not for complete quadrics in
    $\mathbb P^4$.)*
  - [ ] S7.3c A canonical integral basis when cell closures are not unions
    of cells; the oracle is DGMP 1988 for complete quadrics.
- [ ] **S7.4 Toroidal $X$ with $\Lambda\supsetneq\mathbb Z\Sigma$** (finite covers).
- [ ] **S7.5 Real incidences for non-GKM varieties** (M5 beyond GKM).
  *(In progress, `docs/real.md` §§3.2, 5.
  - Non-graded toric surfaces are solved by a deflection rule, checked
    against Choi–Park on 164 decompositions.
  - Exact real BB incidences for toric varieties by discrete Morse theory
    (`realtoric.py`) confirm H1′ entry by entry on 96 graded decompositions
    of dimension 2–4.
  - The rule applied to the curves of the Brion components
    (`brion.invariant_curves`) is correct on graded non-GKM decompositions:
    $\mathbb P^q\times\check{\mathbb P}^q$, $\mathrm{Gr}(2,6)$ and
    $\mathrm{Gr}(2,8)$ with symplectic tori, and $E_6/P_1$ with the
    $F_4$-torus (`docs/real.md` §4).
  - Open: non-graded decompositions in dimension $\ge3$. Every rank-two
    complete symmetric variety is non-graded. For complete conics (oracle
    $1,0,0,0,0,1$) one unique term must be removed in each chamber.)*
- [x] **S7.6 Cell counts without listing fixed points** (`docs/spherical.md` §10).
  *(Done for $E_7$: EII, EI, EVI (758079 fixed points, 48 s), EVII and EV
  (28373976 fixed points, 30 min), and for EIX (7445880). EVIII has more
  than $6\cdot10^8$ fixed points on its closed orbit alone.)*

## Original order and parallelism

```
S0 → S1.1 → S1.2 → S1.3 ┬→ S1.4
                        ├→ S1.5 → S1.6 → S1.7
                        └→ S2.1 → S2.2 → S2.3
S1.6 + S2.2 → S3.1 → S3.2 → S3.3
S1.6 + S2.2 → S3.4
S2.2 → S4.1
S2.2 + S3.2 → S5.1 → S5.3
S1.5 + S2.2 → S6a.1 → S6a.2 → S6a.3
S1.6 + S2.3 → S6b.1
S3.4 → S6c.1 → S6c.2
```

Toric (S1.5–S1.7) and root systems (S2.1–S2.3) are independent after S1.3
and can proceed in parallel. The first milestone worth showing is **M1 plus
S2.2**: arbitrary toric and flag varieties with all Level-0 invariants.

## Original first action

Start with **S0**, then **S1.1–S1.3**, as one pull request, "engine and
invariants". Its done-criteria are the union of those steps'.
