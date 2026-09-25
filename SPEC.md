# SPEC — BB cells, motives and stable cell data from combinatorial input

Status: **v0.3** (2026-09-25). The open questions of v0.1 are decided
(§5). Implemented: M1, M2, M4, and parts of M3, M5 and M6. The status of
each step is tracked in `PLAN.md` §3.

Every item carries a maturity tag:

- **[L0]** standard mathematics; implement and test.
- **[L1]** known in the literature for special classes; implement per class and
  cite (see `LITERATURE.md`).
- **[L2]** plausible, but details must be verified before code is written.
- **[R]** research; not a deliverable, but the design must not block it.

---

## 1. Goal

Given combinatorial data for a smooth projective variety $X$ with a split torus
action that has finitely many fixed points (toric fan, $G/P$ data, wonderful
data, …), compute:

1. the Białynicki-Birula (BB) cell decomposition for a chosen or automatically
   picked generic cocharacter; **[L0]**
2. the invariants that depend only on the cell dimensions: Poincaré polynomial,
   Chow motive, Hodge numbers, point counts, class in $K_0(\mathrm{Var}_k)$,
   quadratic Euler characteristic in $GW(k)$; **[L0]**
3. the cellular filtration as a combinatorial object: a closure/adjacency
   relation between cells, where it can be obtained; **[L1]/[L2]**
4. eventually, stable motivic attaching data: first the $\eta$-components
   between adjacent cells, and via real realization the integral cohomology of
   $X(\mathbb{R})$ and Chow–Witt groups. **[L2]/[R]**

The long-term aim is item 4 for **smooth complete spherical varieties** given
by their Luna–Vust data (colored fan) or spherical system.

### Non-goals (for now)

- Non-split forms and Galois descent. Everything is split over $k$.
- Singular varieties. Rationally smooth ones may come later through
  Brion/Gonzales-style rational cells.
- Fixed loci that are not isolated. The plus-decomposition then has
  positive-dimensional bases; the code may later allow this, but v1 assumes
  $X^T$ is finite.
- Sheaf cohomology of line bundles. For that, use cohomCalg or Sage/M2 on
  toric varieties.

---

## 2. Conventions (fixed, to avoid confusion)

**C1. Base field.** $k$ is a field and $T \cong \mathbb{G}_m^r$ is a split
torus. Characters $X^*(T) \cong \mathbb{Z}^r$ and cocharacters
$X_*(T) \cong \mathbb{Z}^r$ are integer vectors, paired by the standard dot
product $\langle \lambda, \chi \rangle$.

**C2. Tangent weights.** For a fixed point $p \in X^T$, the multiset
$\mathrm{wt}(p) \subset X^*(T)$ is the set of characters by which $T$ acts on
the tangent space $T_pX$, where $t\cdot v = \chi(t)\,v$. Isolatedness means
$0 \notin \mathrm{wt}(p)$. Warning: the characters of the *coordinate
functions* on an invariant affine chart are the negatives of the tangent
weights. Each front end states which one it produces and converts to
tangent weights.

**C3. Genericity.** A cocharacter $\lambda$ is *generic* if
$\langle\lambda,\chi\rangle \neq 0$ for all $\chi \in \bigcup_p \mathrm{wt}(p)$.
Then $X^{\lambda(\mathbb{G}_m)} = X^T$.

**C4. Plus-cells.** $X_p^+ = \{x : \lim_{t\to 0}\lambda(t)x = p\}$, which is
isomorphic to $\mathbb{A}^{d^+(p)}$, where
$d^+(p) = \#\{\chi \in \mathrm{wt}(p) : \langle\lambda,\chi\rangle > 0\}$.
*The code always reports plus-cells.* Replacing $\lambda$ by $-\lambda$ gives
minus-cells of dimension $n - d^+(p)$. `quadric.py` in effect counts the
negative weights, which is the plus-cell convention for $-\lambda$. Do not mix
the two.

**C5. Motives.** Chow motives are covariant, with Tate motive $\mathbb{Z}(i)[2i]$
(Voevodsky's $DM$ convention), so
$M(\mathbb{P}^1) = \mathbb{Z} \oplus \mathbb{Z}(1)[2]$. A cell of dimension
$d$ contributes $\mathbb{Z}(d)[2d]$ to $M(X)$ in the convention where plus-cell
closures give the homology classes (Brosnan / Chernousov–Gille–Merkurjev).

**C6. Motivic spheres.** $S^{p,q} = (S^1)^{\wedge(p-q)} \wedge \mathbb{G}_m^{\wedge q}$,
so $\mathbb{P}^1 \simeq S^{2,1}$ and $\mathbb{A}^d/(\mathbb{A}^d\smallsetminus 0) \simeq S^{2d,d}$.
$\eta : \mathbb{A}^2\smallsetminus 0 \to \mathbb{P}^1$ is the Hopf map, and
stably $\eta \in \pi_{1,1}(\mathbb{S}) \cong K^{MW}_{-1}(k) \cong W(k)$.

**C7. Grothendieck–Witt.** Whenever $GW$ or $W$ appear, $\operatorname{char} k \neq 2$.
$\langle a\rangle$ is the class of the form $ax^2$,
and $\mathbb{H} = \langle 1\rangle + \langle -1\rangle$. With compact support,
$\chi_c^{\mathbb{A}^1}(\mathbb{A}^d) = \langle -1\rangle^d$.

**C8. Indexing.** Roots, simple roots and Weyl group elements follow Bourbaki
numbering, with 1-based labels in the user interface. The legacy `quadric.py`
uses 0-based labels, so tests translate between them.

---

## 3. Architecture

Pure Python ≥ 3.10 with the standard library only, using exact arithmetic
through `fractions.Fraction` and `int` (decision D1). Tests use `unittest`
and `doctest`, so no test dependency is needed; `pytest` can still run them.
Optional SageMath adapters live in a separate module, are never imported by
the core, and their tests are skipped when Sage is absent. The package
`bbcells/` lives in this repository next to the untouched legacy script
`quadric.py` (decision D2).

```
bbcells/
  core.py          FixedPointData, choose_generic_cocharacter, bb_cells
  invariants.py    Poincaré polynomial, motive, Hodge, K0, point counts
  gw.py            GW(k) elements (split case: pairs (a, b) = a<1> + b<-1>)
  rootsystem.py    Cartan matrix → roots, Weyl group, reflections (all types)
  output.py        text / LaTeX / JSON renderers
  frontends/
    raw.py         user-supplied fixed points + weights (+ optional GKM edges)
    toric.py       smooth complete fan
    flag.py        G/P from (Cartan type, subset of simple roots)
    quadric.py     split quadrics (via flag.py; regression vs legacy script)
    product.py     products of any of the above
    wonderful.py   wonderful compactification of adjoint G           [L1]
    horospherical.py  smooth toroidal horospherical varieties         [L2]
    spherical.py   colored fan / spherical system (placeholder)       [R]
  cli.py
  sage_adapter.py  optional cross-checks against SageMath (never required)
tools/
  fetch_arxiv.py   literature fetcher (see LITERATURE.md §8)
literature/
  arxiv_metadata.json   committed
  cache/                git-ignored PDFs and TeX sources
pyproject.toml
quadric.py         legacy, kept as regression oracle
tests/
```

The authoritative module list, with `operations.py` (product, restriction
to a subtorus, blow-up) and `oracles.py` (independent formulas used as test
oracles), is in `PLAN.md` §2.

### 3.1 Core data type [L0]

```python
@dataclass(frozen=True)
class FixedPointData:
    dim: int                                    # n = dim X
    rank: int                                   # r = rank T
    points: tuple[Hashable, ...]                # labels of X^T
    weights: Mapping[Hashable, tuple[Vector, ...]]  # tangent weights, C2
    edges: tuple[Edge, ...] | None = None       # optional GKM data, §3.4
    name: str = ""
```

Validation:

- every point has exactly `dim` weights, all nonzero;
- every weight has length `rank`;
- if `edges` is given, the GKM compatibility conditions hold (§3.4).

### 3.2 Choosing a cocharacter [L0]

- The user may supply $\lambda$; it is rejected if it is not generic.
- Otherwise the code picks one deterministically: for example
  $\lambda = (1, N, N^2, \dots)$ with $N$ larger than twice the maximal absolute
  weight coordinate, then checks genericity and increases $N$ if needed.
  For $G$-varieties, the default is a dominant regular coweight such as
  $\rho^\vee$ where that is generic, so that the cells are Schubert cells.

### 3.3 Invariants from the cell dimensions [L0]

Let $c_d = \#\{p : d^+(p) = d\}$. Everything in this subsection depends only
on the vector $(c_0, \dots, c_n)$:

| Output | Formula |
|---|---|
| Betti numbers | $b_{2d} = c_d$, $b_{\text{odd}} = 0$ |
| Poincaré polynomial | $P_X(t) = \sum_d c_d t^{2d}$ |
| Chow motive | $M(X) \cong \bigoplus_d \mathbb{Z}(d)[2d]^{c_d}$ |
| Hodge numbers | $h^{d,d} = c_d$, all other $h^{p,q} = 0$ |
| $K_0(\mathrm{Var}_k)$ | $[X] = \sum_d c_d\,\mathbb{L}^d$ |
| $\lvert X(\mathbb{F}_q)\rvert$ | $\sum_d c_d q^d$ |
| $\chi_{\mathrm{top}}(X(\mathbb{C}))$ | $\sum_d c_d = \lvert X^T\rvert$ |
| $\chi^{\mathbb{A}^1}(X) \in GW(k)$ | $\sum_d c_d \langle -1\rangle^d$ |
| $\chi(X(\mathbb{R}))$ | signature of the above, $\sum_d (-1)^d c_d$ |
| $\dim_{\mathbb{F}_2} H^*(X(\mathbb{R});\mathbb{F}_2)$ | $\sum_d c_d$ |

Note that $\chi^{\mathbb{A}^1}$ adds no information beyond the $c_d$. It is
listed because it is the first "quadratic" invariant and a useful check.

Built-in consistency checks, run on every computation:

1. independence from $\lambda$: recompute with a second generic cocharacter,
   and the vector $(c_d)$ must agree;
2. Poincaré duality: $c_d = c_{n-d}$;
3. $c_0 = c_n = 1$ when $X$ is connected.

### 3.4 Cellular filtration and GKM data [L1]/[L2]

Optional input: the **GKM graph**, with vertices $X^T$ and one edge $p - q$ for
each $T$-invariant curve $C \cong \mathbb{P}^1$. The edge is labelled by the
weight $\chi \in \mathrm{wt}(p)$ of $T_pC$, and $-\chi \in \mathrm{wt}(q)$.
This is available for toric varieties (walls of the fan), for $G/P$ (edges
$w \to ws_\alpha$), and for wonderful and many spherical varieties when they
are GKM.

Outputs derived from it:

- **[L1]** the *BB order*: $q \preceq p$ if some chain of $\lambda$-increasing
  invariant curves leads from $q$ to $p$, which is a necessary condition for
  $X_q^+ \subset \overline{X_p^+}$. For $G/P$ it is the Bruhat order.
- **[L1]** equivariant cohomology $H_T^*(X)$ as a GKM ring, given as generators
  and relations, over $\mathbb{Z}$ where the GKM conditions hold integrally,
  otherwise over $\mathbb{Q}$.
- **[L2]** whether the BB decomposition is a *filtrable stratification*, and
  whether cell closure agrees with the BB order. Report this as "not
  determined" unless the front end guarantees it (true for $G/P$ and toric).

### 3.5 Stable motivic attaching data [L2]/[R]

By Voelkel (arXiv:1805.04338), every spherical variety is stably cellular
and has a mixed Tate motive. The questions below are therefore well posed for
the whole target class; the problem is to compute the attaching data, not to
show that cells exist.

The filtration $\emptyset = X_{<0} \subset X_{\le 0} \subset \dots \subset X_{\le n} = X$
by unions of cells gives cofiber sequences
$X_{\le d-1} \to X_{\le d} \to \bigvee_{c_d} S^{2d,d}$ in $SH(k)$. The attaching
map between cells of adjacent dimensions has a component in
$[S^{2d+2,d+1}, S^{2d+1,d}] = \pi_{1,1}(\mathbb{S}) \cong W(k)\cdot\eta$.

- **[L2] Level A: the $\eta$-incidence matrix.** For each $d$, a matrix
  $E_d \in W(k)^{c_{d+1}\times c_d}$. In the split case over $\mathbb{R}$, the
  real realization of $E_d$ should give the incidence numbers of the real cell
  complex of $X(\mathbb{R})$, which are in $\{0, \pm 2\}$ (Kocherlakota for
  $G/P$). Target output: $H^*(X(\mathbb{R});\mathbb{Z})$ and, through the real
  cycle class map for cellular varieties (Hornbostel–Wendt–Xie–Zibrowius), the
  Chow–Witt groups $\widetilde{CH}^*(X)$. Plan: implement for $G/P$ first,
  using known real Bruhat incidences, and for toric varieties using the real
  toric cell structure. Then check whether GKM data plus orientation signs
  determine $E_d$ in general. **To verify before coding.**

  *Data model (decision D4).* Every variety handled here, its torus action
  and its BB cells are defined over $\mathbb{Z}$ (Chevalley groups, fans).
  Hypothesis **H1 [L2]:** the $\eta$-components of the attaching maps come
  from $SH(\mathbb{Z})$ and lie in the image of
  $W(\mathbb{Z}) \cdot \eta$, where $W(\mathbb{Z}) \cong \mathbb{Z}$
  ($\langle 1\rangle \mapsto 1$, $\langle -1\rangle \mapsto -1$,
  $\mathbb{H}\eta = 0$). Under H1, one integer $e$ per pair of adjacent cells
  determines the component over every $k$ by base change, and real
  realization turns $e\,\eta$ into an incidence number $2e$, since
  $\eta_{\mathbb{R}} : S^1 \to S^1$ has degree $\pm 2$ depending on
  orientation conventions. So $E_d$ is stored as an **integer matrix**, and
  the documented meaning of each entry is "coefficient in $W(\mathbb{Z})$".
  Before H1 is proved or found in the literature, every output that depends
  on it is labelled `conditional_on="H1"`. Supporting evidence: the unstable
  Thom-space cell structures of arXiv:1805.04338 (`thm:unstable-thom-cells`)
  are constructed over any base smooth over a Dedekind ring, in particular
  over $\mathbb{Z}$. Test oracles for H1: Kocherlakota
  (real $G/P$), Hudson–Matszangosz–Wendt (type A flags; all torsion is
  2-torsion), Casian–Kodama (Grassmannians), Choi–Park (real toric; odd
  torsion occurs there, which the integer model can represent).
- **[R] Level B.** Attaching maps between cells whose dimensions differ by
  2 or more lie in higher stems $\pi_{m,m'}$. No computation is planned; the
  data model only has to allow storing them.

### 3.6 Front ends

| Front end | Input | Fixed points | Tangent weights | Tag |
|---|---|---|---|---|
| raw | JSON/dict | given | given | L0 |
| toric | rays $u_\rho \in \mathbb{Z}^r$ and maximal cones of a smooth complete fan | maximal cones $\sigma$ | the dual basis $m_1,\dots,m_n$ of the generators of $\sigma$, with the sign from C2 | L0 |
| flag | Cartan type, rank, subset $I$ of simple roots ($P = P_I$) | $W/W_I$, minimal coset representatives | $w(\Phi^- \smallsetminus \Phi_I^-)$ at $wP$ (sign convention fixed in code) | L0 |
| quadric | $n$ | via flag ($B$ or $D$ type, $P_1$) | via flag | L0 |
| product | list of front ends | products | concatenation over $T_1 \times T_2$ | L0 |
| wonderful | adjoint type | $W \times W$ in the closed orbit $G/B \times G/B^-$ for $T\times T$ | tangent to closed orbit plus normal weights from simple roots (De Concini–Procesi) | L1 |
| horospherical | $G$, $P$, a smooth fan for the toric fibre | pairs (coset, cone) | base weights plus twisted fibre weights | L2 |
| spherical | colored fan / spherical system | open | open | R |

Minimum test cases. All must pass before a front end counts as done.

- $\mathbb{P}^n$ (toric and flag $A_n$, $P_1$): $P = \sum_{i\le n} t^{2i}$.
- $\mathrm{Gr}(k,n)$: Gaussian binomial $\binom{n}{k}_{t^2}$.
- $\mathbb{P}^1\times\mathbb{P}^1$ and Hirzebruch surfaces $F_a$: $1 + 2t^2 + t^4$.
- Split quadrics $Q_n$: $\sum_{i=0}^n t^{2i}$, plus an extra $t^{n}$ when $n$ is
  even; must match `quadric.py` cell for cell for $n \le 8$.
- Full flags $G/B$ for all types of rank $\le 4$: $\sum_{w\in W} t^{2\ell(w)}$.
- Wonderful compactification of $PGL_2$, which is $\mathbb{P}^3$.
- $\chi^{\mathbb{A}^1}(\mathbb{P}^1) = \mathbb{H}$ and
  $\chi^{\mathbb{A}^1}(\mathbb{P}^2) = 2\langle 1\rangle + \langle -1\rangle$.

### 3.7 Interfaces

- **Python API:** `bb_cells(data, lam=None) -> CellDecomposition`, where
  `CellDecomposition` has `.cells`, `.dims`, `.poincare()`, `.motive()`,
  `.hodge()`, `.gw_euler()`, `.to_latex()` and `.to_json()`.
- **CLI:** `python -m bbcells flag B3 --parabolic 1`,
  `python -m bbcells toric fan.json`, `python -m bbcells quadric 5`.
- **Output formats:** human-readable text, LaTeX (as in `quadric.py`), and
  JSON for machine use and regression tests.

---

## 4. Milestones

1. **M1:** `core`, `invariants`, `gw`, `raw`, `toric`, `product`, checks, tests.
2. **M2:** `rootsystem` (generic, from Cartan matrix), `flag`, `quadric`;
   regression against `quadric.py`.
3. **M3:** GKM input, BB order, equivariant cohomology ring for toric and flag.
4. **M4:** `wonderful` for adjoint group compactifications, tested against
   De Concini–Springer.
5. **M5:** Level A of §3.5. Real incidences for $G/P$ and
   $H^*(X(\mathbb{R});\mathbb{Z})$; check hypothesis H1 against the oracles.
6. **M6:** spherical classes in the order fixed by decision D3:
   - **M6a** smooth toroidal horospherical varieties (toric bundles over
     $G/P$). Test oracle: the Batyrev–Moreau formula
     $E(X) = E(G/H)\sum_{n\in|\Sigma|\cap N}(uv)^{\omega_X(n)}$
     (arXiv:1203.0671), which gives the Betti numbers straight from the
     colored fan;
   - **M6b** smooth complete spherical varieties of rank one, with
     arXiv:1805.04338 as reference. By Akhiezer's classification, the
     two-orbit completion $\overline{X}$ is a flag variety $G'/P'$ of a
     larger group with boundary a flag variety $G/Q$, so this reduces to the
     flag front end plus restriction to a subtorus (`PLAN.md` S6b.1);
   - **M6c** complete symmetric varieties (De Concini–Procesi);
   - **M6d [R]** general smooth complete toroidal spherical varieties from
     Luna–Vust data, using Brion's GKM description and the local structure
     theorem, with smoothness checked by Gagliardi's criterion.

## 5. Decisions (v0.2)

The four open questions of v0.1 were delegated by KV and are decided as
follows. Each can be revisited; a change is recorded here with a date.

**D1 — language and dependencies.** Pure Python ≥ 3.10, standard library
only. Reasons: KV prefers Python; the legacy script is pure Python; the
tool should run anywhere, including CI and teaching settings, without a
SageMath install. Exact arithmetic uses `int` and `Fraction`; polynomials
are small dict-based classes. SageMath is used only through the optional
`sage_adapter.py` for cross-checks (toric Betti numbers, Weyl group lengths),
and its tests are skipped when Sage is absent.

**D2 — repository layout.** Restructure *inside* this repository: add the
package `bbcells/`, `tests/`, `tools/`, `literature/` and a `pyproject.toml`,
and keep `quadric.py` unchanged at the top level as a regression oracle, since
the README examples call it directly. Renaming the GitHub repository is left
to KV: it changes public URLs and is not needed for the work.

**D3 — spherical classes after M4.** Order: horospherical, then rank one,
then complete symmetric varieties, then general toroidal (M6a–d above).
Reasons:
- Smooth toroidal horospherical varieties are toric bundles over $G/P$, so
  they reuse the flag and toric front ends. They are the cheapest class that
  is genuinely given by Luna–Vust data.
- Rank one is where arXiv:1805.04338 already has explicit unstable cell
  structures, so we have answers to test against and a direct link to the
  stable questions in §3.5.
- Complete symmetric varieties come with classical Betti numbers
  (De Concini–Springer) but need the symmetric-pair machinery (restricted
  roots, the little Weyl group), which is more work to implement.

**D4 — attaching data.** Track $\eta$-coefficients as integers interpreted in
$W(\mathbb{Z}) \cong \mathbb{Z}$, conditional on hypothesis H1 (§3.5), and
compute their real realizations ($\{0,\pm 2\}$ incidences) as the primary
checkable quantity. Do not implement general $W(k)$ arithmetic: under H1 it
would carry no extra information for split varieties over $\mathbb{Z}$, and it can
be added later if a non-split front end ever appears (a non-goal for now).
