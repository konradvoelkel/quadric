# SPEC — BB cells, motives and stable cell data from combinatorial input

Status: **draft v0.1** (2026-09). Nothing below is implemented yet except the
legacy script `quadric.py`, which covers the case of split quadrics.

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

**C7. Grothendieck–Witt.** $\langle a\rangle$ is the class of the form $ax^2$,
and $\mathbb{H} = \langle 1\rangle + \langle -1\rangle$. With compact support,
$\chi_c^{\mathbb{A}^1}(\mathbb{A}^d) = \langle -1\rangle^d$.

**C8. Indexing.** Roots, simple roots and Weyl group elements follow Bourbaki
numbering, with 1-based labels in the user interface. The legacy `quadric.py`
uses 0-based labels, so tests translate between them.

---

## 3. Architecture

Pure Python ≥ 3.10 with the standard library only, using exact arithmetic
through `fractions.Fraction` and `int`. Optional adapters for SageMath live in
a separate module and are never required.

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
tests/
```

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
4. **M4:** `wonderful` (adjoint group compactifications), then complete
   symmetric varieties.
5. **M5:** literature check for §3.5 Level A; real incidences for $G/P$ and
   $H^*(X(\mathbb{R});\mathbb{Z})$.
6. **M6 [R]:** horospherical and toroidal spherical varieties from Luna–Vust
   data; strategy via the local structure theorem.

## 5. Open questions (for KV)

- Q1: Pure Python confirmed, with Sage only as an optional adapter?
- Q2: Should the repo be renamed or restructured (e.g. `bbcells`), keeping
  `quadric.py` as legacy?
- Q3: Which spherical class matters most after the wonderful compactifications:
  complete symmetric varieties, horospherical varieties, or the smooth
  projective spherical varieties of small rank from the classification
  literature?
- Q4: For §3.5, is the target $W(k)$-valued incidences, or only their real
  realizations? This decides whether we track Witt classes or just integers.
