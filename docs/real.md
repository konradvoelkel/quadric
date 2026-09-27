# Real points and $\eta$-incidences

Status (2026-09-27). This note collects what the package knows about the
real cell complex of $X(\mathbb R)$ and hypothesis H1 (`SPEC.md` §3.5). The
code is in `realcells.py` (flag varieties, the GKM rule), `h1.py`
(predictions and experiments), `realtoric.py` (exact incidences for toric
varieties) and `brion.invariant_curves` (curves beyond GKM); the tests are
`tests/test_realcells.py`, `tests/test_h1.py` and `tests/test_realtoric.py`.
Everything below is experimental evidence, not proof, except where marked
*proved*.

## 1. The question

Let $X$ be smooth projective over $\mathbb{Z}$ with a split torus action and a BB
decomposition for a generic $\lambda$. In $SH$, the component of the attaching map
from a cell $x$ of dimension $d+1$ to a cell $y$ of dimension $d$ is an element of
$\pi_{1,1} = W\cdot\eta$ (SPEC §3.5). Hypothesis H1 says that it comes from
$W(\mathbb{Z})\cdot\eta \cong \mathbb{Z}\cdot\eta$. Two realizations make it testable:

- **real:** $\eta_{\mathbb{R}}$ has degree $\pm 2$, so the incidence numbers of
  the real cell complex $X(\mathbb{R})$ are twice the $W(\mathbb{Z})$-coefficients;
- **complex:** $\eta_{\mathbb{C}} = \eta_{\mathrm{top}} \in \pi_1^s = \mathbb{Z}/2$
  is detected by $Sq^2$, so the coefficients mod 2 are visible in the Steenrod
  squares of $H^*(X(\mathbb{C});\mathbb{F}_2)$.

## 2. The GKM form of Kocherlakota's rule

For a fixed point $p$ let
$$\sigma(p) = \sum_{\chi \in \mathrm{wt}(p),\ \langle\lambda,\chi\rangle > 0} \chi ,$$
the weight of $\det T_p X_p^+$. For an invariant curve joining cells $x$ and $y$
with $\dim x = \dim y + 1$ and weight $\varphi$ at $x$ (with $\langle\lambda,\varphi\rangle > 0$),
write $\sigma(x) - \sigma(y) = m\,\varphi$ whenever possible.

**Prediction.** The $\eta$-component of the attaching map of $x$ to $y$ is
nonzero iff $m$ is even. Cells not joined by an invariant curve have
component 0. The coefficient in $W(\mathbb{Z})$ is $\pm 1$, and the real
incidence is $\pm 2$.

*Proved (in the code's conventions).* For $G/P$ with $\lambda$ dominant regular,
Kocherlakota's set $\mathcal{N}(x) = \{\varphi>0 : \varphi(x) < 0\}$ for
$x = w\omega_I$ is exactly the set of $\lambda$-positive tangent weights at $wP$:
$\varphi = -w\beta > 0$ with $\beta \in \Phi^+\smallsetminus\Phi_I^+$ gives
$\langle\varphi^\vee, w\omega_I\rangle = -\langle\beta^\vee, \omega_I\rangle < 0$.
So on $G/P$ the prediction is Kocherlakota's theorem (Adv. Math. 110, 1995)
for the real points. The code checks the identification for every parabolic
of $A_2, A_3, B_2, B_3, C_3, G_2, D_4$.

*Local picture.* Along a curve $C \cong \mathbb{P}^1$ with
$N_C = \bigoplus_i \mathcal{O}(a_i)$, the normal weights at the two ends are
$\chi_i$ and $\chi_i - a_i\varphi$. If the sign patterns of the normal
weights under $\lambda$ agree at both ends, then
$m = 1 + \sum_{i \in S} a_i$, with $S$ the $\lambda$-positive normal
directions. For $\mathbb{P}^n$ this gives $m = d + 1$, so the attaching map from cell
$d+1$ to cell $d$ is nonzero iff $d$ is odd. This matches the known
$n_\varepsilon\eta$ attaching maps of $\mathbb{P}^n$ (for example $\mathbb{P}^2 = \mathrm{cofib}(\eta)$
and $\mathbb{P}^3/\mathbb{P}^1 \simeq S^{4,2}\vee S^{6,3}$ stably).

## 3. Evidence for GKM varieties

### 3.1 Sign completions against oracles

**Real realization, toric varieties.** The fans were $\mathbb{P}^2$, $\mathbb{P}^3$, $F_0,\dots,F_4$,
$\mathrm{Bl}_{pt}\mathbb{P}^2$, $\mathrm{Bl}_{line}\mathbb{P}^3$, $\mathrm{Bl}_{pt}\mathbb{P}^3$,
$\mathbb{P}^1\times\mathbb{P}^2$ and $F_1\times\mathbb{P}^1$, each with a
cocharacter whose decomposition is graded. In every case:
- the predicted unsigned incidences admit exactly one sign completion with
  $\partial^2 = 0$, up to reorienting cells;
- the resulting rational Betti numbers of $X(\mathbb{R})$ equal those of the
  Choi–Park / Suciu–Trevisan formula. That formula is independent of cells;
  it uses full subcomplexes of the fan (`oracles.real_toric_rational_betti`).

This includes the Klein bottles $F_{odd}(\mathbb{R})$, where $\partial e^2 = 2e^1$ is
forced.

**Real realization, flag varieties.** Here the prediction equals
Kocherlakota's theorem (§2). In type A, the signed incidences of
arXiv:1910.11149 give $H^*(X(\mathbb{R});\mathbb{Z})$, which matches all
available checks (PLAN S5.1).

**Complex realization.** For every cell $y$ of dimension 1 and $x$ of
dimension 2, the coefficient of $x^*$ in $(y^*)^2 \bmod 2$ was computed in the
GKM ring. It equals the parity prediction, with no mismatch, on $\mathbb{P}^3$,
$F_1$, $F_2$, $\mathrm{Bl}_{line}\mathbb{P}^3$, $\mathrm{Gr}(2,4)$, $A_2/B$,
$A_3/B$, $B_2/B$, $G_2/B$, $Q_4$ and $Q_5$. The dual cochain $y^*$ is the
class of the closure of the minus-cell, i.e. the flow-up class for
$-\lambda$, and $Sq^2 a = a^2$ on $H^2$.

### 3.2 Exact incidences for toric varieties

`realtoric.RealToricComplex` computes the real BB incidences of a smooth
complete toric variety exactly.

$X(\mathbb R) = P\times(\mathbb Z/2)^n/\!\sim$ has a regular CW structure
(Davis–Januszkiewicz).
- **Fine cells.** A cone $\tau$ contributes one cell per class of
  $(\mathbb Z/2)^n/\Lambda_\tau$, oriented by its face. The boundary is
  $\partial(\tau,e) = \sum_r (-1)^{n-|\tau|+\#\{i\in\tau: i<r\}}(\tau+r,e)$.
- **BB cells as unions of fine cells.** The real BB cell of $v$ is, in the
  ray coordinates of $v$, the set $\{+,-,0\}^{d_v}$ of fine cells.
- **Morse complex.** The lexicographic matching (pair $0$ with $-$ at the
  first coordinate that is not $+$) leaves one critical cell per fixed point
  and is acyclic. Algebraic Morse theory then gives an integral chain
  complex on the BB cells, whether the decomposition is graded or not.
- **Checks.** On nine fans at three cocharacters each:
  - both the fine complex and the Morse complex have the Choi–Park Betti
    numbers;
  - the integral homology has only 2-torsion (e.g. $\mathbb{RP}^3$ and the
    Klein bottle $F_1(\mathbb R)$).

**H1′ entrywise.** For graded decompositions the cellular complex is
canonical, so the rule of §2 can be compared entry by entry, not only
through Betti numbers after a sign completion. On random smooth complete
toric varieties the exact $|[x:y]|$ equals the prediction
($2$ iff $x$, $y$ are joined by a curve with $m$ even) in every entry:

| dimension | decompositions | nonzero incidences |
|---|---|---|
| 2 | 14 | 9 |
| 3 | 42 | 70 |
| 4 | 40 | 151 |

There was no mismatch.

## 4. Beyond GKM: the curves of the Brion components

For a non-GKM variety the rule of §2 is applied to the invariant curves that
the Brion components provide (`docs/cohomology.md` §2):
- every $\mathbb P^1$ component;
- the two lines of each $\mathbb P(\mathfrak{sl}_2)$ plane, of weight $\pm a$
  (the conics of weight $2a$ join cells two apart);
- the four boundary curves of each ruled surface, from each extreme point
  to each saddle, also when $a=b$.

Parities come from pairing the normal weights at the two ends. Equal
weights are paired first, then the rest by congruence modulo $\varphi$ with
the smallest $|a|$. When normal weights are congruent modulo $\varphi$, the
weights do not show the splitting of the normal bundle; flip-free pairings
all give the same $m$.

**Graded non-GKM decompositions.** The prediction is unique and equals the
oracle in every tested chamber:

| variety | torus | oracle | chambers |
|---|---|---|---|
| $\mathbb P^2\times\check{\mathbb P}^2$ (AIII(1,2)) | $PGL_3$ | $\mathbb{RP}^2\times\mathbb{RP}^2$ | 20 |
| $\mathbb P^3\times\check{\mathbb P}^3$ (AIII(1,3)) | $PGL_4$ | $\mathbb{RP}^3\times\mathbb{RP}^3$ | 8 |
| $\mathbb P^4\times\check{\mathbb P}^4$ (AIII(1,4)) | $PGL_5$ | $\mathbb{RP}^4\times\mathbb{RP}^4$ | 12 |
| $\mathrm{Gr}(2,6)$ (CII(1,2)) | $Sp_6$ | Casian–Kodama | 20 |
| $\mathrm{Gr}(2,8)$ (CII(1,3)) | $Sp_8$ | Casian–Kodama | 12 |
| $E_6/P_1$ (FII) | $F_4$ | Kocherlakota with the $E_6$-torus | 6 |

In each case $X(\mathbb R)$ is the same real variety as for a larger torus,
where the answer is known. So this is a genuine test of the rule on
non-GKM data.

## 5. Non-graded decompositions

A decomposition is *graded* if every invariant curve goes down in cell
dimension. Otherwise the cells do not form a CW complex filtered by
dimension, and the adjacent-cell formula of §2 alone is not defined.
Already $dP_6$ admits no graded cocharacter: on a hexagon with a generic
height function two 1-cells are always adjacent. With rational
coefficients a chain complex on the BB cells still exists (a Morse complex
after perturbation), but it is only defined up to filtered change of basis.

### 5.1 Surfaces: the deflection rule

`h1.surface_prediction`, tested in `tests/test_h1.py`.

A curve $a\to b$ between cells of **equal** dimension breaks the
Morse–Smale condition for the real Morse function $\langle\mu,\lambda\rangle$
on $X(\mathbb R)$. Its real locus is a circle, i.e. two flow lines from $a$
to $b$.

- **Parity.** Exactly one normal direction $\psi$ is negative at $a$ and
  positive at $b$. Put $m = (\sigma(a)-\sigma(b)+\psi_b)/\varphi$.
- **Deflection.** After a perturbation, each of the two arcs from $b$ that
  reach $a$ continues along *one* ascending arc of $a$. So every curve
  $c\to a$ with $\dim c = \dim a+1$ contributes $\pm2$ to
  $\langle\partial c, b\rangle$ if $m$ is odd.

  This holds even when the two arcs from $a$ to $c$ cancel, which is why
  the usual handle-slide product $\langle\partial c,a\rangle\cdot n_{ab}$ is
  not enough.
- **Direct terms.** Adjacent pairs contribute as in §2: $\pm2$ if $m$ is
  even.

Signs are free, subject to $\partial^2=0$.

**Tests against Choi–Park.**
- $dP_6$ and four other non-graded fans: each at 4–6 cocharacters.
- Random smooth complete toric surfaces: 30 fans by iterated star
  subdivisions of $\mathbb P^2$ and $F_a$, 144 decompositions.

Every sign completion gives the oracle's Betti numbers, with no mismatch.

Variants all fail on non-graded pentagons:
- magnitude 2 iff $m$ even;
- no slides;
- slides into $\partial a$ instead of $\partial c$;
- the product rule.

### 5.2 Dimension three

In dimension 3 the rule is wrong: 23 of 72 random
  decompositions agree, 6 disagree, and the rest admit no sign completion.
  There the ascending manifolds of index-1 points are 2-dimensional, and
  slides can chain. A rule would need the flow on those manifolds.

The exact complexes of §3.2 show why.

In 12 random threefolds at 3 cocharacters each:
- A curve with $m$ even always gives $|[x:y]|=2$ (123 of 123).
- A curve with $m$ odd always gives $0$ (238 of 238).
- A curve along which normal directions change sign gives $0$ or $2$, with
  no evident rule.
- There are also nonzero incidences between cells that no curve joins. In
  13 cases the cells do not even share a neighbour.

In the non-graded case the complex is only defined up to filtered change
of basis, so these entries are not invariants, and a local rule can only
aim at some representative. The deflection rule of §5.1 does not give one in
dimension 3.

### 5.3 Complete conics

These are the first non-GKM, non-graded target, and their real points are
known. $X(\mathbb R)$ is the real blow-up of $\mathbb{RP}^5$ along the
Veronese $\mathbb{RP}^2$. The exceptional divisor is an
$\mathbb{RP}^2$-bundle over $\mathbb{RP}^2$, and $\mathbb{RP}^2$ is
$\mathbb Q$-acyclic. By the five lemma applied to the blow-up square,
$H^*(X(\mathbb R);\mathbb Q)\cong H^*(\mathbb{RP}^5;\mathbb Q)$ (*proved*), so the
rational Betti numbers are $1,0,0,0,0,1$.

No rank-two complete symmetric variety tested has a graded cocharacter
along the curves of §4: AI(3), AI(4), AII(3), AIII(2,2), AIII(2,3),
BI(2,3), CI(2), CI(3), CII(2,2), DI(2,4) and $G_2$, with 576 random generic
cocharacters. For complete conics:
- the direct prediction admits no sign completion;
- in each of 21 chambers, removing exactly one term gives the oracle
  $1,0,0,0,0,1$, and that term is unique;
- the removed term always runs from the upper end of the equal-dimension
  curve with $m$ even to the lower end of the one with $m$ odd; there are
  always exactly two such curves.

A rule that explains this is open. Neither the deflection rule of §5.1 nor a
rule summing over paths of slides reproduces it. The closest consistent
complex also drops the paths that stay inside one
$\mathbb P(\mathfrak{sl}_2)$ plane, which suggests that the planes need
their own local model.

### 5.4 Complete quadric surfaces

The second non-graded oracle (PLAN R1). $X$ is the blow-up of
$\mathbb P^9=\mathbb P(\mathrm{Sym}^2k^4)$ along the Veronese $\mathbb P^3$
(rank one), followed by the blow-up along the strict transform $W$ of the
rank-$\le2$ locus. Both centres are smooth and defined over $\mathbb R$, so
$X(\mathbb R)$ is the corresponding pair of real blow-ups. Result (*proved*
below): $H_*(X(\mathbb R);\mathbb Q)$ is $\mathbb Q$ in degrees 0 and 5 and
zero otherwise, i.e. rational Betti numbers $1,0,0,0,0,1,0,0,0,0$.

- **First blow-up**, $Y=\mathrm{Bl}_{\mathbb{RP}^3}\mathbb{RP}^9$ (the real
  rank-one quadrics are $\pm\ell^2$, one point each, so the centre is
  $\mathbb{RP}^3$; codimension 6).
  - $H_*(\mathbb{RP}^9,\mathbb{RP}^3;\mathbb Q)$ is $\mathbb Q$ in degrees 4 and 9:
    $\mathbb{RP}^3$ is orientable and $[\mathbb{RP}^3]$ dies in $\mathbb{RP}^9$.
  - The exceptional divisor $E=\mathbb P(N)$ is an $\mathbb{RP}^5$-bundle.
    Its fibre class lives in the local system of $w_1(N)$, and
    $w_1(N) = w_1(T\mathbb{RP}^9)|-w_1(T\mathbb{RP}^3) = 10a|-4a = 0$. So
    $H_*(E;\mathbb Q)=H_*(\mathbb{RP}^3)\otimes H_*(\mathbb{RP}^5)$, in degrees
    $0,3,5,8$ (no room for differentials).
  - The sequence of $(Y,E)$ with $H_*(Y,E)\cong H_*(\mathbb{RP}^9,\mathbb{RP}^3)$
    (excision): $\partial\colon H_4(Y,E)\to H_3(E)$ maps onto the class over
    $[\mathbb{RP}^3]$, so $H_3(Y)=H_4(Y)=0$. $Y$ is non-orientable
    ($w_1(Y)=5e=e\ne0$, since $e|_E$ is the tautological class), so
    $H_9(Y)=0$, $\partial\colon H_9(Y,E)\to H_8(E)$ is an isomorphism, and
    $H_8(Y)=0$. Finally $H_5(E)\cong H_5(Y)$, generated by a fibre
    $\mathbb{RP}^5$ of $E$.
  - So $H_*(Y;\mathbb Q)=\mathbb Q_0\oplus\mathbb Q_5$.
- **Second blow-up**, along $W(\mathbb R)$ of codimension 3. The exceptional
  divisor is an $\mathbb{RP}^2$-bundle over $W(\mathbb R)$, rationally a copy
  of $W(\mathbb R)$, and the five lemma on the blow-up square gives
  $H_*(X(\mathbb R);\mathbb Q)\cong H_*(Y;\mathbb Q)$, as for complete conics.

Test against the rules (`tests/test_h1.py`): at random generic cocharacters
the direct rule of §2 on the Brion curves admits no sign completion, as for
conics. But the data are much richer: 66 cells, 24 to 26 curves between cells
of equal dimension, a few curves between adjacent dimensions whose parity is
undefined (normal weights that do not pair), and curves that go up in
dimension. A correction rule would have to be found on this example.

First attempt (negative): for complete conics the removed term joins the
top of the even equal-dimension curve to the bottom of the odd one; the two
curves lie in different $\mathbb P(\mathfrak{sl}_2)$ planes, and the removed
term runs along a curve of the closed orbit between them. The literal
generalization, removing every direct term from the top of an even
equal-dimension curve to the bottom of an odd one, reproduces the conics in
11 of 11 chambers, but for the surfaces (2 to 4 such terms) it admits no
sign completion in any of 10 chambers. The curves that go up in dimension
and the adjacent curves of undefined parity are not used by it, so a rule
will have to account for them.

## 6. Conjecture H1′ and open points

> **H1′.** Let $X$ be a smooth projective GKM variety over $\mathbb{Z}$ with a
> generic cocharacter whose BB decomposition is graded (every invariant curve
> goes down in dimension). Then the stable attaching map between cells of
> adjacent dimensions has $\eta$-component $\varepsilon_{xy}\,\eta$ with
> $\varepsilon_{xy} \in W(\mathbb{Z}) = \mathbb{Z}$, where $\varepsilon_{xy} = \pm 1$ if $x, y$ are joined by an
> invariant curve with $m$ even, and $\varepsilon_{xy} = 0$ otherwise.

The evidence is consistent in both realizations. It says nothing about the
motivic statement beyond them: the realizations detect $\varepsilon_{xy}$ mod 2
(complex) and as an integer up to gauge (real), and that is all that was tested.

Beyond the graded GKM case, §4 extends the evidence to graded non-GKM
decompositions along the curves of the Brion components.

Open points:

- **Undetermined curves.** The rule needs $\sigma(x)-\sigma(y) \in \mathbb{Z}\varphi$.
  This held in every GKM example so far; `gkm_incidences` reports `None`
  otherwise. Beyond GKM, `h1.curve_parities` pairs the normal weights
  explicitly (§4).
- **Signs.** They were determined only by $\partial^2 = 0$, which happened to be
  unique up to gauge in all examples. A sign rule from GKM data (the analogue
  of Matszangosz's $s(I,J)$) is open.
- **Beyond realizations.** A proof of H1′ would need the attaching maps in
  $SH(\mathbb{Z})$ themselves, for example via the Thom-space cell structures of
  arXiv:1805.04338 restricted to invariant curves.
- **Chow–Witt.** Done without twist (`bbcells.chowwitt`, `real … --chow-witt`,
  `real-toric … --chow-witt`), and with twists for toric varieties
  (`real-toric … --twist`); see §7. Open: twists for flag varieties
  (incidences twisted along the invariant curves on which $\mathcal L$ has odd
  degree), and the ring structure.
- **Non-graded decompositions** in dimension $\ge3$ (§§5.2–5.3), in
  particular every rank-two complete symmetric variety.

## 7. Chow–Witt groups over $\mathbb R$

For a smooth cellular $X$ over $\mathbb R$, the Chow ring has no 2-torsion, so
$\widetilde{CH}^q(X)=H^q(X,\mathbf I^q)\times_{\mathrm{Ch}^q(X)}CH^q(X)$
(Hornbostel–Wendt, as stated in Hudson–Matszangosz–Wendt, arXiv:2302.11003,
§2). The real cycle class map of Hornbostel–Wendt–Xie–Zibrowius
(arXiv:1911.04150) is an isomorphism
$H^q(X,\mathbf I^q)\cong H^q(X(\mathbb R);\mathbb Z)$ for cellular $X$, and
it is compatible with the reduction to $\mathrm{Ch}^q(X)\cong
H^q(X(\mathbb R);\mathbb Z/2)$ (Borel–Haefliger). Hence
$$\widetilde{CH}^q(X) \cong CH^q(X)\times_{H^q(X(\mathbb R);\mathbb Z/2)}H^q(X(\mathbb R);\mathbb Z).$$

- Both sides have bases indexed by the cells, and the cellular coboundaries of
  $X(\mathbb R)$ are even. The isomorphism type of the fibre product does not
  depend on how the bases correspond modulo 2, since every matrix in
  $GL(\mathbb F_2)$ lifts to $GL(\mathbb Z)$. With $Z^q$ the cocycles,
  $$\widetilde{CH}^q \cong (Z^q\oplus\mathbb Z^{c_q})\,/\,\{(\delta w,-\delta w/2)\},$$
  which `chowwitt.chow_witt_groups` computes by Smith normal form.
- Its rank is $\operatorname{rank} CH^q + b_q(X(\mathbb R);\mathbb Q)$. Its torsion is
  the torsion of $H^q(X(\mathbb R);\mathbb Z)$ that is divisible by 2, so 2-torsion
  disappears.
- Checks (`tests/test_chowwitt.py`):
  - $\widetilde{CH}^0 = GW(\mathbb R)=\mathbb Z^2$;
  - $\widetilde{CH}^q(\mathbb P^n)=\mathbb Z$ for $0<q<n$, and the top group is
    $GW(\mathbb R)$ for $n$ odd and $\mathbb Z$ for $n$ even, from both the toric
    and the flag complex;
  - the ranks on flag varieties.
- Input: the exact toric Morse complexes (§3.2) and the signed real Schubert
  complexes. Complexes predicted under H1 could be used the same way, and
  would then be conditional on H1.

### 7.1 Twisted coefficients (toric varieties)

For a divisor $D=\sum a_\rho D_\rho$ the orientation local system
$\mathbb Z(\mathcal L)$ of the real line bundle $\mathcal L=\mathcal O(D)$ is trivialized on each copy
$P\times\{e\}$ of the Davis–Januszkiewicz model by the section $s_D$, which
changes sign across the facet of $\rho$ iff $a_\rho$ is odd. So a cell
$(\tau,e)$ has one generator per representative, with
$[\tau,e+r_\rho]=(-1)^{a_\rho}[\tau,e]$ for $\rho\in\tau$, and the boundary formula of
§3.2 holds for every representative; with the reduced representatives each
incidence gets a sign. The Morse matching is unchanged
(`RealToricComplex(fan, lam, twist=a)`). The twisted Chow–Witt groups are
then computed by the fibre product of §7 with $H^q(X(\mathbb R);\mathbb Z(\mathcal L))$; that
the identifications of §7 hold with twists is assumed, and checked below.

Checks (`tests/test_realtoric.py`, `tests/test_chowwitt.py`):
- twisted Poincaré duality: with $a_\rho=1$ for all $\rho$ ($\mathcal L=\omega_X$, whose real
  points have $w_1 = w_1(X(\mathbb R))$), $H_d(X(\mathbb R);\mathbb Z(\omega))\cong H^{n-d}(X(\mathbb R);\mathbb Z)$ for
  all nine test fans, torsion included;
- only the class of $D$ modulo $2\,\mathrm{Pic}$ matters: $D+\operatorname{div}\chi^m$ gives the same
  groups, and $2D$ gives the untwisted ones;
- $H_*(\mathbb{RP}^n;\mathbb Z(\mathcal O(1)))$ is the homology with the non-trivial local system
  (e.g. $\mathbb Z/2,0,\mathbb Z$ for $n=2$);
- $\widetilde{CH}^q(\mathbb P^n,\mathcal O(1))$: $\mathbb Z$ for $q<n$, and the top group is $GW(\mathbb R)$ iff $n$
  is even, i.e. iff $\mathcal O(1)\equiv\omega$ modulo squares; and
  $\widetilde{CH}^n(X,\omega_X)=GW(\mathbb R)$ for $F_1$, $F_2$, $dP_6$ and $\mathbb P^3$.

