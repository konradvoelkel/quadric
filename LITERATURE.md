# LITERATURE — annotated overview

Status (2026-09-25): all papers cited here with an arXiv ID were downloaded
with `tools/fetch_arxiv.py`, and their titles and authors were matched
against the arXiv metadata in `literature/arxiv_metadata.json`. Entries
without an arXiv ID are checked against web search results only. Crossref and
zbMATH are still unreachable, so journal data for entries without an arXiv
journal-ref is not independently confirmed.

Markers:

- ✓a title and authors match the arXiv record; the journal data comes from
  arXiv's journal-ref where one is given;
- ✓ author, title and venue confirmed by a search result;
- ✓m from memory and very likely right, but not confirmed this round;
- **?** check before citing.

The numbers in brackets (§…) point to the relevant section of `SPEC.md`.

---

## 0. Directly on target — read first

- ✓a K. Voelkel, *Motivic cell structures for spherical varieties*,
  arXiv:1805.04338 (2018).
  A general method for unstable motivic cell structures, following Wendt's
  use of Białynicki-Birula "algebraic Morse theory". It is applied to
  spherical varieties, with special attention to rank one, giving unstable
  cell structures after finitely many $\mathbb{P}^1$-suspensions. Checked in
  the TeX source: Theorem `thm:spherical-stably-cellular` says that **every
  spherical $k$-variety is motivic stably cellular** in the sense of
  Dugger–Isaksen, with the corollary that its Voevodsky motive is mixed Tate.
  So the stable question in SPEC §3.5 is well posed for the whole target
  class. Also relevant: Theorem `thm:unstable-thom-cells` (Thom spaces over a
  base with a totally affinely contractible cover are unstably cellular, over
  a base smooth over a Dedekind ring or a field), which supports hypothesis H1;
  and the conjecture that two-orbit completable homogeneous spaces are
  unstably cellular.
- ✓ K. Voelkel, *Motivic cell structures for projective spaces over split
  quaternions*, PhD thesis, Freiburg (2016), advisor M. Wendt; available via
  FreiDok and the DNB. Quaternionic projective spaces and the Cayley plane;
  affine quadrics as motivic spheres.
- ✓a M. Wendt, *More examples of motivic cell structures*, arXiv:1012.0454.
- ✓m D. Dugger, D. Isaksen, *Motivic cell structures*, Algebr. Geom. Topol. 5
  (2005), 615–652.
- Blog posts by KV (konradvoelkel.com, 2012–2013) on cellular objects in the
  motivic model category and on the motivic cell structure of toric surfaces.
  These are useful for worked toric examples in M1.

## 1. Białynicki-Birula decomposition and motives (§3.3)

- ✓m A. Białynicki-Birula, *Some theorems on actions of algebraic groups*,
  Ann. of Math. 98 (1973), 480–497.
- **?** A. Białynicki-Birula, *Some properties of the decompositions of
  algebraic varieties determined by actions of a torus*, Bull. Acad. Polon.
  Sci. 24 (1976). On filtrability.
- ✓a P. Brosnan, *On motivic decompositions arising from the method of
  Białynicki-Birula*, Invent. Math. 161 (2005), 91–111; doi:10.1007/s00222-004-0419-7;
  arXiv:math/0407305.
- ✓m V. Chernousov, S. Gille, A. Merkurjev, *Motivic decomposition of
  isotropic projective homogeneous varieties*, Duke Math. J. 126 (2005),
  137–159.
- ✓m N. Karpenko, *Cohomology of relative cellular spaces and of isotropic
  flag varieties*, St. Petersburg Math. J. 12 (2001).
- ✓m B. Köck, *Chow motif and higher Chow theory of G/P*, Manuscripta Math. 70
  (1991), 363–372.
- ✓ B. Totaro, *Chow groups, Chow cohomology, and linear varieties*, Forum
  Math. Sigma 2 (2014), e17. For smooth projective linear varieties the motive
  is pure Tate, which justifies the table in SPEC §3.3 for all smooth
  projective spherical varieties.
- ✓ R. Joshua, *Algebraic K-theory and higher Chow groups of linear
  varieties*, Math. Proc. Cambridge Philos. Soc. 130 (2001), 37–60.
- ✓a A. Weber, *Hirzebruch class and Białynicki-Birula decomposition*,
  Transform. Groups 22 (2017), 537–557; arXiv:1411.6594. Refines BB Betti
  numbers to $\chi_y$ and Hirzebruch classes via localization.

## 2. Motivic homotopy and quadratic invariants (§3.5)

- ✓m F. Morel, V. Voevodsky, *$\mathbb{A}^1$-homotopy theory of schemes*,
  Publ. Math. IHÉS 90 (1999), 45–143.
- ✓m F. Morel, *$\mathbb{A}^1$-algebraic topology over a field*, LNM 2052
  (2012). Milnor–Witt K-theory and $\pi_{1,1} = K^{MW}_{-1} = W\cdot\eta$.
- ✓m A. Asok, B. Doran, J. Fasel, *Smooth models of motivic spheres and the
  clutching construction*, IMRN 2017.
- ✓m M. Hoyois, *A quadratic refinement of the Grothendieck–Lefschetz–Verdier
  trace formula*, Algebr. Geom. Topol. 14 (2014), 3603–3658.
- ✓m M. Levine, *Motivic Euler characteristics and Witt-valued characteristic
  classes*, Nagoya Math. J. 236 (2019), 251–310.
- ✓m M. Levine, A. Raksit, *Motivic Gauss–Bonnet formulas*, Algebra Number
  Theory 14 (2020), 1801–1851.
- ✓m M. Levine, *Aspects of enumerative geometry with quadratic forms*,
  Doc. Math. 25 (2020), 2179–2239.
- ✓a N. Arcila-Maya, C. Bethea, M. Opie, K. Wickelgren, I. Zakharevich,
  *Compactly supported $\mathbb{A}^1$-Euler characteristic and the Hochschild
  complex*, Topology Appl. (2022); arXiv:2003.09457. The source for
  $\chi_c(\mathbb{A}^n) = \langle -1\rangle^n$ and the motivic measure
  $K_0(\mathrm{Var}_k) \to GW(k)$ (SPEC C7).

## 3. Chow–Witt groups and real realization (target of SPEC §3.5 Level A)

- ✓m J. Hornbostel, M. Wendt, H. Xie, M. Zibrowius, *The real cycle class
  map*, Ann. K-Theory 6 (2021), 239–317. For cellular varieties, $I^j$-cohomology
  is identified with $H^*(X(\mathbb{R});\mathbb{Z})$.
- ✓m J. Hornbostel, M. Wendt, *Chow–Witt rings of classifying spaces for
  symplectic and special linear groups*, J. Topol. 12 (2019).
- ✓a M. Wendt, *Chow–Witt rings of Grassmannians*, Algebr. Geom. Topol. 24
  (2024); arXiv:1805.06142.
- ✓a T. Hudson, Á. Matszangosz, M. Wendt, *Chow–Witt rings and topology of flag
  varieties*, J. Topol. 17 (2024); arXiv:2302.11003. Witt-sheaf cohomology of
  type A partial flag varieties; all torsion in $H^*(\mathrm{Fl}(\mathbb{R});\mathbb{Z})$
  is 2-torsion. **The main test oracle for M5.**
- ✓ R. R. Kocherlakota, *Integral homology of real flag manifolds and loop
  spaces of symmetric spaces*, Adv. Math. 110 (1995), 1–46. (Also confirmed
  by the bibliography of arXiv:1910.11149.)
- ✓a Á. K. Matszangosz, *On the cohomology rings of real flag manifolds:
  Schubert cycles*, arXiv:1910.11149. Incidence coefficients of real Schubert
  cells in type A partial flag manifolds **with signs**:
  $[\Omega_I,\Omega_J] = 0$ or $(-1)^{s(I,J)}2$ according to the parity of
  $N_I(a,b)$. **The algorithm for PLAN S5.1** (implemented in
  `bbcells/realcells.py`). Two caveats found while implementing: the worked
  sign example ($I = (4,5,6,1,2,3)$, $J = (4,2,6,1,5,3)$) has $N_I(a,b) = 2$,
  so its incidence is 0, not $+2$ (Kocherlakota's rule agrees:
  $\sigma(I)-\sigma(J) = 3e_{25}$); and the identity $m = N_I(a,b)+1$ used
  to rederive Kocherlakota's theorem holds for complete flags only.
- ✓ L. Rabelo, L. A. B. San Martin, *Cellular homology of real flag
  manifolds*, Indag. Math. 30 (2019), 745–772 (from the bibliography of
  arXiv:1910.11149). Signs for general real flag manifolds; PLAN S5.4. Real Bruhat cell
  incidences are $0$ or $\pm 2$; the model for the matrices $E_d$ in SPEC §3.5.
- ✓a L. Casian, Y. Kodama, *On the cohomology of real Grassmann manifolds*,
  arXiv:1309.5520. Explicit incidence graphs via checkered Young diagrams.
- ✓a S. Choi, H. Park, *On the cohomology and their torsion of real toric
  objects*, Forum Math. (2017); arXiv:1311.7056. Real toric manifolds; odd
  torsion occurs, so $\mathbb{Z}$ must be tracked and not only $\mathbb{F}_2$.
- ✓ A. Suciu, A. Trevisan, *Real toric varieties and abelian covers of
  generalized Davis–Januszkiewicz spaces* (preprint, 2012). Rational Betti
  numbers of real toric varieties.
- ✓a M. Franz, *The cohomology rings of real toric spaces and smooth real toric
  varieties*, Proc. Roy. Soc. Edinburgh Sect. A 152 (2022), 720–737;
  arXiv:2008.08961.

## 4. Spherical varieties: foundations

- ✓m D. Luna, T. Vust, *Plongements d'espaces homogènes*, Comment. Math. Helv.
  58 (1983), 186–245.
- ✓m F. Knop, *The Luna–Vust theory of spherical embeddings*, Proc. Hyderabad
  Conf. on Algebraic Groups (1991), 225–249.
- ✓m D. Timashev, *Homogeneous spaces and equivariant embeddings*, Encyclopaedia
  Math. Sci. 138, Springer 2011.
- ✓a N. Perrin, *On the geometry of spherical varieties*, Transform. Groups 19
  (2014), 171–223; arXiv:1211.1277.
- ✓ G. Pezzini, *Lectures on spherical and wonderful varieties*, Les cours du
  CIRM 1 (2010), 33–53.
- ✓m F. Knop, *On the set of orbits for a Borel subgroup*, Comment. Math. Helv.
  70 (1995), 285–309.
- ✓a F. Knop, *Spherical roots of spherical varieties*, Ann. Inst. Fourier 64
  (2014), 2503–2526; arXiv:1303.2466. Contains Akhiezer's classification of
  rank-one spherical varieties extended to all characteristics $\neq 2$,
  with a **table of cuspidal rank-one spherical varieties for adjoint
  groups** (§`sec:TABLE`), which is the case list for PLAN S6b.1.
- **?** D. Akhiezer, *Equivariant completions of homogeneous algebraic
  varieties by homogeneous divisors*, Ann. Global Anal. Geom. 1 (1983). The
  original rank-one list, cited as [Ahi83] in arXiv:1805.04338; venue from
  memory.
- ✓a F. Knop, *Localization of spherical varieties*, Algebra Number Theory 8
  (2014), 703–728; arXiv:1303.2561.
- ✓a G. Gagliardi, *A combinatorial smoothness criterion for spherical
  varieties*, Manuscripta Math. 146 (2015), 445–461; arXiv:1307.7702.
  **Needed for input validation in M6**: decide smoothness from the colored
  fan before running BB.
- ✓a G. Gagliardi, J. Hofscheier, H. Pearson, *Toricness and smoothness
  criteria for spherical varieties*, arXiv:2601.06376 (2026).
- ✓a G. Gagliardi, J. Hofscheier, H. Pearson, *The generalised Mukai conjecture
  for spherical varieties*, arXiv:2502.21155 (2025).
- ✓a K. Altmann, V. Kiritchenko, L. Petersen, *Merging divisorial with colored
  fans*, Michigan Math. J. 64 (2015), 3–38; arXiv:1210.4523.
- ✓a J. Tevelev, T. Vogiannou, *Spherical tropicalization*, arXiv:1511.02203.

**Classification by spherical systems**

- ✓m D. Luna, *Variétés sphériques de type A*, Publ. Math. IHÉS 94 (2001),
  161–226.
- ✓m I. Losev, *Proof of the Knop conjecture*, Ann. Inst. Fourier 59 (2009);
  I. Losev, *Uniqueness property for spherical homogeneous spaces*, Duke
  Math. J. 147 (2009).
- ✓a P. Bravi, G. Pezzini, *Wonderful subgroups of reductive groups and
  spherical systems*, J. Algebra 409 (2014), 101–147; arXiv:1103.0380.
- ✓a P. Bravi, G. Pezzini, *Primitive wonderful varieties*, Math. Z. (2016);
  arXiv:1106.3187.
- ✓a P. Bravi, G. Pezzini, *Wonderful varieties of type D*, Represent. Theory 9
  (2005); arXiv:math/0410472.
- ✓a P. Bravi, *Primitive spherical systems*, arXiv:0909.3765 (journal version
  to check).
- P. Bravi, G. Pezzini, *Wonderful varieties of type B and C* (0909.3771) was
  **withdrawn** by the authors, as superseded by 1103.0380, 1106.3187 and
  *The spherical systems of the wonderful reductive subgroups*,
  arXiv:1109.6777. Cite those instead.
- ✓a S. Cupit-Foutou, *Wonderful varieties: a geometrical realization*,
  arXiv:0907.2852.
- ✓a R. Avdeev, *On computing the spherical roots for a class of spherical
  subgroups*, arXiv:2604.07056 (2026).

## 5. Cohomology and cells of special spherical classes (SPEC §3.6)

**Wonderful and complete symmetric varieties (M4)**

- ✓m C. De Concini, C. Procesi, *Complete symmetric varieties*, LNM 996 (1983),
  1–44; and *Complete symmetric varieties II*, Adv. Stud. Pure Math. 6 (1985),
  481–513.
- ✓ C. De Concini, T. A. Springer, *Betti numbers of complete symmetric
  varieties*, Geometry Today (Roma 1984), Progr. Math. 60 (1985), 87–104.
  **Main reference and test oracle for M4.**
- ✓m E. Bifet, C. De Concini, C. Procesi, *Cohomology of regular embeddings*,
  Adv. Math. 82 (1990), 1–34.
- ✓ E. Strickland, *Equivariant cohomology of the wonderful group
  compactification*, J. Algebra (2006).
- ✓a M. Brion, R. Joshua, *Equivariant Chow ring and Chern classes of wonderful
  symmetric varieties of minimal rank*, Transform. Groups 13 (2008), 471–493;
  arXiv:0705.1035.

**General smooth projective spherical varieties**

- ✓a M. Brion, *Equivariant cohomology and equivariant intersection theory*,
  Montréal lectures 1997; arXiv:math/9802063. GKM-type description of
  $H_T^*$ for spherical varieties; **the main reference for M3 and M6**.
- ✓m M. Brion, *Rational smoothness and fixed points of torus actions*,
  Transform. Groups 4 (1999), 127–156.
- ✓a S. Banerjee, M. B. Can, *Equivariant K-theory of smooth projective
  spherical varieties*, arXiv:1603.04926.
- ✓a R. Gonzales, *Algebraic rational cells and equivariant intersection
  theory*, Math. Z. 282 (2016), 79–97; arXiv:1404.2486.

**Horospherical (M6a)**

- ✓m B. Pasquier, *Variétés horosphériques de Fano*, Bull. Soc. Math. France
  136 (2008), 195–225.
- ✓a S. Monahan, *Horospherical stacks and stacky coloured fans*, Trans. AMS 378
  (2025), 1167–1214; arXiv:2305.01571.
- ✓a V. Batyrev, A. Moreau, *The arc space of horospherical varieties and
  motivic integration*, Compositio Math. (to appear per arXiv comments);
  arXiv:1203.0671. For a $\mathbb{Q}$-Gorenstein horospherical $X$ with
  colored fan $\Sigma$:
  $E_{st}(X;u,v) = E(G/H;u,v)\sum_{n\in|\Sigma|\cap N}(uv)^{\omega_X(n)}$,
  computed through a weighted Stanley–Reisner ring. For smooth $X$ this is
  the E-polynomial, so **it is a closed formula for the Betti numbers
  straight from the colored fan and the test oracle for M6a**.
- ✓a J. Hofscheier, A. Khovanskii, L. Monin, *Cohomology rings of toric bundles
  and the ring of conditions*, arXiv:2006.12043. BKK-type description of
  $H^*$ of toric bundles and of the ring of conditions of horospherical
  homogeneous spaces.
- ✓m P. Sankaran, V. Uma, *Cohomology of toric bundles*, Comment. Math. Helv.
  78 (2003). The Stanley–Reisner description of $H^*$ of toric bundles, which
  covers smooth toroidal horospherical varieties.

## 6. Toric varieties and GKM theory (M1, M3)

- ✓m V. Danilov, *The geometry of toric varieties*, Russian Math. Surveys 33
  (1978), 97–154.
- ✓m W. Fulton, *Introduction to toric varieties*, Ann. of Math. Stud. 131
  (1993). §5.2: Betti numbers from the $h$-vector.
- ✓m D. Cox, J. Little, H. Schenck, *Toric varieties*, GSM 124 (2011).
- ✓a M. Goresky, R. Kottwitz, R. MacPherson, *Equivariant cohomology, Koszul
  duality, and the localization theorem*, Invent. Math. 131 (1998), 25–83.

## 7. Existing software (quick search, 2026-09)

| Tool | Scope | Our use |
|---|---|---|
| SageMath `ToricVariety` (Braun, Novoseltsev) | fans, cohomology and Chow rings, Chern classes | optional cross-check (SPEC D1) |
| Macaulay2 `NormalToricVarieties` (G. Smith) | same, plus sheaf cohomology | cross-check by hand |
| OSCAR (Julia) | toric geometry | cross-check by hand |
| SageMath root systems / Weyl groups | Bruhat order, lengths | optional cross-check |
| PALP, cohomCalg | Hodge numbers of CY hypersurfaces; line-bundle cohomology | out of scope |
| **spherical varieties** | **no general package found** | ad hoc code in classification papers |

## 8. Getting the papers

`python3 tools/fetch_arxiv.py` downloads PDFs and TeX sources of every paper
cited in this file as `arXiv:ID` into `literature/cache/`, which is
git-ignored (arXiv's licence lets arXiv distribute the papers, not us), and
writes metadata to `literature/arxiv_metadata.json`, which is committed.
`--list` prints the IDs; `--metadata-only` skips the downloads.

Notes from the first run (2026-09-25):
- `arxiv.org` and `export.arxiv.org` are allowed in this environment now.
  The API endpoint `export.arxiv.org/api/query` answers 406 to requests from
  here, so the script takes the metadata from the abstract pages
  (`"metadata_source": "abs-page"` in the JSON).
- 30 of 31 papers were downloaded (about 24 MB). The one failure is the
  withdrawn 0909.3771, which has no PDF.
- `api.crossref.org`, `zbmath.org`, `api.openalex.org` and
  `api.semanticscholar.org` are still blocked. Allowing Crossref or zbMATH
  would let us check journal data (volumes, pages, DOIs) automatically.
