# LITERATURE — annotated overview

Bibliographic details were written from memory and from a quick web search on
2026-09-25; arXiv could not be reached from the build environment. Markers:

- ✓ confident about author, title and venue;
- **?** check before citing: the venue, year or exact title may be off, or the
  reference was only seen in search snippets.

The numbers in brackets (§…) point to the relevant section of `SPEC.md`.

---

## 1. Białynicki-Birula decomposition and motives (§3.3)

- ✓ A. Białynicki-Birula, *Some theorems on actions of algebraic groups*,
  Ann. of Math. 98 (1973), 480–497.
  The decomposition into plus-cells for smooth projective $X$ with a
  $\mathbb{G}_m$-action; cells are affine bundles over the fixed components.
- ? A. Białynicki-Birula, *Some properties of the decompositions of algebraic
  varieties determined by actions of a torus*, Bull. Acad. Polon. Sci. 24
  (1976). Filtrability of the decomposition.
- ✓ P. Brosnan, *On motivic decompositions arising from the method of
  Białynicki-Birula*, Invent. Math. 161 (2005), 91–111.
  Chow motive of $X$ as a sum of Tate twists of the motives of the fixed
  components.
- ✓ V. Chernousov, S. Gille, A. Merkurjev, *Motivic decomposition of isotropic
  projective homogeneous varieties*, Duke Math. J. 126 (2005), 137–159.
- ✓ N. Karpenko, *Cohomology of relative cellular spaces and of isotropic flag
  varieties*, Algebra i Analiz 12 (2000); English translation in
  St. Petersburg Math. J. 12 (2001).
- ✓ B. Köck, *Chow motif and higher Chow theory of G/P*, Manuscripta Math. 70
  (1991), 363–372.
- ✓ B. Totaro, *Chow groups, Chow cohomology, and linear varieties*,
  Forum Math. Sigma 2 (2014), e17.
  For smooth projective linear varieties, in particular spherical ones, the
  cycle class map is an isomorphism and the motive is pure Tate. This is the
  reason the table in SPEC §3.3 applies to *all* smooth projective spherical
  varieties, not only those with a known BB decomposition.
- ? R. Joshua, *Algebraic K-theory and higher Chow groups of linear varieties*,
  Math. Proc. Cambridge Philos. Soc. 130 (2001).

## 2. Motivic homotopy, cell structures, quadratic invariants (§3.5)

- ✓ F. Morel, V. Voevodsky, *$\mathbb{A}^1$-homotopy theory of schemes*,
  Publ. Math. IHÉS 90 (1999), 45–143.
- ✓ F. Morel, *$\mathbb{A}^1$-algebraic topology over a field*, LNM 2052,
  Springer 2012. Milnor–Witt K-theory; $\pi_{1,1}$ and the role of $\eta$.
- ✓ D. Dugger, D. Isaksen, *Motivic cell structures*, Algebr. Geom. Topol. 5
  (2005), 615–652. The definition of cellular objects in $SH(k)$. Varieties
  with a filtrable affine stratification are stably cellular.
- ✓ M. Wendt, *More examples of motivic cell structures*, arXiv:1012.0454.
  BB and Bruhat-type cell structures in the motivic homotopy category.
- ✓ A. Asok, B. Doran, J. Fasel, *Smooth models of motivic spheres and the
  clutching construction*, IMRN 2017. Affine quadrics $Q_{2n}$ as motivic
  spheres; relevant to the TODO in `quadric.py` about $AQ_n$.
- TODO(KV): add your own papers and thesis on motivic cell structures (e.g.
  quadrics and projective spaces over split quaternions) with exact data.

**Quadratic Euler characteristics**

- ✓ M. Hoyois, *A quadratic refinement of the Grothendieck–Lefschetz–Verdier
  trace formula*, Algebr. Geom. Topol. 14 (2014), 3603–3658.
- ✓ M. Levine, *Motivic Euler characteristics and Witt-valued characteristic
  classes*, Nagoya Math. J. 236 (2019), 251–310.
- ✓ M. Levine, A. Raksit, *Motivic Gauss–Bonnet formulas*, Algebra Number
  Theory 14 (2020), 1801–1851.
- ✓ M. Levine, *Aspects of enumerative geometry with quadratic forms*,
  Doc. Math. 25 (2020), 2179–2239. Includes $\chi^{\mathbb{A}^1}$ of cellular
  varieties.
- ? N. Arcila-Maya, C. Bethea, M. Opie, K. Wickelgren, I. Zakharevich,
  *Compactly supported $\mathbb{A}^1$-Euler characteristic and the Hochschild
  complex*, Topology Appl. (2022?). $\chi_c(\mathbb{A}^n) = \langle -1\rangle^n$
  (SPEC C7).

**Chow–Witt groups and real realization (target of §3.5 Level A)**

- ✓ J. Hornbostel, M. Wendt, H. Xie, M. Zibrowius, *The real cycle class map*,
  Ann. K-Theory 6 (2021), 239–317. For cellular varieties, the $I^j$-cohomology
  is identified with the singular cohomology of $X(\mathbb{R})$, which
  determines $\widetilde{CH}^*$ from $CH^*$ and $H^*(X(\mathbb{R});\mathbb{Z})$.
- ✓ J. Hornbostel, M. Wendt, *Chow–Witt rings of classifying spaces for
  symplectic and special linear groups*, J. Topol. 12 (2019).
- ? M. Wendt, *Chow–Witt rings of Grassmannians*, arXiv 2018; later
  Algebr. Geom. Topol.?
- ? R. Kocherlakota, *Integral homology of real flag manifolds and loop spaces
  of symmetric spaces*, Adv. Math. 110 (1995), 1–46. Incidence numbers
  $0, \pm 2$ of real Bruhat cells; the model for $E_d$ in SPEC §3.5.
- ? L. Casian, Y. Kodama, *On the cohomology of real Grassmann manifolds*,
  arXiv ~2013. Explicit incidence graphs.
- ? Real toric varieties: A. Suciu, A. Trevisan, and S. Choi, H. Park have
  papers on the cohomology of real toric varieties; exact references to be
  added.

## 3. Spherical varieties: foundations (§3.6 "spherical")

- ✓ D. Luna, T. Vust, *Plongements d'espaces homogènes*, Comment. Math. Helv.
  58 (1983), 186–245. Classification of embeddings of $G/H$ by colored fans.
- ✓ F. Knop, *The Luna–Vust theory of spherical embeddings*, Proc. Hyderabad
  Conf. on Algebraic Groups (1991), 225–249. The standard modern account.
- ✓ D. Timashev, *Homogeneous spaces and equivariant embeddings*, Encyclopaedia
  Math. Sci. 138, Springer 2011. Reference monograph.
- ✓ N. Perrin, *On the geometry of spherical varieties*, Transform. Groups 19
  (2014), 171–223. Survey.
- ? G. Pezzini, *Lectures on spherical and wonderful varieties*, Les cours du
  CIRM 1 (2010).
- ? M. Brion, *Variétés sphériques*, lecture notes (1997). On B-orbits, local
  structure, and fixed points.
- ✓ F. Knop, *On the set of orbits for a Borel subgroup*, Comment. Math. Helv.
  70 (1995), 285–309. Finitely many B-orbits and the weak order; relevant to
  counting points via B-orbits.
- ? M. Brion, *On orbit closures of spherical subgroups in flag varieties*,
  Comment. Math. Helv. 76 (2001).

**Classification: spherical systems and wonderful varieties**

- ✓ D. Luna, *Variétés sphériques de type A*, Publ. Math. IHÉS 94 (2001),
  161–226.
- ✓ I. Losev, *Proof of the Knop conjecture*, Ann. Inst. Fourier 59 (2009).
  Also Losev, *Uniqueness property for spherical homogeneous spaces*,
  Duke Math. J. 147 (2009).
- ✓ P. Bravi, G. Pezzini, *Primitive wonderful varieties*, Math. Z. 282 (2016).
  arXiv:1106.3187. Completes the existence part of Luna's conjecture.
- ✓ P. Bravi, G. Pezzini, *Wonderful subgroups of reductive groups and
  spherical systems*, J. Algebra (2014?), arXiv:1103.0380.
- ✓ S. Cupit-Foutou, *Wonderful varieties: a geometrical realization*,
  arXiv:0907.2852.
- ✓ P. Bravi, G. Pezzini, *Wonderful varieties of type D*, Represent. Theory 9
  (2005), arXiv:math/0410472. P. Bravi, S. Cupit-Foutou, *Classification of
  strict wonderful varieties* / *Wonderful varieties of type B and C*
  (arXiv:0909.3771)?
- ? Recent: *Toricness and smoothness criteria for spherical varieties*,
  arXiv:2601.06376; *On computing the spherical roots for a class of
  spherical subgroups*, arXiv:2604.07056. Authors not yet checked. These could
  be the closest to an algorithmic treatment and are **the first to read for
  M6**.
- ? G. Gagliardi, *The Cox ring of a spherical embedding*, J. Algebra (2014);
  G. Gagliardi, J. Hofscheier, *The generalized Mukai conjecture for
  symmetric varieties* (Trans. AMS 2017?) and *… for spherical varieties*,
  arXiv:2502.21155 (the last one possibly with other authors).
- ? *Merging divisorial with colored fans*, arXiv:1210.4523. Complexity-one
  point of view.
- ? *Spherical tropicalization*, arXiv:1511.02203 (Nash? Kaveh–Manon?).

## 4. Cohomology and cells of special spherical classes (§3.6 L1/L2)

**Wonderful and complete symmetric varieties**

- ✓ C. De Concini, C. Procesi, *Complete symmetric varieties*, in Invariant
  Theory (Montecatini 1982), LNM 996 (1983), 1–44.
- ✓ C. De Concini, C. Procesi, *Complete symmetric varieties II. Intersection
  theory*, Adv. Stud. Pure Math. 6 (1985), 481–513.
- ? C. De Concini, T. A. Springer, *Betti numbers of complete symmetric
  varieties*, in Geometry Today, Progr. Math. 60 (1985). The Poincaré
  polynomial via a BB decomposition; **main reference for M4**.
- ✓ E. Bifet, C. De Concini, C. Procesi, *Cohomology of regular embeddings*,
  Adv. Math. 82 (1990), 1–34.
- ? E. Strickland, *Equivariant cohomology of the wonderful group
  compactification*, J. Algebra 306 (2006).
- ? M. Brion, R. Joshua, *Equivariant Chow ring and Chern classes of wonderful
  symmetric varieties of minimal rank*, Transform. Groups 13 (2008).

**Fixed points, rational smoothness, rational cells**

- ✓ M. Brion, *Rational smoothness and fixed points of torus actions*,
  Transform. Groups 4 (1999), 127–156.
- ? M. Brion, *Equivariant cohomology and equivariant intersection theory*,
  notes (1998), arXiv:math/9802063. GKM for spherical varieties.
- ? R. Gonzales, *Algebraic rational cells, equivariant intersection theory,
  and Poincaré duality*, preprint (IHÉS; PDF on his homepage). Rational
  cells; group embeddings and spherical varieties.
- ? R. Gonzales, *Equivariant cohomology of rationally smooth group
  embeddings*, Transform. Groups (2014?).
- ? *Localization of spherical varieties*, arXiv:1303.2561. Reduction via the
  BB decomposition of a $\mathbb{G}_m$-variety.
- ? *Hirzebruch class and Białynicki-Birula decomposition*, arXiv:1411.6594
  (Weber?). Refines Betti numbers to $\chi_y$ and Hirzebruch classes.

**Horospherical**

- ✓ B. Pasquier, *Variétés horosphériques de Fano*, Bull. Soc. Math. France 136
  (2008), 195–225.
- ? Cohomology of smooth projective horospherical varieties from colored fans:
  a Stanley–Reisner-type description in the toroidal case. Seen in search
  snippets; exact reference to be found.
- ? *Horospherical stacks and stacky coloured fans*, arXiv:2305.01571.

## 5. Toric varieties (§3.6 "toric")

- ✓ V. Danilov, *The geometry of toric varieties*, Russian Math. Surveys 33
  (1978), 97–154.
- ✓ W. Fulton, *Introduction to toric varieties*, Ann. of Math. Stud. 131
  (1993). §5.2 computes Betti numbers via the $h$-vector; this is the BB
  decomposition in disguise.
- ✓ D. Cox, J. Little, H. Schenck, *Toric varieties*, GSM 124, AMS 2011.
- ✓ V. Batyrev, *Dual polyhedra and mirror symmetry for Calabi–Yau
  hypersurfaces in toric varieties*, J. Algebraic Geom. 3 (1994).
  Hodge numbers of hypersurfaces; out of scope for us.
- ? *Betti numbers of toric varieties and eulerian polynomials*,
  arXiv:1009.1817.

## 6. GKM theory (§3.4)

- ✓ M. Goresky, R. Kottwitz, R. MacPherson, *Equivariant cohomology, Koszul
  duality, and the localization theorem*, Invent. Math. 131 (1998), 25–83.
- ? T. Holm, J. Tymoczko and others: surveys on GKM combinatorics.

---

## 7. Existing software (as of 2026-09, from a quick search)

| Tool | Scope | Relevance |
|---|---|---|
| SageMath `ToricVariety` / `CPRFanoToricVariety` (Braun, Novoseltsev) | fans, cohomology and Chow rings, Chern and Todd classes, Betti numbers | cross-check for the toric front end |
| Macaulay2 `NormalToricVarieties` (G. Smith) | same, plus sheaf cohomology | cross-check |
| OSCAR (Julia) toric geometry | same | cross-check |
| polymake | polytopes and fans | fan input |
| PALP (Kreuzer–Skarke) | reflexive polytopes, Hodge numbers of CY hypersurfaces | out of scope |
| cohomCalg (Blumenhagen–Jurke–Rahn–Roschy) | line-bundle cohomology on toric varieties | out of scope |
| SageMath root systems / Weyl groups, LiE, CHEVIE | Weyl groups, Bruhat order | cross-check for `rootsystem.py` |
| **Spherical varieties** | **no general package found** | ad hoc code exists in classification papers (Bravi–Pezzini and others); ask the authors |

## 8. Reading order for the milestones

- **M1/M2:** Białynicki-Birula (1973), Brosnan, Fulton §5.2, Köck, and
  Bourbaki Lie IV–VI for conventions.
- **M3:** Goresky–Kottwitz–MacPherson; Brion's equivariant cohomology notes.
- **M4:** De Concini–Procesi I and II, De Concini–Springer, Bifet–De
  Concini–Procesi, Strickland.
- **M5:** Dugger–Isaksen, Wendt (1012.0454), Hornbostel–Wendt–Xie–Zibrowius,
  Kocherlakota, Levine (2020).
- **M6:** Knop (1991), Timashev, Perrin, Pasquier, then the 2026 arXiv
  preprints listed in §3.
