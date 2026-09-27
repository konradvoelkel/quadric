"""
Command line interface.

    python3 -m bbcells toric --named P3
    python3 -m bbcells flag E6 --parabolic 1
    python3 -m bbcells quadric 4 --cocharacter=-1,-2,-3
    python3 -m bbcells wonderful A2
    python3 -m bbcells example complete-conics
    python3 -m bbcells two-orbit HP 2
    python3 -m bbcells spherical complete-quadrics 5
    python3 -m bbcells symmetric AIII 2 3
    python3 -m bbcells symmetric AIII 2 5 --certify symmetry
    python3 -m bbcells symmetric EVIII --counts-only --method orbits --processes 4
    python3 -m bbcells real A3 --parabolic 2
    python3 -m bbcells real-toric --named P1xF1
    python3 -m bbcells flag A3 --real-prediction
    python3 -m bbcells characteristic 3 --tangency     # 3264 conics
    python3 -m bbcells toric fan.json --cocharacter 1,5,25 --format latex
    python3 -m bbcells raw fixed_points.json --format json
"""

import argparse
import re
import sys

from bbcells.core import bb_cells
from bbcells.output import render


def _parse_vector(text):
    try:
        return tuple(int(x) for x in text.replace(" ", "").split(",") if x)
    except ValueError:
        raise argparse.ArgumentTypeError("expected integers separated by commas: %r" % text)


def _named_fan(spec):
    from bbcells.frontends import toric
    match = re.fullmatch(r"P(\d+)", spec)
    if match:
        return toric.projective_space(int(match.group(1)))
    match = re.fullmatch(r"F(\d+)", spec)
    if match:
        return toric.hirzebruch(int(match.group(1)))
    if spec == "dP6":
        return toric.del_pezzo_6()
    if "x" in spec:
        factors = [_named_fan(part) for part in spec.split("x")]
        result = factors[0]
        for factor in factors[1:]:
            result = toric.product(result, factor)
        return result
    raise ValueError("unknown named fan %r (try P<n>, F<a>, dP6, or products like P1xP2)"
                     % spec)


def _data_toric(args):
    from bbcells.frontends import toric
    if (args.file is None) == (args.named is None):
        raise ValueError("give either a fan file or --named")
    fan = toric.load(args.file) if args.file else _named_fan(args.named)
    return toric.fixed_point_data(fan)


def _data_flag(args):
    from bbcells.frontends import flag
    return flag.flag_variety(args.cartan_type, set(args.parabolic) if args.parabolic else None)


def _data_quadric(args):
    from bbcells.frontends import quadric
    return quadric.quadric(args.n)


def _data_wonderful(args):
    from bbcells.frontends import wonderful
    return wonderful.wonderful_compactification(args.cartan_type)


EXAMPLES = {
    "complete-conics": ("bbcells.frontends.complete_conics", "complete_conics"),
    "complete-quadrics-p3": ("bbcells.frontends.complete_conics", "complete_quadrics_p3"),
}


def _data_example(args):
    import importlib
    if args.name not in EXAMPLES:
        raise ValueError("unknown example %r; available: %s"
                         % (args.name, ", ".join(sorted(EXAMPLES))))
    module, function = EXAMPLES[args.name]
    return getattr(importlib.import_module(module), function)()


SPHERICAL = {
    "complete-quadrics": "complete_quadrics",
    "complete-skew-forms": "complete_skew_forms",
}


def _data_spherical(args):
    from bbcells.frontends import spherical
    return getattr(spherical, SPHERICAL[args.family])(args.n)


def _run_symmetric_counts(args):
    import json
    from bbcells.frontends import symmetric
    from bbcells.invariants import Invariants
    if args.method == "orbits":
        D = symmetric.diagram(args.kind, *args.parameters)
        if D.black or any(D.eps[i] != i for i in D.white):
            raise ValueError("--method orbits needs a split form (no black nodes, no arrows)")
        counts = symmetric.split_cell_counts(D.R.name, processes=args.processes)
    else:
        counts = symmetric.cell_counts(args.kind, *args.parameters, processes=args.processes)
    inv = Invariants(len(counts) - 1, counts)
    if args.format == "json":
        print(json.dumps({"name": "complete symmetric variety %s%s" % (args.kind, tuple(args.parameters)),
                          "dim": inv.dim, "counts": list(counts),
                          "gw_euler": {"plus": inv.gw_euler.plus, "minus": inv.gw_euler.minus}},
                         indent=1))
        return 0
    print("complete symmetric variety %s %s (dimension %d, %d fixed points; counts only)"
          % (args.kind, " ".join(map(str, args.parameters)), inv.dim, sum(counts)))
    print("Poincare polynomial: %s" % inv.poincare.format("t"))
    print("class in K_0(Var): %s" % inv.k0_class.format("L"))
    gw = inv.gw_euler
    print("chi^A1 in GW: %s  (rank %d, signature %d = chi(X(R)))" % (gw, gw.rank, gw.signature))
    return 0


def _data_symmetric(args):
    from bbcells.frontends import symmetric
    if args.fan is None and not args.cover:
        return symmetric.complete_symmetric_variety(args.kind, *args.parameters)
    import json
    from bbcells.frontends import toroidal
    D = symmetric.diagram(args.kind, *args.parameters)
    lattice = D.lattice() if args.cover else None
    if args.fan is not None:
        with open(args.fan) as handle:
            cones = json.load(handle)["cones"]
    else:
        rows = toroidal.lattice_in_sigma(D.R.name, D.spherical_roots, lattice)
        cones = toroidal.resolve(toroidal.orthant(len(D.spherical_roots), rows), rows)
    name = "toroidal %s%s" % ("G/G^theta, " if args.cover else "", D.name)
    if args.cover:
        name += " (fan %s)" % json.dumps([[list(v) for v in cone] for cone in cones])
    return toroidal.toroidal_variety(D.R.name, D.spherical_roots, D.satellite, cones,
                                     parabolic=tuple(sorted(D.black)), strict=False,
                                     name=name, lattice=lattice)


def _two_orbit_case(args):
    from bbcells.frontends import two_orbit
    family = args.family.upper()
    fixed = {"OP2": two_orbit.octonionic_projective_plane, "G2": two_orbit.g2_on_p6,
             "SPIN7": two_orbit.spin7_on_p7}
    if family in fixed:
        return fixed[family]()
    if args.n is None:
        raise ValueError("%s needs a dimension parameter n" % args.family)
    builders = {"AQ": two_orbit.affine_quadric, "HP": two_orbit.quaternionic_projective_space,
                "PGL": two_orbit.pgl_mod_gl, "PQ": two_orbit.projective_space_minus_quadric}
    if family not in builders:
        raise ValueError("unknown family %r; use AQ, PQ, HP, PGL (with n) or OP2, G2, SPIN7"
                         % args.family)
    return builders[family](args.n)


def _run_two_orbit(args):
    import json
    from bbcells.output import to_json_dict
    case = _two_orbit_case(args)
    if args.format == "json":
        k0, gw = case.k0_class(), case.gw_euler_compact_support()
        print(json.dumps({"name": case.name, "k0_class_L": list(k0.coefficients),
                          "gw_euler_compact_support": {"plus": gw.plus, "minus": gw.minus},
                          "open_fixed_points": list(case.open_fixed_points),
                          "completion": to_json_dict(case.completion_cells(), checks=False),
                          "boundary": to_json_dict(case.boundary_cells(), checks=False)},
                         indent=1))
    else:
        print(case.summary())
    return 0


def _run_real(args):
    import json
    from bbcells import realcells
    X = realcells.real_flag_variety(args.cartan_type,
                                    set(args.parabolic) if args.parabolic else None)
    if not X.signed:
        nonzero = sum(1 for v in X.incidences.values() if v)
        print("%s: %d adjacent pairs, %d with incidence +-2 (Kocherlakota, unsigned); "
              "the signs are not determined by d o d = 0 here" % (X.name, len(X.incidences),
                                                                 nonzero))
        return 0
    cohomology = X.cohomology()
    if args.chow_witt:
        return _print_chow_witt(X.name, X, args.format)
    if args.format == "json":
        print(json.dumps({"name": X.name, "cohomology": [
            {"degree": c, "free": free, "torsion": torsion}
            for c, (free, torsion) in enumerate(cohomology)]}, indent=1))
        return 0
    source = {"cooriented": "Matszangosz, arXiv:1910.11149",
              "cellular": "Kocherlakota magnitudes; signs from Rabelo-San Martin "
                          "(classical types) or d o d = 0"}[X.convention]
    print("%s: H^*(X(R); Z) (%s)" % (X.name, source))
    for c, (free, torsion) in enumerate(cohomology):
        parts = (["Z^%d" % free if free > 1 else "Z"] if free else []) + \
                ["Z/%d" % t for t in torsion]
        print("  H^%d = %s" % (c, " + ".join(parts) if parts else "0"))
    return 0


def _run_symmetric_certify(args):
    """how the normal weights beyond condition (R) are decided"""
    import json
    from bbcells.frontends import spherical, symmetric
    D = symmetric.diagram(args.kind, *args.parameters)
    orbits = D.orbits(strict=False)
    unknowns = sorted((o.name, g) for o in orbits if o.note == spherical.BEYOND_R
                      for g in o.normal_roots)
    if args.certify == "symmetry":
        decided = spherical.certify_by_symmetry(D.R.name, orbits, D.spherical_roots)
        methods = {u: ("opposition symmetry", ()) for u in decided}
    else:
        methods = {}
        decided = spherical.certify_by_closures(D.R.name, orbits, methods=methods)
    rows = [{"orbit": name, "spherical_root": gamma,
             "admissible_c": decided.get((name, gamma)),
             "method": methods.get((name, gamma), ("undecided", ()))[0],
             "closure": list(methods.get((name, gamma), (None, ()))[1])}
            for name, gamma in unknowns]
    if args.format == "json":
        print(json.dumps({"name": D.name, "unknowns": rows}, indent=1))
        return 0
    if not rows:
        print("%s: condition (R) holds, no unknown normal weights" % D.name)
        return 0
    print("%s: %d unknown normal weight%s beyond condition (R), chi = pr(-gamma) + c zeta"
          % (D.name, len(rows), "" if len(rows) == 1 else "s"))
    for row in rows:
        where = " on the closure %s" % row["closure"] if row["closure"] else ""
        print("  orbit %s, gamma_%d: c in %s (%s%s)" % (row["orbit"], row["spherical_root"],
                                                      row["admissible_c"], row["method"], where))
    return 0


def _run_characteristic(args):
    from bbcells import brion, oracles
    n = args.n
    if n < 2:
        raise ValueError("n must be at least 2 (complete quadrics in P^{n-1})")
    given = [x is not None for x in (args.monomial, args.divisor)] + [args.tangency]
    if sum(given) != 1:
        raise ValueError("give exactly one of --monomial, --divisor, --tangency")
    coefficients = (2,) * (n - 1) if args.tangency else args.divisor
    if args.method == "sections":
        if coefficients is None:
            raise ValueError("--method sections needs --divisor or --tangency")
        if len(coefficients) != n - 1:
            raise ValueError("need %d coefficients" % (n - 1))
        value = oracles.complete_quadrics_degree(n, coefficients)
    else:
        value = brion.characteristic_number(n, exponents=args.monomial,
                                            coefficients=coefficients)
    print(value)
    return 0


def _run_real_toric(args):
    import json
    from bbcells.frontends import toric
    from bbcells.realtoric import RealToricComplex
    if (args.file is None) == (args.named is None):
        raise ValueError("give either a fan file or --named")
    fan = toric.load(args.file) if args.file else _named_fan(args.named)
    cells = bb_cells(toric.fixed_point_data(fan), args.cocharacter)
    X = RealToricComplex(fan, cells.cocharacter)
    if args.chow_witt:
        return _print_chow_witt(fan.name, X, args.format)
    homology, betti = X.integral_homology(), X.betti()
    if args.format == "json":
        print(json.dumps({"name": fan.name, "cocharacter": list(cells.cocharacter),
                          "incidences": [[x, y, v] for (x, y), v in sorted(X.incidences().items())],
                          "homology": homology, "rational_betti": betti}, indent=1))
        return 0
    print("%s(R): exact real BB incidences (discrete Morse theory), cocharacter %s"
          % (fan.name, ",".join(map(str, cells.cocharacter))))
    for d, summands in enumerate(homology):
        free = summands.count(0)
        parts = (["Z^%d" % free if free > 1 else "Z"] if free else []) + \
                ["Z/%d" % t for t in summands if t]
        print("  H_%d = %s" % (d, " + ".join(parts) if parts else "0"))
    print("  rational Betti numbers: %s" % ", ".join(map(str, betti)))
    return 0


def _print_chow_witt(name, complex_, form):
    import json
    from bbcells import chowwitt
    groups = chowwitt.chow_witt(complex_)
    if form == "json":
        print(json.dumps({"name": name, "chow_witt": [
            {"degree": q, "free": free, "torsion": torsion}
            for q, (free, torsion) in enumerate(groups)]}, indent=1))
        return 0
    print("%s: Chow-Witt groups CH~^q over R (untwisted; CH^q x_{Ch^q} H^q(X(R); Z))" % name)
    for q, (free, torsion) in enumerate(groups):
        parts = (["Z^%d" % free if free > 1 else "Z"] if free else []) + \
                ["Z/%d" % t for t in torsion]
        print("  CH~^%d = %s" % (q, " + ".join(parts) if parts else "0"))
    return 0


def _real_prediction(data, cells):
    """(rule, set of rational Betti vectors of X(R)) under hypothesis H1"""
    from bbcells import brion, h1
    try:
        return "the GKM rule", h1.real_prediction(cells)
    except (TypeError, KeyError):
        pass
    curves = brion.with_invariant_curves(data)
    return "the rule on the Brion curves", h1.brion_prediction(bb_cells(curves, cells.cocharacter))


def _data_raw(args):
    from bbcells.frontends import raw
    return raw.load(args.file)


def _add_common(parser):
    parser.add_argument("--cocharacter", type=_parse_vector, default=None,
                        help="generic cocharacter, e.g. 1,2,3; write --cocharacter=-1,2,3 "
                             "if it starts with a minus sign (default: automatic)")
    parser.add_argument("--format", choices=("text", "latex", "json"), default="text")
    parser.add_argument("--weights", action="store_true",
                        help="show tangent weights and their signs (text format)")
    parser.add_argument("--real-prediction", action="store_true",
                        help="also print the rational Betti numbers of X(R) predicted under "
                             "hypothesis H1 (experimental; graded decompositions only)")
    parser.add_argument("--no-check", action="store_true",
                        help="skip the consistency checks")


def build_parser():
    """the argparse parser of the command line, one subcommand per front end
    >>> args = build_parser().parse_args(["flag", "E6", "--parabolic", "1"])
    >>> args.command, args.cartan_type, args.parabolic, args.format
    ('flag', 'E6', (1,), 'text')
    """
    parser = argparse.ArgumentParser(
        prog="bbcells",
        description="Bialynicki-Birula cells, motives and quadratic invariants "
                    "from combinatorial data.")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = commands.add_parser("toric", help="smooth complete toric variety from a fan")
    p.add_argument("file", nargs="?", help="fan as JSON: {\"rays\": [...], \"cones\": [...]}")
    p.add_argument("--named", help="P<n>, F<a> (Hirzebruch), dP6, or products like P1xF2")
    _add_common(p)
    p.set_defaults(build=_data_toric)

    p = commands.add_parser("flag", help="flag variety G/P of a split reductive group")
    p.add_argument("cartan_type", help="A<n>, B<n>, C<n>, D<n>, E6, E7, E8, F4 or G2")
    p.add_argument("--parabolic", type=_parse_vector, default=None,
                   help="crossed nodes J of P_J, Bourbaki numbering, e.g. 1 for E6/P1 "
                        "(default: all nodes, i.e. G/B)")
    _add_common(p)
    p.set_defaults(build=_data_flag)

    p = commands.add_parser("quadric", help="smooth split quadric Q_n in P^{n+1}")
    p.add_argument("n", type=int)
    _add_common(p)
    p.set_defaults(build=_data_quadric)

    p = commands.add_parser("wonderful",
                            help="wonderful compactification of an adjoint group")
    p.add_argument("cartan_type")
    _add_common(p)
    p.set_defaults(build=_data_wonderful)

    p = commands.add_parser("example", help="named examples: " + ", ".join(sorted(EXAMPLES)))
    p.add_argument("name")
    _add_common(p)
    p.set_defaults(build=_data_example)

    p = commands.add_parser("spherical",
                            help="wonderful varieties assembled orbit by orbit: "
                                 "complete-quadrics n (in P^{n-1}), complete-skew-forms n (on k^{2n})")
    p.add_argument("family", choices=sorted(SPHERICAL))
    p.add_argument("n", type=int)
    _add_common(p)
    p.set_defaults(build=_data_spherical)

    p = commands.add_parser("symmetric",
                            help="complete symmetric varieties of G_ad/G_ad^theta from Satake "
                                 "diagrams: AI n, AII n, AIII p q, BI p q, CI n, CII p q, DI p q, "
                                 "DIII n, EI, EII, EIII, EIV, FI, FII, G")
    p.add_argument("kind")
    p.add_argument("parameters", type=int, nargs="*")
    p.add_argument("--fan", help="JSON {\"cones\": [...]}: a smooth fan subdividing the valuation "
                   "cone, rays in the coordinates <gamma_i, n> (all <= 0); gives the toroidal "
                   "variety over the complete symmetric variety")
    p.add_argument("--cover", action="store_true",
                   help="G/G^theta with G simply connected instead of G/N(G^theta): its "
                   "weight lattice (Helgason) is finer than Z Sigma; the fan (rays in "
                   "Hom(Lambda, Z)) defaults to the orthant, resolved; weights in "
                   "fundamental-weight coordinates")
    p.add_argument("--certify", choices=("symmetry", "points"), default=None,
                   help="show how the normal weights beyond condition (R) are decided: by the "
                        "opposition symmetry, or by point counts of orbit closures")
    p.add_argument("--processes", type=int, default=1,
                   help="with --counts-only: count the orbits in this many processes")
    p.add_argument("--method", choices=("stream", "orbits"), default="stream",
                   help="with --counts-only: stream over the fixed points, or (split forms "
                        "only) sum over the orbits with Brion-Peyre counts (EVIII)")
    p.add_argument("--counts-only", action="store_true",
                   help="stream the cell counts without listing fixed points (large E7/E8 cases)")
    _add_common(p)
    p.set_defaults(build=_data_symmetric)

    p = commands.add_parser("two-orbit",
                            help="rank-one two-orbit completions: AQ n, PQ n, HP n, PGL n, OP2, G2, SPIN7")
    p.add_argument("family", help="AQ (affine quadric), PQ (P^{n+1} - Q_n), HP, PGL "
                   "(PGL_n/GL_{n-1}), OP2, G2 (P^6 - Q_5), SPIN7 (P^7 - Q_6)")
    p.add_argument("n", type=int, nargs="?")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.set_defaults(run=_run_two_orbit)

    p = commands.add_parser("real", help="H^*(G/P(R); Z) from real Schubert cells")
    p.add_argument("cartan_type")
    p.add_argument("--parabolic", type=_parse_vector, default=None)
    p.add_argument("--chow-witt", action="store_true",
                   help="the Chow-Witt groups over R instead (untwisted)")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.set_defaults(run=_run_real)

    p = commands.add_parser("real-toric",
                            help="H_*(X(R); Z) of a smooth complete toric variety, exactly")
    p.add_argument("file", nargs="?", help="fan as JSON")
    p.add_argument("--named", help="P<n>, F<a>, dP6, or products like P1xF2")
    p.add_argument("--cocharacter", type=_parse_vector, default=None)
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--chow-witt", action="store_true",
                   help="the Chow-Witt groups over R instead (untwisted)")
    p.set_defaults(run=_run_real_toric)

    p = commands.add_parser("characteristic",
                            help="characteristic numbers of complete quadrics in P^{n-1}")
    p.add_argument("n", type=int, help="size of the symmetric matrices (3: conics)")
    p.add_argument("--monomial", type=_parse_vector, default=None,
                   help="exponents e_1,...,e_{n-1}: the integral of prod mu_k^e_k")
    p.add_argument("--divisor", type=_parse_vector, default=None,
                   help="coefficients a_1,...,a_{n-1}: the integral of (sum a_k mu_k)^dim")
    p.add_argument("--tangency", action="store_true",
                   help="the number of quadrics tangent to dim general quadrics")
    p.add_argument("--method", choices=("localization", "sections"), default="localization",
                   help="localization at the fixed points, or the Hilbert function of "
                        "De Concini-Procesi (independent, slower)")
    p.set_defaults(run=_run_characteristic)

    p = commands.add_parser("raw", help="fixed points and tangent weights as JSON")
    p.add_argument("file")
    _add_common(p)
    p.set_defaults(build=_data_raw)
    return parser


def main(argv=None):
    """run the command line on argv (default: sys.argv[1:]); returns the exit
    status, 2 on invalid input. The quadric surface Q_2 = P^1 x P^1:
    >>> main(["quadric", "2", "--no-check"])                  # doctest: +ELLIPSIS
    Q_2 (dimension 2, torus rank 2, 4 fixed points)
    ...
    Poincare polynomial: 1 + 2t^2 + t^4
    ...
    0
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if hasattr(args, "run"):
            return args.run(args)
        if getattr(args, "certify", None):
            return _run_symmetric_certify(args)
        if getattr(args, "counts_only", False):
            return _run_symmetric_counts(args)
        data = args.build(args)
        cells = bb_cells(data, args.cocharacter)
        prediction = _real_prediction(data, cells) if args.real_prediction else None
    except (ValueError, OSError) as error:
        print("bbcells: error: %s" % error, file=sys.stderr)
        return 2
    print(render(cells, args.format, show_weights=args.weights, checks=not args.no_check))
    if prediction is not None:
        rule, bettis = prediction
        print("real points, predicted by %s (conditional on H1, experimental):" % rule)
        for betti in sorted(bettis):
            print("  rational Betti numbers of X(R): %s" % ", ".join(map(str, betti)))
    return 0
