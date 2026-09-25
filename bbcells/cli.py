"""
Command line interface.

    python3 -m bbcells toric --named P3
    python3 -m bbcells flag E6 --parabolic 1
    python3 -m bbcells quadric 4 --cocharacter=-1,-2,-3
    python3 -m bbcells wonderful A2
    python3 -m bbcells example complete-conics
    python3 -m bbcells two-orbit HP 2
    python3 -m bbcells real A3 --parabolic 2
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
}


def _data_example(args):
    import importlib
    if args.name not in EXAMPLES:
        raise ValueError("unknown example %r; available: %s"
                         % (args.name, ", ".join(sorted(EXAMPLES))))
    module, function = EXAMPLES[args.name]
    return getattr(importlib.import_module(module), function)()


def _two_orbit_case(args):
    from bbcells.frontends import two_orbit
    family = args.family.upper()
    if family == "OP2":
        return two_orbit.octonionic_projective_plane()
    if args.n is None:
        raise ValueError("%s needs a dimension parameter n" % args.family)
    builders = {"AQ": two_orbit.affine_quadric, "HP": two_orbit.quaternionic_projective_space,
                "PGL": two_orbit.pgl_mod_gl}
    if family not in builders:
        raise ValueError("unknown family %r; use AQ, HP, OP2 or PGL" % args.family)
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
              "signs are implemented for type A only" % (X.name, len(X.incidences), nonzero))
        return 0
    cohomology = X.cohomology()
    if args.format == "json":
        print(json.dumps({"name": X.name, "cohomology": [
            {"degree": c, "free": free, "torsion": torsion}
            for c, (free, torsion) in enumerate(cohomology)]}, indent=1))
        return 0
    print("%s: H^*(X(R); Z), from signed incidences (arXiv:1910.11149)" % X.name)
    for c, (free, torsion) in enumerate(cohomology):
        parts = (["Z^%d" % free if free > 1 else "Z"] if free else []) + \
                ["Z/%d" % t for t in torsion]
        print("  H^%d = %s" % (c, " + ".join(parts) if parts else "0"))
    return 0


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
    parser.add_argument("--no-check", action="store_true",
                        help="skip the consistency checks")


def build_parser():
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

    p = commands.add_parser("two-orbit",
                            help="rank-one two-orbit completions: AQ n, HP n, OP2, PGL n")
    p.add_argument("family", help="AQ (affine quadric), HP, OP2 or PGL (PGL_n/GL_{n-1})")
    p.add_argument("n", type=int, nargs="?")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.set_defaults(run=_run_two_orbit)

    p = commands.add_parser("real", help="H^*(G/P(R); Z) from real Schubert cells (type A)")
    p.add_argument("cartan_type")
    p.add_argument("--parabolic", type=_parse_vector, default=None)
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.set_defaults(run=_run_real)

    p = commands.add_parser("raw", help="fixed points and tangent weights as JSON")
    p.add_argument("file")
    _add_common(p)
    p.set_defaults(build=_data_raw)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if hasattr(args, "run"):
            return args.run(args)
        data = args.build(args)
        cells = bb_cells(data, args.cocharacter)
    except (ValueError, OSError) as error:
        print("bbcells: error: %s" % error, file=sys.stderr)
        return 2
    print(render(cells, args.format, show_weights=args.weights, checks=not args.no_check))
    return 0
