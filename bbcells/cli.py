"""
Command line interface.

    python3 -m bbcells toric --named P3
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


def _data_raw(args):
    from bbcells.frontends import raw
    return raw.load(args.file)


def _add_common(parser):
    parser.add_argument("--cocharacter", type=_parse_vector, default=None,
                        help="generic cocharacter, e.g. 1,2,3 (default: automatic)")
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
        data = args.build(args)
        cells = bb_cells(data, args.cocharacter)
    except (ValueError, OSError) as error:
        print("bbcells: error: %s" % error, file=sys.stderr)
        return 2
    print(render(cells, args.format, show_weights=args.weights, checks=not args.no_check))
    return 0
