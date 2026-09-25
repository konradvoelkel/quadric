"""command line interface (filled in by PLAN step S1.7)"""

import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        prog="bbcells",
        description="Bialynicki-Birula cells, motives and quadratic invariants "
                    "from combinatorial data.")
    parser.add_subparsers(dest="command", metavar="COMMAND")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0
