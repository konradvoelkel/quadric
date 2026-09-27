import ast
import unittest
from pathlib import Path

import bbcells
from bbcells import cli, linalg
from tests._util import doctests_for

load_tests = doctests_for(bbcells, cli, linalg)

PACKAGE = Path(__file__).resolve().parent.parent / "bbcells"


class TestPackage(unittest.TestCase):

    def test_cli_help_runs(self):
        self.assertEqual(cli.main([]), 0)

    def test_every_public_function_has_a_doctest(self):
        """PLAN.md section 0, rule 5: public top-level functions and classes
        carry a docstring with a doctest (methods need not)"""
        missing = []
        for path in sorted(PACKAGE.rglob("*.py")):
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and \
                        not node.name.startswith("_") and \
                        ">>>" not in (ast.get_docstring(node) or ""):
                    missing.append("%s:%d %s" % (path.relative_to(PACKAGE.parent),
                                                 node.lineno, node.name))
        self.assertEqual(missing, [], "public functions without a doctest:\n" +
                         "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
