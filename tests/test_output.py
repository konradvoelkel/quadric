"""golden-file tests of the renderers and the CLI (PLAN.md S1.7).
Regenerate with BBCELLS_UPDATE_GOLDEN=1 python3 -m unittest tests.test_output"""

import contextlib
import io
import json
import os
import unittest
from pathlib import Path

from bbcells import cli, output
from bbcells.core import bb_cells
from bbcells.frontends import toric
from tests._util import doctests_for

load_tests = doctests_for(output)
GOLDEN = Path(__file__).resolve().parent / "golden"


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class TestGolden(unittest.TestCase):

    def compare(self, name, text):
        path = GOLDEN / name
        if os.environ.get("BBCELLS_UPDATE_GOLDEN"):
            path.write_text(text)
        self.assertEqual(text, path.read_text(), "output differs from %s" % path)

    def test_p2_all_formats(self):
        for fmt, suffix in (("text", "txt"), ("latex", "tex"), ("json", "json")):
            code, out, err = run_cli("toric", "--named", "P2", "--format", fmt)
            self.assertEqual((code, err), (0, ""))
            self.compare("P2." + suffix, out)

    def test_json_is_valid_and_complete(self):
        cells = bb_cells(toric.fixed_point_data(toric.hirzebruch(3)))
        document = json.loads(output.to_json(cells))
        self.assertEqual(document["counts"], [1, 2, 1])
        self.assertEqual(document["gw_euler"], {"plus": 2, "minus": 2, "rank": 4, "signature": 0})
        self.assertTrue(all(item["passed"] for item in document["checks"]))

    def test_cli_flag(self):
        code, out, err = run_cli("flag", "E6", "--parabolic", "1", "--format", "json")
        self.assertEqual(code, 0, err)
        document = json.loads(out)
        self.assertEqual(sum(document["counts"]), 27)
        self.assertEqual(document["cocharacter"], [1] * 6)

    def test_cli_quadric_with_negative_cocharacter(self):
        code, out, err = run_cli("quadric", "4", "--cocharacter=-1,-2,-3", "--format", "json")
        self.assertEqual(code, 0, err)
        cells = {c["label"]: c["dim"] for c in json.loads(out)["cells"]}
        # the legacy example in README.md: x_2 is the open cell for (1, 2, 3)
        self.assertEqual(cells["x_2"], 4)
        self.assertEqual(cells["y_2"], 0)

    def test_cli_errors_are_reported(self):
        code, out, err = run_cli("toric", "--named", "P2", "--cocharacter", "1,1")
        self.assertEqual(code, 2)
        self.assertIn("not generic", err)
        code, out, err = run_cli("toric", "--named", "Q7")
        self.assertEqual(code, 2)
        self.assertIn("unknown named fan", err)


if __name__ == "__main__":
    unittest.main()
