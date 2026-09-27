"""the fast claims of the companion paper (paper/reproduce.py, PLAN.md P2)"""

import contextlib
import importlib.util
import io
import os
import unittest

PATH = os.path.join(os.path.dirname(__file__), "..", "paper", "reproduce.py")


class TestPaperClaims(unittest.TestCase):

    def test_fast_claims(self):
        spec = importlib.util.spec_from_file_location("reproduce", PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fast = [c for c in module.CLAIMS if not c[2]]
        self.assertGreater(len(fast), 20)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            failures = module.run(fast)
        self.assertEqual(failures, 0, output.getvalue())


if __name__ == "__main__":
    unittest.main()
