import unittest

import bbcells
from bbcells import cli
from tests._util import doctests_for

load_tests = doctests_for(bbcells)


class TestPackage(unittest.TestCase):

    def test_cli_help_runs(self):
        self.assertEqual(cli.main([]), 0)


if __name__ == "__main__":
    unittest.main()
