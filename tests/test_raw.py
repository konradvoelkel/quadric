import json
import tempfile
import unittest
from pathlib import Path

from bbcells import schema
from bbcells.core import bb_cells
from bbcells.frontends import raw
from tests._util import doctests_for
from tests.test_core import projective_space

load_tests = doctests_for(raw, schema)


class TestRaw(unittest.TestCase):

    def test_round_trip(self):
        data = projective_space(3)
        again = raw.from_dict(json.loads(raw.dump(data)))
        self.assertEqual(again, data)
        self.assertEqual(bb_cells(again).counts, (1, 1, 1, 1))

    def test_file_round_trip_with_edges(self):
        document = {"name": "P^1", "dim": 1, "rank": 1,
                    "points": [{"label": "0", "weights": [[1]]},
                               {"label": "oo", "weights": [[-1]]}],
                    "edges": [{"from": "0", "to": "oo", "weight": [1]}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p1.json"
            path.write_text(json.dumps(document))
            data = raw.load(path)
            self.assertEqual(data.edges, (("0", "oo", (1,)),))
            self.assertEqual(raw.to_dict(data)["edges"], document["edges"])

    def test_schema_rejects_booleans_and_unknown_keys(self):
        with self.assertRaisesRegex(ValueError, "expected integer"):
            raw.from_dict({"dim": True, "rank": 1, "points": []})
        with self.assertRaisesRegex(ValueError, "unexpected key 'colour'"):
            raw.from_dict({"dim": 1, "rank": 1, "points": [], "colour": "red"})


if __name__ == "__main__":
    unittest.main()
