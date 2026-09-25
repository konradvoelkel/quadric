"""offline tests for tools/fetch_arxiv.py (no network access needed)"""

import doctest
import gzip
import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("fetch_arxiv", ROOT / "tools" / "fetch_arxiv.py")
fetch_arxiv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_arxiv)

ATOM_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1805.04338v2</id>
    <updated>2018-06-01T00:00:00Z</updated>
    <published>2018-05-11T00:00:00Z</published>
    <title>Motivic Cell Structures for
      Spherical Varieties</title>
    <summary>We give a general method ...</summary>
    <author><name>Konrad Voelkel</name></author>
    <arxiv:primary_category term="math.AG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format_for_9999.99999</id>
    <title>Error</title>
  </entry>
</feed>
"""


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(fetch_arxiv))
    return tests


def make_tar_gz(members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


class TestFetchArxiv(unittest.TestCase):

    def test_parse_atom(self):
        parsed = fetch_arxiv.parse_atom(ATOM_SAMPLE)
        self.assertEqual(list(parsed), ["1805.04338"])
        entry = parsed["1805.04338"]
        self.assertEqual(entry["title"], "Motivic Cell Structures for Spherical Varieties")
        self.assertEqual(entry["authors"], ["Konrad Voelkel"])
        self.assertEqual(entry["latest_version"], "1805.04338v2")
        self.assertEqual(entry["primary_category"], "math.AG")
        self.assertIsNone(entry["journal_ref"])

    def test_literature_ids_are_well_formed(self):
        ids = fetch_arxiv.extract_arxiv_ids((ROOT / "LITERATURE.md").read_text())
        self.assertIn("1805.04338", ids)
        self.assertEqual(len(ids), len(set(ids)))
        for arxiv_id in ids:
            self.assertRegex(arxiv_id, fetch_arxiv.ID_PATTERN)

    def test_unpack_tarball_refuses_path_traversal(self):
        data = make_tar_gz({"paper.tex": b"\\documentclass{amsart}",
                            "figs/a.tex": b"%",
                            "../evil.tex": b"no"})
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "src"
            written = fetch_arxiv.unpack_source(data, target)
            names = sorted(p.relative_to(target).as_posix() for p in written)
            self.assertEqual(names, ["figs/a.tex", "paper.tex"])
            self.assertFalse((Path(tmp) / "evil.tex").exists())

    def test_unpack_single_gzipped_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = fetch_arxiv.unpack_source(gzip.compress(b"\\begin{document}"),
                                                Path(tmp))
            self.assertEqual([p.name for p in written], ["main.tex"])

    def test_unpack_pdf_only_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = fetch_arxiv.unpack_source(b"%PDF-1.5 ...", Path(tmp))
            self.assertEqual([p.name for p in written], ["source.pdf"])


if __name__ == "__main__":
    unittest.main()
