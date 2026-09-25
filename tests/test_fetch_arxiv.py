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

ABS_SAMPLE = """<html><head>
<meta name="citation_title" content="Chow-Witt rings of Grassmannians" />
<meta name="citation_author" content="Wendt, Matthias" />
<meta name="citation_date" content="2018/05/16" />
<meta name="citation_abstract" content="We compute &amp; describe ..." />
</head><body><table>
<td class="tablecell jref">Algebr. Geom. Topol. 24 (2024) 1-48</td>
<td class="tablecell doi"><a href="https://doi.org/10.2140/agt.2024.24.1">https://doi.org/10.2140/agt.2024.24.1</a></td>
</table>
<h2>Submission history</h2> <strong><a href="/abs/1805.06142v1">[v1]</a></strong>
        Wed, 16 May 2018 06:04:35 UTC (35 KB)<br/>
    <strong>[v2]</strong>
        Mon, 23 Mar 2020 10:00:00 UTC (40 KB)<br/>
</body></html>"""


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

    def test_parse_abs_page(self):
        entry = fetch_arxiv.parse_abs_page(ABS_SAMPLE, "1805.06142")
        self.assertEqual(entry["title"], "Chow-Witt rings of Grassmannians")
        self.assertEqual(entry["authors"], ["Matthias Wendt"])
        self.assertEqual(entry["latest_version"], "1805.06142v2")
        self.assertEqual(entry["published"], "Wed, 16 May 2018")
        self.assertEqual(entry["updated"], "Mon, 23 Mar 2020")
        self.assertEqual(entry["journal_ref"], "Algebr. Geom. Topol. 24 (2024) 1-48")
        self.assertEqual(entry["doi"], "10.2140/agt.2024.24.1")
        self.assertEqual(entry["abstract"], "We compute & describe ...")
        self.assertIsNone(fetch_arxiv.parse_abs_page("<html></html>", "0000.00000"))

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
