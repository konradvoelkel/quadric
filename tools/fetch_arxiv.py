#! /usr/bin/env python3
"""
Fetch metadata, PDFs and TeX sources of the arXiv papers cited in LITERATURE.md.

    python3 tools/fetch_arxiv.py                 # all IDs found in LITERATURE.md
    python3 tools/fetch_arxiv.py 1805.04338      # just these IDs
    python3 tools/fetch_arxiv.py --metadata-only # only refresh the metadata file
    python3 tools/fetch_arxiv.py --list          # print the IDs and exit

Output:
  literature/arxiv_metadata.json   title, authors, dates, journal_ref, doi (committed)
  literature/cache/<id>/<id>.pdf   the PDF (git-ignored, not ours to redistribute)
  literature/cache/<id>/src/       the unpacked e-print source (git-ignored)

Needs outbound HTTPS to arxiv.org and export.arxiv.org. Requests are spaced
3 seconds apart, as the arXiv API terms ask. Standard library only.
"""

import argparse
import gzip
import io
import json
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LITERATURE = ROOT / "LITERATURE.md"
OUT_DIR = ROOT / "literature"
CACHE_DIR = OUT_DIR / "cache"
METADATA_FILE = OUT_DIR / "arxiv_metadata.json"

API_URL = "https://export.arxiv.org/api/query?id_list=%s&max_results=%d"
PDF_URL = "https://arxiv.org/pdf/%s"
SRC_URL = "https://arxiv.org/e-print/%s"
USER_AGENT = "quadric-bbcells-literature-fetch/0.1 (https://github.com/konradvoelkel/quadric)"
DELAY_SECONDS = 3.0

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"

# new style 1805.04338 (optionally v2), old style math/0410472 or math.AG/0410472
ID_PATTERN = re.compile(
    r"(?<![\w./])"
    r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)"
    r"(?![\w/])")
# only take IDs that are marked as arXiv references in the text
CONTEXT_PATTERN = re.compile(r"arXiv:\s*(\S+)|arxiv\.org/(?:abs|pdf)/(\S+)", re.I)


def extract_arxiv_ids(text):
    """IDs cited as 'arXiv:ID' or as arxiv.org links, in order of appearance
    >>> extract_arxiv_ids("see arXiv:1805.04338 and arXiv:math/0410472v2, arxiv.org/abs/1012.0454.")
    ['1805.04338', 'math/0410472', '1012.0454']
    >>> extract_arxiv_ids("arXiv:1805.04338 again arXiv:1805.04338v3; not an id: 2014.1")
    ['1805.04338']
    """
    ids = []
    for match in CONTEXT_PATTERN.finditer(text):
        candidate = (match.group(1) or match.group(2)).rstrip(".,;:)]*")
        found = ID_PATTERN.match(candidate)
        if found:
            arxiv_id = strip_version(found.group(1))
            if arxiv_id not in ids:
                ids.append(arxiv_id)
    return ids


def strip_version(arxiv_id):
    """
    >>> strip_version("1805.04338v2")
    '1805.04338'
    >>> strip_version("math/0410472")
    'math/0410472'
    """
    return re.sub(r"v\d+$", "", arxiv_id)


def safe_name(arxiv_id):
    """file system name for an ID
    >>> safe_name("math/0410472")
    'math_0410472'
    """
    return arxiv_id.replace("/", "_")


def parse_atom(xml_bytes):
    """parse an arXiv API Atom response into {id: metadata}"""
    root = ET.fromstring(xml_bytes)
    result = {}
    for entry in root.findall(ATOM + "entry"):
        raw_id = entry.findtext(ATOM + "id", default="")
        # the API reports missing IDs as entries without a proper abs URL
        if "/abs/" not in raw_id:
            continue
        versioned = raw_id.rsplit("/abs/", 1)[1]
        arxiv_id = strip_version(versioned)

        def text(tag, ns=ATOM):
            value = entry.findtext(ns + tag)
            return " ".join(value.split()) if value else None

        result[arxiv_id] = {
            "id": arxiv_id,
            "latest_version": versioned,
            "title": text("title"),
            "authors": [" ".join(a.findtext(ATOM + "name", "").split())
                        for a in entry.findall(ATOM + "author")],
            "published": text("published"),
            "updated": text("updated"),
            "journal_ref": text("journal_ref", ARXIV),
            "doi": text("doi", ARXIV),
            "primary_category": (entry.find(ARXIV + "primary_category").get("term")
                                 if entry.find(ARXIV + "primary_category") is not None
                                 else None),
            "abstract": text("summary"),
        }
    return result


def unpack_source(data, target_dir):
    """unpack an arXiv e-print: a gzipped tarball, a gzipped single file,
    or (rarely) a plain PDF. Returns the list of written paths."""
    target_dir.mkdir(parents=True, exist_ok=True)
    if data[:4] == b"%PDF":
        path = target_dir / "source.pdf"
        path.write_bytes(data)
        return [path]
    raw = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
    try:
        tar = tarfile.open(fileobj=io.BytesIO(raw))
    except tarfile.ReadError:
        path = target_dir / "main.tex"
        path.write_bytes(raw)
        return [path]
    written = []
    with tar:
        for member in tar.getmembers():
            destination = (target_dir / member.name).resolve()
            # refuse absolute paths, "..", links and devices
            if (not destination.is_relative_to(target_dir.resolve())
                    or not (member.isfile() or member.isdir())):
                continue
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(tar.extractfile(member).read())
            written.append(destination)
    return written


class Fetcher(object):

    def __init__(self, delay=DELAY_SECONDS):
        self.delay = delay
        self._last = 0.0

    def get(self, url):
        wait = self._last + self.delay - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        finally:
            self._last = time.monotonic()


def explain_network_error(error):
    message = str(error)
    if "403" in message or "Tunnel connection failed" in message:
        return ("%s\n  The network policy of this environment probably blocks "
                "arxiv.org / export.arxiv.org. Allow both hosts (or run this "
                "script on a machine with normal internet access)." % message)
    return message


def load_metadata():
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text())
    return {}


def save_metadata(metadata):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.write_text(
        json.dumps(dict(sorted(metadata.items())), indent=2, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("ids", nargs="*", help="arXiv IDs (default: all in LITERATURE.md)")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--no-source", action="store_true", help="skip TeX sources")
    parser.add_argument("--no-pdf", action="store_true", help="skip PDFs")
    parser.add_argument("--force", action="store_true", help="re-download cached files")
    parser.add_argument("--list", action="store_true", help="print the IDs and exit")
    args = parser.parse_args(argv)

    ids = ([strip_version(i) for i in args.ids] if args.ids
           else extract_arxiv_ids(LITERATURE.read_text()))
    if args.list:
        print("\n".join(ids))
        return 0

    fetcher = Fetcher()
    metadata = load_metadata()
    failures = 0

    # metadata in batches; the API accepts comma separated id lists
    for start in range(0, len(ids), 20):
        batch = ids[start:start + 20]
        try:
            metadata.update(parse_atom(fetcher.get(
                API_URL % (",".join(batch), len(batch)))))
        except (urllib.error.URLError, OSError) as error:
            print("metadata request failed: %s" % explain_network_error(error),
                  file=sys.stderr)
            return 2
    save_metadata(metadata)
    missing = [i for i in ids if i not in metadata]
    print("metadata: %d of %d IDs resolved -> %s"
          % (len(ids) - len(missing), len(ids), METADATA_FILE.relative_to(ROOT)))
    for arxiv_id in missing:
        print("  not found on arXiv: %s" % arxiv_id, file=sys.stderr)
        failures += 1
    if args.metadata_only:
        return 1 if failures else 0

    for arxiv_id in ids:
        if arxiv_id in missing:
            continue
        folder = CACHE_DIR / safe_name(arxiv_id)
        pdf_path = folder / (safe_name(arxiv_id) + ".pdf")
        src_dir = folder / "src"
        try:
            if not args.no_pdf and (args.force or not pdf_path.exists()):
                folder.mkdir(parents=True, exist_ok=True)
                pdf_path.write_bytes(fetcher.get(PDF_URL % arxiv_id))
                print("  pdf    %s" % pdf_path.relative_to(ROOT))
            if not args.no_source and (args.force or not src_dir.exists()):
                files = unpack_source(fetcher.get(SRC_URL % arxiv_id), src_dir)
                print("  source %s (%d files)" % (src_dir.relative_to(ROOT), len(files)))
        except (urllib.error.URLError, OSError, tarfile.TarError) as error:
            print("  %s failed: %s" % (arxiv_id, explain_network_error(error)),
                  file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
