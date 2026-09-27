# Companion paper

`bbcells.tex` is the companion paper of the package: methods and results,
with *theorem*, *proposition* and *lemma* reserved for proved statements, and
*computation*, *observation* and *conjecture* for the rest. Its status and
the remaining steps (proofs, the $\mathbb P^5$ number, review, venue) are
track P of [`PLAN.md`](../PLAN.md) §4.1.

Build (TeX Live with `amsart`, `booktabs`):

    cd paper
    pdflatex bbcells && bibtex bbcells && pdflatex bbcells && pdflatex bbcells

`references.bib` marks every entry whose bibliographic data could not be
confirmed with a comment `% UNVERIFIED`; these must be checked before
submission.

## Reproducing the numbers

Every number in the paper is a claim in `reproduce.py`:

    python3 paper/reproduce.py            # fast claims, about 25 s
    python3 paper/reproduce.py --all      # everything, several hours
    python3 paper/reproduce.py --list

The fast claims run in the test suite (`tests/test_paper.py`). The output of
the last full run, with timings, is in `reproduce.log`.
