from __future__ import annotations

"""End-to-end driver for the cross-provider full-text test run (scheme 3).

``run_fulltext_testrun_pipeline`` chains the whole thing: build or load the
open-access corpus from the abstract-level candidate set, code every paper with
every model, check integrity and the quoted evidence, compute reliability, render
the figures, and write the summary. It is what the notebook and the CLI call.
"""

import pandas as pd

from bps_review.fulltext.analysis.integrity import build_integrity
from bps_review.fulltext.analysis.reliability import build_reliability
from bps_review.fulltext.coding.runner import load_items, load_or_run, run_fulltext_testrun
from bps_review.fulltext.config import corpus_csv
from bps_review.fulltext.corpus.pmc import build_corpus, load_corpus, load_corpus_records
from bps_review.fulltext.report import write_fulltext_summary
from bps_review.fulltext.visualization.figures import build_figures


def ensure_corpus(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Retrieve the open-access corpus, or load the one already on disk."""
    if corpus_csv().exists() and not force:
        return load_corpus()
    return build_corpus(verbose=verbose)


def run_fulltext_testrun_pipeline(
    force_corpus: bool = False,
    force_coding: bool = False,
    make_figures: bool = True,
    verbose: bool = True,
) -> dict:
    """Run (or load) the corpus and the coding, then verify, analyse, and report.

    Parameters
    ----------
    force_corpus:
        Retrieve the full texts again instead of reusing the cached corpus.
    force_coding:
        Re-code every paper with every model instead of reusing the cached long
        table. This is the step that calls the API.
    make_figures:
        Render the four multi-panel figures.
    """
    corpus = ensure_corpus(force=force_corpus, verbose=verbose)
    long_df = run_fulltext_testrun(verbose=verbose) if force_coding else load_or_run(force=False)
    items_df = load_items()
    records = load_corpus_records()

    integrity = build_integrity(long_df, items_df, records, write=True)
    results = build_reliability(long_df, write=True)
    figures = build_figures(long_df, results, integrity) if make_figures else []
    summary_text = write_fulltext_summary(long_df, results, integrity, corpus)

    return {
        "corpus": corpus,
        "long_df": long_df,
        "items_df": items_df,
        "results": results,
        "integrity": integrity,
        "figures": figures,
        "summary_text": summary_text,
    }
