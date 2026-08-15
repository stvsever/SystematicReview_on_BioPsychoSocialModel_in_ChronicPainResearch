from __future__ import annotations

"""End-to-end driver for the cross-provider full-text test run (scheme 3).

``run_fulltext_testrun_pipeline`` chains the whole thing: build or load the
open-access corpus from the abstract-level candidate set, code every paper with
every model, check integrity and the quoted evidence, compute reliability,
quantify semantic overlap of the open extraction lists, render the figures, build
the interactive knowledge graph, and write the summary. It is what the notebook
and the CLI call.

Two of those steps are enrichments rather than requirements. The semantic overlap
calls an embedding endpoint, so a network or credential failure is reported and
skipped instead of invalidating a complete lexical result. The knowledge graph is
a review surface built from tables that are already on disk, so it can be rebuilt
at any time without re-coding anything.
"""

import pandas as pd

from bps_review.fulltext.analysis.integrity import build_integrity
from bps_review.fulltext.analysis.reliability import build_reliability
from bps_review.fulltext.analysis.semantic import build_semantic_overlap
from bps_review.fulltext.coding.runner import load_items, load_or_run, run_fulltext_testrun
from bps_review.fulltext.config import corpus_csv, graph_dir, reliability_dir
from bps_review.fulltext.corpus.pmc import build_corpus, load_corpus, load_corpus_records
from bps_review.fulltext.report import write_fulltext_summary
from bps_review.fulltext.visualization.figures import build_figures


GRAPH_TITLE = "FULL-TEXT CODING SCHEME (test run)"


def ensure_corpus(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Retrieve the open-access corpus, or load the one already on disk."""
    if corpus_csv().exists() and not force:
        return load_corpus()
    return build_corpus(verbose=verbose)


def run_fulltext_testrun_pipeline(
    force_corpus: bool = False,
    force_coding: bool = False,
    make_figures: bool = True,
    make_semantic: bool = True,
    make_graph: bool = True,
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
        Render the multi-panel figures.
    make_semantic:
        Embed the extraction labels and quantify semantic list overlap next to
        the lexical one. Cached vectors make every rerun free.
    make_graph:
        Write the local interactive knowledge graph over the coded run.
    """
    corpus = ensure_corpus(force=force_corpus, verbose=verbose)
    long_df = run_fulltext_testrun(verbose=verbose) if force_coding else load_or_run(force=False)
    items_df = load_items()
    records = load_corpus_records()

    integrity = build_integrity(long_df, items_df, records, write=True)
    results = build_reliability(long_df, write=True)

    semantic = None
    if make_semantic:
        if verbose:
            print("Semantic label overlap:")
        try:
            semantic = build_semantic_overlap(
                long_df,
                corpus_df=corpus,
                write=True,
                out_dir=reliability_dir(),
                verbose=verbose,
            )
        except Exception as exc:  # network, credentials, or provider outage
            # The semantic layer enriches an already complete lexical result, so
            # a failure here must not invalidate the run. It is reported, skipped,
            # and picked up on the next execution from the cached vectors.
            print(f"  skipped semantic overlap: {exc}")
            semantic = None

    figures = build_figures(long_df, results, integrity, semantic=semantic) if make_figures else []

    graph_path = None
    if make_graph:
        from bps_review.graph import build_knowledge_graph

        graph_path = build_knowledge_graph(
            corpus_df=corpus,
            long_df=long_df,
            items_df=items_df,
            output_dir=graph_dir(),
            run_title=GRAPH_TITLE,
            run_subtitle=(
                f"{long_df['record_id'].nunique()} open-access reviews coded by "
                f"{long_df['model_label'].nunique()} independent providers"
            ),
            verification_df=integrity["quote_verification"],
        )

    summary_text = write_fulltext_summary(long_df, results, integrity, corpus, semantic)

    return {
        "corpus": corpus,
        "long_df": long_df,
        "items_df": items_df,
        "results": results,
        "integrity": integrity,
        "semantic": semantic,
        "figures": figures,
        "graph_path": graph_path,
        "summary_text": summary_text,
    }
