from __future__ import annotations

"""End-to-end driver for the cross-provider abstract-level test run.

``run_abstract_testrun`` chains the whole thing: build or load the PubMed
sample, code every abstract with every model, compute reliability, render the
figures, write the summary, and export the filtered candidate set that the
full-text stage reads. It is what the notebook and the CLI call.
"""

import pandas as pd

from bps_review.pilot.analysis.reliability import build_reliability
from bps_review.pilot.coding.data import ensure_corpus, load_testrun_records
from bps_review.pilot.coding.runner import load_or_run, run_testrun
from bps_review.pilot.config import candidate_set_csv
from bps_review.pilot.report import write_summary
from bps_review.pilot.visualization.figures import build_figures
from bps_review.utils.io import write_csv


CANDIDATE_COLUMNS = [
    "record_id", "pmid", "pmcid", "doi", "title", "journal", "year",
    "stage3_candidate", "stage3_priority", "candidate_consensus_depth",
    "musculoskeletal_flag", "icd11_pain_category", "provisional_typology",
    "bio_mentioned", "psych_mentioned", "social_mentioned", "n_domains_consensus",
]


def export_candidate_set(consensus: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """The abstract-level filter, as the hand-off to the full-text stage.

    A record is carried forward when the majority of the models call it a Stage 3
    candidate. The consensus row carries the majority value of every coded field,
    so the hand-off is one table rather than three disagreeing ones.
    """
    merged = corpus.merge(consensus, on="record_id", how="inner", suffixes=("", "_consensus"))
    candidates = merged[merged["stage3_candidate"] == "yes"].copy()
    columns = [column for column in CANDIDATE_COLUMNS if column in candidates.columns]
    candidates = candidates[columns].sort_values(
        ["stage3_priority", "candidate_consensus_depth", "record_id"],
        ascending=[True, False, True],
    )
    write_csv(candidate_set_csv(), candidates)
    return candidates


def run_abstract_testrun(
    force_corpus: bool = False,
    force_coding: bool = False,
    make_figures: bool = True,
    verbose: bool = True,
) -> dict:
    """Run (or load) the sample and the coding, then analyse and report.

    Parameters
    ----------
    force_corpus:
        Retrieve a fresh sample from PubMed instead of reusing the cached one.
    force_coding:
        Re-code every abstract with every model instead of reusing the cached
        long table. This is the step that calls the API.
    make_figures:
        Render the four multi-panel figures.
    """
    corpus = ensure_corpus(force=force_corpus, verbose=verbose)
    if force_coding:
        long_df = run_testrun(records=load_testrun_records(), verbose=verbose)
    else:
        long_df = load_or_run(force=False)

    results = build_reliability(long_df, write=True)
    figures = build_figures(long_df, results) if make_figures else []
    candidates = export_candidate_set(results["consensus"], corpus)
    summary_text = write_summary(long_df, results, corpus)

    return {
        "corpus": corpus,
        "long_df": long_df,
        "results": results,
        "candidates": candidates,
        "figures": figures,
        "summary_text": summary_text,
    }
