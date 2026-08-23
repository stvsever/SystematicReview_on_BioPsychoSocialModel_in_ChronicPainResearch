from __future__ import annotations

"""End-to-end driver for the cross-provider full-text test run (scheme 3).

``run_fulltext_testrun_pipeline`` chains the whole thing: build or load the
open-access corpus from the abstract-level candidate set, code every paper with
every model, check integrity and the quoted evidence, compute reliability,
quantify semantic overlap of the open extraction lists, write every output as
the tables it is read in, render the figures, build the interactive knowledge
graph, and write the summary. It is what the notebook and the CLI call.

The published tables are the run's store. A cached run is therefore reloaded
from them rather than from a second, unreadable set of files, and the runner's
native artifacts are absorbed into them and removed once written.

Two steps are enrichments rather than requirements. The semantic overlap calls
an embedding endpoint, so a network or credential failure is reported and
skipped instead of invalidating a complete lexical result. The knowledge graph
is a review surface built from tables that are already on disk, so it can be
rebuilt at any time without re-coding anything.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bps_review.fulltext.analysis.integrity import build_integrity
from bps_review.fulltext.analysis.reliability import build_reliability
from bps_review.fulltext.analysis.semantic import build_semantic_overlap
from bps_review.fulltext.coding.runner import (
    load_items,
    load_or_run,
    repair_failed_codings,
    run_fulltext_testrun,
)
from bps_review.fulltext.config import (
    api_calls_dir,
    codings_dir,
    corpus_csv,
    fulltext_root,
    graph_dir,
    reliability_dir,
    run_overview_json,
    run_staging_dir,
)
from bps_review.fulltext.corpus.pmc import build_corpus, load_corpus, load_corpus_records
from bps_review.fulltext.publish import (
    CODINGS_DIRNAME,
    CODINGS_LONG_FILE,
    build_output_tables,
    load_run_from_tables,
    load_run_manifest,
)
from bps_review.fulltext.report import write_fulltext_summary
from bps_review.fulltext.visualization.figures import build_figures
from bps_review.utils.io import ensure_parent, write_json


GRAPH_TITLE = "FULL-TEXT CODING SCHEME (test run)"
TEST_RUN_ID = "01_fulltext_coding_open_access_corpus"


def published_codings_csv() -> Path:
    """The published coding table, and the file a cached run is reloaded from."""
    return codings_dir() / CODINGS_DIRNAME / CODINGS_LONG_FILE


def staged_manifest() -> dict:
    """The runner's manifest, from staging or from what was published.

    A cached run has no staging folder left, so the provider and usage tables are
    rebuilt from what was written the first time rather than lost.
    """
    return load_run_manifest()


def staged_audit_paths() -> list[Path]:
    """The runner's per-provider call trail, from staging or from what was kept."""
    staged = sorted((run_staging_dir() / "audit").glob("*.jsonl"))
    return staged or sorted((api_calls_dir() / "04_raw_calls").glob("*.jsonl"))


def clear_staging() -> None:
    """Remove the runner's native artifacts once they are published.

    Everything in staging is present in the published tables: the coding table
    and the category tables together carry every field the schema defines, and
    the call table carries every field of the audit trail. Keeping a second copy
    only invites the question of which one is authoritative. The manifest, the
    log, and the raw call trail are preserved verbatim under the API-call tables,
    because they are the provenance of the run rather than a shape of its
    content.
    """
    staging = run_staging_dir()
    if not staging.exists():
        return
    for name in ("run_manifest.json", "run.log"):
        source = staging / name
        if source.exists():
            ensure_parent(api_calls_dir() / name).write_text(
                source.read_text(encoding="utf-8"), encoding="utf-8"
            )
    for audit in sorted((staging / "audit").glob("*.jsonl")):
        ensure_parent(api_calls_dir() / "04_raw_calls" / audit.name).write_text(
            audit.read_text(encoding="utf-8"), encoding="utf-8"
        )
    shutil.rmtree(staging)


def ensure_corpus(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Retrieve the open-access corpus, or load the one already on disk."""
    if corpus_csv().exists() and not force:
        return load_corpus()
    return build_corpus(verbose=verbose)


def _load_cached_run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The coded run, read back from wherever it currently lives.

    The published tables are the store. A run coded before they existed still has
    its staged long tables, so those are used when the published ones are not
    there yet, and the publishing step below writes them out.
    """
    if published_codings_csv().exists():
        return load_run_from_tables(codings_dir())
    return load_or_run(force=False), load_items()


def run_fulltext_testrun_pipeline(
    force_corpus: bool = False,
    force_coding: bool = False,
    repair_coding: bool = False,
    make_figures: bool = True,
    make_semantic: bool = True,
    make_graph: bool = True,
    verbose: bool = True,
) -> dict:
    """Run (or load) the corpus and the coding, then verify, analyse, and publish.

    Parameters
    ----------
    force_corpus:
        Retrieve the full texts again instead of reusing the cached corpus.
    force_coding:
        Re-code every paper with every model instead of reusing the published
        tables. This is the step that calls the API.
    repair_coding:
        Re-code only the cells written as ``coding_failed`` and splice them into
        the cached run. A failed coding is almost always a provider outage rather
        than a paper the model cannot read, so filling one in should not cost the
        whole grid. Ignored when ``force_coding`` is set, which recodes anyway.
    make_figures:
        Render the multi-panel figures.
    make_semantic:
        Embed the extraction labels and quantify semantic list overlap next to
        the lexical one. Cached vectors make every rerun free.
    make_graph:
        Write the local interactive knowledge graph over the coded run.
    """
    corpus = ensure_corpus(force=force_corpus, verbose=verbose)
    if force_coding:
        long_df = run_fulltext_testrun(verbose=verbose)
        items_df = load_items()
    elif repair_coding:
        long_df = repair_failed_codings(verbose=verbose)
        items_df = load_items()
    else:
        long_df, items_df = _load_cached_run()
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

    if verbose:
        print("Output tables:")
    published = build_output_tables(
        codings_dir(),
        long_df,
        items_df,
        manifest=staged_manifest(),
        audit_paths=staged_audit_paths(),
        verbose=verbose,
    )
    clear_staging()

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
    write_json(
        run_overview_json(),
        {
            "run_id": TEST_RUN_ID,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_papers": int(long_df["record_id"].nunique()),
            "n_providers": int(long_df["model_label"].nunique()),
            "n_codings": int(len(long_df)),
            "n_extracted_items": int(len(items_df)),
            "n_failed_codings": int(integrity["completeness"]["n_failed_codings"]),
            "graph_index": str(graph_path.relative_to(fulltext_root())) if graph_path else "",
            "outputs": str(Path(published["output_dir"]).relative_to(fulltext_root())),
            "summary": "TEST_RUN_SUMMARY.md",
        },
    )

    return {
        "corpus": corpus,
        "long_df": long_df,
        "items_df": items_df,
        "results": results,
        "integrity": integrity,
        "semantic": semantic,
        "published": published,
        "figures": figures,
        "graph_path": graph_path,
        "summary_text": summary_text,
        "output_root": fulltext_root(),
    }
