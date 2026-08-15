from __future__ import annotations

"""Rebuild the knowledge graph of a full-text run from the tables already on disk.

The graph is a review surface, not a result: everything it shows was written by
the coding step and checked by the integrity step. This module therefore rebuilds
it from the persisted tables alone, with no API call and no re-derivation, so the
graph can be regenerated after a change to the layout or the grouping without
touching the run it describes.

``run_fulltext_testrun_pipeline`` builds the graph as part of a full run. This
module is the entry point for the other case: an existing run whose coding is
final.
"""

from pathlib import Path

import pandas as pd

from bps_review.fulltext.analysis.integrity import verify_all_quotes
from bps_review.fulltext.config import (
    graph_dir,
    items_csv,
    long_codings_csv,
)
from bps_review.fulltext.corpus.pmc import load_corpus, load_corpus_records
from bps_review.graph import build_knowledge_graph


GRAPH_TITLE = "FULL-TEXT CODING SCHEME (test run)"


def _quote_verification(items_df: pd.DataFrame) -> pd.DataFrame:
    """Quote verdicts for the graph, or an empty frame when the texts are local-only.

    The full texts are deliberately not carried in Git, so a clone without them
    still builds a complete graph; it simply cannot show whether a quote was
    found in its source article.
    """
    if items_df.empty:
        return pd.DataFrame()
    try:
        records = load_corpus_records()
    except FileNotFoundError:
        return pd.DataFrame()
    if not any(record.get("body_text") for record in records):
        return pd.DataFrame()
    return verify_all_quotes(items_df, records)


def export_fulltext_graph(
    output_dir: Path | None = None,
    run_title: str = GRAPH_TITLE,
    run_subtitle: str | None = None,
    verbose: bool = True,
) -> dict:
    """Write the knowledge graph of the cached full-text run and report what it holds."""
    path = long_codings_csv()
    if not path.exists():
        raise FileNotFoundError(
            f"No cached full-text coding table at {path}. Run the full-text test run first."
        )
    long_df = pd.read_csv(path).fillna("")
    items_df = pd.read_csv(items_csv()).fillna("") if items_csv().exists() else pd.DataFrame()
    corpus = load_corpus()
    verification = _quote_verification(items_df)
    if verbose:
        print(f"  codings: {len(long_df)}, items: {len(items_df)}, "
              f"verified quotes: {len(verification)}")

    n_papers = int(long_df["record_id"].nunique())
    n_providers = int(long_df["model_label"].nunique())
    subtitle = run_subtitle or (
        f"{n_papers} open-access reviews coded by {n_providers} independent providers"
    )
    index_path = build_knowledge_graph(
        corpus_df=corpus,
        long_df=long_df,
        items_df=items_df,
        output_dir=Path(output_dir) if output_dir is not None else graph_dir(),
        run_title=run_title,
        run_subtitle=subtitle,
        verification_df=verification if not verification.empty else None,
    )
    return {
        "index": str(index_path),
        "papers": n_papers,
        "providers": n_providers,
        "codings": int(len(long_df)),
        "extracted_items": int(len(items_df)),
        "quotes_verified": int(len(verification)),
    }
