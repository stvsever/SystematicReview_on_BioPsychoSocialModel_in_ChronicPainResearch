from __future__ import annotations

"""Build and load the abstract-level test-run corpus.

The corpus is a fresh sample from the registered review's operational PubMed
query, so the test run starts where the real review starts rather than from a
convenience file. The sampling rule is deliberately simple and reproducible:

1. run the operational query inside the protocol's date window, sorted by
   publication date, and retrieve a candidate pool;
2. drop records without a usable abstract, because an abstract-level scheme
   cannot be applied to a title alone;
3. keep the most recent N records that survive.

Everything about the retrieval is written to a manifest, so the sample can be
described exactly in the notebook and rebuilt later.
"""

import time
import xml.etree.ElementTree as ET

import pandas as pd

from bps_review.pilot.config import (
    CANDIDATE_POOL,
    PUBMED_QUERY_KEY,
    TESTRUN_SAMPLE_SIZE,
    corpus_manifest_json,
    input_csv,
)
from bps_review.search.pubmed import (
    _parse_article,
    _request,
    load_query,
    operational_date_window,
)
from bps_review.utils.io import utc_timestamp, write_csv, write_json


FETCH_BATCH = 100
NCBI_PAUSE_SECONDS = 0.34
MIN_ABSTRACT_CHARS = 250

CORPUS_COLUMNS = [
    "record_id",
    "pmid",
    "pmcid",
    "doi",
    "title",
    "abstract",
    "journal",
    "year",
    "publication_date",
    "publication_types",
    "keywords",
    "mesh_terms",
    "language",
    "authors",
    "pubmed_url",
]


def build_corpus(
    sample_size: int = TESTRUN_SAMPLE_SIZE,
    pool: int = CANDIDATE_POOL,
    query_key: str = PUBMED_QUERY_KEY,
    verbose: bool = True,
) -> pd.DataFrame:
    """Retrieve, filter, and persist the abstract-level test-run sample."""
    query = load_query(query_key)
    window = operational_date_window()
    search_date = utc_timestamp().split("T", 1)[0]

    esearch = _request(
        "esearch.fcgi",
        {
            "db": "pubmed",
            "term": query["string"],
            "retmode": "json",
            "retmax": pool,
            "mindate": window["start"],
            "maxdate": window["end"],
            "datetype": "pdat",
            "sort": "pub date",
        },
    ).json()
    result = esearch["esearchresult"]
    total_hits = int(result.get("count", 0))
    pmids = list(result.get("idlist", []))
    if verbose:
        print(f"{total_hits} records match the operational query; retrieving {len(pmids)} for screening", flush=True)

    records: list[dict[str, str]] = []
    for start in range(0, len(pmids), FETCH_BATCH):
        batch = pmids[start : start + FETCH_BATCH]
        response = _request(
            "efetch.fcgi",
            {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"},
        )
        root = ET.fromstring(response.text)
        for article in root.findall("PubmedArticle"):
            records.append(_parse_article(article, query_key, query["label"], search_date))
        if verbose:
            print(f"  parsed {min(start + FETCH_BATCH, len(pmids))}/{len(pmids)} records", flush=True)
        time.sleep(NCBI_PAUSE_SECONDS)

    frame = pd.DataFrame(records).fillna("")
    n_parsed = len(frame)
    frame["abstract"] = frame["abstract"].astype(str)
    with_abstract = frame[frame["abstract"].str.len() >= MIN_ABSTRACT_CHARS].copy()
    n_with_abstract = len(with_abstract)

    with_abstract["_year"] = pd.to_numeric(with_abstract.get("year"), errors="coerce")
    with_abstract = with_abstract.sort_values(["_year", "pmid"], ascending=[False, False])
    selected = with_abstract.head(sample_size).drop(columns=["_year"]).reset_index(drop=True)
    # A test-run record id that is stable, sortable, and traceable to its PMID.
    selected["record_id"] = [f"A{index + 1:03d}_{row.pmid}" for index, row in selected.iterrows()]

    columns = [column for column in CORPUS_COLUMNS if column in selected.columns]
    selected = selected[columns]
    write_csv(input_csv(), selected)

    manifest = {
        "built_at_utc": utc_timestamp(),
        "source": "MEDLINE via PubMed (NCBI E-utilities)",
        "query_key": query_key,
        "query_label": query["label"],
        "query_string": query["string"],
        "date_window": window,
        "sort": "publication date, most recent first",
        "total_query_hits": total_hits,
        "candidates_retrieved": len(pmids),
        "candidates_parsed": n_parsed,
        "with_usable_abstract": n_with_abstract,
        "min_abstract_chars": MIN_ABSTRACT_CHARS,
        "sample_size": len(selected),
        "with_pmcid": int((selected.get("pmcid", pd.Series(dtype=str)).astype(str).str.strip() != "").sum()),
        "year_range": [
            str(pd.to_numeric(selected["year"], errors="coerce").min()),
            str(pd.to_numeric(selected["year"], errors="coerce").max()),
        ],
    }
    write_json(corpus_manifest_json(), manifest)

    if verbose:
        print(f"  {n_with_abstract}/{n_parsed} records carry a usable abstract", flush=True)
        print(f"  sample: {len(selected)} records written to {input_csv().name}", flush=True)
    return selected


def load_corpus() -> pd.DataFrame:
    """Load the persisted abstract sample."""
    return pd.read_csv(input_csv()).fillna("")


def ensure_corpus(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Build the sample, or load the one already on disk."""
    if input_csv().exists() and not force:
        return load_corpus()
    return build_corpus(verbose=verbose)


def load_testrun_records() -> list[dict[str, str]]:
    """The sample as coding records: exactly the fields the scheme may read."""
    frame = load_corpus()
    fields = ["record_id", "title", "abstract", "journal", "year", "publication_types"]
    available = [field for field in fields if field in frame.columns]
    return frame[available].astype(str).to_dict(orient="records")
