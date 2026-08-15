from __future__ import annotations

"""Did the coding actually work? Integrity and evidence checks for scheme 3.

Reliability tells you whether the coders agree. It does not tell you whether they
read the paper. At full-text level that second question is the more important
one, because the scheme asks for verbatim evidence, and a quote is the one part
of the output that can be checked against ground truth.

Three checks are implemented.

1. **Completeness.** Every (paper, model) cell present, no failed codings, no
   empty required fields.
2. **Quote verification.** Every verbatim quote is looked up in the source text.
   A quote is ``exact`` when its normalized form appears literally, ``near`` when
   most of its word 5-grams appear (a quote with a typo, an expanded ligature, or
   a dropped citation marker), and ``unverified`` otherwise. An unverified quote
   is either a paraphrase presented as a quote or an invention, and both are
   defects for this scheme.
3. **Evidence discipline.** Whether the integration ladder is actually backed by
   evidence: for every coding that grades a pair above ``mentioned``, is there a
   quoted claim behind it? This is the check that decides whether the integration
   numbers can be trusted at all.
"""

import json
import re
from pathlib import Path

import pandas as pd

from bps_review.fulltext.config import (
    COUNT_FIELDS,
    MODEL_LABELS,
    PAIRWISE_DEPTH,
    reliability_dir,
)
from bps_review.utils.io import write_csv, write_json


NGRAM = 5
NEAR_MATCH_THRESHOLD = 0.6
MIN_QUOTE_WORDS = 4

PAIR_TO_FIELD = {
    "bio_psych": "integration_bio_psych",
    "psych_social": "integration_psych_social",
    "bio_social": "integration_bio_social",
}


def normalize_for_match(text: str) -> str:
    """Lowercase, strip punctuation and whitespace differences, keep word order."""
    lowered = str(text or "").lower()
    lowered = lowered.replace("’", "'").replace("‘", "'")
    lowered = lowered.replace("“", '"').replace("”", '"')
    lowered = re.sub(r"[^a-z0-9']+", " ", lowered)
    return " ".join(lowered.split())


def _ngrams(words: list[str], size: int = NGRAM) -> list[str]:
    if len(words) <= size:
        return [" ".join(words)] if words else []
    return [" ".join(words[index : index + size]) for index in range(len(words) - size + 1)]


def verify_quote(quote: str, normalized_source: str) -> tuple[str, float]:
    """Return (status, coverage) for one quote against one normalized source text."""
    normalized = normalize_for_match(quote)
    words = normalized.split()
    if len(words) < MIN_QUOTE_WORDS:
        return "too_short_to_check", float("nan")
    if normalized in normalized_source:
        return "exact", 1.0
    grams = _ngrams(words)
    if not grams:
        return "unverified", 0.0
    hits = sum(1 for gram in grams if gram in normalized_source)
    coverage = hits / len(grams)
    if coverage >= NEAR_MATCH_THRESHOLD:
        return "near", coverage
    return "unverified", coverage


def verify_all_quotes(items_df: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    """Check every extracted quote against the full text it was taken from."""
    if items_df.empty:
        return pd.DataFrame()
    sources = {
        record["record_id"]: normalize_for_match(
            f"{record.get('title', '')} {record.get('abstract', '')} {record.get('body_text', '')}"
        )
        for record in records
    }
    rows = []
    for row in items_df.to_dict(orient="records"):
        quote = str(row.get("quote", "") or "")
        if not quote.strip():
            continue
        status, coverage = verify_quote(quote, sources.get(row["record_id"], ""))
        rows.append(
            {
                "record_id": row["record_id"],
                "model_label": row["model_label"],
                "extraction_field": row["extraction_field"],
                "label_normalized": row.get("label_normalized", ""),
                "quote_words": len(quote.split()),
                "verification": status,
                "ngram_coverage": coverage,
                "quote": quote,
            }
        )
    return pd.DataFrame(rows)


def quote_verification_summary(verification: pd.DataFrame) -> pd.DataFrame:
    """Per model: how much of the quoted evidence is really in the source text."""
    if verification.empty:
        return pd.DataFrame()
    checkable = verification[verification["verification"] != "too_short_to_check"]
    rows = []
    for model in MODEL_LABELS:
        subset = checkable[checkable["model_label"] == model]
        if subset.empty:
            continue
        rows.append(
            {
                "model_label": model,
                "n_quotes": int(len(subset)),
                "exact_rate": float((subset["verification"] == "exact").mean()),
                "near_rate": float((subset["verification"] == "near").mean()),
                "verified_rate": float(subset["verification"].isin(["exact", "near"]).mean()),
                "unverified_rate": float((subset["verification"] == "unverified").mean()),
                "mean_quote_words": float(subset["quote_words"].mean()),
                "mean_ngram_coverage": float(subset["ngram_coverage"].mean()),
            }
        )
    return pd.DataFrame(rows)


def quote_verification_by_field(verification: pd.DataFrame) -> pd.DataFrame:
    if verification.empty:
        return pd.DataFrame()
    checkable = verification[verification["verification"] != "too_short_to_check"]
    return (
        checkable.assign(verified=checkable["verification"].isin(["exact", "near"]).astype(float))
        .groupby("extraction_field")
        .agg(n_quotes=("verified", "size"), verified_rate=("verified", "mean"),
             mean_quote_words=("quote_words", "mean"))
        .reset_index()
        .sort_values("n_quotes", ascending=False)
    )


def completeness_report(long_df: pd.DataFrame, n_expected_papers: int) -> dict[str, object]:
    """Is the (paper x model) grid complete, and did every cell really code?"""
    expected = n_expected_papers * len(MODEL_LABELS)
    method_counts = long_df["coding_method"].value_counts().to_dict()
    per_model = (
        long_df.groupby("model_label")["coding_method"]
        .apply(lambda values: float((values == "llm_structured").mean()))
        .to_dict()
    )
    return {
        "expected_codings": expected,
        "actual_codings": int(len(long_df)),
        "grid_complete": bool(len(long_df) == expected),
        "coding_method_counts": method_counts,
        "structured_share_per_model": {key: round(value, 4) for key, value in per_model.items()},
        "n_failed_codings": int(method_counts.get("coding_failed", 0)),
        "n_empty_coding_rationale": int((long_df["coding_rationale"].astype(str).str.strip() == "").sum()),
        "papers_missing_any_model": sorted(
            long_df.groupby("record_id")["model_label"].nunique()
            .loc[lambda series: series < len(MODEL_LABELS)].index.tolist()
        ),
    }


def integration_evidence_discipline(long_df: pd.DataFrame) -> pd.DataFrame:
    """Is a graded integration actually backed by a quoted claim?

    For every coding and every domain pair graded above ``mentioned``, this asks
    whether the coder returned at least one integration claim for that pair. A
    graded ladder with no evidence behind it is the failure mode this scheme is
    most exposed to, so it is measured rather than assumed.
    """
    rows = []
    for row in long_df.to_dict(orient="records"):
        try:
            claims = json.loads(row.get("integration_claims") or "[]")
        except json.JSONDecodeError:
            claims = []
        claimed_pairs = {
            str(claim.get("domains_linked", "")) for claim in claims
            if isinstance(claim, dict) and str(claim.get("claim_verbatim", "")).strip()
        }
        for pair, field in PAIR_TO_FIELD.items():
            level = str(row.get(field, "none"))
            if PAIRWISE_DEPTH.get(level, 0) >= 2:   # descriptive or higher
                rows.append(
                    {
                        "record_id": row["record_id"],
                        "model_label": row["model_label"],
                        "domain_pair": pair,
                        "integration_level": level,
                        "has_quoted_claim": "yes" if pair in claimed_pairs else "no",
                    }
                )
        triadic = str(row.get("integration_triadic", "none"))
        if triadic != "none":
            rows.append(
                {
                    "record_id": row["record_id"],
                    "model_label": row["model_label"],
                    "domain_pair": "triadic",
                    "integration_level": triadic,
                    "has_quoted_claim": "yes" if "triadic" in claimed_pairs else "no",
                }
            )
    return pd.DataFrame(rows)


def evidence_discipline_summary(discipline: pd.DataFrame) -> pd.DataFrame:
    if discipline.empty:
        return pd.DataFrame()
    return (
        discipline.assign(backed=(discipline["has_quoted_claim"] == "yes").astype(float))
        .groupby("model_label")
        .agg(n_graded_links=("backed", "size"), share_backed_by_quote=("backed", "mean"))
        .reset_index()
    )


def extraction_yield(long_df: pd.DataFrame) -> pd.DataFrame:
    """Total and per-paper extraction volume, per model and category."""
    rows = []
    for model in MODEL_LABELS:
        subset = long_df[long_df["model_label"] == model]
        if subset.empty:
            continue
        entry: dict[str, object] = {"model_label": model, "n_papers": int(subset["record_id"].nunique())}
        for column in COUNT_FIELDS:
            if column in subset.columns:
                values = pd.to_numeric(subset[column], errors="coerce")
                entry[f"total_{column[2:]}"] = int(values.sum())
                entry[f"mean_{column[2:]}"] = round(float(values.mean()), 2)
        rows.append(entry)
    return pd.DataFrame(rows)


def spine_coverage(items_df: pd.DataFrame) -> pd.DataFrame:
    """How much of what was extracted lands on the project vocabularies.

    Every extracted item carries both readings of its label: what the coder wrote
    and, where a vocabulary applies, what that maps onto. This table compares the
    two per extraction list. A low controlled share is a statement about the
    ontology rather than about the coder: it says the literature is naming things
    the project vocabulary does not yet carry, and the off-spine labels are the
    candidate list for extending it.
    """
    if items_df.empty or "anchor_controlled" not in items_df.columns:
        return pd.DataFrame()
    mapped = items_df[items_df["anchor_vocabulary"].astype(str).str.len() > 0].copy()
    if mapped.empty:
        return pd.DataFrame()
    mapped["on_spine"] = (mapped["anchor_controlled"] == "yes").astype(float)
    mapped["has_anchor"] = (mapped["anchor_label"].astype(str).str.len() > 0).astype(float)
    table = (
        mapped.groupby("extraction_field")
        .agg(n_items=("on_spine", "size"),
             anchored_share=("has_anchor", "mean"),
             controlled_share=("on_spine", "mean"),
             distinct_raw_labels=("label_raw", "nunique"),
             distinct_anchors=("anchor_label", "nunique"))
        .reset_index()
        .sort_values("n_items", ascending=False)
    )
    return table


def off_spine_labels(items_df: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
    """What the reviews named that the project vocabularies do not carry.

    Shown in the coders' own words, most frequent first. This is the working list
    for extending the ontology after expert evaluation, so it deliberately keeps
    the raw label rather than a normalized approximation of it.
    """
    if items_df.empty or "anchor_controlled" not in items_df.columns:
        return pd.DataFrame()
    subset = items_df[(items_df["anchor_controlled"] == "no")
                      & (items_df["anchor_vocabulary"].astype(str).str.len() > 0)]
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.groupby(["extraction_field", "label_normalized"])
        .agg(n_extractions=("record_id", "size"), n_papers=("record_id", "nunique"),
             n_models=("model_label", "nunique"))
        .reset_index()
        .sort_values(["n_papers", "n_extractions"], ascending=False)
        .head(top_n)
    )


def corpus_extraction_totals(long_df: pd.DataFrame, items_df: pd.DataFrame) -> dict[str, object]:
    """Headline numbers on how much material the run harvested."""
    numeric = lambda column: pd.to_numeric(long_df[column], errors="coerce")  # noqa: E731
    totals = {
        "total_extracted_items": int(numeric("n_extracted_items").sum()),
        "total_evidence_quotes": int(numeric("n_evidence_quotes").sum()),
        "mean_items_per_coding": round(float(numeric("n_extracted_items").mean()), 2),
        "mean_items_per_paper_across_models": round(
            float(numeric("n_extracted_items").sum() / max(1, long_df["record_id"].nunique())), 2
        ),
    }
    if not items_df.empty:
        totals["distinct_raw_labels"] = int(items_df["label_raw"].nunique())
        totals["distinct_normalized_labels"] = int(items_df["label_normalized"].nunique())
        totals["items_per_field"] = items_df["extraction_field"].value_counts().to_dict()
        if "anchor_controlled" in items_df.columns:
            mapped = items_df[items_df["anchor_vocabulary"].astype(str).str.len() > 0]
            if not mapped.empty:
                totals["controlled_anchor_share"] = round(
                    float((mapped["anchor_controlled"] == "yes").mean()), 4)
                totals["n_off_spine_labels"] = int(
                    mapped[mapped["anchor_controlled"] == "no"]["label_normalized"].nunique())
    if "n_open_list_entries" in long_df.columns:
        totals["total_open_list_entries"] = int(
            pd.to_numeric(long_df["n_open_list_entries"], errors="coerce").sum())
    return totals


def build_integrity(long_df: pd.DataFrame, items_df: pd.DataFrame, records: list[dict],
                    write: bool = True, out_dir: Path | None = None) -> dict[str, object]:
    completeness = completeness_report(long_df, n_expected_papers=len(records))
    verification = verify_all_quotes(items_df, records)
    per_model_quotes = quote_verification_summary(verification)
    per_field_quotes = quote_verification_by_field(verification)
    yield_table = extraction_yield(long_df)
    discipline = integration_evidence_discipline(long_df)
    discipline_summary = evidence_discipline_summary(discipline)
    totals = corpus_extraction_totals(long_df, items_df)
    spine = spine_coverage(items_df)
    off_spine = off_spine_labels(items_df)

    checkable = (verification[verification["verification"] != "too_short_to_check"]
                 if not verification.empty else verification)
    summary = {
        "completeness": completeness,
        "extraction_totals": totals,
        "quote_verification": {
            "n_quotes_checked": int(len(checkable)) if not verification.empty else 0,
            "verified_rate": float(checkable["verification"].isin(["exact", "near"]).mean())
            if not verification.empty and len(checkable) else None,
            "exact_rate": float((checkable["verification"] == "exact").mean())
            if not verification.empty and len(checkable) else None,
            "per_model_verified_rate": {
                row["model_label"]: round(float(row["verified_rate"]), 4)
                for _, row in per_model_quotes.iterrows()
            } if not per_model_quotes.empty else {},
        },
        "evidence_discipline": {
            "n_graded_links": int(len(discipline)),
            "share_backed_by_quote": float((discipline["has_quoted_claim"] == "yes").mean())
            if not discipline.empty else None,
            "per_model": {
                row["model_label"]: round(float(row["share_backed_by_quote"]), 4)
                for _, row in discipline_summary.iterrows()
            } if not discipline_summary.empty else {},
        },
    }

    if write:
        out = Path(out_dir) if out_dir is not None else reliability_dir()
        write_csv(out / "09_extraction_yield.csv", yield_table)
        if not per_model_quotes.empty:
            write_csv(out / "10_quote_verification_by_model.csv", per_model_quotes)
        if not per_field_quotes.empty:
            write_csv(out / "11_quote_verification_by_field.csv", per_field_quotes)
        if not verification.empty:
            write_csv(out / "12_unverified_quotes.csv",
                      verification[verification["verification"] == "unverified"])
        if not discipline.empty:
            write_csv(out / "13_integration_evidence_discipline.csv", discipline)
        if not spine.empty:
            write_csv(out / "14_ontology_spine_coverage.csv", spine)
        if not off_spine.empty:
            write_csv(out / "15_off_spine_labels.csv", off_spine)
        write_json(out / "integrity_summary.json", summary)

    return {
        "completeness": completeness,
        "quote_verification": verification,
        "quote_verification_by_model": per_model_quotes,
        "quote_verification_by_field": per_field_quotes,
        "extraction_yield": yield_table,
        "evidence_discipline": discipline,
        "evidence_discipline_by_model": discipline_summary,
        "spine_coverage": spine,
        "off_spine_labels": off_spine,
        "extraction_totals": totals,
        "summary": summary,
    }


def label_catalog(items_df: pd.DataFrame, field: str, top_n: int = 30) -> pd.DataFrame:
    """The most frequently extracted labels for one field, across all models.

    This is a look at what the corpus contains, not a synthesis: the actual
    synthesis happens later, on the screened corpus.
    """
    if items_df.empty:
        return pd.DataFrame()
    subset = items_df[items_df["extraction_field"] == field]
    if subset.empty:
        return pd.DataFrame()
    grouped = (
        subset.groupby("label_normalized")
        .agg(n_extractions=("record_id", "size"), n_papers=("record_id", "nunique"),
             n_models=("model_label", "nunique"))
        .reset_index()
        .sort_values(["n_papers", "n_extractions"], ascending=False)
    )
    return grouped[grouped["label_normalized"].astype(str).str.len() > 1].head(top_n)
