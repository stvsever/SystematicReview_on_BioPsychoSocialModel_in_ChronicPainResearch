from __future__ import annotations

"""Every output of a full-text coding run, written in readable form.

The runner produces one row per (paper, provider) with each structured
extraction list held as a JSON blob and each open list crammed into a single
cell. That shape is right for a program and wrong for a person: it shows a
record id instead of a citation, and a paragraph of JSON instead of the twelve
biological factors a provider actually named.

This module writes the same run as the tables it is read in:

* every table carries the citation, title, authors, journal, and DOI next to the
  record id, so a row can be traced to a paper without a lookup;
* every open list is split across numbered columns, one item per column;
* every extracted item becomes its own row with its own typed columns, filed per
  extraction category, and pivoted wide by paper, by provider, and by both;
* every API call behind the run is a row, with its status, retries, timing,
  tokens, and cost;
* everything is also collected into one Excel workbook with frozen headers.

The extraction categories are ordered for this review rather than for the coding
form: the factors that carry each domain come first, then the links drawn
between them, then the biopsychosocial label itself, then the apparatus of
frameworks and instruments, then the critique. A directory listing therefore
answers "where are the biological factors" without opening anything.

These tables are the run's store, not a copy of one. Everything the coding schema
defines is present in them, so :func:`load_run_from_tables` reads a cached run
back from here and no second, unreadable set of files is kept alongside.

Nothing here derives, judges, or recomputes. The derived columns come from the
pipeline unchanged.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from bps_review.fulltext.coding import schema as S
from bps_review.utils.io import ensure_parent, write_csv, write_json


# --------------------------------------------------------------------------
# Column groups of the paper-by-provider table, in reading order.
# --------------------------------------------------------------------------
BIBLIOGRAPHY_COLUMNS = (
    "record_id",
    "citation",
    "title",
    "authors",
    "publication_year",
    "journal",
    "doi",
    "doi_url",
)

CODER_COLUMNS = ("model_order", "model_label", "provider", "model_id")

# The headline reading of a paper: what it was judged to be, and how much it
# yielded. These are the columns a reviewer scans first.
VERDICT_COLUMNS = (
    "fulltext_eligibility",
    "fulltext_exclusion_reason",
    "bps_typology",
    "derived_typology",
    "typology_matches_derived",
    "overall_balance",
    "conceptual_yield",
    "synthesis_priority",
    "n_extracted_items",
    "n_evidence_quotes",
)

# What kind of paper it is, and about which pain, in which population.
CONTEXT_COLUMNS = (
    "review_track",
    "source_type",
    "icd11_pain_category",
    "population",
    "care_setting",
    "primary_discipline",
    "pain_condition_detail",
    "context_note",
    "quality_assessment_reported",
)

# How the biopsychosocial label itself is used and defined (RQ1).
BPS_LABEL_COLUMNS = (
    "bps_label_used",
    "bps_primary_function",
    "bps_definition_status",
    "bps_has_substantive_function",
    "bps_function_set",
    "n_bps_functions",
    "bps_usage_sections",
    "n_bps_usage_sections",
    "bps_operationalization_summary",
)

# How deeply each domain is covered, on the four-rung ladder and as a depth score.
COVERAGE_COLUMNS = (
    "domain_coverage_bio",
    "domain_coverage_psych",
    "domain_coverage_social",
    "coverage_lifestyle",
    "coverage_spiritual_existential",
    "coverage_depth_bio",
    "coverage_depth_psych",
    "coverage_depth_social",
    "coverage_total",
    "domains_present",
)

# Whether the domains are actually related to each other, and how deeply (RQ2).
INTEGRATION_COLUMNS = (
    "integration_bio_psych",
    "integration_psych_social",
    "integration_bio_social",
    "integration_triadic",
    "pairwise_depth_total",
    "pairwise_depth_max",
    "triadic_depth",
    "integration_index",
    "n_triadic_claims",
    "n_named_integration_edges",
    "integration_mechanism_summary",
)

# The psychological concepts and the relations drawn between them (RQ3).
CONCEPT_COLUMNS = (
    "concept_definitions_present",
    "n_defined_concepts",
    "n_hierarchical_relations",
)

# How much of the extraction landed on the project vocabularies.
ONTOLOGY_COLUMNS = (
    "n_subdomains_bio",
    "n_subdomains_psych",
    "n_subdomains_social",
    "n_subdomains_named",
    "n_emergent_labels",
    "n_labels_checked",
    "controlled_label_share",
)

# Open lists of the record itself, semicolon-joined by the runner. Each becomes
# one column per item.
SPLIT_LIST_COLUMNS = tuple(S.OPEN_LIST_FIELDS)

FREE_TEXT_COLUMNS = ("synthesis_note", "coding_rationale")

PRESENCE_COLUMNS = tuple(
    f"present_{name}"
    for name in (
        "bps_usage_evidence",
        "bps_definition",
        "integration_evidence",
        "triadic_claim",
        "named_integration_edge",
        "biological_factors",
        "social_factors",
        "other_domain_factors",
        "psychological_concepts",
        "defined_concepts",
        "concept_relations",
        "hierarchical_relation",
        "theoretical_frameworks",
        "instruments",
        "conceptual_problems",
        "domain_evidence_bio",
        "domain_evidence_psych",
        "domain_evidence_social",
    )
)

COUNT_COLUMNS = (*(f"n_{name}" for name in S.ITEM_MODELS), "n_open_list_entries")

# The headline fields repeated per provider in the paper-level wide table.
WIDE_FIELDS = (
    "fulltext_eligibility",
    "bps_typology",
    "derived_typology",
    "overall_balance",
    "domain_coverage_bio",
    "domain_coverage_psych",
    "domain_coverage_social",
    "integration_bio_psych",
    "integration_psych_social",
    "integration_bio_social",
    "integration_triadic",
    "integration_index",
    "conceptual_yield",
    "synthesis_priority",
    "n_extracted_items",
)

# The fields repeated per paper in the provider-level wide table.
WIDE_FIELDS_PER_PAPER = (
    "fulltext_eligibility",
    "bps_typology",
    "integration_index",
    "n_extracted_items",
)

# Extraction categories, ordered for this review rather than for the coding
# form: first the named things that carry each domain, then the links drawn
# between them, then the biopsychosocial label itself, then the apparatus, then
# the critique and the quotes kept for the synthesis.
CATEGORY_ORDER = (
    ("biological_factors", "01_biological_factors"),
    ("psychological_concepts", "02_psychological_concepts"),
    ("social_factors", "03_social_factors"),
    ("other_domain_factors", "04_other_domain_factors"),
    ("domain_evidence", "05_domain_evidence"),
    ("integration_claims", "06_integration_claims"),
    ("concept_relations", "07_concept_relations"),
    ("bps_usage_instances", "08_bps_usage_instances"),
    ("bps_definitions", "09_bps_definitions"),
    ("theoretical_frameworks", "10_theoretical_frameworks"),
    ("instruments", "11_instruments"),
    ("conceptual_problems", "12_conceptual_problems"),
    ("key_quotes", "13_key_quotes"),
)

# One readable header order per structured item type. Every field of the item is
# listed; what changes is the order, so the thing's own name and its typed
# attributes come before the passage it was read from.
ITEM_FIELD_ORDER: dict[str, tuple[str, ...]] = {
    "biological_factors": (
        "factor_label",
        "subdomain_label",
        "mechanism_level",
        "factor_role",
        "evidence_basis",
        "factor_verbatim",
        "section_located",
    ),
    "psychological_concepts": (
        "concept_label",
        "concept_family",
        "definitional_status",
        "definition_source",
        "measure_named",
        "factor_role",
        "definition_verbatim",
        "section_located",
    ),
    "social_factors": (
        "factor_label",
        "subdomain_label",
        "social_level",
        "factor_role",
        "evidence_basis",
        "factor_verbatim",
        "section_located",
    ),
    "other_domain_factors": (
        "factor_label",
        "domain",
        "factor_role",
        "factor_verbatim",
        "section_located",
    ),
    "domain_evidence": (
        "domain",
        "coverage_level",
        "constructs_named",
        "subdomains_named",
        "evidence_verbatim",
        "section_located",
    ),
    "integration_claims": (
        "domains_linked",
        "integration_level",
        "source_factor_label",
        "target_factor_label",
        "direction",
        "mediator_or_moderator",
        "evidence_basis",
        "claim_verbatim",
        "mechanism_note",
        "section_located",
    ),
    "concept_relations": (
        "source_concept",
        "relation_type",
        "target_concept",
        "explicitly_stated",
        "relation_verbatim",
        "section_located",
    ),
    "bps_usage_instances": (
        "bps_function",
        "is_definitional",
        "attributed_source",
        "usage_verbatim",
        "section_located",
        "note",
    ),
    "bps_definitions": (
        "definition_type",
        "attributed_source",
        "elements_named",
        "definition_verbatim",
        "section_located",
    ),
    "theoretical_frameworks": (
        "framework_label",
        "role",
        "domains_covered",
        "attributed_source",
        "framework_verbatim",
        "section_located",
    ),
    "instruments": (
        "instrument_label",
        "abbreviation",
        "domain_measured",
        "construct_measured_as_stated",
        "role",
        "instrument_verbatim",
    ),
    "conceptual_problems": (
        "problem_type",
        "problem_scope",
        "affected_labels",
        "named_by_authors",
        "problem_verbatim",
        "note",
    ),
    "key_quotes": (
        "claim_type",
        "claim_verbatim",
        "section_located",
        "why_it_matters",
    ),
}

# The singular of each category, used in the file name of its item-level table:
# "one row per biological factor" reads better than "long".
CATEGORY_SINGULAR: dict[str, str] = {
    "biological_factors": "biological_factor",
    "psychological_concepts": "psychological_concept",
    "social_factors": "social_factor",
    "other_domain_factors": "other_domain_factor",
    "domain_evidence": "domain_evidence_passage",
    "integration_claims": "integration_claim",
    "concept_relations": "concept_relation",
    "bps_usage_instances": "bps_usage_instance",
    "bps_definitions": "bps_definition",
    "theoretical_frameworks": "theoretical_framework",
    "instruments": "instrument",
    "conceptual_problems": "conceptual_problem",
    "key_quotes": "key_quote",
}

# What an item is called in a table: its identifying keys, joined. Two of the
# thirteen are edges, so their name is the pair they connect.
ITEM_LABEL_COLUMN = "item_label"


def category_item_file(field: str) -> str:
    return f"01_one_row_per_{CATEGORY_SINGULAR.get(field, 'item')}.csv"


# --------------------------------------------------------------------------
# Bibliography.
# --------------------------------------------------------------------------
def _lead_author(authors: str) -> str:
    """The short author form of a citation: one name, two names, or et al."""
    names = [name.strip() for name in str(authors or "").split("|") if name.strip()]
    if not names:
        return "Unknown author"
    surname = lambda name: name.split(",")[0].split()[-1] if name.split() else name
    if len(names) == 1:
        return surname(names[0])
    if len(names) == 2:
        return f"{surname(names[0])} & {surname(names[1])}"
    return f"{surname(names[0])} et al."


def load_bibliography(corpus_dir: Path) -> pd.DataFrame:
    """The verified bibliography of a run, one row per record id.

    The corpus was retrieved from PubMed Central, so every record carries the
    publisher's own metadata: the DOI, the journal, the author list, and the year
    come from the article record rather than from a filename or a guess.
    """
    corpus = pd.read_csv(Path(corpus_dir) / "articles.csv").fillna("")
    doi = corpus["doi"].astype(str).str.strip()
    return pd.DataFrame(
        {
            "record_id": corpus["record_id"],
            "citation": [
                f"{_lead_author(authors)} ({year}), {journal}".strip().rstrip(",")
                for authors, year, journal in zip(
                    corpus["authors"], corpus["year"], corpus["journal"]
                )
            ],
            "title": corpus["title"],
            "authors": corpus["authors"].astype(str).str.replace(" | ", "; ", regex=False),
            "publication_year": corpus["year"],
            "journal": corpus["journal"],
            "doi": doi,
            "doi_url": doi.map(lambda value: f"https://doi.org/{value}" if value else ""),
        }
    ).sort_values("record_id").reset_index(drop=True)


def citation_columns(bibliography: pd.DataFrame) -> pd.DataFrame:
    """The bibliographic block prefixed to every table."""
    keep = [column for column in BIBLIOGRAPHY_COLUMNS if column in bibliography.columns]
    return bibliography[keep]


def build_papers_table(corpus_dir: Path) -> pd.DataFrame:
    """The run's paper list: full citation plus where the text came from.

    This is the table to open to find a paper again. It carries the identifiers
    that resolve to the source (DOI, PubMed, PubMed Central), the licence the
    text was retrieved under, the size of what was read, and the abstract-level
    reading that put the paper in this corpus.
    """
    corpus_dir = Path(corpus_dir)
    bibliography = load_bibliography(corpus_dir)
    corpus = pd.read_csv(corpus_dir / "articles.csv").fillna("")
    keep = [
        "record_id",
        "pmid",
        "pmcid",
        "pmc_url",
        "publication_types",
        "license",
        "n_sections",
        "abstract_chars",
        "body_chars",
        "text_file",
        "abstract_record_id",
        "abstract_stage3_priority",
        "abstract_typology",
        "abstract_msk_flag",
    ]
    keep = [column for column in keep if column in corpus.columns]
    merged = bibliography.merge(corpus[keep], on="record_id", how="left")
    return merged.rename(
        columns={
            "n_sections": "n_body_sections",
            "text_file": "fulltext_file",
            "abstract_record_id": "abstract_level_record_id",
            "abstract_stage3_priority": "abstract_level_priority",
            "abstract_typology": "abstract_level_typology",
            "abstract_msk_flag": "abstract_level_musculoskeletal_flag",
        }
    )


# --------------------------------------------------------------------------
# Splitting lists into columns.
# --------------------------------------------------------------------------
# The runner joins a record-level open list with a semicolon, and a list inside
# a structured item with a pipe. Both separators are undone here, each where it
# belongs, so nothing is silently glued together.
OPEN_LIST_SEPARATOR = ";"
ITEM_LIST_SEPARATOR = "|"


def _split_cell(value: object, separator: str = OPEN_LIST_SEPARATOR) -> list[str]:
    """The items of a joined cell. Only a genuinely missing value yields nothing."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(separator) if part.strip()]


def split_list_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Replace one joined column with a count plus one column per item.

    The width is the widest cell actually present, so no empty tail columns are
    written. The original column is dropped: its content is fully recoverable
    from the numbered columns, and keeping both is what made the table hard to
    read in the first place.
    """
    if column not in frame.columns:
        return frame
    parts = frame[column].map(_split_cell)
    width = int(parts.map(len).max() or 0)
    position = frame.columns.get_loc(column)
    frame = frame.drop(columns=[column])
    inserted = [(f"{column}_count", parts.map(len))]
    for index in range(width):
        inserted.append(
            (
                f"{column}_{index + 1}",
                parts.map(lambda items, i=index: items[i] if len(items) > i else ""),
            )
        )
    for offset, (name, values) in enumerate(inserted):
        frame.insert(position + offset, name, values)
    return frame


# --------------------------------------------------------------------------
# Paper-by-provider long table.
# --------------------------------------------------------------------------
def _bibliography_columns_in(frame: pd.DataFrame) -> list[str]:
    """Bibliography columns already on a frame, including merge-suffixed copies.

    A frame read back from these tables carries the citation block. Merging it
    again would not fail, it would quietly produce ``citation_x`` and
    ``citation_y``, so both the plain names and any pandas suffix are removed
    before the merge that attaches the block exactly once.
    """
    names = {column for column in BIBLIOGRAPHY_COLUMNS if column != "record_id"}
    suffixed = {f"{name}{suffix}" for name in names for suffix in ("_x", "_y")}
    return [column for column in frame.columns if column in names or column in suffixed]


def _flattened_item_pattern() -> re.Pattern[str]:
    return re.compile(rf"^({'|'.join(S.ITEM_MODELS)})_\d{{2}}_")


def build_codings_long(long_df: pd.DataFrame, bibliography: pd.DataFrame) -> pd.DataFrame:
    """One row per (paper, coder), citation first and every open list split out."""
    frame = long_df.copy().fillna("")
    frame = frame.drop(columns=[c for c in S.ITEM_MODELS if c in frame.columns])
    frame = frame.drop(columns=[c for c in ("llm_model",) if c in frame.columns])
    # A frame read back from these tables already carries the citation block and
    # the flattened item blocks. Drop both, so the merge below is the single
    # place that attaches the citation and the items are appended exactly once.
    frame = frame.drop(columns=_bibliography_columns_in(frame))
    flattened = _flattened_item_pattern()
    frame = frame.drop(columns=[c for c in frame.columns if flattened.match(c)])

    ordered = [
        *CODER_COLUMNS,
        "coding_method",
        *VERDICT_COLUMNS,
        *CONTEXT_COLUMNS,
        *BPS_LABEL_COLUMNS,
        *COVERAGE_COLUMNS,
        *INTEGRATION_COLUMNS,
        *CONCEPT_COLUMNS,
        *ONTOLOGY_COLUMNS,
        *SPLIT_LIST_COLUMNS,
        *COUNT_COLUMNS,
        *PRESENCE_COLUMNS,
        *FREE_TEXT_COLUMNS,
    ]
    present = [column for column in ordered if column in frame.columns]
    trailing = [
        column for column in frame.columns if column not in present and column != "record_id"
    ]
    frame = frame[["record_id", *present, *trailing]]

    merged = bibliography.merge(frame, on="record_id", how="right")
    lead = [column for column in BIBLIOGRAPHY_COLUMNS if column in merged.columns]
    rest = [column for column in merged.columns if column not in lead]
    merged = merged[[*lead, *rest]]
    for column in SPLIT_LIST_COLUMNS:
        merged = split_list_column(merged, column)
    # One list already has a count elsewhere in the row. Writing a second one
    # under a different name only invites the question of which is right.
    duplicate_counts = [
        name
        for name, existing in (("emergent_labels_count", "n_emergent_labels"),)
        if name in merged.columns and existing in merged.columns
    ]
    merged = merged.drop(columns=duplicate_counts)
    return merged.sort_values(["record_id", "model_order", "model_label"]).reset_index(drop=True)


def flatten_items_into_codings(
    codings_long: pd.DataFrame, items_df: pd.DataFrame
) -> pd.DataFrame:
    """Append every extracted item to its coding row, one column per field.

    The coding row summarizes what a provider found; without this it counts
    twelve biological factors without saying which ones. The runner kept them as
    a JSON blob, which is complete and unreadable. Here each item becomes its own
    block of named columns, so one row is the provider's entire reading of one
    paper at full resolution: every label, every verbatim quote, every typed
    attribute.

    The width follows the longest list actually present, so no empty tail columns
    are written, and the blocks come after the summary columns so the readable
    part of the row stays at the front.
    """
    if items_df.empty:
        return codings_long
    frame = codings_long.copy()
    keys = list(zip(frame["record_id"].astype(str), frame["model_label"].astype(str)))
    blocks: list[pd.DataFrame] = []

    for field, _ in CATEGORY_ORDER:
        subset = items_df[items_df["extraction_field"] == field]
        if subset.empty:
            continue
        columns = ITEM_FIELD_ORDER.get(field) or tuple(S.ITEM_MODELS[field].model_fields)
        by_key: dict[tuple[str, str], list[dict]] = {}
        for row in subset.sort_values("item_index").to_dict(orient="records"):
            payload = json.loads(row["item_json"]) if str(row.get("item_json", "")).strip() else {}
            by_key.setdefault((str(row["record_id"]), str(row["model_label"])), []).append(payload)
        width = max(len(values) for values in by_key.values())
        block: dict[str, list[str]] = {}
        for position in range(width):
            for column in columns:
                values = []
                for key in keys:
                    items = by_key.get(key, [])
                    payload = items[position] if len(items) > position else {}
                    value = payload.get(column, "")
                    if isinstance(value, list):
                        value = f" {ITEM_LIST_SEPARATOR} ".join(str(part) for part in value)
                    values.append("" if value is None else str(value))
                block[f"{field}_{position + 1:02d}_{column}"] = values
        blocks.append(pd.DataFrame(block, index=frame.index))
    return pd.concat([frame, *blocks], axis=1) if blocks else frame


# --------------------------------------------------------------------------
# Wide pivots of the coding table.
# --------------------------------------------------------------------------
def _model_slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")


def _provider_order(long_df: pd.DataFrame) -> list[str]:
    if "model_order" in long_df.columns:
        return list(dict.fromkeys(long_df.sort_values("model_order")["model_label"]))
    return sorted(long_df["model_label"].unique())


def _modal(values: list[str]) -> str:
    """The value most providers gave, or empty when they all gave a different one.

    With three coders a tie is a real outcome, not a rounding problem, and naming
    one of the tied values as the mode would invent a majority that does not
    exist. The counter preserves provider order, so the answer is the same on
    every run over the same coding.
    """
    counts = Counter(values)
    if not counts:
        return ""
    top = max(counts.values())
    winners = [value for value, count in counts.items() if count == top]
    return winners[0] if len(winners) == 1 else ""


def _agreement(values: list[str]) -> str:
    """How the providers landed: all on one value, most on one, or all apart."""
    counts = Counter(values)
    if not counts:
        return ""
    top = max(counts.values())
    if top == len(values):
        return "unanimous"
    if top > 1 and list(counts.values()).count(top) == 1:
        return "majority"
    return "no majority"


def build_codings_wide_per_paper(
    long_df: pd.DataFrame, bibliography: pd.DataFrame
) -> pd.DataFrame:
    """One row per paper, with a column block per provider side by side.

    This is the table that answers "did the providers read this paper the same
    way". The per-provider blocks are preceded by the agreement columns that
    summarize them, so disagreement on the two headline judgements, eligibility
    and typology, is visible without scrolling.
    """
    frame = long_df.copy().fillna("")
    providers = _provider_order(frame)

    rows: list[dict[str, object]] = []
    for record_id, group in frame.groupby("record_id"):
        indexed = group.set_index("model_label")
        eligibility = [
            str(indexed.loc[name, "fulltext_eligibility"])
            for name in providers
            if name in indexed.index
        ]
        typology = [
            str(indexed.loc[name, "bps_typology"])
            for name in providers
            if name in indexed.index and "bps_typology" in indexed.columns
        ]
        row: dict[str, object] = {
            "record_id": record_id,
            "n_providers": len(eligibility),
            "n_providers_include": sum(1 for value in eligibility if value == "include"),
            "eligibility_agreement": _agreement(eligibility),
            "modal_eligibility": _modal(eligibility),
            "typology_agreement": _agreement(typology),
            "modal_bps_typology": _modal(typology),
            "mean_integration_index": round(
                float(pd.to_numeric(group["integration_index"], errors="coerce").fillna(0).mean()), 4
            ),
            "total_extracted_items": int(
                pd.to_numeric(group["n_extracted_items"], errors="coerce").fillna(0).sum()
            ),
        }
        for name in providers:
            if name not in indexed.index:
                continue
            slug = _model_slug(name)
            for field in WIDE_FIELDS:
                if field in indexed.columns:
                    row[f"{slug}__{field}"] = indexed.loc[name, field]
        rows.append(row)

    wide = pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)
    merged = bibliography.merge(wide, on="record_id", how="right")
    lead = [column for column in BIBLIOGRAPHY_COLUMNS if column in merged.columns]
    rest = [column for column in merged.columns if column not in lead]
    return merged[[*lead, *rest]]


def build_codings_wide_per_provider(
    long_df: pd.DataFrame, bibliography: pd.DataFrame
) -> pd.DataFrame:
    """One row per provider, with a column block per paper side by side.

    The mirror image of the per-paper pivot: it answers "how did this provider
    behave across the whole corpus", which is the view that exposes a provider
    that includes everything, or that grades every integration one rung higher
    than the others.
    """
    frame = long_df.copy().fillna("")
    providers = _provider_order(frame)
    record_ids = sorted(frame["record_id"].unique())
    citations = dict(zip(bibliography["record_id"], bibliography["citation"]))

    rows: list[dict[str, object]] = []
    for order, name in enumerate(providers, start=1):
        group = frame[frame["model_label"] == name].set_index("record_id")
        eligibility = [
            str(group.loc[rid, "fulltext_eligibility"]) for rid in record_ids if rid in group.index
        ]
        methods = (
            group["coding_method"].astype(str).tolist() if "coding_method" in group.columns else []
        )
        row: dict[str, object] = {
            "provider_order": order,
            "model_label": name,
            "provider": group["provider"].iloc[0] if "provider" in group.columns and len(group) else "",
            "model_id": group["model_id"].iloc[0] if "model_id" in group.columns and len(group) else "",
            "n_papers": len(group),
            "n_coding_failed": sum(1 for value in methods if value == "coding_failed"),
            "n_include": sum(1 for value in eligibility if value == "include"),
            "n_exclude": sum(1 for value in eligibility if value == "exclude"),
            "n_uncertain": sum(1 for value in eligibility if value == "uncertain"),
            "mean_integration_index": round(
                float(pd.to_numeric(group["integration_index"], errors="coerce").fillna(0).mean()), 4
            ),
            "total_extracted_items": int(
                pd.to_numeric(group["n_extracted_items"], errors="coerce").fillna(0).sum()
            ),
            "mean_extracted_items_per_paper": round(
                float(pd.to_numeric(group["n_extracted_items"], errors="coerce").fillna(0).mean()), 2
            ),
        }
        for record_id in record_ids:
            if record_id not in group.index:
                continue
            for field in WIDE_FIELDS_PER_PAPER:
                if field in group.columns:
                    row[f"{record_id}__{field}"] = group.loc[record_id, field]
        rows.append(row)

    wide = pd.DataFrame(rows)
    # The paper column blocks are keyed by record id, so the citations they stand
    # for are carried alongside rather than left implicit.
    wide.attrs["citations"] = citations
    return wide


# --------------------------------------------------------------------------
# Extracted items.
# --------------------------------------------------------------------------
def _item_label(payload: dict, field: str) -> str:
    """What the item is called: its identifying keys, joined.

    Eleven of the thirteen categories are named things, so their name is one
    field. Concept relations and integration claims are edges, so their name is
    the pair they connect with the relation between them.
    """
    parts = [
        str(payload.get(key, "") or "").strip() for key in S.ITEM_LABEL_KEY.get(field, ())
    ]
    return " | ".join(part for part in parts if part)


def _item_frame(items_df: pd.DataFrame, field: str) -> pd.DataFrame:
    """Explode one extraction category's JSON payload into typed columns."""
    subset = items_df[items_df["extraction_field"] == field].copy()
    if subset.empty:
        return subset
    payloads = subset["item_json"].map(
        lambda raw: json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    )
    columns = ITEM_FIELD_ORDER.get(field) or tuple(S.ITEM_MODELS[field].model_fields)
    detail = pd.DataFrame(index=subset.index)
    detail[ITEM_LABEL_COLUMN] = payloads.map(lambda payload: _item_label(payload, field))
    for column in columns:
        detail[column] = payloads.map(
            lambda payload, key=column: f" {ITEM_LIST_SEPARATOR} ".join(
                str(part) for part in payload.get(key)
            )
            if isinstance(payload.get(key), list)
            else str(payload.get(key, "") or "")
        )
    # Two normalizations sit beside the item, and each is written only where it
    # means something. The first maps the item's own name onto its vocabulary,
    # and only three categories have one. The second says where the item attaches
    # to the project ontology, which is always a different field from the name,
    # so a filled cell never means the coder happened to use our word for the
    # thing itself. Writing either column where no vocabulary applies would
    # suggest a mapping that never happened.
    keep = ["record_id", "model_label", "item_index"]
    rename: dict[str, str] = {}
    if field in S.ITEM_IDENTITY_VOCAB:
        keep += ["label_normalized", "label_controlled"]
        rename.update(
            {
                "label_normalized": "label_normalized_for_matching",
                "label_controlled": "label_is_controlled",
            }
        )
    if field in S.ITEM_ANCHOR:
        keep += ["anchor_label", "anchor_controlled"]
        rename.update(
            {"anchor_label": "ontology_anchor", "anchor_controlled": "ontology_anchor_is_controlled"}
        )
    base = subset[keep].rename(columns=rename)
    return pd.concat([base, detail], axis=1)


def build_items_long(items_df: pd.DataFrame, bibliography: pd.DataFrame) -> pd.DataFrame:
    """Every extracted item as its own row, with the quote in its own column."""
    frame = items_df.copy().fillna("")
    frame = frame.drop(columns=[c for c in ("item_json", "model_id") if c in frame.columns])
    frame = frame.rename(
        columns={
            "extraction_field": "extraction_category",
            "label_raw": "label",
            "label_normalized": "label_normalized_for_matching",
            "label_vocabulary": "label_vocabulary",
            "label_controlled": "label_is_controlled",
            "anchor_label": "ontology_anchor",
            "anchor_vocabulary": "ontology_anchor_vocabulary",
            "anchor_controlled": "ontology_anchor_is_controlled",
            "quote": "verbatim_quote",
        }
    )
    # Where no vocabulary applies, the normalized label is only the cleaned
    # original. Blank it there rather than repeat the label, so a filled cell
    # always means a real mapping onto a controlled list happened.
    open_identity = ~frame["extraction_category"].isin(S.ITEM_IDENTITY_VOCAB)
    for column in ("label_normalized_for_matching", "label_is_controlled", "label_vocabulary"):
        if column in frame.columns:
            frame[column] = frame[column].astype(object)
            frame.loc[open_identity, column] = ""
    merged = bibliography.merge(frame, on="record_id", how="right")
    lead = [column for column in BIBLIOGRAPHY_COLUMNS if column in merged.columns]
    tail = [
        "model_label",
        "extraction_category",
        "item_index",
        "label",
        "verbatim_quote",
        "label_normalized_for_matching",
        "label_vocabulary",
        "label_is_controlled",
        "ontology_anchor",
        "ontology_anchor_vocabulary",
        "ontology_anchor_is_controlled",
    ]
    tail = [column for column in tail if column in merged.columns]
    rest = [column for column in merged.columns if column not in lead and column not in tail]
    merged = merged[[*lead, *tail, *rest]]
    category_rank = {field: index for index, (field, _) in enumerate(CATEGORY_ORDER)}
    merged["_rank"] = merged["extraction_category"].map(category_rank).fillna(99)
    merged = merged.sort_values(["record_id", "model_label", "_rank", "item_index"])
    return merged.drop(columns=["_rank"]).reset_index(drop=True)


def build_items_by_category(
    items_df: pd.DataFrame, bibliography: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """One table per extraction category, carrying only that category's fields."""
    tables: dict[str, pd.DataFrame] = {}
    for field, filename in CATEGORY_ORDER:
        detail = _item_frame(items_df, field)
        if detail.empty:
            continue
        merged = bibliography.merge(detail, on="record_id", how="right")
        lead = [column for column in BIBLIOGRAPHY_COLUMNS if column in merged.columns]
        rest = [column for column in merged.columns if column not in lead]
        merged = merged[[*lead, *rest]]
        tables[filename] = merged.sort_values(
            ["record_id", "model_label", "item_index"]
        ).reset_index(drop=True)
    return tables


# --------------------------------------------------------------------------
# Wide pivots of one extraction category.
# --------------------------------------------------------------------------
def build_category_wide_per_paper_and_provider(
    category_long: pd.DataFrame, field: str, bibliography: pd.DataFrame
) -> pd.DataFrame:
    """One row per (paper, provider): everything that provider found, in columns.

    The most direct answer to "which biological factors did this model read out
    of this paper", with the item's name and the sentence it came from side by
    side.
    """
    quote_column = S.ITEM_QUOTE_KEY.get(field, "")
    rows: list[dict[str, object]] = []
    for (record_id, provider), group in category_long.groupby(["record_id", "model_label"]):
        group = group.sort_values("item_index")
        row: dict[str, object] = {
            "record_id": record_id,
            "model_label": provider,
            "n_items": len(group),
        }
        for position, (_, item) in enumerate(group.iterrows(), start=1):
            row[f"item_{position}_label"] = item.get(ITEM_LABEL_COLUMN, "")
            if quote_column and quote_column in group.columns:
                row[f"item_{position}_quote"] = item.get(quote_column, "")
        rows.append(row)
    wide = pd.DataFrame(rows).sort_values(["record_id", "model_label"]).reset_index(drop=True)
    merged = bibliography.merge(wide, on="record_id", how="right")
    lead = [column for column in BIBLIOGRAPHY_COLUMNS if column in merged.columns]
    rest = [column for column in merged.columns if column not in lead]
    return merged[[*lead, *rest]]


def build_category_wide_per_paper(
    category_long: pd.DataFrame,
    field: str,
    bibliography: pd.DataFrame,
    providers: list[str] | None = None,
) -> pd.DataFrame:
    """One row per paper, with the names every provider gave, side by side.

    This is the agreement view for one category: the providers' lists next to
    each other show at a glance where they converged and where only one of them
    saw something.
    """
    providers = providers or sorted(category_long["model_label"].unique())
    rows: list[dict[str, object]] = []
    for record_id, group in category_long.groupby("record_id"):
        row: dict[str, object] = {"record_id": record_id, "n_items_total": len(group)}
        for provider in providers:
            subset = group[group["model_label"] == provider].sort_values("item_index")
            slug = _model_slug(provider)
            row[f"{slug}__n_items"] = len(subset)
            for position, label in enumerate(subset[ITEM_LABEL_COLUMN].tolist(), start=1):
                row[f"{slug}__item_{position}"] = label
        rows.append(row)
    wide = pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)
    merged = bibliography.merge(wide, on="record_id", how="right")
    lead = [column for column in BIBLIOGRAPHY_COLUMNS if column in merged.columns]
    rest = [column for column in merged.columns if column not in lead]
    return merged[[*lead, *rest]]


def build_category_wide_per_provider(
    category_long: pd.DataFrame, field: str, providers: list[str] | None = None
) -> pd.DataFrame:
    """One row per provider, with the names it found in every paper.

    The mirror image of the per-paper pivot, and the quickest way to see one
    provider's whole vocabulary for one category across the corpus.
    """
    providers = providers or sorted(category_long["model_label"].unique())
    record_ids = sorted(category_long["record_id"].unique())
    rows: list[dict[str, object]] = []
    for provider in providers:
        group = category_long[category_long["model_label"] == provider]
        row: dict[str, object] = {
            "model_label": provider,
            "n_items_total": len(group),
            "n_distinct_labels": int(group[ITEM_LABEL_COLUMN].nunique()),
        }
        for record_id in record_ids:
            subset = group[group["record_id"] == record_id].sort_values("item_index")
            row[f"{record_id}__n_items"] = len(subset)
            for position, label in enumerate(subset[ITEM_LABEL_COLUMN].tolist(), start=1):
                row[f"{record_id}__item_{position}"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def build_domain_factor_inventory(
    by_category: dict[str, pd.DataFrame], bibliography: pd.DataFrame
) -> pd.DataFrame:
    """Every named factor of every domain, in one table, one row per item.

    The three factor lists plus the fourth domain answer the review's scope
    question together, and reading them together is what shows that a paper
    carries eleven biological factors and one social one. The four tables have
    different typed columns, so this view keeps what they share (the domain, the
    name, the ontology subdomain, the role, the level, and the quote) and files
    the rest under its own table.
    """
    domain_of = {
        "01_biological_factors": "biological",
        "02_psychological_concepts": "psychological",
        "03_social_factors": "social",
        "04_other_domain_factors": "beyond the triad",
    }
    name_column = {
        "01_biological_factors": "factor_label",
        "02_psychological_concepts": "concept_label",
        "03_social_factors": "factor_label",
        "04_other_domain_factors": "factor_label",
    }
    level_column = {
        "01_biological_factors": "mechanism_level",
        "03_social_factors": "social_level",
        "04_other_domain_factors": "domain",
    }
    subdomain_column = {
        "01_biological_factors": "subdomain_label",
        "02_psychological_concepts": "concept_family",
        "03_social_factors": "subdomain_label",
    }
    frames: list[pd.DataFrame] = []
    for filename, domain in domain_of.items():
        table = by_category.get(filename)
        if table is None or table.empty:
            continue
        quote = S.ITEM_QUOTE_KEY.get(
            {name: field for field, name in CATEGORY_ORDER}[filename], ""
        )
        block = pd.DataFrame(
            {
                "record_id": table["record_id"],
                "model_label": table["model_label"],
                "item_index": table["item_index"],
                "domain": domain,
                "factor_label": table[name_column[filename]],
                "ontology_subdomain": table.get(subdomain_column.get(filename, ""), ""),
                "level_or_mechanism": table.get(level_column.get(filename, ""), ""),
                "factor_role": table.get("factor_role", ""),
                "verbatim_quote": table.get(quote, ""),
                "section_located": table.get("section_located", ""),
                "source_table": filename,
            }
        )
        frames.append(block)
    if not frames:
        return pd.DataFrame()
    stacked = pd.concat(frames, ignore_index=True)
    domain_rank = {"biological": 0, "psychological": 1, "social": 2, "beyond the triad": 3}
    stacked["_rank"] = stacked["domain"].map(domain_rank).fillna(9)
    stacked = stacked.sort_values(["record_id", "model_label", "_rank", "item_index"]).drop(
        columns=["_rank"]
    )
    merged = bibliography.merge(stacked, on="record_id", how="right")
    lead = [column for column in BIBLIOGRAPHY_COLUMNS if column in merged.columns]
    rest = [column for column in merged.columns if column not in lead]
    return merged[[*lead, *rest]].reset_index(drop=True)


# --------------------------------------------------------------------------
# The API calls behind the run.
# --------------------------------------------------------------------------
def _flatten(payload: dict, prefix: str = "") -> dict[str, object]:
    """Flatten a nested usage dict into `parent_child` columns."""
    flat: dict[str, object] = {}
    for key, value in (payload or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}_"))
        else:
            flat[name] = value
    return flat


def build_api_calls(
    audit_paths: list[Path], bibliography: pd.DataFrame, long_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per API call: what was sent, what came back, and what it cost.

    This is the complete content of the runner's per-provider audit trail, with
    the nested token-usage payload flattened into columns, so the provenance of
    every coding is readable without opening a JSON file.
    """
    rows: list[dict[str, object]] = []
    for path in sorted(audit_paths):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            usage = entry.pop("usage", {}) or {}
            rows.append({**entry, **_flatten(usage)})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    # The audit names the model by its OpenRouter id. The label the rest of the
    # run uses is added here, so a call can be filed under the same provider name
    # as everything else.
    if long_df is not None and "model_id" in frame.columns and "model_label" not in frame.columns:
        labels = dict(zip(long_df["model_id"].astype(str), long_df["model_label"].astype(str)))
        frame.insert(1, "model_label", frame["model_id"].astype(str).map(labels).fillna(""))
    merged = bibliography.merge(frame, on="record_id", how="right")
    lead = [column for column in BIBLIOGRAPHY_COLUMNS if column in merged.columns]
    order = [
        "model_label",
        "model_id",
        "status",
        "attempts",
        "seconds",
        "coding_text_chars",
        "text_reduced",
        "kept_share",
        "prompt_tokens",
        "completion_tokens",
        "completion_tokens_details_reasoning_tokens",
        "total_tokens",
        "cost",
    ]
    order = [column for column in order if column in merged.columns]
    rest = [column for column in merged.columns if column not in lead and column not in order]
    merged = merged[[*lead, *order, *rest]]
    sort_keys = [key for key in ("model_label", "model_id", "record_id") if key in merged.columns]
    return merged.sort_values(sort_keys).reset_index(drop=True)


def build_providers(manifest: dict) -> pd.DataFrame:
    """One row per provider: which model was called, and how it was configured."""
    runtime = manifest.get("model_runtime") or {}
    rows = []
    for entry in manifest.get("models") or []:
        label = entry.get("label", "")
        config = runtime.get(label) or {}
        reasoning = config.get("reasoning") or {}
        rows.append(
            {
                "provider_order": entry.get("order", ""),
                "model_label": label,
                "provider": entry.get("provider", ""),
                "model_id": entry.get("openrouter_id", ""),
                "max_output_tokens": config.get("max_output_tokens", ""),
                "reasoning_enabled": reasoning.get(
                    "enabled", "" if "effort" not in reasoning else True
                ),
                "reasoning_effort": reasoning.get("effort", ""),
                "max_workers": manifest.get("max_workers", ""),
            }
        )
    return pd.DataFrame(rows)


def build_usage_per_provider(manifest: dict) -> pd.DataFrame:
    """One row per provider: tokens spent and dollars billed."""
    usage = manifest.get("token_usage_by_model") or {}
    order = {entry.get("label"): entry.get("order", 99) for entry in manifest.get("models") or []}
    rows = [{"model_label": label, **totals} for label, totals in usage.items()]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["provider_order"] = frame["model_label"].map(order)
    frame = frame.sort_values("provider_order")
    lead = ["provider_order", "model_label"]
    rest = [column for column in frame.columns if column not in lead]
    total = {
        "provider_order": "",
        "model_label": "TOTAL",
        **(manifest.get("token_usage_total") or {}),
    }
    return pd.concat([frame[[*lead, *rest]], pd.DataFrame([total])], ignore_index=True).fillna("")


# --------------------------------------------------------------------------
# Reading a run back from these tables.
# --------------------------------------------------------------------------
def load_run_manifest() -> dict:
    """The runner's manifest for the current run, wherever it currently sits.

    During a run it is in staging; after one it has been moved beside the API
    call tables. One lookup, so the summary, the pipeline, and the notebook never
    disagree about where the run's own bookkeeping lives.
    """
    from bps_review.fulltext.config import api_calls_dir, run_staging_dir

    for path in (run_staging_dir() / "run_manifest.json", api_calls_dir() / "run_manifest.json"):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}



def _rejoin_split_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Undo `split_list_column`, rebuilding the joined cell it replaced."""
    frame = frame.copy()
    for column in SPLIT_LIST_COLUMNS:
        parts = sorted(
            (name for name in frame.columns if re.fullmatch(rf"{re.escape(column)}_\d+", name)),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        if not parts:
            # A list that was empty in every row got no numbered columns at all.
            # It is still a field of the schema, so it comes back as empty rather
            # than as a column that quietly stopped existing.
            if f"{column}_count" in frame.columns or column not in frame.columns:
                frame = frame.drop(columns=[f"{column}_count"], errors="ignore")
                frame[column] = ""
            continue
        joined = (
            frame[parts]
            .fillna("")
            .astype(str)
            .apply(
                lambda row: f"{OPEN_LIST_SEPARATOR} ".join(
                    value.strip() for value in row if value.strip()
                ),
                axis=1,
            )
        )
        frame = frame.drop(columns=[*parts, f"{column}_count"], errors="ignore")
        frame[column] = joined
    return frame


def load_run_from_tables(codings_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the pipeline's own frames from the published tables.

    These tables are the run's store, not a copy of one. Everything the coding
    schema defines is present in them, so a cached run is reloaded from here
    rather than from a second, unreadable set of files: the coding table carries
    every record-level field, and the category tables carry every field of every
    extracted item, in their original order.

    Returns the wide (paper x provider) frame and the item-level frame in exactly
    the shape the runner produces.
    """
    from bps_review.fulltext.coding.derive import item_rows, record_from_row, rederive_frame

    codings_dir = Path(codings_dir)
    long_path = codings_dir / CODINGS_DIRNAME / CODINGS_LONG_FILE
    frame = _rejoin_split_columns(pd.read_csv(long_path).fillna(""))
    # The published row also carries every extracted item flattened into its own
    # columns, for reading. Those are a second rendering of the category tables
    # below, so they are dropped here: what this function returns is the frame
    # the runner produces, not the frame a person opens.
    flattened = _flattened_item_pattern()
    frame = frame.drop(columns=[c for c in frame.columns if flattened.match(c)])
    frame = frame.drop(columns=_bibliography_columns_in(frame))

    # Rebuild each extraction list from its own category table, in item order.
    items_by_key: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for field, filename in CATEGORY_ORDER:
        path = codings_dir / ITEMS_DIRNAME / filename / category_item_file(field)
        if not path.exists():
            continue
        table = pd.read_csv(path).fillna("")
        columns = tuple(S.ITEM_MODELS[field].model_fields)
        list_fields = {
            name
            for name, info in S.ITEM_MODELS[field].model_fields.items()
            if str(info.annotation).startswith("list")
        }
        for row in table.sort_values("item_index").to_dict(orient="records"):
            payload = {}
            for column in columns:
                value = str(row.get(column, "") or "")
                payload[column] = (
                    [
                        part.strip()
                        for part in value.split(ITEM_LIST_SEPARATOR)
                        if part.strip()
                    ]
                    if column in list_fields
                    else value
                )
            key = (str(row["record_id"]), str(row["model_label"]))
            items_by_key.setdefault(key, {}).setdefault(field, []).append(payload)

    records = []
    for row in frame.to_dict(orient="records"):
        key = (str(row["record_id"]), str(row["model_label"]))
        for field in S.ITEM_MODELS:
            row[field] = json.dumps(items_by_key.get(key, {}).get(field, []), ensure_ascii=False)
        row.setdefault("llm_model", row.get("model_id", ""))
        records.append(row)
    long_df = rederive_frame(pd.DataFrame(records).fillna(""))

    item_frames = []
    for row in long_df.to_dict(orient="records"):
        record = record_from_row(row)
        item_frames.extend(
            item_rows(record, str(row.get("model_id", "")), str(row.get("model_label", "")))
        )
    items_df = pd.DataFrame(item_frames)
    return long_df, items_df


# --------------------------------------------------------------------------
# Excel workbook.
# --------------------------------------------------------------------------
def _write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> Path:
    """One workbook, one sheet per table, headers frozen and filterable."""
    ensure_parent(path)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            sheet = name[:31]
            frame.to_excel(writer, sheet_name=sheet, index=False)
            worksheet = writer.sheets[sheet]
            worksheet.freeze_panes = "B2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for index, column in enumerate(frame.columns, start=1):
                letter = worksheet.cell(row=1, column=index).column_letter
                longest = int(frame[column].astype(str).str.len().max() or 0)
                worksheet.column_dimensions[letter].width = min(
                    max(len(str(column)) + 2, longest + 2), 60
                )
    return path


# --------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------
CODINGS_DIRNAME = "01_codings"
ITEMS_DIRNAME = "02_extracted_items"
PROVIDER_DIRNAME = "03_by_provider"
API_DIRNAME = "04_api_calls"

# Every file is named after what one of its rows is, so a directory listing
# answers "which one do I open" without opening anything.
CODINGS_LONG_FILE = "01_one_row_per_paper_and_provider.csv"
CODINGS_PER_PAPER_FILE = "02_one_row_per_paper.csv"
CODINGS_PER_PROVIDER_FILE = "03_one_row_per_provider.csv"
CODINGS_BY_PROVIDER_DIR = "04_the_same_rows_split_per_provider"
CODINGS_BY_PAPER_DIR = "05_the_same_rows_split_per_paper"

ALL_ITEMS_DIR = "00_all_categories"
ALL_ITEMS_FILE = "01_one_row_per_item.csv"
DOMAIN_FACTORS_FILE = "02_one_row_per_named_factor_of_any_domain.csv"


def _paper_file_stem(row: pd.Series) -> str:
    """A file name a person recognizes: the record id plus first author and year."""
    author = str(row.get("citation", "")).split("(")[0].strip()
    author = re.sub(r"[^A-Za-z0-9]+", "_", author).strip("_") or "unknown"
    year = str(row.get("publication_year", "")).strip()
    return f"{row['record_id']}_{author}_{year}".rstrip("_")


def build_output_tables(
    codings_dir: Path,
    long_df: pd.DataFrame,
    items_df: pd.DataFrame,
    corpus_dir: Path | None = None,
    manifest: dict | None = None,
    audit_paths: list[Path] | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Write every output of one coding run, in readable form.

    ``long_df`` and ``items_df`` are the pipeline frames exactly as produced. No
    API call, no re-derivation, and no judgement is added here: this is the same
    content in the shape it is read in.
    """
    out_dir = Path(codings_dir)
    corpus_dir = Path(corpus_dir) if corpus_dir is not None else out_dir.parent / "01_corpus"

    papers = build_papers_table(corpus_dir)
    bibliography = citation_columns(load_bibliography(corpus_dir))
    codings_long = flatten_items_into_codings(
        build_codings_long(long_df, bibliography), items_df
    )
    codings_per_paper = build_codings_wide_per_paper(long_df, bibliography)
    codings_per_provider = build_codings_wide_per_provider(long_df, bibliography)
    items_long = build_items_long(items_df, bibliography)
    by_category = build_items_by_category(items_df, bibliography)
    domain_factors = build_domain_factor_inventory(by_category, bibliography)
    provider_order = _provider_order(long_df)

    codings_dir_out = out_dir / CODINGS_DIRNAME
    write_csv(codings_dir_out / CODINGS_LONG_FILE, codings_long)
    write_csv(codings_dir_out / CODINGS_PER_PAPER_FILE, codings_per_paper)
    write_csv(codings_dir_out / CODINGS_PER_PROVIDER_FILE, codings_per_provider)

    # The same rows, cut the two ways a reader asks for them: everything one
    # provider did, and everything that was said about one paper.
    for order, provider in enumerate(provider_order, start=1):
        write_csv(
            codings_dir_out / CODINGS_BY_PROVIDER_DIR / f"{order:02d}_{_model_slug(provider)}.csv",
            codings_long[codings_long["model_label"] == provider],
        )
    for _, paper in papers.iterrows():
        subset = codings_long[codings_long["record_id"] == paper["record_id"]]
        if subset.empty:
            continue
        write_csv(codings_dir_out / CODINGS_BY_PAPER_DIR / f"{_paper_file_stem(paper)}.csv", subset)

    # The paper list belongs to the corpus, not to the codings.
    write_csv(corpus_dir / "papers.csv", papers)

    write_csv(out_dir / ITEMS_DIRNAME / ALL_ITEMS_DIR / ALL_ITEMS_FILE, items_long)
    if not domain_factors.empty:
        write_csv(out_dir / ITEMS_DIRNAME / ALL_ITEMS_DIR / DOMAIN_FACTORS_FILE, domain_factors)
    field_by_filename = {filename: field for field, filename in CATEGORY_ORDER}
    for filename, table in by_category.items():
        field = field_by_filename[filename]
        folder = out_dir / ITEMS_DIRNAME / filename
        write_csv(folder / category_item_file(field), table)
        write_csv(
            folder / "02_one_row_per_paper_and_provider.csv",
            build_category_wide_per_paper_and_provider(table, field, bibliography),
        )
        write_csv(
            folder / "03_one_row_per_paper.csv",
            build_category_wide_per_paper(table, field, bibliography, provider_order),
        )
        write_csv(
            folder / "04_one_row_per_provider.csv",
            build_category_wide_per_provider(table, field, provider_order),
        )

    api_calls = build_api_calls(list(audit_paths or []), bibliography, long_df)
    providers = build_providers(manifest or {})
    usage = build_usage_per_provider(manifest or {})
    if not api_calls.empty:
        write_csv(out_dir / API_DIRNAME / "01_one_row_per_call.csv", api_calls)
    if not usage.empty:
        write_csv(out_dir / API_DIRNAME / "02_one_row_per_provider.csv", usage)
    if not providers.empty:
        write_csv(out_dir / API_DIRNAME / "03_provider_settings.csv", providers)

    provider_dir = out_dir / PROVIDER_DIRNAME
    for order, provider in enumerate(provider_order, start=1):
        folder = provider_dir / f"{order:02d}_{_model_slug(provider)}"
        write_csv(
            folder / "01_codings_one_row_per_paper.csv",
            codings_long[codings_long["model_label"] == provider],
        )
        write_csv(
            folder / "02_extracted_items_one_row_per_item.csv",
            items_long[items_long["model_label"] == provider],
        )
        for filename, table in by_category.items():
            write_csv(
                folder / "03_extracted_items_per_category" / f"{filename}.csv",
                table[table["model_label"] == provider],
            )
        if not api_calls.empty and "model_label" in api_calls.columns:
            write_csv(
                folder / "04_api_calls.csv", api_calls[api_calls["model_label"] == provider]
            )

    workbook = _write_workbook(
        out_dir / "00_workbook.xlsx",
        {
            "papers": papers,
            "codings_per_paper_provider": codings_long,
            "codings_per_paper": codings_per_paper,
            "codings_per_provider": codings_per_provider,
            **({"named_factors_all_domains": domain_factors} if not domain_factors.empty else {}),
            **{filename[3:]: table for filename, table in by_category.items()},
            **({"api_calls": api_calls} if not api_calls.empty else {}),
            **({"usage_per_provider": usage} if not usage.empty else {}),
        },
    )

    summary = {
        "n_papers": int(papers["record_id"].nunique()),
        "n_papers_with_doi": int((papers["doi"].astype(str).str.strip() != "").sum()),
        "n_providers": int(len(provider_order)),
        "n_coding_rows": int(len(codings_long)),
        "n_coding_columns": int(len(codings_long.columns)),
        "n_extracted_items": int(len(items_long)),
        "items_per_category": {
            filename[3:]: int(len(table)) for filename, table in by_category.items()
        },
        "n_api_calls": int(len(api_calls)),
        "outputs": {
            "workbook": workbook.name,
            "codings": f"{CODINGS_DIRNAME}/",
            "extracted_items": f"{ITEMS_DIRNAME}/",
            "by_provider": f"{PROVIDER_DIRNAME}/",
            "api_calls": f"{API_DIRNAME}/",
            "papers": "../01_corpus/papers.csv",
            "coding_scheme": "../../../02_coding_schemes/scheme_3/",
        },
    }
    write_json(out_dir / "outputs_manifest.json", summary)

    if verbose:
        print(
            f"  {len(papers)} papers, {len(codings_long)} coding rows "
            f"({len(codings_long.columns)} columns), {len(items_long)} items"
        )
        print(f"  written to {out_dir}")
    return {
        "output_dir": out_dir,
        "papers": papers,
        "codings_long": codings_long,
        "codings_per_paper": codings_per_paper,
        "codings_per_provider": codings_per_provider,
        "items_long": items_long,
        "items_by_category": by_category,
        "domain_factors": domain_factors,
        "api_calls": api_calls,
        "manifest": summary,
    }
