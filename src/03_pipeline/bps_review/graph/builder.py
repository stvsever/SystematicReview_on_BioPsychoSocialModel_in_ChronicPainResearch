from __future__ import annotations

"""Build a local static knowledge graph from scheme 3 coding outputs.

The graph is the review surface for a full-text run: it turns the wide coding
table, the item table, and the quote-verification table into one browsable
hierarchy, so a reviewer can walk from the scheme itself down to the sentence a
judgement rests on.

The hierarchy is

    run -> field group -> [entity] -> coding field -> provider -> article -> item

where the entity level appears only under "Biopsychosocial entities". That group
is the one place where the coded fields are not siblings: the biological, the
psychological, the social, the lifestyle, and the spiritual or existential
entities are five different kinds of thing, and each carries several fields, so
each gets a node of its own and the fields hang beneath it.

Two of the scheme's lists hold more than one entity at a time. The domain
evidence is a single list covering all three domains, and the beyond-the-triad
factors are a single list covering lifestyle, existential, and environmental
factors together. Both are therefore split into item-filtered views, so the
biological evidence appears under the biological entity rather than in one
undifferentiated list. See ``FieldView``.

The first view shows only the scheme: the field groups, the entities, and every
canonical coding field of scheme 3. Providers, articles, and extracted items are
complete descendants of that overview, expanded on demand, so the opening picture
stays a picture of the coding scheme rather than of a few hundred coded cells.

Grouping is the one part of this module that is specific to scheme 3. Fields are
laid out along the review's own questions (how the biopsychosocial label is used,
how deep each domain goes, what the model is made of, how the domains are linked,
what is measured, what is conceptually wrong with it), and any column the table
carries that this file does not name still appears, under "Other coded fields".
That keeps the surface correct across scheme revisions: a new coded field shows
up without a code change, and a retired one disappears.
"""

import colorsys
import hashlib
import json
import math
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bps_review.fulltext.coding import schema
from bps_review.fulltext.config import FIELD_LABELS
from bps_review.utils.io import ensure_parent


@dataclass(frozen=True)
class FieldView:
    """One coding-field node: a column, optionally restricted to some of its items.

    Most field nodes are a whole column. A few are a slice of one: the domain
    evidence is a single list carrying all three domains, and the beyond-the-triad
    factors are a single list carrying lifestyle, existential, and environmental
    factors together. Reading those as one node each would hide exactly the
    distinction this review is about, so a view can carry a filter and appear as
    its own node under the entity it belongs to.
    """

    column: str
    label: str = ""
    key: str = ""
    filter_key: str = ""
    filter_values: tuple[str, ...] = ()

    def resolved_key(self) -> str:
        return self.key or self.column

    def resolved_label(self) -> str:
        return self.label or FIELD_LABELS.get(self.column, self.column.replace("_", " ").capitalize())


def _domain_slice(domain: str, label: str) -> FieldView:
    """The domain-evidence items for one domain: the passage and the constructs named."""
    return FieldView(
        column="domain_evidence",
        label=label,
        key=f"domain_evidence__{domain.replace(' ', '_')}",
        filter_key="domain",
        filter_values=(domain,),
    )


def _beyond_triad_slice(domain: str, label: str) -> FieldView:
    """Factors of one kind, out of the single list that holds everything beyond the triad."""
    return FieldView(
        column="other_domain_factors",
        label=label,
        key=f"other_domain_factors__{domain.replace(' ', '_')}",
        filter_key="domain",
        filter_values=(domain,),
    )


# The entity layer: what the review says the biopsychosocial model is made of.
#
# This is the one group that nests, because it is the one place where the coded
# fields are not siblings. The biological, psychological, and social entities are
# the triad the model names. What the registration adds beyond the triad, the
# lifestyle and the spiritual or existential factors, is a fourth thing of a
# different order: not a fourth domain, but the account of what falls outside the
# three. So it sits under one heading of its own, with its own children, and the
# depth of the tree says that directly.
#
#     Biopsychosocial entities
#     |- Biological factors
#     |- Psychological factors
#     |- Social factors
#     +- Other factors
#        |- Lifestyle factors
#        +- Spiritual and existential factors
#
# Flattening any of this into one ring would put a concept definition next to a
# social factor next to a lifestyle factor and imply the three are alike.
BPS_ENTITY_SUBGROUPS: OrderedDict[str, Any] = OrderedDict(
    [
        (
            "Biological factors",
            [
                FieldView("biological_factors"),
                _domain_slice("biological", "Biological evidence and constructs"),
            ],
        ),
        (
            "Psychological factors",
            [
                FieldView("psychological_concepts"),
                FieldView("concept_definitions_present"),
                FieldView("concept_relations"),
                _domain_slice("psychological", "Psychological evidence and constructs"),
            ],
        ),
        (
            "Social factors",
            [
                FieldView("social_factors"),
                _domain_slice("social", "Social evidence and constructs"),
            ],
        ),
        (
            "Other factors",
            OrderedDict(
                [
                    (
                        "Lifestyle factors",
                        [
                            _beyond_triad_slice("lifestyle", "Lifestyle factors named"),
                            FieldView("coverage_lifestyle"),
                        ],
                    ),
                    (
                        "Spiritual and existential factors",
                        [
                            _beyond_triad_slice("spiritual or existential",
                                                "Existential factors named"),
                            FieldView("coverage_spiritual_existential"),
                            _beyond_triad_slice("environmental", "Environmental factors named"),
                        ],
                    ),
                ]
            ),
        ),
    ]
)


# group -> either a flat list of columns, or an ordered map of subgroups.
FIELD_GROUPS: OrderedDict[str, Any] = OrderedDict(
    [
        (
            "Article context",
            [
                "review_track",
                "source_type",
                "icd11_pain_category",
                "population",
                "care_setting",
                "primary_discipline",
                "pain_condition_detail",
                "pain_conditions",
                "context_note",
                "quality_assessment_reported",
                "quality_assessment_tools",
            ],
        ),
        (
            "Biopsychosocial label",
            [
                "bps_label_used",
                "bps_primary_function",
                "bps_functions_present",
                "bps_definition_status",
                "bps_model_variants",
                "bps_usage_instances",
                "bps_definitions",
                "bps_operationalization_summary",
                "bps_function_set",
                "bps_has_substantive_function",
                "bps_usage_sections",
            ],
        ),
        (
            "Domain coverage",
            ["domain_coverage_bio", "domain_coverage_psych", "domain_coverage_social"],
        ),
        ("Biopsychosocial entities", BPS_ENTITY_SUBGROUPS),
        (
            "Integration",
            [
                "integration_bio_psych",
                "integration_psych_social",
                "integration_bio_social",
                "integration_triadic",
                "integration_claims",
                "integration_mechanism_summary",
            ],
        ),
        (
            "Typology and balance",
            ["overall_balance", "bps_typology", "derived_typology", "typology_matches_derived"],
        ),
        (
            "Frameworks and instruments",
            ["theoretical_frameworks", "instruments"],
        ),
        (
            "Conceptual problems",
            ["conceptual_problems"],
        ),
        (
            "Synthesis hooks",
            [
                "key_quotes",
                "emergent_labels",
                "conceptual_tensions",
                "additional_observations",
                "synthesis_note",
                "coding_rationale",
            ],
        ),
        (
            "Presence flags",
            [
                "present_bps_usage_evidence",
                "present_bps_definition",
                "present_integration_evidence",
                "present_triadic_claim",
                "present_named_integration_edge",
                "present_biological_factors",
                "present_social_factors",
                "present_other_domain_factors",
                "present_psychological_concepts",
                "present_defined_concepts",
                "present_concept_relations",
                "present_hierarchical_relation",
                "present_theoretical_frameworks",
                "present_instruments",
                "present_conceptual_problems",
                "present_domain_evidence_bio",
                "present_domain_evidence_psych",
                "present_domain_evidence_social",
            ],
        ),
        (
            "Eligibility and yield",
            [
                "fulltext_eligibility",
                "fulltext_exclusion_reason",
                "conceptual_yield",
                "synthesis_priority",
                "integration_index",
                "coverage_total",
                "domains_present",
                "coverage_depth_bio",
                "coverage_depth_psych",
                "coverage_depth_social",
                "pairwise_depth_total",
                "pairwise_depth_max",
                "triadic_depth",
            ],
        ),
        (
            "Counts and provenance",
            [
                "n_bps_usage_instances",
                "n_bps_definitions",
                "n_domain_evidence",
                "n_biological_factors",
                "n_social_factors",
                "n_other_domain_factors",
                "n_psychological_concepts",
                "n_defined_concepts",
                "n_concept_relations",
                "n_hierarchical_relations",
                "n_integration_claims",
                "n_triadic_claims",
                "n_named_integration_edges",
                "n_theoretical_frameworks",
                "n_instruments",
                "n_conceptual_problems",
                "n_key_quotes",
                "n_emergent_labels",
                "n_subdomains_bio",
                "n_subdomains_psych",
                "n_subdomains_social",
                "n_subdomains_named",
                "n_bps_functions",
                "n_bps_usage_sections",
                "n_open_list_entries",
                "n_labels_checked",
                "controlled_label_share",
                "n_evidence_quotes",
                "n_extracted_items",
                "coding_method",
                "llm_model",
            ],
        ),
    ]
)


GROUP_COLORS = {
    "Article context": "#5ca6c9",
    "Biopsychosocial label": "#9677d6",
    "Domain coverage": "#6daee8",
    "Biopsychosocial entities": "#42c1a1",
    "Integration": "#ee9b5c",
    "Typology and balance": "#e27ba6",
    "Frameworks and instruments": "#d8bb55",
    "Conceptual problems": "#eb6f75",
    "Synthesis hooks": "#58b6b2",
    "Presence flags": "#7fa8bd",
    "Eligibility and yield": "#cf8f6a",
    "Counts and provenance": "#8793a6",
    "Other coded fields": "#9aa5b1",
}

# One colour per entity, so a biological factor reads as biological wherever it
# appears. These are the domain colours the static figures already use.
SUBGROUP_COLORS = {
    "Biological factors": "#0e8f80",
    "Psychological factors": "#6d5ae0",
    "Social factors": "#d98016",
    "Other factors": "#8fa06b",
    "Lifestyle factors": "#7fae4a",
    "Spiritual and existential factors": "#a8809f",
}

ARTICLE_COLORS = [
    "#4b8bd8",
    "#55a5b5",
    "#6ca96b",
    "#c69a45",
    "#cf7a61",
    "#c7678c",
    "#8c74cf",
    "#667fb4",
    "#3f9b85",
    "#86a94d",
    "#d58b3d",
    "#bd6755",
    "#b65e9d",
    "#7867c2",
    "#4a99c2",
    "#7a8a9f",
]

PROVIDER_COLORS = ["#a97ff2", "#4da3e5", "#e09443", "#4cb78c", "#e26789"]

IDENTITY_COLUMNS = {"record_id", "model_order", "model_label", "provider", "model_id"}

# The thirteen JSON-serialized extraction lists of scheme 3, and the free-text
# lists the wide table stores as semicolon-joined strings. Both expand into their
# own leaf nodes; everything else is one coded value.
STRUCTURED_FIELDS = set(schema.ITEM_MODELS)
FLAT_LIST_FIELDS = set(schema.OPEN_LIST_FIELDS) | {"bps_function_set", "bps_usage_sections"}

# Item-table columns worth showing on an extracted item, in reading order. The
# item carries its own wording and its place on the project ontology side by
# side, and the graph never shows one without the other.
ITEM_METADATA_LABELS: OrderedDict[str, str] = OrderedDict(
    [
        ("label_normalized", "Normalized label"),
        ("label_vocabulary", "Label vocabulary"),
        ("label_controlled", "Label on the controlled list"),
        ("anchor_label", "Ontology anchor"),
        ("anchor_vocabulary", "Anchor vocabulary"),
        ("anchor_controlled", "Anchor on the controlled list"),
    ]
)


def _json_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.replace(chr(0x2014), " - ")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value) if not isinstance(value, (int, float, bool)) else value


def _field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " ").capitalize())


def _view_has_items(view: FieldView, rows: list[dict[str, Any]]) -> bool:
    """Whether a filtered view matches anything at all in this run.

    An unfiltered column always earns its node: a field nobody filled is itself a
    finding. A filtered slice with no matching item is not a finding, it is an
    empty subdivision of a list, so it is dropped.
    """
    if not view.filter_key:
        return True
    return any(
        _filtered_items(_parse_structured(row.get(view.column, "")), view)
        for row in rows
    )


@dataclass(frozen=True)
class Branch:
    """One heading in the scheme overview: its own fields, and its own children.

    A branch with no name holds fields directly under its group, which is the
    shape of every group except the entity layer. Branches nest to any depth, so
    a heading can stand for a kind of entity that has kinds of its own.
    """

    name: str
    views: tuple[FieldView, ...] = ()
    children: tuple["Branch", ...] = ()

    def all_views(self) -> list[FieldView]:
        return list(self.views) + [view for child in self.children for view in child.all_views()]

    def depth(self) -> int:
        """How many heading levels this branch spans, itself included."""
        return 1 + max((child.depth() for child in self.children), default=0)


def _resolve_branch(
    name: str,
    spec: Any,
    rows: list[dict[str, Any]],
    available: set[str],
    claimed: set[str],
) -> Branch | None:
    """One branch of the scheme overview, restricted to what this run carries.

    Returns ``None`` when nothing under the branch survives, so a heading is never
    drawn over an empty subtree.
    """
    if isinstance(spec, dict):
        children = [
            child for child in (
                _resolve_branch(child_name, child_spec, rows, available, claimed)
                for child_name, child_spec in spec.items()
            )
            if child is not None
        ]
        return Branch(name, (), tuple(children)) if children else None

    views = [
        view if isinstance(view, FieldView) else FieldView(view)
        for view in spec
    ]
    present = [
        view for view in views
        # An explicit view names its column on purpose, so it is never suppressed
        # by an earlier group having taken that column. A bare column in a flat
        # list is a catch-all, and the first group to name it wins.
        if view.column in available
        and (view.filter_key or view.label or view.column not in claimed)
        and _view_has_items(view, rows)
    ]
    if not present:
        return None
    claimed.update(view.column for view in present)
    return Branch(name, tuple(present), ())


def _resolve_groups(
    columns: list[str], rows: list[dict[str, Any]]
) -> "OrderedDict[str, Branch]":
    """The grouping this run actually supports, as one branch tree per group.

    Columns the table carries and no group names still reach the reviewer, under
    "Other coded fields", so a scheme revision can never silently drop a field
    from the review surface.
    """
    available = set(columns)
    claimed: set[str] = set()
    groups: "OrderedDict[str, Branch]" = OrderedDict()

    for group, spec in FIELD_GROUPS.items():
        branch = _resolve_branch("", spec, rows, available, claimed)
        if branch is not None:
            groups[group] = branch

    remaining = [
        column for column in columns
        if column not in IDENTITY_COLUMNS and column not in claimed
    ]
    if remaining:
        groups["Other coded fields"] = Branch(
            "", tuple(FieldView(column) for column in remaining), ()
        )
    return groups


def _parse_structured(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def _filtered_items(items: list[dict[str, Any]], view: FieldView) -> list[dict[str, Any]]:
    """The items of one cell that belong to this view."""
    if not view.filter_key:
        return items
    return [
        item for item in items
        if str(item.get(view.filter_key, "") or "").strip() in view.filter_values
    ]


def _flat_items(value: Any) -> list[str]:
    """Split a record-level open list. The wide table joins these with semicolons."""
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _item_label(field: str, item: dict[str, Any], index: int) -> str:
    """What identifies one extracted item, using the scheme's own identity keys.

    A relation and an integration claim are edges, so their identity is the pair
    they connect joined by the relation, exactly as the reliability metrics read
    them. Everything else is identified by its own label.
    """
    parts = [
        str(item.get(key, "") or "").strip()
        for key in schema.ITEM_LABEL_KEY.get(field, ())
    ]
    label = " -> ".join(part for part in parts if part)
    if label:
        return label
    return f"{_field_label(field)} item {index + 1}"


def _short(value: Any, limit: int = 76) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def _panel_sections(branch: Branch, path: tuple[str, ...]) -> list[tuple[tuple[str, ...], list[FieldView]]]:
    """Every heading that owns coding fields directly, with the path that names it."""
    here = path + (branch.name,) if branch.name else path
    sections: list[tuple[tuple[str, ...], list[FieldView]]] = []
    if branch.views:
        sections.append((here, list(branch.views)))
    for child in branch.children:
        sections.extend(_panel_sections(child, here))
    return sections


def _branch_color(group: str, path: tuple[str, ...] = ()) -> str:
    """The palette a node varies within: the nearest named heading above it."""
    for name in reversed(path):
        if name in SUBGROUP_COLORS:
            return SUBGROUP_COLORS[name]
    return GROUP_COLORS.get(group, GROUP_COLORS["Other coded fields"])


def _field_color(group: str, field: str, path: tuple[str, ...] = ()) -> str:
    """Vary hue, saturation, and lightness within a stable field-group palette."""
    base = _branch_color(group, path).lstrip("#")
    red, green, blue = (int(base[index : index + 2], 16) / 255 for index in (0, 2, 4))
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    digest = hashlib.sha1(field.encode("utf-8")).digest()
    hue = (hue + ((digest[0] / 255) - 0.5) * 0.055) % 1.0
    saturation = max(0.48, min(0.88, saturation + ((digest[1] / 255) - 0.5) * 0.22))
    lightness = max(0.48, min(0.70, lightness + ((digest[2] / 255) - 0.5) * 0.16))
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def graph_payload(
    corpus_df: pd.DataFrame,
    long_df: pd.DataFrame,
    items_df: pd.DataFrame | None = None,
    verification_df: pd.DataFrame | None = None,
    run_title: str = "Full-text coding knowledge graph",
    run_subtitle: str = "Cross-provider scheme 3 review",
) -> dict[str, Any]:
    """Return the complete browser-ready graph payload."""
    if long_df.empty:
        raise ValueError("Cannot build a knowledge graph from an empty coding table")
    required = {"record_id", "model_label", "provider"}
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"Coding table is missing required columns: {sorted(missing)}")

    columns = list(long_df.columns)
    rows = long_df.sort_values(["record_id", "model_order"]).to_dict(orient="records")
    groups = _resolve_groups(columns, rows)
    corpus = {
        str(row["record_id"]): {key: _json_value(value) for key, value in row.items()}
        for row in corpus_df.to_dict(orient="records")
    }
    providers = (
        long_df[["model_order", "model_label", "provider", "model_id"]]
        .drop_duplicates()
        .sort_values("model_order")
        .to_dict(orient="records")
    )
    item_metadata: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    if items_df is not None and not items_df.empty:
        available = [column for column in ITEM_METADATA_LABELS if column in items_df.columns]
        for item in items_df.to_dict(orient="records"):
            key = (
                str(item.get("record_id", "")),
                str(item.get("model_label", "")),
                str(item.get("extraction_field", "")),
                int(item.get("item_index", 0)),
            )
            item_metadata[key] = {
                ITEM_METADATA_LABELS[column]: _json_value(item.get(column, ""))
                for column in available
            }
    verification_metadata: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if verification_df is not None and not verification_df.empty:
        for item in verification_df.to_dict(orient="records"):
            key = (
                str(item.get("record_id", "")),
                str(item.get("model_label", "")),
                str(item.get("extraction_field", "")),
                str(item.get("quote", "")),
            )
            verification_metadata[key] = {
                "Quote verification": _json_value(item.get("verification", "")),
                "Quote n-gram coverage": _json_value(item.get("ngram_coverage", "")),
                "Quote words": _json_value(item.get("quote_words", "")),
            }
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_counter = 0

    def add_node(**payload: Any) -> str:
        nonlocal node_counter
        node_counter += 1
        node_id = f"n{node_counter}"
        nodes.append({"id": node_id, **payload})
        return node_id

    def add_edge(source: str, target: str, kind: str) -> None:
        edges.append({"source": source, "target": target, "kind": kind})

    root_id = add_node(
        label=run_title,
        type="run",
        level=0,
        size=30,
        color="#d7e5ff",
        article_id="",
        provider="",
        field="",
        field_group="",
        value=run_subtitle,
        detail={
            "Papers": int(long_df["record_id"].nunique()),
            "Providers": int(long_df["model_label"].nunique()),
            "Codings": int(len(long_df)),
            "Structured items": int(len(items_df)) if items_df is not None else 0,
        },
    )

    article_ids = sorted(long_df["record_id"].astype(str).unique())
    rows_by_provider = {
        str(provider["model_label"]): [
            row for row in rows if str(row.get("model_label", "")) == str(provider["model_label"])
        ]
        for provider in providers
    }
    article_colors = {
        record_id: ARTICLE_COLORS[index % len(ARTICLE_COLORS)]
        for index, record_id in enumerate(article_ids)
    }
    provider_colors = {
        str(item["model_label"]): PROVIDER_COLORS[index % len(PROVIDER_COLORS)]
        for index, item in enumerate(providers)
    }

    def emit_field(
        view: FieldView,
        parent_id: str,
        group: str,
        path: tuple[str, ...],
        depth: int,
        sibling_count: int,
        group_index: int,
        field_index: int,
    ) -> None:
        """One coding field, and every provider, article, and item beneath it."""
        field = view.resolved_key()
        column = view.column
        label = view.resolved_label()
        subgroup = path[-1] if path else ""
        color = _field_color(group, field, path)
        structured_column = column in STRUCTURED_FIELDS
        flat_column = column in FLAT_LIST_FIELDS

        def items_of(row: dict[str, Any]) -> list[dict[str, Any]]:
            return _filtered_items(_parse_structured(row.get(column, "")), view)

        if structured_column:
            populated = sum(1 for row in rows if items_of(row))
            extracted_count = sum(len(items_of(row)) for row in rows)
        else:
            populated = sum(bool(str(row.get(column, "") or "").strip()) for row in rows)
            extracted_count = (sum(len(_flat_items(row.get(column, ""))) for row in rows)
                               if flat_column else 0)
        restriction = (
            f"{view.filter_key} is {' or '.join(view.filter_values)}" if view.filter_key else ""
        )
        field_node = add_node(
            label=label,
            type="field",
            level=depth,
            size=9,
            color=color,
            article_id="",
            provider="",
            field=field,
            field_group=group,
            field_subgroup=subgroup,
            field_path=list(path),
            value=f"{populated} recorded values",
            detail={
                "Coding field": label,
                "Field key": field,
                "Coded column": column,
                **({"Restricted to items where": restriction} if restriction else {}),
                "Field group": group,
                **({"Entity": " / ".join(path)} if path else {}),
                "Article-provider codings": int(len(rows)),
                "Recorded values": populated,
                "Extracted entries": extracted_count,
                "Value type": "structured extraction list" if structured_column
                else "open list" if flat_column else "coded value",
            },
            group_index=group_index,
            field_index=field_index,
            sibling_count=sibling_count,
        )
        add_edge(parent_id, field_node, "contains_field")

        for provider_index, provider_info in enumerate(providers):
            model_label = str(provider_info["model_label"])
            provider = str(provider_info["provider"])
            provider_rows = rows_by_provider[model_label]
            provider_node = add_node(
                label=f"{model_label} | {provider}",
                type="provider",
                level=depth + 1,
                size=8.2,
                color=provider_colors[model_label],
                article_id="",
                article_title="",
                provider=model_label,
                provider_name=provider,
                field=field,
                field_group=group,
                field_subgroup=subgroup,
                value=str(provider_info.get("model_id", "")),
                detail={
                    "Provider": {
                        "Model label": model_label,
                        "Provider": provider,
                        "Model ID": provider_info.get("model_id", ""),
                    },
                    "Coding field": label,
                    "Field group": group,
                    **({"Entity": " / ".join(path)} if path else {}),
                    "Available article codings": len(provider_rows),
                },
                provider_index=provider_index,
            )
            add_edge(field_node, provider_node, "provider_branch")

            for article_index, row in enumerate(provider_rows):
                record_id = str(row.get("record_id", ""))
                article = corpus.get(record_id, {"record_id": record_id})
                title = str(article.get("title") or record_id)
                value = _json_value(row.get(column, ""))
                structured = items_of(row) if structured_column else []
                flat = _flat_items(value) if flat_column else []
                if structured:
                    summary = f"{len(structured)} extracted entries"
                    rendered_value: Any = _json_value(structured)
                elif flat:
                    summary = " | ".join(flat)
                    rendered_value = flat
                elif structured_column:
                    summary = "Not recorded"
                    rendered_value = []
                else:
                    summary = str(value or "Not recorded")
                    rendered_value = value
                article_node = add_node(
                    label=f"{record_id} | {_short(title, 48)}: {_short(summary, 42)}",
                    type="article",
                    level=depth + 2,
                    size=5.7,
                    color=article_colors[record_id],
                    article_id=record_id,
                    article_title=title,
                    provider=model_label,
                    provider_name=provider,
                    field=field,
                    field_group=group,
                    field_subgroup=subgroup,
                    value=_json_value(rendered_value),
                    detail={
                        "Article": {"Record ID": record_id, "Title": title},
                        "Provider": {
                            "Model label": model_label,
                            "Provider": provider,
                            "Model ID": row.get("model_id", ""),
                        },
                        "Coding field": label,
                        "Field group": group,
                        **({"Entity": " / ".join(path)} if path else {}),
                        "Recorded value": rendered_value,
                    },
                    article_index=article_index,
                )
                add_edge(provider_node, article_node, "article_coding")

                item_values: list[tuple[str, Any]] = []
                if structured:
                    all_items = _parse_structured(row.get(column, ""))
                    for item in structured:
                        # The item table is indexed by position in the unfiltered
                        # list, so a slice looks its metadata up by that index.
                        index = all_items.index(item)
                        detail = dict(item)
                        detail.update(item_metadata.get((record_id, model_label, column, index), {}))
                        quote_key = schema.ITEM_QUOTE_KEY.get(column, "")
                        quote = str(item.get(quote_key, "")) if quote_key else ""
                        detail.update(verification_metadata.get((record_id, model_label, column, quote), {}))
                        item_values.append((_item_label(column, item, index), detail))
                elif flat:
                    item_values = [(item, {"Value": item}) for item in flat]
                for item_index, (item_label, item_detail) in enumerate(item_values):
                    item_node = add_node(
                        label=_short(item_label, 88),
                        type="item",
                        level=depth + 3,
                        size=3.8,
                        color=color,
                        article_id=record_id,
                        article_title=title,
                        provider=model_label,
                        provider_name=provider,
                        field=field,
                        field_group=group,
                        field_subgroup=subgroup,
                        value=item_label,
                        detail=_json_value(item_detail),
                        item_index=item_index,
                    )
                    add_edge(article_node, item_node, "extracts")

    def emit_branch(
        branch: Branch,
        parent_id: str,
        group: str,
        path: tuple[str, ...],
        depth: int,
        group_index: int,
        counter: list[int],
    ) -> None:
        """One heading and everything under it, to whatever depth the group nests."""
        holder = parent_id
        here = path
        if branch.name:
            here = path + (branch.name,)
            branch_views = branch.all_views()
            recorded = sum(
                bool(str(row.get(view.column, "") or "").strip())
                for row in rows
                for view in branch_views
            )
            holder = add_node(
                label=branch.name,
                type="subgroup",
                level=depth,
                size=13.5 - 1.2 * (len(here) - 1),
                color=_branch_color(group, here),
                article_id="",
                provider="",
                field="",
                field_group=group,
                field_subgroup=branch.name,
                field_path=list(here),
                value=(f"{len(branch.children)} kinds, {len(branch_views)} coding fields"
                       if branch.children else f"{len(branch_views)} coding fields"),
                detail={
                    "Entity": branch.name,
                    "Field group": group,
                    **({"Sits under": " / ".join(path)} if path else {}),
                    **({"Kinds": [child.name for child in branch.children]}
                       if branch.children else {}),
                    "Coding fields": [view.resolved_label() for view in branch_views],
                    "Number of fields": len(branch_views),
                    "Recorded coding cells": recorded,
                    "Available article-provider codings": int(len(rows)),
                },
                group_index=group_index,
            )
            add_edge(parent_id, holder, "contains_subgroup")
            depth += 1

        for view in branch.views:
            emit_field(view, holder, group, here, depth, len(branch.views),
                       group_index, counter[0])
            counter[0] += 1
        for child in branch.children:
            emit_branch(child, holder, group, here, depth, group_index, counter)

    # The browser opens on one canonical overview of the scheme. Article and
    # provider-specific values remain complete descendants, but they do not
    # duplicate the visible field layer once per paper.
    for group_index, (group, branch) in enumerate(groups.items()):
        group_views = branch.all_views()
        recorded_cells = sum(
            bool(str(row.get(view.column, "") or "").strip())
            for row in rows
            for view in group_views
        )
        entities = [child.name for child in branch.children]
        group_node = add_node(
            label=group,
            type="group",
            level=1,
            size=18,
            color=GROUP_COLORS.get(group, GROUP_COLORS["Other coded fields"]),
            article_id="",
            provider="",
            field="",
            field_group=group,
            field_subgroup="",
            field_path=[],
            value=(f"{len(entities)} entities, {len(group_views)} coding fields"
                   if entities else f"{len(group_views)} coding fields"),
            detail={
                "Field group": group,
                **({"Entities": entities} if entities else {}),
                "Coding fields": [view.resolved_label() for view in group_views],
                "Number of fields": len(group_views),
                "Recorded coding cells": recorded_cells,
                "Available article-provider codings": int(len(rows)),
            },
            group_index=group_index,
        )
        add_edge(root_id, group_node, "contains_group")
        emit_branch(branch, group_node, group, (), 2, group_index, [0])

    for node in nodes:
        searchable = [
            node.get("label", ""),
            node.get("article_id", ""),
            node.get("provider", ""),
            node.get("field", ""),
            node.get("field_group", ""),
            json.dumps(node.get("detail", {}), ensure_ascii=False),
        ]
        node["search"] = " ".join(str(part) for part in searchable).lower()

    return {
        "meta": {
            "title": run_title,
            "subtitle": run_subtitle,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_papers": len(article_ids),
            "n_providers": len(providers),
            "n_codings": int(len(long_df)),
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_field_groups": len(groups),
            "n_coding_fields": sum(len(branch.all_views()) for branch in groups.values()),
        },
        "nodes": nodes,
        "edges": edges,
        "filters": {
            "articles": [
                {
                    "id": record_id,
                    "label": str(corpus.get(record_id, {}).get("title") or record_id),
                    "color": article_colors[record_id],
                }
                for record_id in article_ids
            ],
            "providers": [
                {
                    "id": str(item["model_label"]),
                    "label": str(item["model_label"]),
                    "provider": str(item["provider"]),
                    "model_id": str(item["model_id"]),
                    "color": provider_colors[str(item["model_label"])],
                }
                for item in providers
            ],
            # One panel section per leaf heading, named by its path, so the panel
            # has the same shape as the graph and a reviewer can switch off
            # "Other factors / Lifestyle factors" as one thing.
            "field_groups": [
                {
                    "name": " \u00b7 ".join((group,) + path) if path else group,
                    "group": group,
                    "subgroup": path[-1] if path else "",
                    "path": list(path),
                    "color": _branch_color(group, path),
                    "fields": [
                        {
                            "id": view.resolved_key(),
                            "label": view.resolved_label(),
                            "color": _field_color(group, view.resolved_key(), path),
                        }
                        for view in views
                    ],
                }
                for group, branch in groups.items()
                for path, views in _panel_sections(branch, ())
            ],
        },
    }


def build_knowledge_graph(
    corpus_df: pd.DataFrame,
    long_df: pd.DataFrame,
    items_df: pd.DataFrame | None,
    output_dir: Path,
    run_title: str = "Full-text coding knowledge graph",
    run_subtitle: str = "Cross-provider scheme 3 review",
    verification_df: pd.DataFrame | None = None,
) -> Path:
    """Write a complete static graph bundle and return its index path."""
    output_dir = Path(output_dir)
    assets_out = output_dir / "assets"
    assets_source = Path(__file__).resolve().parent / "assets"
    payload = graph_payload(
        corpus_df,
        long_df,
        items_df,
        verification_df,
        run_title,
        run_subtitle,
    )

    ensure_parent(assets_out / "styles.css")
    shutil.copy2(assets_source / "dashboard.css", assets_out / "styles.css")
    shutil.copy2(assets_source / "dashboard.js", assets_out / "app.js")
    template = (assets_source / "dashboard.html").read_text(encoding="utf-8")
    index_text = template.replace("{{RUN_TITLE}}", run_title).replace("{{RUN_SUBTITLE}}", run_subtitle)
    index_path = output_dir / "index.html"
    ensure_parent(index_path).write_text(index_text, encoding="utf-8")
    (assets_out / "graph_data.js").write_text(
        "window.BPS_GRAPH_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Knowledge graph review surface\n\n"
        "Open `index.html` in a desktop browser. The bundle is fully local and requires no server.\n\n"
        f"- Papers: {payload['meta']['n_papers']}\n"
        f"- Providers: {payload['meta']['n_providers']}\n"
        f"- Coding cells: {payload['meta']['n_codings']}\n"
        f"- Graph nodes: {payload['meta']['n_nodes']}\n"
        f"- Graph links: {payload['meta']['n_edges']}\n\n"
        "The first view shows the field groups, the biopsychosocial entities, and all canonical "
        "scheme 3 coding fields. The entity level holds the triad as three siblings and everything "
        "beyond it under Other factors, which carries lifestyle and spiritual or existential as its "
        "own children, so the evidence for one domain sits under that domain rather than in one "
        "undifferentiated list. Double-click a "
        "field or use its Explore button to reveal provider hubs with papers grouped beneath them, then "
        "expand an article coding to reveal extracted items. With one selected provider, papers connect "
        "directly to the field. Use Show all to render every selected layer. Use "
        "the left panel to filter articles, providers, and coding fields. Drag nodes to pin them, drag the "
        "background to pan, use the mouse wheel to zoom, switch theme, move or disable the node preview, and "
        "click a node for its formatted inspector. The complete root-to-leaf path stays highlighted while the "
        "scheme overview remains visible as context. Whenever Labels is enabled, the run root, field-group "
        "labels, and canonical coding-field labels stay visible at every drill-down depth. Back one level and "
        "parent-node double-clicks move upward. "
        "Deep article views use compact automatically sized rings and collision-aware leaf labels. Context "
        "fitting frames the active branch, and dragging a parent moves its complete descendant subtree, "
        "including hidden descendants expanded later. Manual zoom supports up to 1000 percent. Reset view "
        "returns to the complete scheme overview and clears manual placement.\n",
        encoding="utf-8",
    )
    return index_path
