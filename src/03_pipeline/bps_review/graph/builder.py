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
from bps_review.fulltext.coding.prompt import (
    CONTROLLED_LIST_VALUES,
    CONTROLLED_VALUES,
    ITEM_VALUE_LISTS,
    LADDERS,
)
from bps_review.fulltext.config import (
    FIELD_LABELS,
    ITEM_CAPS,
    ITEM_SUBLIST_CAP,
    MAX_NOTE_WORDS,
    MAX_QUOTE_WORDS,
    MAX_SUMMARY_WORDS,
    OPEN_LIST_CAPS,
    PRESENCE_ORDER,
)
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


# --------------------------------------------------------------------------
# The reader-facing reference layer.
#
# A coding-field node used to carry only counts, which says how often a field was
# filled but never what the field means or which values it can take. Every field
# therefore also carries a one or two sentence explanation and its value space:
# the closed vocabulary when it has one, the rung-by-rung rule when the field is
# a ladder, the format when it is open, and the item fields with their own
# vocabularies for the structured extraction lists.
#
# The explanations are written here. Every value list, ladder, and cap is read
# from the schema, the prompt, and the configuration, so the graph cannot drift
# from what the coder was asked for or from what the validator accepts.
# --------------------------------------------------------------------------
FIELD_DESCRIPTIONS: dict[str, str] = {
    # Article context
    "review_track": (
        "Which of the two planned reviews this record belongs to, read from the pain condition the "
        "paper actually studies. Musculoskeletal covers low back, neck, osteoarthritis, fibromyalgia "
        "and similar; neuropathic covers painful neuropathy, radicular pain and similar."
    ),
    "source_type": (
        "What kind of evidence synthesis this is, read from how the paper describes itself in its "
        "abstract and methods. The most specific applicable value wins, so meta-analysis outranks "
        "systematic review when effect sizes are pooled, and umbrella review applies when the units "
        "reviewed are themselves reviews."
    ),
    "icd11_pain_category": (
        "The ICD-11 aligned pain category the paper is about, read from the full text. Mixed or "
        "unspecified covers a paper that genuinely spans several categories."
    ),
    "population": (
        "The population the reviewed evidence concerns. Mixed ages covers adults and younger "
        "participants together, and not applicable fits a purely theoretical paper with no population."
    ),
    "care_setting": (
        "The care setting the paper is about, when it reports one. Not reported is the honest answer "
        "for most reviews and is preferred over a guess."
    ),
    "primary_discipline": (
        "The disciplinary home of the paper, read from the journal, the framing, and the vocabulary "
        "rather than from author affiliations. Multidisciplinary describes the writing, not the "
        "author list."
    ),
    "pain_condition_detail": (
        "The exact pain condition or conditions the paper studies, in the paper's own words."
    ),
    "pain_conditions": (
        "The specific pain conditions the paper names, as short labels. The project vocabulary is a "
        "preferred spine, and the paper's own wording is kept when it is more precise."
    ),
    "context_note": (
        "The cultural, geographic, or healthcare-system context, when the paper states one. Empty "
        "when it does not."
    ),
    "quality_assessment_reported": (
        "Whether the paper reports a formal quality or risk-of-bias assessment of the evidence it "
        "reviews. This is descriptive only, since this review does not appraise those papers itself."
    ),
    "quality_assessment_tools": (
        "The appraisal tools the paper names, such as AMSTAR, ROBIS, GRADE, or the Cochrane "
        "risk-of-bias tool. Empty when none is named."
    ),
    # Biopsychosocial label
    "bps_label_used": (
        "Which biopsychosocial vocabulary the paper actually uses, from the explicit term through a "
        "variant such as psychosocial or multidimensional, down to domain language carrying no model "
        "label at all."
    ),
    "bps_primary_function": (
        "The single dominant work the biopsychosocial label does, judged over the paper as a whole. "
        "Operational definition is reserved for a paper that turns the model into the variables it "
        "codes or measures, and rhetorical label for ceremonial use with no analytic follow-through."
    ),
    "bps_functions_present": (
        "Every function the label performs anywhere in the paper. A paper routinely does two or three "
        "of these at once, and which ones it combines is itself a finding."
    ),
    "bps_definition_status": (
        "How the paper handles the meaning of the model itself, from a stated definition down to a "
        "label used with no meaning given anywhere. Undefined is a finding, not a coding failure."
    ),
    "bps_model_variants": (
        "The model labels the paper actually uses, verbatim and de-duplicated, such as "
        "biopsychosocial model, bio-psycho-social framework, or sociopsychobiological model. This is "
        "what makes terminological drift visible."
    ),
    "bps_usage_instances": (
        "One item for every distinct passage where the biopsychosocial label does work, with the "
        "function it serves there, whether it is definitional, who the model is credited to, and the "
        "quoted passage. A paper that invokes the model in the introduction and again in the "
        "discussion yields two items, not one."
    ),
    "bps_definitions": (
        "One item for every place where the paper says what the biopsychosocial model is, with the "
        "quoted definition, its type, its attributed source, and the components it names. The list is "
        "empty when the paper never says what the model is."
    ),
    "bps_operationalization_summary": (
        "What this paper actually does with the biopsychosocial model, as opposed to what it says "
        "about it, named as a mechanism of use and written in the coder's own words."
    ),
    "bps_function_set": (
        "Derived: every distinct function the label performs in this coding, pooled from the coded "
        "function list and from the functions attached to the individual usage passages."
    ),
    "bps_has_substantive_function": (
        "Derived yes or no: whether the label does substantive work anywhere in the paper, meaning it "
        "serves as an explanatory framework, an operational definition, an intervention rationale, or "
        "an organizing principle rather than only ceremonial or background use."
    ),
    "bps_usage_sections": (
        "Derived: the distinct sections of the paper in which the biopsychosocial label is used, "
        "pooled from the usage passages. Passages whose location is unclear are left out."
    ),
    # Domain coverage
    "domain_coverage_bio": (
        "How deeply the paper treats biological content: anatomy, physiology, pathophysiology, "
        "nociception, central or peripheral sensitization, inflammation, imaging, genetics, "
        "pharmacology, tissue pathology. The word biopsychosocial is never evidence of coverage."
    ),
    "domain_coverage_psych": (
        "How deeply the paper treats psychological content: cognition, affect, behaviour, beliefs, "
        "coping, catastrophizing, fear-avoidance, self-efficacy, depression, anxiety, acceptance, "
        "psychological treatment."
    ),
    "domain_coverage_social": (
        "How deeply the paper treats social content: work and occupational context, family and "
        "relationships, culture, socioeconomic position, healthcare system, social support, stigma, "
        "policy."
    ),
    "coverage_lifestyle": (
        "How deeply the paper treats lifestyle content, on the same ladder as the triad: physical "
        "activity and exercise behaviour, sleep, diet and weight, smoking, alcohol. Lifestyle is "
        "registered as a domain in its own right and is not folded into the three."
    ),
    "coverage_spiritual_existential": (
        "How deeply the paper treats spiritual or existential content, on the same ladder: meaning, "
        "faith or religion, hope, existential suffering. Absent is the expected value for most papers "
        "and is a finding in itself."
    ),
    "domain_evidence": (
        "One item per core domain that was not scored as absent, carrying the passage that justifies "
        "the coverage level given to it, together with the constructs the paper names and the "
        "ontology subdomains that content belongs to."
    ),
    "domain_evidence__biological": (
        "The domain-evidence item for the biological domain: the passage that justifies the "
        "biological coverage level, with the biological constructs the paper names and the ontology "
        "subdomains they belong to."
    ),
    "domain_evidence__psychological": (
        "The domain-evidence item for the psychological domain: the passage that justifies the "
        "psychological coverage level, with the constructs the paper names and the ontology "
        "subdomains they belong to."
    ),
    "domain_evidence__social": (
        "The domain-evidence item for the social domain: the passage that justifies the social "
        "coverage level, with the social constructs the paper names and the ontology subdomains they "
        "belong to."
    ),
    # Biopsychosocial entities
    "biological_factors": (
        "Every biological factor the paper names, one item each, in the paper's own wording, with the "
        "ontology subdomain it maps onto, the mechanism level it sits at, the role it plays in this "
        "paper, and the passage that shows it. Psychological constructs belong under psychological "
        "concepts instead."
    ),
    "social_factors": (
        "Every social factor the paper names, one item each, with its ontology subdomain, the level "
        "of social organization it sits at, its role, and the passage that shows it. The social "
        "domain is the one this literature is thinnest on, so even brief mentions are recorded."
    ),
    "other_domain_factors": (
        "Factors outside the triad, held in one list: lifestyle, spiritual or existential, "
        "environmental, and anything else that matters to the paper's account without belonging to a "
        "named domain."
    ),
    "other_domain_factors__lifestyle": (
        "The lifestyle factors the paper names, taken from the single list that holds everything "
        "beyond the triad: activity and exercise, sleep, diet and weight, smoking, alcohol, each with "
        "its role and its quoted passage."
    ),
    "other_domain_factors__spiritual_or_existential": (
        "The spiritual or existential factors the paper names, out of the same beyond-the-triad list: "
        "meaning, faith, hope, existential suffering, each with its role and its quoted passage."
    ),
    "other_domain_factors__environmental": (
        "The environmental factors the paper names, out of the same beyond-the-triad list: the "
        "physical environment, housing, climate, and the built environment, each with its role and "
        "its quoted passage."
    ),
    "psychological_concepts": (
        "Every psychological construct the paper uses, one item each, at the resolution the paper "
        "uses it, with the concept family it belongs to, whether and how the paper defines it, the "
        "measure it is operationalized with, and the role it plays."
    ),
    "concept_definitions_present": (
        "Whether the review defines the psychological constructs it uses. Partial means some are "
        "defined or clearly operationalized while others are only named."
    ),
    "concept_relations": (
        "Every relation the paper draws between two constructs, as a source, a relation type, and a "
        "target. These are the edges that turn a list of concepts into a map, and silent conflation "
        "of two constructs is recorded here as a relation type of its own."
    ),
    # Integration
    "integration_bio_psych": (
        "How far the paper integrates the biological and the psychological domain, on the pairwise "
        "ladder. Integration is a claim about a relation, never about co-occurrence in a sentence."
    ),
    "integration_psych_social": (
        "How far the paper integrates the psychological and the social domain, on the pairwise ladder."
    ),
    "integration_bio_social": (
        "How far the paper integrates the biological and the social domain, on the pairwise ladder. "
        "This is the pair this literature articulates least often."
    ),
    "integration_triadic": (
        "How far the paper integrates all three domains at once. Serial listing of biological, then "
        "psychological, then social content is none, however long the lists are. This is the most "
        "consequential judgement in the scheme."
    ),
    "integration_claims": (
        "One item for every passage that relates two or three domains to one another, naming the two "
        "specific factors that are linked, the direction, any named mediator or moderator, the quoted "
        "claim, and the pathway when one is given. This is the evidence base behind the four "
        "integration ladders."
    ),
    "integration_mechanism_summary": (
        "The cross-domain pathways this paper actually proposes, in the coder's own words, or 'none "
        "proposed' when it proposes none."
    ),
    # Typology and balance
    "overall_balance": (
        "Relative emphasis across the three core domains, judged on how much of the paper each one "
        "occupies. Dyadic means two domains carry the paper and the third is marginal."
    ),
    "bps_typology": (
        "What this review does with the biopsychosocial model at full-text depth, from a genuinely "
        "integrative account down to a single-domain review that claims a biopsychosocial frame."
    ),
    "derived_typology": (
        "The same typology recomputed from coverage and integration by a fixed rule, so the coded "
        "judgement can be checked against the rule the codebook states."
    ),
    "typology_matches_derived": (
        "Whether the coded typology and the rule-derived typology agree for this coding. Disagreement "
        "tests how tightly the typology is defined, and is not by itself a coding error."
    ),
    # Frameworks and instruments
    "theoretical_frameworks": (
        "Every theoretical model or framework the paper invokes, with the role it plays here, which "
        "of the three domains it actually spans, its attributed source, and the quoted passage."
    ),
    "instruments": (
        "Every measurement or appraisal instrument the paper names, with what the paper says it "
        "captures, the domain it measures, and its role. What a review measures is the most concrete "
        "form its operationalization of the model takes."
    ),
    # Conceptual problems
    "conceptual_problems": (
        "Conceptual problems the paper names or merely displays, each with its type, what it is "
        "about, the constructs it concerns, whether the authors point it out themselves, and the "
        "passage that shows it."
    ),
    # Synthesis hooks
    "key_quotes": (
        "The most conceptually load-bearing passages in the paper, quoted for the later synthesis, "
        "each typed and carrying one sentence on why it matters."
    ),
    "emergent_labels": (
        "Every conceptually important term this paper uses that the project vocabularies do not "
        "contain. This list is the review's own error signal: it is how the ontology finds out what "
        "it is missing."
    ),
    "conceptual_tensions": (
        "Contradictions, ambiguities, unresolved debates, and gaps the paper names or displays, "
        "including tensions visible inside the paper itself."
    ),
    "additional_observations": (
        "Anything else conceptually relevant that no other field captures, so an observation never "
        "has to be forced into a field where it does not belong."
    ),
    "synthesis_note": (
        "What this paper contributes to the question of how the biopsychosocial model is "
        "operationalized, and what it does not, written for a reviewer who has not read it."
    ),
    "coding_rationale": (
        "The coder's short justification of its main judgements: the typology, the triadic "
        "integration level, and anything that was a close call."
    ),
    # Presence flags
    "present_bps_usage_evidence": (
        "Derived yes or no: whether this coder returned at least one passage in which the "
        "biopsychosocial label does work."
    ),
    "present_bps_definition": (
        "Derived yes or no: whether this coder returned at least one definition of the "
        "biopsychosocial model."
    ),
    "present_integration_evidence": (
        "Derived yes or no: whether this coder returned at least one integration claim."
    ),
    "present_triadic_claim": (
        "Derived yes or no: whether at least one integration claim links all three domains at once."
    ),
    "present_named_integration_edge": (
        "Derived yes or no: whether at least one integration claim names both the source and the "
        "target factor, which is what makes a claim usable as an ontology edge."
    ),
    "present_biological_factors": (
        "Derived yes or no: whether this coder named at least one biological factor."
    ),
    "present_social_factors": (
        "Derived yes or no: whether this coder named at least one social factor."
    ),
    "present_other_domain_factors": (
        "Derived yes or no: whether this coder named at least one factor beyond the triad."
    ),
    "present_psychological_concepts": (
        "Derived yes or no: whether this coder named at least one psychological construct."
    ),
    "present_defined_concepts": (
        "Derived yes or no: whether at least one psychological construct is formally defined or "
        "operationalized through a measure rather than only named."
    ),
    "present_concept_relations": (
        "Derived yes or no: whether this coder recorded at least one relation between two constructs."
    ),
    "present_hierarchical_relation": (
        "Derived yes or no: whether at least one concept relation is hierarchical, meaning a subtype, "
        "a part or component, or a synonym relation."
    ),
    "present_theoretical_frameworks": (
        "Derived yes or no: whether this coder named at least one theoretical framework."
    ),
    "present_instruments": (
        "Derived yes or no: whether this coder named at least one instrument."
    ),
    "present_conceptual_problems": (
        "Derived yes or no: whether this coder recorded at least one conceptual problem."
    ),
    "present_domain_evidence_bio": (
        "Derived yes or no: whether a quoted evidence passage was returned for the biological domain."
    ),
    "present_domain_evidence_psych": (
        "Derived yes or no: whether a quoted evidence passage was returned for the psychological "
        "domain."
    ),
    "present_domain_evidence_social": (
        "Derived yes or no: whether a quoted evidence passage was returned for the social domain."
    ),
    # Eligibility and yield
    "fulltext_eligibility": (
        "The derived post-retrieval verdict, computed from the coded content rather than asked of the "
        "coder. It protects recall, so anything doubtful becomes uncertain rather than exclude, and "
        "it stays a recommendation for human adjudication."
    ),
    "fulltext_exclusion_reason": (
        "The rule that produced a verdict other than include, empty for an included paper."
    ),
    "conceptual_yield": (
        "The derived measure of how much conceptual material this paper actually yielded. It counts "
        "what was extracted, weighted towards integration claims, named factors, and defined "
        "concepts, because those carry the review's questions."
    ),
    "synthesis_priority": (
        "The derived reading order for the later synthesis, combining the eligibility verdict with "
        "the conceptual yield."
    ),
    "integration_index": (
        "One number per coding: the three pairwise ladders averaged and the triadic ladder, each "
        "normalized to its own top rung and then averaged, so papers can be compared on integration "
        "without treating a ladder as an interval scale."
    ),
    "coverage_total": (
        "The three core coverage ladders added up as depth scores, from 0 when all three domains are "
        "absent to 9 when all three are elaborated."
    ),
    "domains_present": (
        "How many of the three core domains are substantively covered, counting a domain as present "
        "from the mentioned rung upward."
    ),
    "coverage_depth_bio": "The biological coverage ladder as a depth score.",
    "coverage_depth_psych": "The psychological coverage ladder as a depth score.",
    "coverage_depth_social": "The social coverage ladder as a depth score.",
    "pairwise_depth_total": (
        "The three pairwise integration ladders added up as depth scores, from 0 when no pair is "
        "related to 12 when all three pairs are mechanistic."
    ),
    "pairwise_depth_max": (
        "The highest of the three pairwise integration ladders, as a depth score."
    ),
    "triadic_depth": "The triadic integration ladder as a depth score.",
    # Counts and provenance
    "n_bps_usage_instances": (
        "How many biopsychosocial usage passages this provider extracted from this paper."
    ),
    "n_bps_definitions": (
        "How many definitions of the biopsychosocial model this provider extracted from this paper."
    ),
    "n_domain_evidence": (
        "How many domain-evidence items this provider extracted, at most one per core domain."
    ),
    "n_biological_factors": "How many biological factors this provider named for this paper.",
    "n_social_factors": "How many social factors this provider named for this paper.",
    "n_other_domain_factors": (
        "How many factors beyond the triad this provider named for this paper."
    ),
    "n_psychological_concepts": (
        "How many psychological constructs this provider named for this paper."
    ),
    "n_defined_concepts": (
        "How many of those constructs are formally defined or operationalized through a measure "
        "rather than only named."
    ),
    "n_concept_relations": (
        "How many relations between constructs this provider extracted from this paper."
    ),
    "n_hierarchical_relations": (
        "How many of those relations are hierarchical: subtype, part or component, or synonym."
    ),
    "n_integration_claims": (
        "How many cross-domain integration claims this provider extracted from this paper."
    ),
    "n_triadic_claims": "How many of those claims link all three domains at once.",
    "n_named_integration_edges": (
        "How many of those claims name both ends of the link, which is how many usable ontology edges "
        "this coding contributes."
    ),
    "n_theoretical_frameworks": (
        "How many theoretical frameworks this provider named for this paper."
    ),
    "n_instruments": "How many instruments this provider named for this paper.",
    "n_conceptual_problems": (
        "How many conceptual problems this provider recorded for this paper."
    ),
    "n_key_quotes": "How many key quotes this provider extracted from this paper.",
    "n_emergent_labels": (
        "How many terms this provider recorded as absent from the project vocabularies."
    ),
    "n_subdomains_bio": (
        "How many distinct biological ontology subdomains this coding touched, across the biological "
        "factors and the biological evidence item."
    ),
    "n_subdomains_psych": (
        "How many distinct psychological concept families this coding touched, across the "
        "psychological concepts and the psychological evidence item."
    ),
    "n_subdomains_social": (
        "How many distinct social ontology subdomains this coding touched, across the social factors "
        "and the social evidence item."
    ),
    "n_subdomains_named": (
        "How many distinct ontology subdomains this coding touched in total, across the three "
        "domains. This is the breadth of the account the paper gives."
    ),
    "n_bps_functions": (
        "How many distinct functions the biopsychosocial label performs in this coding."
    ),
    "n_bps_usage_sections": (
        "In how many distinct sections of the paper the biopsychosocial label is used."
    ),
    "n_open_list_entries": (
        "How many entries this coding wrote across all record-level open lists together."
    ),
    "n_labels_checked": (
        "How many extracted items carried an ontology anchor that could be checked against the "
        "project vocabularies."
    ),
    "controlled_label_share": (
        "The share of those anchors that landed on the controlled vocabulary. It measures the "
        "ontology against the literature, so a low share says the vocabularies need extending rather "
        "than that the coder erred."
    ),
    "n_evidence_quotes": (
        "How many non-empty verbatim quotes this coding carries across every extraction list. Quotes "
        "are checked against the source text after the run."
    ),
    "n_extracted_items": (
        "The total number of structured items in this coding, summed over the thirteen extraction "
        "lists."
    ),
    "coding_method": (
        "How this row was produced. A paper that never coded after every retry is written as an "
        "explicit failed row rather than dropped, so the table stays complete."
    ),
    "llm_model": "The model identifier of the provider that produced this coding.",
}

# Closed value lists that are derived rather than coded, so they are not part of
# the prompt's controlled vocabularies. The strings match ``coding.derive``.
DERIVED_VALUES: dict[str, list[str]] = {
    "fulltext_eligibility": list(schema.ELIGIBILITY_OPTIONS),
    "conceptual_yield": list(schema.YIELD_OPTIONS),
    "synthesis_priority": list(schema.SYNTHESIS_PRIORITY_OPTIONS),
    # The rule never returns "unclear": it always lands on one of the five.
    "derived_typology": [
        value for value in schema.TYPOLOGY_OPTIONS if value != "unclear"
    ],
    "typology_matches_derived": list(PRESENCE_ORDER),
    "bps_has_substantive_function": list(PRESENCE_ORDER),
    "coding_method": ["llm_structured", "coding_failed"],
    "bps_function_set": list(schema.BPS_FUNCTION_OPTIONS),
    "bps_usage_sections": [
        value for value in schema.SECTION_OPTIONS if value != "unclear"
    ],
    "fulltext_exclusion_reason": [
        "(empty, the paper is included)",
        "no biopsychosocial domain content in the full text",
        "single-domain review with no cross-domain claim",
        "reads as a primary study rather than an evidence synthesis",
        "fewer than two domains substantively covered",
        "no biopsychosocial vocabulary and no readable typology",
        "typology not readable and no triadic integration found",
    ],
}

# The ladder scores, spelled out as the rung each number stands for.
COVERAGE_SCALE = ["0 = absent", "1 = minimal", "2 = mentioned", "3 = elaborated"]
PAIRWISE_SCALE = ["0 = none", "1 = mentioned", "2 = descriptive", "3 = directional", "4 = mechanistic"]
TRIADIC_SCALE = ["0 = none", "1 = partial", "2 = descriptive", "3 = mechanistic"]

SCORE_VALUES: dict[str, list[str]] = {
    "coverage_depth_bio": COVERAGE_SCALE,
    "coverage_depth_psych": COVERAGE_SCALE,
    "coverage_depth_social": COVERAGE_SCALE,
    "pairwise_depth_max": PAIRWISE_SCALE,
    "triadic_depth": TRIADIC_SCALE,
}

# Which ladder rules belong to which coded field.
FIELD_LADDERS: dict[str, str] = {
    **{name: "domain_coverage" for name in (
        "domain_coverage_bio",
        "domain_coverage_psych",
        "domain_coverage_social",
        "coverage_lifestyle",
        "coverage_spiritual_existential",
    )},
    **{name: "pairwise_integration" for name in (
        "integration_bio_psych",
        "integration_psych_social",
        "integration_bio_social",
    )},
    "integration_triadic": "triadic_integration",
}

# The item-count columns and the extraction list each one counts.
COUNT_SOURCE_FIELDS: dict[str, str] = {f"n_{name}": name for name in schema.ITEM_MODELS}
COUNT_SOURCE_FIELDS.update(
    {
        "n_defined_concepts": "psychological_concepts",
        "n_hierarchical_relations": "concept_relations",
        "n_triadic_claims": "integration_claims",
        "n_named_integration_edges": "integration_claims",
        "n_emergent_labels": "emergent_labels",
    }
)

_TOTAL_ITEM_CAP = sum(ITEM_CAPS.values())
_TOTAL_OPEN_LIST_CAP = sum(OPEN_LIST_CAPS.values())
_ANCHORED_ITEM_CAP = sum(ITEM_CAPS[name] for name in schema.ITEM_ANCHOR)
_N_SECTIONS = len([value for value in schema.SECTION_OPTIONS if value != "unclear"])

# Formats for the fields with no closed value list.
OPEN_VALUE_FORMATS: dict[str, str] = {
    "pain_condition_detail": f"Free text, at most {MAX_NOTE_WORDS} words",
    "context_note": f"Free text, at most {MAX_NOTE_WORDS} words",
    "coding_rationale": f"Free text, at most {MAX_NOTE_WORDS} words",
    "bps_operationalization_summary": f"Free text, at most {MAX_SUMMARY_WORDS} words",
    "integration_mechanism_summary": f"Free text, at most {MAX_SUMMARY_WORDS} words",
    "synthesis_note": f"Free text, at most {MAX_SUMMARY_WORDS} words",
    "llm_model": "Provider model identifier",
    "integration_index": "Number between 0 and 1, rounded to four decimals",
    "controlled_label_share": "Number between 0 and 1, rounded to four decimals",
    "coverage_total": "Whole number, 0 to 9",
    "domains_present": "Whole number, 0 to 3",
    "pairwise_depth_total": "Whole number, 0 to 12",
    "n_evidence_quotes": f"Whole number, 0 to {_TOTAL_ITEM_CAP}",
    "n_extracted_items": f"Whole number, 0 to {_TOTAL_ITEM_CAP}",
    "n_open_list_entries": f"Whole number, 0 to {_TOTAL_OPEN_LIST_CAP}",
    "n_labels_checked": f"Whole number, 0 to {_ANCHORED_ITEM_CAP}",
    "n_bps_functions": f"Whole number, 0 to {len(schema.BPS_FUNCTION_OPTIONS)}",
    "n_bps_usage_sections": f"Whole number, 0 to {_N_SECTIONS}",
    "n_subdomains_bio": "Whole number",
    "n_subdomains_psych": "Whole number",
    "n_subdomains_social": "Whole number",
    "n_subdomains_named": "Whole number",
    **{
        column: f"Whole number, 0 to {ITEM_CAPS[source]}"
        for column, source in COUNT_SOURCE_FIELDS.items()
        if source in ITEM_CAPS
    },
    **{
        column: f"Whole number, 0 to {OPEN_LIST_CAPS[source]}"
        for column, source in COUNT_SOURCE_FIELDS.items()
        if source in OPEN_LIST_CAPS
    },
}


def _closed_values(field: str, column: str) -> list[str] | None:
    """The closed vocabulary of a field, coded or derived, if it has one."""
    for name in (field, column):
        if name in CONTROLLED_VALUES:
            return list(CONTROLLED_VALUES[name])
        if name in CONTROLLED_LIST_VALUES:
            return list(CONTROLLED_LIST_VALUES[name])
        if name in DERIVED_VALUES:
            return list(DERIVED_VALUES[name])
        if name in SCORE_VALUES:
            return list(SCORE_VALUES[name])
        if name.startswith("present_"):
            return list(PRESENCE_ORDER)
    return None


def _value_format(column: str) -> str:
    """A short statement of the value space of a field with no closed vocabulary."""
    if column in STRUCTURED_FIELDS:
        cap = ITEM_CAPS.get(column)
        quoted = " Every item carries a verbatim quote." if column in schema.ITEM_QUOTE_KEY else ""
        return (f"Structured extraction list, at most {cap} items.{quoted}" if cap
                else "Structured extraction list.")
    if column in FLAT_LIST_FIELDS:
        cap = OPEN_LIST_CAPS.get(column)
        limit = f", at most {cap} entries" if cap else ""
        return f"Open list{limit}, written as one value per entry"
    return OPEN_VALUE_FORMATS.get(column, "Free text")


def _item_value_space(column: str) -> dict[str, Any]:
    """Every field inside one extracted item, with its vocabulary or its format."""
    model = schema.ITEM_MODELS[column]
    value_lists = ITEM_VALUE_LISTS.get(column, {})
    space: dict[str, Any] = {}
    for name, info in model.model_fields.items():
        if name in value_lists:
            space[name] = list(value_lists[name])
        elif name.endswith("verbatim"):
            space[name] = f"Verbatim quote copied from the article, at most {MAX_QUOTE_WORDS} words"
        elif info.annotation == list[str]:
            space[name] = f"List of short free-text labels, at most {ITEM_SUBLIST_CAP}"
        elif name in ("note", "why_it_matters", "mechanism_note"):
            space[name] = f"Free text, at most {MAX_NOTE_WORDS} words"
        else:
            space[name] = "Free-text label"
    return space


def _field_reference(view: "FieldView") -> "OrderedDict[str, Any]":
    """The explanation and the value space shown on a coding-field node.

    A filtered view explains its own slice when it has an explanation of its own,
    and otherwise falls back to the explanation of the column it reads.
    """
    field = view.resolved_key()
    column = view.column
    reference: "OrderedDict[str, Any]" = OrderedDict()
    description = FIELD_DESCRIPTIONS.get(field) or FIELD_DESCRIPTIONS.get(column)
    if description:
        reference["What this field records"] = description
    values = _closed_values(field, column)
    if values:
        reference["Possible values"] = values
    else:
        reference["Value format"] = _value_format(column)
    ladder = FIELD_LADDERS.get(column)
    if ladder:
        reference["What each value means"] = dict(LADDERS[ladder])
    if column in STRUCTURED_FIELDS:
        reference["Item fields and their values"] = _item_value_space(column)
    return reference


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
                **_field_reference(view),
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


def bundle_readme(payload: dict[str, Any]) -> str:
    """The run-specific opening instructions written next to the bundle."""
    return (
        "# Knowledge graph review surface\n\n"
        "Open `index.html` in a desktop browser. The bundle is fully local and requires no server.\n\n"
        f"- Papers: {payload['meta']['n_papers']}\n"
        f"- Providers: {payload['meta']['n_providers']}\n"
        f"- Coding cells: {payload['meta']['n_codings']}\n"
        f"- Graph nodes: {payload['meta']['n_nodes']}\n"
        f"- Graph links: {payload['meta']['n_edges']}\n\n"
        "Search accepts several words at once, all of which must match, and ranks what it finds: a "
        "quoted phrase stays contiguous, a leading minus excludes a word, and field:, group:, "
        "provider:, article:, label:, and type: aim a word at one part of a node. The filter panel "
        "and the inspector each fold away from the toolbar to give the canvas their width.\n\n"
        "The first view shows the field groups, the biopsychosocial entities, and all canonical "
        "scheme 3 coding fields. The entity level holds the triad as three siblings and everything "
        "beyond it under Other factors, which carries lifestyle and spiritual or existential as its "
        "own children, so the evidence for one domain sits under that domain rather than in one "
        "undifferentiated list.\n\n"
        "Every coding field explains itself: its card and its inspector state what the field records, "
        "list its possible values when the field has a closed vocabulary, give its value format when it "
        "does not, spell out the rung-by-rung rule for the coverage and integration ladders, and, for a "
        "structured extraction list, name every item field with its own vocabulary.\n\n"
        "Double-click a "
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
        "returns to the complete scheme overview and clears manual placement.\n"
    )


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
    (output_dir / "README.md").write_text(bundle_readme(payload), encoding="utf-8")
    return index_path
