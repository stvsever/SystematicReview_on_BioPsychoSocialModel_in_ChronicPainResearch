from __future__ import annotations

"""Central configuration for the cross-provider full-text test run (scheme 3).

Scheme 3 applies the Stage 3 full-text deep coding scheme to whole open-access
articles. It differs from scheme 2 (``bps_review.pilot``) in purpose and in
resolution:

* scheme 2 reads a title and an abstract and acts mainly as a routing and
  relevance filter;
* scheme 3 reads the full text and answers the review's central question: does a
  biopsychosocially labelled review actually integrate the three domains, and if
  so, how. It grades the depth of each domain, grades every pairwise and the
  triadic integration on an explicit ladder, and carries a verbatim quote for
  each of those judgements so the ladder is auditable.

Scheme 3 is also the pass that harvests the ontology. Alongside the graded
judgements it extracts the named things a review carries: every biological,
social, and lifestyle or existential factor, every psychological construct with
its definitional status, the relations drawn between constructs, the frameworks
and instruments in use, and every passage where the biopsychosocial label does
work. Those items are nodes and edges, so the caps below are generous and the
completion budget is sized for a long structured answer rather than a verdict.

The same three cheap models from three different providers are used as in the
abstract-level run, so the two stages are directly comparable and swapping in
state-of-the-art models stays a one-line change.
"""

from dataclasses import dataclass

from bps_review.pilot.config import TESTRUN_MODELS, TestRunModel
from bps_review.utils.paths import project_path


# --------------------------------------------------------------------------
# Models. Identical to the abstract-level run, on purpose.
# --------------------------------------------------------------------------
FULLTEXT_MODELS: list[TestRunModel] = list(TESTRUN_MODELS)

MODEL_LABELS: list[str] = [model.label for model in FULLTEXT_MODELS]
MODEL_BY_ID: dict[str, TestRunModel] = {model.openrouter_id: model for model in FULLTEXT_MODELS}
MODEL_BY_LABEL: dict[str, TestRunModel] = {model.label: model for model in FULLTEXT_MODELS}


# --------------------------------------------------------------------------
# Run settings. One paper per request (full texts are far too long to batch),
# high worker counts so the whole grid finishes in one pass.
# --------------------------------------------------------------------------
MAX_WORKERS = 12              # concurrent papers per model
MAX_MODEL_WORKERS = 3         # models coded concurrently (one worker per model)
MAX_RETRIES = 4               # attempts per paper before it is marked failed
RETRY_BACKOFF_SECONDS = 2.5
HARD_TIMEOUT_SECONDS = 420.0  # wall-clock cap per attempt; a stalled provider is abandoned and retried
REQUEST_TIMEOUT_SECONDS = 400

# Every model used here advertises at least a 131k-token context window, so the
# binding constraint is cost, not context. The instruction block is around 12k
# tokens (it carries the anchors, the ladders, and the preferred-label
# vocabularies) and this budget adds at most another 15k for the article, so one
# paper costs roughly 27k input tokens and the whole run stays inside a few
# dollars while still sending most papers whole.
CODING_TEXT_CHAR_BUDGET = 60_000
MAX_OUTPUT_TOKENS = 16000

# Per-model runtime settings. Two of the three answer this task directly. The
# third reasons regardless of the reasoning controls, and on a reasoning model
# the thinking tokens come out of the same completion budget as the answer, so a
# long structured extraction needs a budget that covers both. This is a property
# of the endpoint, not of the coding scheme: every model receives the identical
# prompt and returns the identical schema.
#
# The budgets are sized for the extraction, not for the verdict. A rich paper
# fills thirteen extraction lists, so a truncated completion is the main way this
# run can lose data, and completion tokens are the cheap half of the bill.
MODEL_RUNTIME: dict[str, dict] = {
    "deepseek/deepseek-v4-flash": {"reasoning": {"enabled": False}, "max_output_tokens": 16000},
    "nex-agi/nex-n2-mini": {"reasoning": {"effort": "low"}, "max_output_tokens": 40000},
    "poolside/laguna-xs-2.1": {"reasoning": {"enabled": False}, "max_output_tokens": 16000},
}
DEFAULT_RUNTIME: dict = {"reasoning": {"enabled": False}, "max_output_tokens": MAX_OUTPUT_TOKENS}


def model_runtime(openrouter_id: str) -> dict:
    """Reasoning settings and completion budget for one model."""
    return MODEL_RUNTIME.get(openrouter_id, DEFAULT_RUNTIME)


# Caps on how many items the models may return per extraction list, and how long
# a quoted passage may be. They are ceilings, never targets: an empty list is a
# legitimate coding, and a paper that genuinely says more is allowed to fill them.
# The factor and concept lists carry the ontology, so their ceilings are set to
# what a rich umbrella review can actually contain rather than to a round number.
ITEM_CAPS: dict[str, int] = {
    "bps_usage_instances": 8,
    "bps_definitions": 3,
    "domain_evidence": 5,
    "biological_factors": 12,
    "social_factors": 12,
    "psychological_concepts": 16,
    "other_domain_factors": 6,
    "concept_relations": 12,
    "integration_claims": 12,
    "theoretical_frameworks": 8,
    "instruments": 8,
    "conceptual_problems": 8,
    "key_quotes": 6,
}

# Caps on the record-level free-text lists. emergent_labels is deliberately the
# most generous of them: it is where a term that the project vocabularies do not
# carry gets recorded, and those terms are how the ontology finds out what it is
# missing.
OPEN_LIST_CAPS: dict[str, int] = {
    "pain_conditions": 6,
    "quality_assessment_tools": 4,
    "bps_model_variants": 5,
    "bps_functions_present": 6,
    "emergent_labels": 12,
    "conceptual_tensions": 5,
    "additional_observations": 6,
}

# Caps on the short list fields inside structured items.
ITEM_SUBLIST_CAP = 6

MAX_QUOTE_WORDS = 60
MAX_NOTE_WORDS = 40
MAX_SUMMARY_WORDS = 90


# --------------------------------------------------------------------------
# Coded fields, grouped for the reliability analysis.
#
# Agreement is quantified on three kinds of variable, kept apart because they
# answer different questions.
#
# * **Ordered ladders.** Domain coverage and the four integration fields. These
#   carry the review's central construct, and a disagreement between two coders
#   is a real disagreement about the paper.
# * **Binary presence.** One yes or no per extraction element, derived from the
#   coded content rather than asked. Whether two coders both found a theoretical
#   framework in a paper has one answer; whether they wrote the same free-text
#   label for it does not, and that second question is answered by set overlap.
# * **Open lists.** Concepts, frameworks, and conceptual problems, compared with
#   Jaccard overlap over normalized labels.
# --------------------------------------------------------------------------
COVERAGE_FIELDS: list[str] = [
    "domain_coverage_bio",
    "domain_coverage_psych",
    "domain_coverage_social",
]

# The two domains the registration names alongside the triad. They ride the same
# ladder but stay out of COVERAGE_FIELDS, because every triad-level derivation
# (balance, typology, eligibility) is a statement about the three core domains.
AUXILIARY_COVERAGE_FIELDS: list[str] = [
    "coverage_lifestyle",
    "coverage_spiritual_existential",
]

INTEGRATION_FIELDS: list[str] = [
    "integration_bio_psych",
    "integration_psych_social",
    "integration_bio_social",
    "integration_triadic",
]

NOMINAL_FIELDS: list[str] = [
    "review_track",
    "source_type",
    "icd11_pain_category",
    "population",
    "care_setting",
    "primary_discipline",
    "quality_assessment_reported",
    "bps_label_used",
    "bps_primary_function",
    "bps_definition_status",
    "overall_balance",
    "bps_typology",
    "concept_definitions_present",
    "fulltext_eligibility",
    "synthesis_priority",
]

PRESENCE_FIELDS: list[str] = [
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
]

CATEGORICAL_FIELDS: list[str] = (
    COVERAGE_FIELDS + AUXILIARY_COVERAGE_FIELDS + INTEGRATION_FIELDS + NOMINAL_FIELDS
)

# Every field on which agreement is quantified, in a readable order.
RELIABILITY_FIELDS: list[str] = CATEGORICAL_FIELDS + PRESENCE_FIELDS

# The extraction lists compared across models with set overlap. Two coders can
# both be right and still write different strings, so these are never scored with
# kappa. Every list of the scheme is compared, not a chosen few: a list left out
# of the overlap metrics is a part of the extraction nobody ever checks.
LIST_FIELDS: list[str] = [
    "bps_usage_instances",
    "bps_definitions",
    "domain_evidence",
    "biological_factors",
    "social_factors",
    "psychological_concepts",
    "other_domain_factors",
    "concept_relations",
    "integration_claims",
    "theoretical_frameworks",
    "instruments",
    "conceptual_problems",
    "key_quotes",
]

# The extraction lists whose items carry a verbatim quote from the source text.
QUOTED_ITEM_FIELDS: list[str] = [
    "bps_usage_instances",
    "bps_definitions",
    "domain_evidence",
    "biological_factors",
    "social_factors",
    "psychological_concepts",
    "other_domain_factors",
    "concept_relations",
    "integration_claims",
    "theoretical_frameworks",
    "instruments",
    "conceptual_problems",
    "key_quotes",
]

# What identifies an item for the overlap metrics: one key, or several joined,
# because a relation and an integration claim are edges rather than labels.
#
# This mirrors ``schema.ITEM_LABEL_KEY`` and is written out rather than imported,
# because this module is the leaf that the schema package itself imports. A test
# asserts the two tables are identical, so they cannot drift apart in silence.
LIST_LABEL_KEY: dict[str, tuple[str, ...]] = {
    "bps_usage_instances": ("bps_function",),
    "bps_definitions": ("attributed_source",),
    "domain_evidence": ("domain",),
    "biological_factors": ("factor_label",),
    "social_factors": ("factor_label",),
    "psychological_concepts": ("concept_label",),
    "other_domain_factors": ("factor_label",),
    "concept_relations": ("source_concept", "relation_type", "target_concept"),
    "integration_claims": ("source_factor_label", "domains_linked", "target_factor_label"),
    "theoretical_frameworks": ("framework_label",),
    "instruments": ("instrument_label",),
    "conceptual_problems": ("problem_type",),
    "key_quotes": ("claim_type",),
}

# Which project vocabulary the identifying label is normalized against before two
# models are compared, so "catastrophising" and "catastrophizing" are one label.
LIST_LABEL_VOCAB: dict[str, str] = {
    "biological_factors": "bio_subdomain",
    "social_factors": "social_subdomain",
    "psychological_concepts": "psych_concept",
    "theoretical_frameworks": "framework",
    "instruments": "instrument",
    "bps_definitions": "bps_source",
}

# Whether an item's identity is the coder's own wording or a value from a closed
# list. The distinction decides how a number should be read, not whether it is
# computed: on a controlled identity the two coders picked from the same menu, so
# the semantic layer has nothing to merge and a gap over the lexical score would
# be an artifact. Reporting both side by side makes that visible instead of
# hiding it.
LIST_LABEL_KIND: dict[str, str] = {
    "bps_usage_instances": "controlled",
    "bps_definitions": "free text",
    "domain_evidence": "controlled",
    "biological_factors": "free text",
    "social_factors": "free text",
    "psychological_concepts": "free text",
    "other_domain_factors": "free text",
    "concept_relations": "free text",
    "integration_claims": "free text",
    "theoretical_frameworks": "free text",
    "instruments": "free text",
    "conceptual_problems": "controlled",
    "key_quotes": "controlled",
}

# --------------------------------------------------------------------------
# Comparison spaces: every vocabulary two coders could have written differently.
#
# Comparing extraction lists by item identity answers one question, "did the two
# coders find the same things", and it is the question the lexical table answers.
# It is not the only question the extraction supports, and on its own it wastes
# most of the resolution the scheme was built for.
#
# A scheme 3 item is not a label. It is a small record, and several of its fields
# are themselves open vocabularies: which constructs a coder says carry the
# biological domain, which measure a construct is operationalized with, which
# elements a definition of the model lists, which constructs a conceptual problem
# concerns, who the model is attributed to. Each of those is a place where two
# coders can read a paper identically and still write different words, which is
# exactly what the semantic layer exists to separate from real disagreement.
#
# A space is therefore any set of labels that can be pulled out of one extraction
# list and compared across coders. Three kinds are declared below.
#
# * **Identity spaces**, one per extraction list: what was extracted.
# * **Vocabulary spaces**, reading one field or one sublist inside the items: what
#   the coder called it, and where it was anchored.
# * **Filtered spaces**, restricting a list to the items that carry weight, so a
#   construct the review actually defines is not averaged together with one it
#   merely names, and a framework the review is built on is not averaged together
#   with one mentioned in passing.
#
# Spaces whose source list a run does not carry are skipped, so this table can
# describe the whole scheme while an older run reports only what it holds.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractionSpace:
    """One comparable vocabulary, read out of one extraction list."""

    name: str                            # stable identifier, used as the row key
    label: str                           # how the space is named in a table or a figure
    question: str                        # what comparing it answers, in one line
    field: str                           # the extraction list it is read from
    keys: tuple[str, ...] = ()           # item keys joined into one label
    sublist: str = ""                    # item key holding a list of labels
    vocabulary: str = ""                 # project vocabulary used to normalize, if any
    kind: str = "free text"              # "free text" or "controlled"
    layer: str = "identity"              # "identity", "vocabulary", or "filtered"
    filter_key: str = ""                 # restrict to items whose ...
    filter_values: tuple[str, ...] = ()  # ... value for filter_key is one of these


# The psychological constructs a review states a meaning for, as opposed to the
# ones it only names. RQ3 turns on the difference.
DEFINED_CONCEPT_STATUSES = ("formally_defined", "operationalized_only")

# A framework the review is organized around, tests, revises, or rejects, as
# opposed to one cited in passing.
SUBSTANTIVE_FRAMEWORK_ROLES = (
    "organizing framework",
    "tested or modelled",
    "extended or revised",
    "critiqued or rejected",
    "compared with another model",
)


def _identity_spaces() -> list[ExtractionSpace]:
    """One space per extraction list, in the order the scheme declares them."""
    questions = {
        "bps_usage_instances": "Do the coders find the label doing the same work",
        "bps_definitions": "Do the coders credit the model to the same sources",
        "domain_evidence": "Do the coders carry evidence for the same domains",
        "biological_factors": "Do the coders name the same biological factors",
        "social_factors": "Do the coders name the same social factors",
        "psychological_concepts": "Do the coders name the same psychological constructs",
        "other_domain_factors": "Do the coders reach beyond the triad in the same places",
        "concept_relations": "Do the coders draw the same edges between constructs",
        "integration_claims": "Do the coders draw the same links between domains",
        "theoretical_frameworks": "Do the coders find the same frameworks",
        "instruments": "Do the coders find the same instruments",
        "conceptual_problems": "Do the coders see the same conceptual problems",
        "key_quotes": "Do the coders quote the same kinds of claim",
    }
    return [
        ExtractionSpace(
            name=field,
            label=FIELD_LABELS.get(field, field.replace("_", " ").capitalize()),
            question=questions.get(field, f"Do the coders agree on {field.replace('_', ' ')}"),
            field=field,
            keys=LIST_LABEL_KEY.get(field, ()),
            vocabulary=LIST_LABEL_VOCAB.get(field, ""),
            kind=LIST_LABEL_KIND.get(field, "free text"),
            layer="identity",
        )
        for field in LIST_FIELDS
    ]


VOCABULARY_SPACES: list[ExtractionSpace] = [
    ExtractionSpace(
        name="domain_evidence_constructs",
        label="Constructs carrying a domain",
        question="Do the coders name the same constructs as carrying the domains",
        field="domain_evidence",
        sublist="constructs_named",
        layer="vocabulary",
    ),
    ExtractionSpace(
        name="domain_evidence_constructs_bio",
        label="Constructs carrying biology",
        question="Do the coders name the same biological content",
        field="domain_evidence",
        sublist="constructs_named",
        layer="vocabulary",
        filter_key="domain",
        filter_values=("biological",),
    ),
    ExtractionSpace(
        name="domain_evidence_constructs_psych",
        label="Constructs carrying psychology",
        question="Do the coders name the same psychological content",
        field="domain_evidence",
        sublist="constructs_named",
        layer="vocabulary",
        filter_key="domain",
        filter_values=("psychological",),
    ),
    ExtractionSpace(
        name="domain_evidence_constructs_social",
        label="Constructs carrying the social",
        question="Do the coders name the same social content",
        field="domain_evidence",
        sublist="constructs_named",
        layer="vocabulary",
        filter_key="domain",
        filter_values=("social",),
    ),
    ExtractionSpace(
        name="domain_evidence_subdomains",
        label="Subdomains touched",
        question="Do the coders anchor a domain to the same ontology subdomains",
        field="domain_evidence",
        sublist="subdomains_named",
        layer="vocabulary",
    ),
    ExtractionSpace(
        name="bio_subdomain_anchors",
        label="Biological subdomain anchors",
        question="Do the coders place biological factors on the same ontology spine",
        field="biological_factors",
        keys=("subdomain_label",),
        vocabulary="bio_subdomain",
        layer="vocabulary",
    ),
    ExtractionSpace(
        name="social_subdomain_anchors",
        label="Social subdomain anchors",
        question="Do the coders place social factors on the same ontology spine",
        field="social_factors",
        keys=("subdomain_label",),
        vocabulary="social_subdomain",
        layer="vocabulary",
    ),
    ExtractionSpace(
        name="concept_families",
        label="Concept families",
        question="Do the coders sort constructs into the same families",
        field="psychological_concepts",
        keys=("concept_family",),
        vocabulary="concept_family",
        layer="vocabulary",
    ),
    ExtractionSpace(
        name="concept_measures",
        label="Measures behind a construct",
        question="Do the coders operationalize constructs with the same measures",
        field="psychological_concepts",
        keys=("measure_named",),
        vocabulary="instrument",
        layer="vocabulary",
    ),
    ExtractionSpace(
        name="instrument_constructs",
        label="What an instrument measures",
        question="Do the coders say an instrument measures the same thing",
        field="instruments",
        keys=("construct_measured_as_stated",),
        layer="vocabulary",
    ),
    ExtractionSpace(
        name="framework_domains",
        label="Domains a framework spans",
        question="Do the coders say a framework covers the same domains",
        field="theoretical_frameworks",
        sublist="domains_covered",
        layer="vocabulary",
    ),
    ExtractionSpace(
        name="bps_definition_elements",
        label="Elements of a BPS definition",
        question="Do the coders read the same components into the model",
        field="bps_definitions",
        sublist="elements_named",
        layer="vocabulary",
    ),
    ExtractionSpace(
        name="bps_attributions",
        label="Who the model is credited to",
        question="Do the coders attribute the model to the same sources",
        field="bps_usage_instances",
        keys=("attributed_source",),
        vocabulary="bps_source",
        layer="vocabulary",
    ),
    ExtractionSpace(
        name="problem_affected_labels",
        label="Constructs a problem concerns",
        question="Do the coders attach a conceptual problem to the same constructs",
        field="conceptual_problems",
        sublist="affected_labels",
        layer="vocabulary",
    ),
    ExtractionSpace(
        name="integration_mediators",
        label="Named mediators and moderators",
        question="Do the coders name the same intermediates between domains",
        field="integration_claims",
        keys=("mediator_or_moderator",),
        layer="vocabulary",
    ),
    ExtractionSpace(
        name="integration_endpoints",
        label="Factors at the ends of a link",
        question="Do the coders link the same named factors, not merely the same domains",
        field="integration_claims",
        keys=("source_factor_label", "target_factor_label"),
        layer="vocabulary",
    ),
]

FILTERED_SPACES: list[ExtractionSpace] = [
    ExtractionSpace(
        name="defined_concepts",
        label="Concepts the review defines",
        question="Do the coders agree on which constructs are actually given a meaning",
        field="psychological_concepts",
        keys=("concept_label",),
        vocabulary="psych_concept",
        layer="filtered",
        filter_key="definitional_status",
        filter_values=DEFINED_CONCEPT_STATUSES,
    ),
    ExtractionSpace(
        name="substantive_frameworks",
        label="Frameworks doing real work",
        question="Do the coders agree on which frameworks the review is built on",
        field="theoretical_frameworks",
        keys=("framework_label",),
        vocabulary="framework",
        layer="filtered",
        filter_key="role",
        filter_values=SUBSTANTIVE_FRAMEWORK_ROLES,
    ),
    ExtractionSpace(
        name="mechanistic_integration_claims",
        label="Mechanistic integration claims",
        question="Do the coders agree on where a real pathway is described",
        field="integration_claims",
        keys=("source_factor_label", "domains_linked", "target_factor_label"),
        layer="filtered",
        filter_key="integration_level",
        filter_values=("mechanistic", "directional"),
    ),
    ExtractionSpace(
        name="author_named_problems",
        label="Problems the authors name themselves",
        question="Do the coders agree on which problems the review notices about itself",
        field="conceptual_problems",
        keys=("problem_type",),
        kind="controlled",
        layer="filtered",
        filter_key="named_by_authors",
        filter_values=("yes",),
    ),
]


COUNT_FIELDS: list[str] = [f"n_{name}" for name in (
    "bps_usage_instances",
    "bps_definitions",
    "integration_claims",
    "named_integration_edges",
    "domain_evidence",
    "biological_factors",
    "social_factors",
    "other_domain_factors",
    "psychological_concepts",
    "defined_concepts",
    "concept_relations",
    "theoretical_frameworks",
    "instruments",
    "conceptual_problems",
    "key_quotes",
    "evidence_quotes",
)]

FIELD_LABELS: dict[str, str] = {
    "domain_coverage_bio": "Biological coverage",
    "domain_coverage_psych": "Psychological coverage",
    "domain_coverage_social": "Social coverage",
    "coverage_lifestyle": "Lifestyle coverage",
    "coverage_spiritual_existential": "Spiritual or existential coverage",
    "integration_bio_psych": "Bio-psych integration",
    "integration_psych_social": "Psych-social integration",
    "integration_bio_social": "Bio-social integration",
    "integration_triadic": "Triadic integration",
    "review_track": "Review track",
    "source_type": "Source type",
    "icd11_pain_category": "ICD-11 pain category",
    "population": "Population",
    "care_setting": "Care setting",
    "primary_discipline": "Primary discipline",
    "quality_assessment_reported": "Quality assessment reported",
    "bps_label_used": "BPS label used",
    "bps_primary_function": "BPS primary function",
    "bps_definition_status": "BPS definition status",
    "overall_balance": "Overall balance",
    "bps_typology": "BPS typology",
    "concept_definitions_present": "Concept definitions",
    "fulltext_eligibility": "Full-text eligibility",
    "synthesis_priority": "Synthesis priority",
    "present_bps_usage_evidence": "BPS usage evidence present",
    "present_bps_definition": "BPS definition present",
    "present_integration_evidence": "Integration evidence present",
    "present_triadic_claim": "Triadic claim present",
    "present_named_integration_edge": "Named integration edge present",
    "present_biological_factors": "Biological factors present",
    "present_social_factors": "Social factors present",
    "present_other_domain_factors": "Other-domain factors present",
    "present_psychological_concepts": "Psychological concepts present",
    "present_defined_concepts": "Defined concepts present",
    "present_concept_relations": "Concept relations present",
    "present_hierarchical_relation": "Hierarchical relation present",
    "present_theoretical_frameworks": "Frameworks present",
    "present_instruments": "Instruments present",
    "present_conceptual_problems": "Conceptual problems present",
    "present_domain_evidence_bio": "Biological evidence present",
    "present_domain_evidence_psych": "Psychological evidence present",
    "present_domain_evidence_social": "Social evidence present",
    "bps_usage_instances": "BPS usage instances",
    "bps_definitions": "BPS definitions",
    "biological_factors": "Biological factors",
    "social_factors": "Social factors",
    "other_domain_factors": "Lifestyle and existential factors",
    "psychological_concepts": "Psychological concepts",
    "concept_relations": "Concept relations",
    "theoretical_frameworks": "Theoretical frameworks",
    "instruments": "Instruments",
    "conceptual_problems": "Conceptual problems",
    "integration_claims": "Integration claims",
    "domain_evidence": "Domain evidence",
    "key_quotes": "Key quotes",
}

# Every comparable vocabulary of the scheme, in reading order: what was extracted,
# what it was called, and the subsets that carry the weight. Built after the label
# table so a space is named the same way everywhere it appears.
EXTRACTION_SPACES: list[ExtractionSpace] = (
    _identity_spaces() + VOCABULARY_SPACES + FILTERED_SPACES
)

SPACE_BY_NAME: dict[str, ExtractionSpace] = {space.name: space for space in EXTRACTION_SPACES}

# Ordered category vocabularies used for consensus resolution and stacked plots.
COVERAGE_ORDER = ["elaborated", "mentioned", "minimal", "absent"]
PAIRWISE_ORDER = ["mechanistic", "directional", "descriptive", "mentioned", "none"]
TRIADIC_ORDER = ["mechanistic", "descriptive", "partial", "none"]
BALANCE_ORDER = ["balanced", "psych-dominant", "bio-dominant", "social-dominant", "dyadic", "unclear"]
TYPOLOGY_ORDER = ["true_integrative", "multifactorial", "pseudo_bps", "rhetorical_bps",
                  "narrow_despite_label", "unclear"]
DEFINITIONS_ORDER = ["yes", "partial", "no"]
ELIGIBILITY_ORDER = ["include", "uncertain", "exclude"]
PRIORITY_ORDER = ["core", "supporting", "background", "not_relevant"]
PRESENCE_ORDER = ["yes", "no"]
TRISTATE_ORDER = ["yes", "no", "unclear"]
TRACK_ORDER = ["musculoskeletal", "neuropathic", "mixed_or_other", "unclear"]
BPS_LABEL_ORDER = ["explicit_bps_term", "variant_term_only", "domain_language_only", "absent"]
BPS_DEFINITION_STATUS_ORDER = ["formally_defined", "described_informally", "cited_only", "undefined"]

# The integration ladder as a depth score, so the analysis can compare papers on
# one number without pretending the ladder is an interval scale.
PAIRWISE_DEPTH: dict[str, int] = {"none": 0, "mentioned": 1, "descriptive": 2, "directional": 3, "mechanistic": 4}
TRIADIC_DEPTH: dict[str, int] = {"none": 0, "partial": 1, "descriptive": 2, "mechanistic": 3}
COVERAGE_DEPTH: dict[str, int] = {"absent": 0, "minimal": 1, "mentioned": 2, "elaborated": 3}


# --------------------------------------------------------------------------
# Paths. Everything this test run produces lives under
# src/05_data/pilot/02_fulltext_level, mirroring the abstract-level layout.
# --------------------------------------------------------------------------
def fulltext_root():
    return project_path("data", "pilot", "02_fulltext_level")


def corpus_dir():
    return fulltext_root() / "01_corpus"


def corpus_text_dir():
    return corpus_dir() / "03_fulltext_txt"


def corpus_csv():
    return corpus_dir() / "02_fulltext_corpus.csv"


def corpus_candidates_csv():
    return corpus_dir() / "01_retrieval_candidates.csv"


def corpus_selection_log_csv():
    return corpus_dir() / "04_retrieval_log.csv"


def corpus_manifest_json():
    return corpus_dir() / "05_corpus_manifest.json"


def codings_dir():
    return fulltext_root() / "02_model_codings"


def long_codings_csv():
    return codings_dir() / "all_model_codings_long.csv"


def items_csv():
    return codings_dir() / "all_extracted_items_long.csv"


def reliability_dir():
    return fulltext_root() / "03_reliability"


def figures_dir():
    return fulltext_root() / "04_figures"


def graph_dir():
    return fulltext_root() / "05_knowledge_graph"


def summary_md():
    return fulltext_root() / "TESTRUN_SUMMARY_FULLTEXT.md"
