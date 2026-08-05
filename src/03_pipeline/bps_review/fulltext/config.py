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

The same three cheap models from three different providers are used as in the
abstract-level run, so the two stages are directly comparable and swapping in
state-of-the-art models stays a one-line change.
"""

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
# binding constraint is cost, not context. This budget keeps one paper at roughly
# 15k input tokens, which holds the whole run comfortably inside a few dollars
# while still sending most papers whole.
CODING_TEXT_CHAR_BUDGET = 60_000
MAX_OUTPUT_TOKENS = 8000

# Per-model runtime settings. Two of the three answer this task directly. The
# third reasons regardless of the reasoning controls, and on a reasoning model
# the thinking tokens come out of the same completion budget as the answer, so a
# long structured extraction needs a budget that covers both. This is a property
# of the endpoint, not of the coding scheme: every model receives the identical
# prompt and returns the identical schema.
MODEL_RUNTIME: dict[str, dict] = {
    "deepseek/deepseek-v4-flash": {"reasoning": {"enabled": False}, "max_output_tokens": 8000},
    "nex-agi/nex-n2-mini": {"reasoning": {"effort": "low"}, "max_output_tokens": 32000},
    "poolside/laguna-xs-2.1": {"reasoning": {"enabled": False}, "max_output_tokens": 8000},
}
DEFAULT_RUNTIME: dict = {"reasoning": {"enabled": False}, "max_output_tokens": MAX_OUTPUT_TOKENS}


def model_runtime(openrouter_id: str) -> dict:
    """Reasoning settings and completion budget for one model."""
    return MODEL_RUNTIME.get(openrouter_id, DEFAULT_RUNTIME)


# Caps on how many items the models may return per extraction list, and how long
# a quoted passage may be. They are ceilings, never targets: an empty list is a
# legitimate coding, and a paper that genuinely says more is allowed to fill them.
ITEM_CAPS: dict[str, int] = {
    "integration_claims": 10,
    "domain_evidence": 6,
    "psychological_concepts": 14,
    "theoretical_frameworks": 6,
    "conceptual_problems": 8,
    "key_quotes": 6,
}
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

INTEGRATION_FIELDS: list[str] = [
    "integration_bio_psych",
    "integration_psych_social",
    "integration_bio_social",
    "integration_triadic",
]

NOMINAL_FIELDS: list[str] = [
    "review_track",
    "source_type",
    "overall_balance",
    "bps_typology",
    "concept_definitions_present",
    "fulltext_eligibility",
    "synthesis_priority",
]

PRESENCE_FIELDS: list[str] = [
    "present_integration_evidence",
    "present_triadic_claim",
    "present_psychological_concepts",
    "present_defined_concepts",
    "present_theoretical_frameworks",
    "present_conceptual_problems",
    "present_domain_evidence_bio",
    "present_domain_evidence_psych",
    "present_domain_evidence_social",
]

CATEGORICAL_FIELDS: list[str] = COVERAGE_FIELDS + INTEGRATION_FIELDS + NOMINAL_FIELDS

# Every field on which agreement is quantified, in a readable order.
RELIABILITY_FIELDS: list[str] = CATEGORICAL_FIELDS + PRESENCE_FIELDS

LIST_FIELDS: list[str] = [
    "psychological_concepts",
    "theoretical_frameworks",
    "conceptual_problems",
]

# The extraction lists whose items carry a verbatim quote from the source text.
QUOTED_ITEM_FIELDS: list[str] = [
    "integration_claims",
    "domain_evidence",
    "psychological_concepts",
    "theoretical_frameworks",
    "conceptual_problems",
    "key_quotes",
]

# The label key inside each structured item, used for the overlap metrics.
LIST_LABEL_KEY: dict[str, str] = {
    "psychological_concepts": "concept_label",
    "theoretical_frameworks": "framework_label",
    "conceptual_problems": "problem_type",
}

COUNT_FIELDS: list[str] = [f"n_{name}" for name in (
    "integration_claims",
    "domain_evidence",
    "psychological_concepts",
    "defined_concepts",
    "theoretical_frameworks",
    "conceptual_problems",
    "key_quotes",
    "evidence_quotes",
)]

FIELD_LABELS: dict[str, str] = {
    "domain_coverage_bio": "Biological coverage",
    "domain_coverage_psych": "Psychological coverage",
    "domain_coverage_social": "Social coverage",
    "integration_bio_psych": "Bio-psych integration",
    "integration_psych_social": "Psych-social integration",
    "integration_bio_social": "Bio-social integration",
    "integration_triadic": "Triadic integration",
    "review_track": "Review track",
    "source_type": "Source type",
    "overall_balance": "Overall balance",
    "bps_typology": "BPS typology",
    "concept_definitions_present": "Concept definitions",
    "fulltext_eligibility": "Full-text eligibility",
    "synthesis_priority": "Synthesis priority",
    "present_integration_evidence": "Integration evidence present",
    "present_triadic_claim": "Triadic claim present",
    "present_psychological_concepts": "Psychological concepts present",
    "present_defined_concepts": "Defined concepts present",
    "present_theoretical_frameworks": "Frameworks present",
    "present_conceptual_problems": "Conceptual problems present",
    "present_domain_evidence_bio": "Biological evidence present",
    "present_domain_evidence_psych": "Psychological evidence present",
    "present_domain_evidence_social": "Social evidence present",
    "psychological_concepts": "Psychological concepts",
    "theoretical_frameworks": "Theoretical frameworks",
    "conceptual_problems": "Conceptual problems",
    "integration_claims": "Integration claims",
    "domain_evidence": "Domain evidence",
    "key_quotes": "Key quotes",
}

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
TRACK_ORDER = ["musculoskeletal", "neuropathic", "mixed_or_other", "unclear"]

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


def summary_md():
    return fulltext_root() / "TESTRUN_SUMMARY_FULLTEXT.md"
