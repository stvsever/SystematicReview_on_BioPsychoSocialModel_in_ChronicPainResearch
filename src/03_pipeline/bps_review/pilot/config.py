from __future__ import annotations

"""Central configuration for the cross-provider abstract-level test run.

This test run applies the Stage 2 abstract coding scheme (scheme 2 of the
dossier, implemented in ``bps_review.extraction.llm_stage2``) to a sample of
PubMed records, once per model, with three deliberately cheap large language
models drawn from three different providers.

The goal is not accuracy. The goal is to exercise the code and the coding logic
end to end and to quantify cross-provider inter-rater reliability before the
workflow is re-run with state-of-the-art models on the full corpus. All three
models here are low-cost stand-ins; swapping them for stronger models is a
one-line change in ``TESTRUN_MODELS``.

Why these three. An earlier cross-provider pilot on a neighbouring review
compared five cheap models on the same task and ranked them by how much of their
quoted evidence could be found back in the source article. The three kept here
are the three that scored highest on that check (99.9, 97.5, and 97.0 percent
verified quotes); the two that were dropped verified only about 90 percent. The
same three models are used at the abstract level and at the full-text level, so
the two stages of this test run stay directly comparable.
"""

from dataclasses import dataclass

from bps_review.utils.paths import project_path


@dataclass(frozen=True)
class TestRunModel:
    """One test-run model: a display label, its OpenRouter id, and its provider."""

    order: int
    label: str
    slug: str
    openrouter_id: str
    provider: str


TESTRUN_MODELS: list[TestRunModel] = [
    TestRunModel(1, "DeepSeek-V4-Flash", "01_deepseek_v4_flash", "deepseek/deepseek-v4-flash", "DeepSeek"),
    TestRunModel(2, "Nex-N2-Mini", "02_nex_n2_mini", "nex-agi/nex-n2-mini", "Nex AGI"),
    TestRunModel(3, "Laguna-XS-2.1", "03_laguna_xs_2_1", "poolside/laguna-xs-2.1", "Poolside"),
]

MODEL_LABELS: list[str] = [model.label for model in TESTRUN_MODELS]
MODEL_BY_ID: dict[str, TestRunModel] = {model.openrouter_id: model for model in TESTRUN_MODELS}
MODEL_BY_LABEL: dict[str, TestRunModel] = {model.label: model for model in TESTRUN_MODELS}


# --------------------------------------------------------------------------
# Corpus. The test run reads a fresh sample from the operational PubMed query.
# --------------------------------------------------------------------------
TESTRUN_SAMPLE_SIZE = 100      # abstracts coded by every model
CANDIDATE_POOL = 220           # records retrieved before the abstract filter
PUBMED_QUERY_KEY = "pubmed_operational_primary"


# --------------------------------------------------------------------------
# Run settings. One record per request, high worker counts so the whole grid
# finishes in one pass.
#
# One abstract per request rather than a batch: it makes the call count equal to
# the coding count (100 records x 3 models = 300 calls), it removes any chance
# that one record in a batch influences the coding of another, and abstracts are
# short enough that the cost difference is negligible.
# --------------------------------------------------------------------------
BATCH_SIZE = 1
MAX_WORKERS = 16        # concurrent requests per model (ThreadPoolExecutor)
MAX_MODEL_WORKERS = 3   # models coded concurrently (one worker per model)
MAX_RETRIES = 4         # attempts per record before the deterministic fallback
RETRY_BACKOFF_SECONDS = 2.0
# Hard wall-clock cap per request attempt. Some endpoints trickle bytes, so the
# socket read timeout never fires and a request hangs indefinitely. This cap
# abandons such a request and retries it, so one slow provider cannot stall the
# whole run.
HARD_TIMEOUT_SECONDS = 120.0


# --------------------------------------------------------------------------
# Coded fields, grouped for the reliability analysis.
#
# Agreement is quantified on three kinds of variable, kept apart because they
# answer different questions:
#
# * nominal decisions, where the value itself carries the meaning;
# * the binary domain-presence flags, which are the raw material of RQ2;
# * ordinal decisions (priority), treated as nominal for the headline
#   coefficients but flagged as ordinal so the analysis can say so.
#
# The open extraction lists are not scored with kappa at all. Two coders can
# both be right and still return different concept strings, so those are
# compared with set overlap instead.
# --------------------------------------------------------------------------
NOMINAL_FIELDS: list[str] = [
    "review_type",
    "objective_category",
    "icd11_pain_category",
    "musculoskeletal_flag",
    "bps_function",
    "quality_assessment_reported",
    "provisional_typology",
    "stage3_candidate",
]

DOMAIN_FIELDS: list[str] = [
    "bio_mentioned",
    "psych_mentioned",
    "social_mentioned",
]

ORDINAL_FIELDS: list[str] = [
    "stage3_priority",
]

# Every field on which agreement is quantified, in a readable order.
RELIABILITY_FIELDS: list[str] = NOMINAL_FIELDS + DOMAIN_FIELDS + ORDINAL_FIELDS

# Open lists, compared with Jaccard set overlap rather than with kappa.
LIST_FIELDS: list[str] = [
    "psychological_concepts_detected",
    "theoretical_frameworks_detected",
    "conceptual_problem_flags",
]

# Fields the pipeline computes rather than asks for. They are listed here so the
# notebook can state plainly which agreement numbers are about a judgement and
# which are about a rule applied to a judgement.
DERIVED_FIELDS: list[str] = [
    "stage3_candidate",
    "stage3_priority",
    "conceptual_problem_flags",
]

FIELD_LABELS: dict[str, str] = {
    "review_type": "Review type",
    "objective_category": "Objective category",
    "icd11_pain_category": "ICD-11 pain category",
    "musculoskeletal_flag": "Musculoskeletal flag",
    "bps_function": "BPS function",
    "quality_assessment_reported": "Quality assessment",
    "provisional_typology": "Provisional typology",
    "stage3_candidate": "Stage 3 candidate",
    "stage3_priority": "Stage 3 priority",
    "bio_mentioned": "Biological mention",
    "psych_mentioned": "Psychological mention",
    "social_mentioned": "Social mention",
    "psychological_concepts_detected": "Psychological concepts",
    "theoretical_frameworks_detected": "Theoretical frameworks",
    "conceptual_problem_flags": "Conceptual problems",
}

# Ordered category vocabularies for consensus resolution and stacked plots.
TYPOLOGY_ORDER = [
    "potential integrative signal",
    "multifactorial signal",
    "pseudo-bps or partial signal",
    "rhetorical label signal",
]
PRIORITY_ORDER = ["high", "medium", "low"]
MSK_ORDER = ["yes", "unclear", "no"]
BINARY_ORDER = ["yes", "no"]
BPS_FUNCTION_ORDER = [
    "explanatory framework",
    "intervention rationale",
    "organizing principle",
    "justification",
    "background framing",
    "conclusion",
    "policy/practice implication",
    "rhetorical label",
    "unclear",
]


# --------------------------------------------------------------------------
# Paths. Everything this test run produces lives under
# src/05_test_runs/tests/01_pilot_abstract, beside the full-text run in
# src/05_test_runs/tests/02_pilot_fulltext, so both stay clearly separated from
# the main pipeline outputs and from each other.
# --------------------------------------------------------------------------
def testrun_root():
    return project_path("test_runs", "tests", "01_pilot_abstract")


def corpus_dir():
    return testrun_root() / "01_abstracts"


def input_csv():
    return corpus_dir() / "articles.csv"


def corpus_manifest_json():
    return corpus_dir() / "corpus_manifest.json"


def codings_dir():
    return testrun_root() / "02_model_codings"


def long_codings_csv():
    return codings_dir() / "all_model_codings_long.csv"


def reliability_dir():
    return testrun_root() / "03_reliability"


def figures_dir():
    return testrun_root() / "04_figures"


def summary_md():
    return testrun_root() / "TEST_RUN_SUMMARY.md"


def candidate_set_csv():
    """The filtered set handed to the full-text stage."""
    return testrun_root() / "05_fulltext_candidate_set.csv"
