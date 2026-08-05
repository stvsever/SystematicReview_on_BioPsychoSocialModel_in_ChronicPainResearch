from __future__ import annotations

"""The Stage 3 full-text deep coding schema (scheme 3), as validated data classes.

The schema has three layers.

1. **Controlled decisions.** Closed vocabularies, one value per field: the
   coverage ladder per domain, the pairwise and triadic integration ladders, the
   balance judgement, the biopsychosocial typology, and whether the review
   defines the psychological constructs it uses. These are the fields on which
   cross-provider agreement is quantified with kappa-style coefficients.
2. **Structured extractions.** Open lists of typed items, each carrying a
   verbatim quote from the source text and the section it came from. This is the
   layer that makes the integration ladder auditable: a paper coded as
   mechanistic on bio-psych has to point at the sentence that says so.
3. **Free-text synthesis hooks.** Short analytic notes that carry nuance no
   controlled vocabulary can hold, written for the later synthesis rather than
   for counting.

Nothing in the derived layer is scored by the model. Eligibility, integration
depth, and synthesis priority are computed deterministically from the coded
content in ``derive.py``, so the filter stays auditable and identical across
providers.
"""

from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Controlled vocabularies (the closed value lists of the scheme).
# --------------------------------------------------------------------------
REVIEW_TRACK_OPTIONS = ["musculoskeletal", "neuropathic", "mixed_or_other", "unclear"]

SOURCE_TYPE_OPTIONS = [
    "systematic review",
    "meta-analysis",
    "network meta-analysis",
    "umbrella review",
    "scoping or mapping review",
    "rapid review",
    "realist review",
    "integrative review",
    "narrative or expert review",
    "other evidence synthesis",
    "primary study",
    "unclear",
]

COVERAGE_OPTIONS = ["elaborated", "mentioned", "minimal", "absent"]
PAIRWISE_INTEGRATION_OPTIONS = ["mechanistic", "directional", "descriptive", "mentioned", "none"]
TRIADIC_INTEGRATION_OPTIONS = ["mechanistic", "descriptive", "partial", "none"]
BALANCE_OPTIONS = ["balanced", "psych-dominant", "bio-dominant", "social-dominant", "dyadic", "unclear"]
TYPOLOGY_OPTIONS = [
    "true_integrative",
    "multifactorial",
    "pseudo_bps",
    "rhetorical_bps",
    "narrow_despite_label",
    "unclear",
]
DEFINITIONS_PRESENT_OPTIONS = ["yes", "partial", "no"]

DOMAIN_OPTIONS = ["biological", "psychological", "social"]
DOMAIN_PAIR_OPTIONS = ["bio_psych", "psych_social", "bio_social", "triadic"]

SECTION_OPTIONS = [
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "table or figure",
    "other",
    "unclear",
]

EVIDENCE_BASIS_OPTIONS = [
    "asserted",
    "theorized",
    "empirically_supported",
    "empirically_contested",
    "cited_from_other_work",
    "clinical_observation",
    "other",
    "unclear",
]

DEFINITIONAL_STATUS_OPTIONS = [
    "formally_defined",       # the review states what the construct means
    "operationalized_only",   # the meaning is fixed only through a measure
    "named_only",             # the construct is used without any meaning given
    "unclear",
]

FRAMEWORK_ROLE_OPTIONS = [
    "organizing framework",
    "tested or modelled",
    "extended or revised",
    "critiqued or rejected",
    "compared with another model",
    "mentioned in passing",
    "other",
    "unclear",
]

CONCEPTUAL_PROBLEM_OPTIONS = [
    "vague_definition",
    "tokenistic_bps",
    "missing_social",
    "missing_biology",
    "mechanistic_absence",
    "construct_overlap",
    "parallel_listing_without_integration",
    "measurement_mismatch",
    "other",
]

CLAIM_TYPE_OPTIONS = [
    "definitional",
    "integrative",
    "critical or problematizing",
    "measurement",
    "theoretical",
    "clinical or applied",
    "other",
]

# Derived vocabularies (never produced by the model, always computed).
ELIGIBILITY_OPTIONS = ["include", "uncertain", "exclude"]
SYNTHESIS_PRIORITY_OPTIONS = ["core", "supporting", "background", "not_relevant"]


# --------------------------------------------------------------------------
# Runtime type aliases, built once from the vocabularies above so the option
# lists stay the single source of truth for both validation and the prompt.
# --------------------------------------------------------------------------
ReviewTrackT = Literal[tuple(REVIEW_TRACK_OPTIONS)]  # type: ignore[valid-type]
SourceTypeT = Literal[tuple(SOURCE_TYPE_OPTIONS)]  # type: ignore[valid-type]
CoverageT = Literal[tuple(COVERAGE_OPTIONS)]  # type: ignore[valid-type]
PairwiseT = Literal[tuple(PAIRWISE_INTEGRATION_OPTIONS)]  # type: ignore[valid-type]
TriadicT = Literal[tuple(TRIADIC_INTEGRATION_OPTIONS)]  # type: ignore[valid-type]
BalanceT = Literal[tuple(BALANCE_OPTIONS)]  # type: ignore[valid-type]
TypologyT = Literal[tuple(TYPOLOGY_OPTIONS)]  # type: ignore[valid-type]
DefinitionsPresentT = Literal[tuple(DEFINITIONS_PRESENT_OPTIONS)]  # type: ignore[valid-type]
DomainT = Literal[tuple(DOMAIN_OPTIONS)]  # type: ignore[valid-type]
DomainPairT = Literal[tuple(DOMAIN_PAIR_OPTIONS)]  # type: ignore[valid-type]
SectionT = Literal[tuple(SECTION_OPTIONS)]  # type: ignore[valid-type]
EvidenceBasisT = Literal[tuple(EVIDENCE_BASIS_OPTIONS)]  # type: ignore[valid-type]
DefinitionalStatusT = Literal[tuple(DEFINITIONAL_STATUS_OPTIONS)]  # type: ignore[valid-type]
FrameworkRoleT = Literal[tuple(FRAMEWORK_ROLE_OPTIONS)]  # type: ignore[valid-type]
ConceptualProblemT = Literal[tuple(CONCEPTUAL_PROBLEM_OPTIONS)]  # type: ignore[valid-type]
ClaimTypeT = Literal[tuple(CLAIM_TYPE_OPTIONS)]  # type: ignore[valid-type]


# --------------------------------------------------------------------------
# Structured extraction items.
# --------------------------------------------------------------------------
class IntegrationClaimItem(BaseModel):
    """One passage in which the review relates two or three domains to each other."""

    domains_linked: DomainPairT = "bio_psych"
    integration_level: PairwiseT = "mentioned"
    claim_verbatim: str = ""
    mechanism_note: str = ""      # the pathway in your words, if one is given
    section_located: SectionT = "unclear"
    evidence_basis: EvidenceBasisT = "asserted"


class DomainEvidenceItem(BaseModel):
    """The passage that carries the coverage judgement for one domain."""

    domain: DomainT = "biological"
    coverage_level: CoverageT = "mentioned"
    constructs_named: list[str] = Field(default_factory=list)
    evidence_verbatim: str = ""
    section_located: SectionT = "unclear"


class ConceptItem(BaseModel):
    """One psychological construct the review uses, with its definitional status."""

    concept_label: str = ""
    definitional_status: DefinitionalStatusT = "named_only"
    definition_verbatim: str = ""
    section_located: SectionT = "unclear"


class FrameworkItem(BaseModel):
    """One theoretical model or framework the review invokes."""

    framework_label: str = ""
    role: FrameworkRoleT = "unclear"
    framework_verbatim: str = ""


class ConceptualProblemItem(BaseModel):
    """One conceptual problem the review displays or names."""

    problem_type: ConceptualProblemT = "other"
    problem_verbatim: str = ""
    note: str = ""


class KeyQuoteItem(BaseModel):
    """One high-value passage, quoted for the later synthesis."""

    claim_verbatim: str = ""
    claim_type: ClaimTypeT = "integrative"
    section_located: SectionT = "unclear"
    why_it_matters: str = ""


# --------------------------------------------------------------------------
# The full record produced for one paper by one model.
# --------------------------------------------------------------------------
class FullTextCodingRecord(BaseModel):
    record_id: str

    # --- routing and provenance ---
    review_track: ReviewTrackT = "unclear"
    source_type: SourceTypeT = "unclear"
    pain_condition_detail: str = ""

    # --- domain coverage (the four-level ladder, one per domain) ---
    domain_coverage_bio: CoverageT = "absent"
    domain_coverage_psych: CoverageT = "absent"
    domain_coverage_social: CoverageT = "absent"
    domain_evidence: list[DomainEvidenceItem] = Field(default_factory=list)

    # --- integration (the core RQ2 contribution) ---
    integration_bio_psych: PairwiseT = "none"
    integration_psych_social: PairwiseT = "none"
    integration_bio_social: PairwiseT = "none"
    integration_triadic: TriadicT = "none"
    integration_claims: list[IntegrationClaimItem] = Field(default_factory=list)
    integration_mechanism_summary: str = ""

    # --- typology and balance ---
    overall_balance: BalanceT = "unclear"
    bps_typology: TypologyT = "unclear"

    # --- psychological concepts and evidence (RQ3) ---
    concept_definitions_present: DefinitionsPresentT = "no"
    psychological_concepts: list[ConceptItem] = Field(default_factory=list)
    theoretical_frameworks: list[FrameworkItem] = Field(default_factory=list)
    conceptual_problems: list[ConceptualProblemItem] = Field(default_factory=list)

    # --- synthesis hooks (free text, deliberately unconstrained) ---
    key_quotes: list[KeyQuoteItem] = Field(default_factory=list)
    synthesis_note: str = ""
    coding_rationale: str = ""


ITEM_MODELS: dict[str, type[BaseModel]] = {
    "integration_claims": IntegrationClaimItem,
    "domain_evidence": DomainEvidenceItem,
    "psychological_concepts": ConceptItem,
    "theoretical_frameworks": FrameworkItem,
    "conceptual_problems": ConceptualProblemItem,
    "key_quotes": KeyQuoteItem,
}

# The verbatim quote key inside each structured item.
ITEM_QUOTE_KEY: dict[str, str] = {
    "integration_claims": "claim_verbatim",
    "domain_evidence": "evidence_verbatim",
    "psychological_concepts": "definition_verbatim",
    "theoretical_frameworks": "framework_verbatim",
    "conceptual_problems": "problem_verbatim",
    "key_quotes": "claim_verbatim",
}

# The label key inside each structured item, used for the overlap metrics.
ITEM_LABEL_KEY: dict[str, str] = {
    "integration_claims": "domains_linked",
    "domain_evidence": "domain",
    "psychological_concepts": "concept_label",
    "theoretical_frameworks": "framework_label",
    "conceptual_problems": "problem_type",
    "key_quotes": "claim_type",
}
