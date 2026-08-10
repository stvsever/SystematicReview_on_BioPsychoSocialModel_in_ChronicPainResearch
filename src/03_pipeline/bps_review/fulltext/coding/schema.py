from __future__ import annotations

"""The Stage 3 full-text deep coding schema (scheme 3), as validated data classes.

The schema has four layers.

1. **Controlled decisions.** Closed vocabularies, one value per field: the
   coverage ladder per domain, the pairwise and triadic integration ladders, the
   balance judgement, the biopsychosocial typology, how the biopsychosocial label
   is used, and how it is defined. These are the fields on which cross-provider
   agreement is quantified with kappa-style coefficients.
2. **Structured extractions.** Open lists of typed items, each carrying the
   thing's own name, its place in the project ontology, the role it plays in the
   review, a verbatim quote, and the section the quote came from. This is the
   layer that answers "which ones", not merely "whether": which biological,
   psychological, and social factors a review actually carries, which constructs
   it defines and how, which relations it draws between them, which frameworks
   and instruments it uses, and where the biopsychosocial label does its work.
3. **Free-text synthesis hooks.** Open lists and short analytic notes that carry
   nuance no controlled vocabulary can hold, written for the later synthesis
   rather than for counting. An unusual, precisely worded observation is worth
   more to this review than a controlled label that flattens it.
4. **Derived fields.** Never scored by the model. Eligibility, integration depth,
   conceptual yield, subdomain breadth, ontology edges, and synthesis priority
   are computed deterministically from the coded content in ``derive.py``, so the
   filter stays auditable and identical across providers.

Two design commitments run through the whole schema.

*Presence is never the answer.* Where the earlier generation of this scheme asked
whether a domain was present, this one asks which factors carry it, at which
subdomain of the ontology, in which role, and on the strength of which sentence.
The ontology the synthesis will build is assembled out of those items, so the
items have to be nameable, countable, and traceable to a passage.

*Free text survives.* Labels are normalized onto the project vocabularies only
when they clearly match (see ``vocabulary.py``); anything else is kept as the
review wrote it and reported as off-spine. A label the ontology cannot hold is a
finding about the ontology, not noise.
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
    "clinical guideline or consensus statement",
    "other evidence synthesis",
    "primary study",
    "unclear",
]

ICD11_CATEGORY_OPTIONS = [
    "chronic secondary musculoskeletal pain",
    "chronic neuropathic pain",
    "chronic cancer-related pain",
    "chronic postsurgical or posttraumatic pain",
    "chronic secondary headache or orofacial pain",
    "chronic secondary visceral pain",
    "chronic primary pain",
    "mixed or unspecified chronic pain",
    "unclear",
]

POPULATION_OPTIONS = ["adult", "older adult", "mixed ages", "pediatric", "unclear", "not applicable"]

CARE_SETTING_OPTIONS = [
    "primary care",
    "secondary or tertiary specialist care",
    "rehabilitation or multidisciplinary programme",
    "occupational or workplace",
    "community or population",
    "mixed",
    "not reported",
]

DISCIPLINE_OPTIONS = [
    "physiotherapy or rehabilitation",
    "clinical or health psychology",
    "rheumatology or orthopaedics",
    "pain medicine or anaesthesiology",
    "neurology or neuroscience",
    "nursing",
    "general or family medicine",
    "public health or epidemiology",
    "multidisciplinary",
    "other",
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
TRISTATE_OPTIONS = ["yes", "no", "unclear"]
YES_NO_OPTIONS = ["yes", "no"]

DOMAIN_OPTIONS = ["biological", "psychological", "social"]
DOMAIN_PAIR_OPTIONS = ["bio_psych", "psych_social", "bio_social", "triadic"]
OTHER_DOMAIN_OPTIONS = [
    "lifestyle",                 # sleep hygiene, activity, diet, smoking, alcohol, weight
    "spiritual or existential",  # meaning, faith, existential suffering, hope
    "environmental",             # physical environment, housing, climate, built environment
    "other",
]

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

# --------------------------------------------------------------------------
# How the biopsychosocial label itself is used and defined (the RQ1 layer).
# The function vocabulary is shared with the abstract-level scheme on purpose,
# so a record's abstract-level reading and its full-text reading are directly
# comparable, with two values that only a full text can support.
# --------------------------------------------------------------------------
BPS_FUNCTION_OPTIONS = [
    "explanatory framework",
    "intervention rationale",
    "organizing principle",
    "justification",
    "background framing",
    "conclusion",
    "policy or practice implication",
    "rhetorical label",
    "critique or problematization",   # full-text only: the review argues about the model
    "operational definition",         # full-text only: the model is turned into coded variables
    "unclear",
]

BPS_LABEL_OPTIONS = [
    "explicit_bps_term",      # the words biopsychosocial or bio-psycho-social appear
    "variant_term_only",      # only a variant such as multidimensional or psychosocial
    "domain_language_only",   # the domains are discussed with no model label at all
    "absent",
]

BPS_DEFINITION_STATUS_OPTIONS = [
    "formally_defined",     # the review states what the model means
    "described_informally", # the meaning is carried by description, not by a definition
    "cited_only",           # a citation stands in for a definition
    "undefined",            # the label is used with no meaning given anywhere
]

BPS_DEFINITION_TYPE_OPTIONS = [
    "explicit_formal",
    "operational",           # defined through the variables or domains actually coded
    "implicit_description",
    "borrowed",              # taken over from a cited source
    "critique_of_definition",
    "other",
]

# --------------------------------------------------------------------------
# What a named factor is doing in the review.
# --------------------------------------------------------------------------
FACTOR_ROLE_OPTIONS = [
    "determinant or risk factor",
    "protective factor",
    "mediator",
    "moderator",
    "outcome",
    "correlate",
    "treatment target",
    "intervention component",
    "contextual condition",
    "descriptive theme",
    "other",
    "unclear",
]

BIO_MECHANISM_LEVEL_OPTIONS = [
    "peripheral or tissue",
    "spinal or central nervous system",
    "systemic or whole body",
    "genetic or molecular",
    "structural or anatomical",
    "treatment related",
    "other",
    "unclear",
]

SOCIAL_LEVEL_OPTIONS = [
    "interpersonal",
    "family or household",
    "workplace",
    "community",
    "healthcare system",
    "societal or policy",
    "cultural",
    "economic",
    "other",
    "unclear",
]

DEFINITIONAL_STATUS_OPTIONS = [
    "formally_defined",       # the review states what the construct means
    "operationalized_only",   # the meaning is fixed only through a measure
    "described_informally",   # the meaning is carried by description alone
    "named_only",             # the construct is used without any meaning given
    "unclear",
]

DEFINITION_SOURCE_OPTIONS = [
    "own definition",
    "cited from other work",
    "taken from an instrument",
    "unattributed",
    "unclear",
]

CONCEPT_RELATION_OPTIONS = [
    "is_a_subtype_of",
    "part_of_or_component_of",
    "synonym_or_used_interchangeably",
    "overlapping_or_related",
    "antecedent_or_cause_of",
    "consequence_or_outcome_of",
    "mediates",
    "moderates",
    "measured_by",
    "contrasted_as_distinct_from",
    "conflated_without_comment",
    "other",
    "unclear",
]

LINK_DIRECTION_OPTIONS = ["unidirectional", "bidirectional or reciprocal", "unspecified"]

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

INSTRUMENT_ROLE_OPTIONS = [
    "primary outcome",
    "secondary outcome",
    "predictor or covariate",
    "mediator or moderator",
    "screening or classification",
    "developed or validated here",
    "discussed conceptually",
    "critiqued",
    "referenced only",
    "other",
    "unclear",
]

MEASURED_DOMAIN_OPTIONS = [
    "biological",
    "psychological",
    "social",
    "pain or symptom",
    "function or disability",
    "quality of life",
    "multiple domains",
    "methodological quality",
    "other",
    "unclear",
]

CONCEPTUAL_PROBLEM_OPTIONS = [
    "vague_definition",
    "tokenistic_bps",
    "missing_social",
    "missing_biology",
    "missing_psychology",
    "mechanistic_absence",
    "construct_overlap",
    "parallel_listing_without_integration",
    "measurement_mismatch",
    "definitional_drift",             # the same term shifts meaning inside the review
    "domain_reductionism",            # one domain is quietly reduced to another
    "unfalsifiable_or_untestable",    # the model is stated so broadly it excludes nothing
    "other",
]

PROBLEM_SCOPE_OPTIONS = [
    "the biopsychosocial model itself",
    "a psychological construct",
    "a biological construct",
    "a social construct",
    "integration between domains",
    "measurement",
    "terminology",
    "scope or coverage",
    "other",
]

CLAIM_TYPE_OPTIONS = [
    "definitional",
    "integrative",
    "operationalizing",
    "critical or problematizing",
    "measurement",
    "theoretical",
    "clinical or applied",
    "other",
]

# Derived vocabularies (never produced by the model, always computed).
ELIGIBILITY_OPTIONS = ["include", "uncertain", "exclude"]
YIELD_OPTIONS = ["high", "moderate", "low", "minimal"]
SYNTHESIS_PRIORITY_OPTIONS = ["core", "supporting", "background", "not_relevant"]


# --------------------------------------------------------------------------
# Runtime type aliases, built once from the vocabularies above so the option
# lists stay the single source of truth for both validation and the prompt.
# --------------------------------------------------------------------------
ReviewTrackT = Literal[tuple(REVIEW_TRACK_OPTIONS)]  # type: ignore[valid-type]
SourceTypeT = Literal[tuple(SOURCE_TYPE_OPTIONS)]  # type: ignore[valid-type]
Icd11CategoryT = Literal[tuple(ICD11_CATEGORY_OPTIONS)]  # type: ignore[valid-type]
PopulationT = Literal[tuple(POPULATION_OPTIONS)]  # type: ignore[valid-type]
CareSettingT = Literal[tuple(CARE_SETTING_OPTIONS)]  # type: ignore[valid-type]
DisciplineT = Literal[tuple(DISCIPLINE_OPTIONS)]  # type: ignore[valid-type]
CoverageT = Literal[tuple(COVERAGE_OPTIONS)]  # type: ignore[valid-type]
PairwiseT = Literal[tuple(PAIRWISE_INTEGRATION_OPTIONS)]  # type: ignore[valid-type]
TriadicT = Literal[tuple(TRIADIC_INTEGRATION_OPTIONS)]  # type: ignore[valid-type]
BalanceT = Literal[tuple(BALANCE_OPTIONS)]  # type: ignore[valid-type]
TypologyT = Literal[tuple(TYPOLOGY_OPTIONS)]  # type: ignore[valid-type]
DefinitionsPresentT = Literal[tuple(DEFINITIONS_PRESENT_OPTIONS)]  # type: ignore[valid-type]
TristateT = Literal[tuple(TRISTATE_OPTIONS)]  # type: ignore[valid-type]
YesNoT = Literal[tuple(YES_NO_OPTIONS)]  # type: ignore[valid-type]
DomainT = Literal[tuple(DOMAIN_OPTIONS)]  # type: ignore[valid-type]
DomainPairT = Literal[tuple(DOMAIN_PAIR_OPTIONS)]  # type: ignore[valid-type]
OtherDomainT = Literal[tuple(OTHER_DOMAIN_OPTIONS)]  # type: ignore[valid-type]
SectionT = Literal[tuple(SECTION_OPTIONS)]  # type: ignore[valid-type]
EvidenceBasisT = Literal[tuple(EVIDENCE_BASIS_OPTIONS)]  # type: ignore[valid-type]
BpsFunctionT = Literal[tuple(BPS_FUNCTION_OPTIONS)]  # type: ignore[valid-type]
BpsLabelT = Literal[tuple(BPS_LABEL_OPTIONS)]  # type: ignore[valid-type]
BpsDefinitionStatusT = Literal[tuple(BPS_DEFINITION_STATUS_OPTIONS)]  # type: ignore[valid-type]
BpsDefinitionTypeT = Literal[tuple(BPS_DEFINITION_TYPE_OPTIONS)]  # type: ignore[valid-type]
FactorRoleT = Literal[tuple(FACTOR_ROLE_OPTIONS)]  # type: ignore[valid-type]
BioMechanismLevelT = Literal[tuple(BIO_MECHANISM_LEVEL_OPTIONS)]  # type: ignore[valid-type]
SocialLevelT = Literal[tuple(SOCIAL_LEVEL_OPTIONS)]  # type: ignore[valid-type]
DefinitionalStatusT = Literal[tuple(DEFINITIONAL_STATUS_OPTIONS)]  # type: ignore[valid-type]
DefinitionSourceT = Literal[tuple(DEFINITION_SOURCE_OPTIONS)]  # type: ignore[valid-type]
ConceptRelationT = Literal[tuple(CONCEPT_RELATION_OPTIONS)]  # type: ignore[valid-type]
LinkDirectionT = Literal[tuple(LINK_DIRECTION_OPTIONS)]  # type: ignore[valid-type]
FrameworkRoleT = Literal[tuple(FRAMEWORK_ROLE_OPTIONS)]  # type: ignore[valid-type]
InstrumentRoleT = Literal[tuple(INSTRUMENT_ROLE_OPTIONS)]  # type: ignore[valid-type]
MeasuredDomainT = Literal[tuple(MEASURED_DOMAIN_OPTIONS)]  # type: ignore[valid-type]
ConceptualProblemT = Literal[tuple(CONCEPTUAL_PROBLEM_OPTIONS)]  # type: ignore[valid-type]
ProblemScopeT = Literal[tuple(PROBLEM_SCOPE_OPTIONS)]  # type: ignore[valid-type]
ClaimTypeT = Literal[tuple(CLAIM_TYPE_OPTIONS)]  # type: ignore[valid-type]


# --------------------------------------------------------------------------
# Structured extraction items.
# --------------------------------------------------------------------------
class BpsUsageItem(BaseModel):
    """One passage where the review invokes the biopsychosocial model.

    The central RQ1 evidence. A review is allowed to use the label for several
    different purposes in one article, and that is exactly the pattern the review
    is looking for, so every distinct use gets its own item rather than being
    collapsed into a single verdict.
    """

    usage_verbatim: str = ""
    bps_function: BpsFunctionT = "unclear"
    is_definitional: YesNoT = "no"
    attributed_source: str = ""     # who the model is credited to here, if anyone
    section_located: SectionT = "unclear"
    note: str = ""


class BpsDefinitionItem(BaseModel):
    """One place where the review says what the biopsychosocial model is."""

    definition_verbatim: str = ""
    definition_type: BpsDefinitionTypeT = "implicit_description"
    attributed_source: str = ""
    elements_named: list[str] = Field(default_factory=list)   # the components the definition lists
    section_located: SectionT = "unclear"


class DomainEvidenceItem(BaseModel):
    """The passage that carries the coverage judgement for one domain."""

    domain: DomainT = "biological"
    coverage_level: CoverageT = "mentioned"
    constructs_named: list[str] = Field(default_factory=list)
    subdomains_named: list[str] = Field(default_factory=list)   # ontology subdomains touched
    evidence_verbatim: str = ""
    section_located: SectionT = "unclear"


class BiologicalFactorItem(BaseModel):
    """One named biological factor, with its place in the ontology and its role."""

    factor_label: str = ""            # the review's own term, kept as written
    subdomain_label: str = ""         # normalized onto the biological ontology when it matches
    mechanism_level: BioMechanismLevelT = "unclear"
    factor_role: FactorRoleT = "unclear"
    factor_verbatim: str = ""
    section_located: SectionT = "unclear"
    evidence_basis: EvidenceBasisT = "asserted"


class SocialFactorItem(BaseModel):
    """One named social factor, with its place in the ontology and its level."""

    factor_label: str = ""
    subdomain_label: str = ""
    social_level: SocialLevelT = "unclear"
    factor_role: FactorRoleT = "unclear"
    factor_verbatim: str = ""
    section_located: SectionT = "unclear"
    evidence_basis: EvidenceBasisT = "asserted"


class ConceptItem(BaseModel):
    """One psychological construct, with what the review says it means.

    The psychological counterpart of the two factor lists above, carrying the
    extra fields RQ3 needs: whether the construct is defined, how, on whose
    authority, and through which measure.
    """

    concept_label: str = ""
    concept_family: str = ""          # normalized onto the concept taxonomy when it matches
    definitional_status: DefinitionalStatusT = "named_only"
    definition_verbatim: str = ""
    definition_source: DefinitionSourceT = "unclear"
    measure_named: str = ""           # the instrument the review operationalizes it with
    factor_role: FactorRoleT = "unclear"
    section_located: SectionT = "unclear"


class OtherDomainFactorItem(BaseModel):
    """One factor outside the biopsychosocial triad.

    The registration lists spiritual or existential and lifestyle factors
    alongside the three core domains. They are coded here rather than forced into
    the triad, so a review that reaches beyond the triad stays visible as such.
    """

    factor_label: str = ""
    domain: OtherDomainT = "other"
    factor_role: FactorRoleT = "unclear"
    factor_verbatim: str = ""
    section_located: SectionT = "unclear"


class ConceptRelationItem(BaseModel):
    """One relation the review draws between two constructs.

    The registration asks for hierarchical and semantic relationships between
    psychological concepts. These items are the edges of the concept ontology:
    which construct is treated as a subtype, a component, a synonym, a cause, or
    a distinct neighbour of which other construct, and whether the review says so
    or merely behaves that way.
    """

    source_concept: str = ""
    target_concept: str = ""
    relation_type: ConceptRelationT = "unclear"
    explicitly_stated: YesNoT = "no"
    relation_verbatim: str = ""
    section_located: SectionT = "unclear"


class IntegrationClaimItem(BaseModel):
    """One passage in which the review relates two or three domains to each other.

    The edges of the domain-level ontology. Naming the two factors that are
    linked, rather than only the domains they belong to, is what turns a count of
    integration statements into a map of what this literature claims connects to
    what.
    """

    domains_linked: DomainPairT = "bio_psych"
    integration_level: PairwiseT = "mentioned"
    source_factor_label: str = ""          # the factor doing the influencing
    target_factor_label: str = ""          # the factor being influenced
    direction: LinkDirectionT = "unspecified"
    mediator_or_moderator: str = ""        # the named intermediate, when one is given
    claim_verbatim: str = ""
    mechanism_note: str = ""               # the pathway in your words, if one is given
    section_located: SectionT = "unclear"
    evidence_basis: EvidenceBasisT = "asserted"


class FrameworkItem(BaseModel):
    """One theoretical model or framework the review invokes."""

    framework_label: str = ""
    role: FrameworkRoleT = "unclear"
    domains_covered: list[str] = Field(default_factory=list)   # which BPS domains the model spans
    attributed_source: str = ""
    framework_verbatim: str = ""
    section_located: SectionT = "unclear"


class InstrumentItem(BaseModel):
    """One measurement or appraisal instrument named in the review.

    What a review measures is the most concrete form its operationalization of
    the model takes: a review that claims a biopsychosocial frame and measures
    only psychological questionnaires has told you something the prose did not.
    """

    instrument_label: str = ""
    abbreviation: str = ""
    domain_measured: MeasuredDomainT = "unclear"
    construct_measured_as_stated: str = ""
    role: InstrumentRoleT = "unclear"
    instrument_verbatim: str = ""


class ConceptualProblemItem(BaseModel):
    """One conceptual problem the review displays or names (the SQ1 evidence)."""

    problem_type: ConceptualProblemT = "other"
    problem_scope: ProblemScopeT = "other"
    affected_labels: list[str] = Field(default_factory=list)   # which constructs it concerns
    named_by_authors: YesNoT = "no"       # did the review notice it, or does it merely display it
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

    # --- routing, context, and provenance ---
    review_track: ReviewTrackT = "unclear"
    source_type: SourceTypeT = "unclear"
    icd11_pain_category: Icd11CategoryT = "unclear"
    population: PopulationT = "unclear"
    care_setting: CareSettingT = "not reported"
    primary_discipline: DisciplineT = "unclear"
    pain_condition_detail: str = ""
    pain_conditions: list[str] = Field(default_factory=list)
    context_note: str = ""                       # cultural or healthcare context, when reported
    quality_assessment_reported: TristateT = "unclear"
    quality_assessment_tools: list[str] = Field(default_factory=list)

    # --- how the biopsychosocial label is used and defined (RQ1) ---
    bps_label_used: BpsLabelT = "absent"
    bps_primary_function: BpsFunctionT = "unclear"
    bps_functions_present: list[str] = Field(default_factory=list)
    bps_definition_status: BpsDefinitionStatusT = "undefined"
    bps_model_variants: list[str] = Field(default_factory=list)
    bps_usage_instances: list[BpsUsageItem] = Field(default_factory=list)
    bps_definitions: list[BpsDefinitionItem] = Field(default_factory=list)
    bps_operationalization_summary: str = ""

    # --- domain coverage (the four-level ladder, one per domain) ---
    domain_coverage_bio: CoverageT = "absent"
    domain_coverage_psych: CoverageT = "absent"
    domain_coverage_social: CoverageT = "absent"
    coverage_lifestyle: CoverageT = "absent"
    coverage_spiritual_existential: CoverageT = "absent"
    domain_evidence: list[DomainEvidenceItem] = Field(default_factory=list)

    # --- which factors actually carry each domain (RQ2 scope, ontology nodes) ---
    biological_factors: list[BiologicalFactorItem] = Field(default_factory=list)
    social_factors: list[SocialFactorItem] = Field(default_factory=list)
    other_domain_factors: list[OtherDomainFactorItem] = Field(default_factory=list)

    # --- integration (the core RQ2 contribution, ontology edges) ---
    integration_bio_psych: PairwiseT = "none"
    integration_psych_social: PairwiseT = "none"
    integration_bio_social: PairwiseT = "none"
    integration_triadic: TriadicT = "none"
    integration_claims: list[IntegrationClaimItem] = Field(default_factory=list)
    integration_mechanism_summary: str = ""

    # --- typology and balance ---
    overall_balance: BalanceT = "unclear"
    bps_typology: TypologyT = "unclear"

    # --- psychological concepts, their relations, frameworks, measures (RQ3) ---
    concept_definitions_present: DefinitionsPresentT = "no"
    psychological_concepts: list[ConceptItem] = Field(default_factory=list)
    concept_relations: list[ConceptRelationItem] = Field(default_factory=list)
    theoretical_frameworks: list[FrameworkItem] = Field(default_factory=list)
    instruments: list[InstrumentItem] = Field(default_factory=list)

    # --- conceptual problems (SQ1) ---
    conceptual_problems: list[ConceptualProblemItem] = Field(default_factory=list)

    # --- synthesis hooks (free text, deliberately unconstrained) ---
    key_quotes: list[KeyQuoteItem] = Field(default_factory=list)
    emergent_labels: list[str] = Field(default_factory=list)
    conceptual_tensions: list[str] = Field(default_factory=list)
    additional_observations: list[str] = Field(default_factory=list)
    synthesis_note: str = ""
    coding_rationale: str = ""


ITEM_MODELS: dict[str, type[BaseModel]] = {
    "bps_usage_instances": BpsUsageItem,
    "bps_definitions": BpsDefinitionItem,
    "domain_evidence": DomainEvidenceItem,
    "biological_factors": BiologicalFactorItem,
    "social_factors": SocialFactorItem,
    "psychological_concepts": ConceptItem,
    "other_domain_factors": OtherDomainFactorItem,
    "concept_relations": ConceptRelationItem,
    "integration_claims": IntegrationClaimItem,
    "theoretical_frameworks": FrameworkItem,
    "instruments": InstrumentItem,
    "conceptual_problems": ConceptualProblemItem,
    "key_quotes": KeyQuoteItem,
}

# Record-level free-text lists: open vocabularies, no item structure. The value
# names the vocabulary a list is compared against in the analysis; an empty value
# means the list is pure free text and is never mapped onto anything.
OPEN_LIST_FIELDS: dict[str, str] = {
    "pain_conditions": "pain_condition",
    "quality_assessment_tools": "instrument",
    "bps_model_variants": "",
    "bps_functions_present": "",
    "emergent_labels": "",
    "conceptual_tensions": "",
    "additional_observations": "",
}

# The verbatim quote key inside each structured item.
ITEM_QUOTE_KEY: dict[str, str] = {
    "bps_usage_instances": "usage_verbatim",
    "bps_definitions": "definition_verbatim",
    "domain_evidence": "evidence_verbatim",
    "biological_factors": "factor_verbatim",
    "social_factors": "factor_verbatim",
    "psychological_concepts": "definition_verbatim",
    "other_domain_factors": "factor_verbatim",
    "concept_relations": "relation_verbatim",
    "integration_claims": "claim_verbatim",
    "theoretical_frameworks": "framework_verbatim",
    "instruments": "instrument_verbatim",
    "conceptual_problems": "problem_verbatim",
    "key_quotes": "claim_verbatim",
}

# What identifies an item, as one key or as several joined together. An
# integration claim and a concept relation are edges, so their identity is the
# pair they connect, not a single label.
ITEM_LABEL_KEY: dict[str, tuple[str, ...]] = {
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

# Which project vocabulary normalizes which item field, in place.
#
# Only the two fields whose whole purpose is to point at the shared spine are
# normalized in the stored item, and each of them sits next to a free-text field
# that holds the review's own wording: ``subdomain_label`` next to
# ``factor_label``, ``concept_family`` next to ``concept_label``. Every other
# label is stored exactly as the review wrote it, and its mapped form is added
# alongside in the item table (``label_normalized``, ``label_controlled``), so a
# label that lands on the spine never costs us the high-resolution original.
ITEM_LABEL_VOCAB: dict[str, dict[str, str]] = {
    "biological_factors": {"subdomain_label": "bio_subdomain"},
    "social_factors": {"subdomain_label": "social_subdomain"},
    "psychological_concepts": {"concept_family": "concept_family"},
}

# The vocabulary an item's own identifying label is compared against when the
# analysis normalizes it (the item table and the cross-model set overlap). The
# two factor lists are absent on purpose: a factor label is the review's own
# wording for a specific thing, and it is not supposed to be a vocabulary entry.
ITEM_IDENTITY_VOCAB: dict[str, str] = {
    "psychological_concepts": "psych_concept",
    "theoretical_frameworks": "framework",
    "instruments": "instrument",
}

# Where each item type attaches to the project ontology: (field, vocabulary).
# This is the anchor whose coverage the analysis reports, and it is always a
# different field from the item's own free-text label, so measuring how much of
# the extraction lands on the spine never means measuring how often the coder
# used our words for the thing itself.
ITEM_ANCHOR: dict[str, tuple[str, str]] = {
    "biological_factors": ("subdomain_label", "bio_subdomain"),
    "social_factors": ("subdomain_label", "social_subdomain"),
    "psychological_concepts": ("concept_family", "concept_family"),
    "theoretical_frameworks": ("framework_label", "framework"),
    "instruments": ("instrument_label", "instrument"),
}
