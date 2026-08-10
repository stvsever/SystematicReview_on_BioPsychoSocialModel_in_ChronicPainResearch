from __future__ import annotations

"""The high-resolution instruction set for the Stage 3 full-text coding scheme.

The prompt is assembled from the schema and the vocabularies rather than written
out by hand, so the instructions can never drift from what the validator accepts
or from what the dossier shows the experts.

It is built in six blocks:

1. **Role and review context.** What review this is, which protocol it follows,
   and what the coder is being asked to produce.
2. **Coding principles.** The evidence discipline: quote, do not paraphrase; code
   what the paper says, not what the field believes; never let the word
   biopsychosocial stand in for coverage; name the thing rather than reporting
   that a thing was present; keep nuance rather than rounding it to the nearest
   controlled value.
3. **Field-by-field instructions.** Every field with its operational anchor, its
   controlled values, and its boundary against the adjacent value on the ladder.
4. **The ladders, spelled out.** Coverage, pairwise integration, and triadic
   integration, each with the rule that separates one rung from the next.
5. **Preferred labels.** The project vocabularies, offered as a spine rather than
   as a closed list: use the canonical label when it fits, and otherwise write
   what the paper wrote.
6. **The output contract.** One JSON object, exact keys, caps, and length
   allowances.
"""

import json

from bps_review.fulltext.coding import schema as S
from bps_review.fulltext.coding import vocabulary as V
from bps_review.fulltext.config import (
    ITEM_CAPS,
    ITEM_SUBLIST_CAP,
    MAX_NOTE_WORDS,
    MAX_QUOTE_WORDS,
    MAX_SUMMARY_WORDS,
    OPEN_LIST_CAPS,
)


REVIEW_CONTEXT = (
    "You are coding for an OSF-registered systematic review titled 'How the biopsychosocial model "
    "frames chronic pain research' (OSF DOI 10.17605/OSF.IO/T4FAM). The review asks how the "
    "biopsychosocial (BPS) model is actually operationalized in chronic pain review literature: what "
    "work the BPS label does in a paper, how much biological, psychological, and social content each "
    "review carries and which factors carry it, whether those domains are genuinely related to one "
    "another or merely listed side by side, which psychological concepts and theoretical frameworks "
    "recur and how they are defined and related, and which conceptual problems keep coming back when "
    "BPS is invoked. The central claim being tested is that BPS language is widespread while "
    "substantive three-domain integration is not, so the integration ladder is the most consequential "
    "judgement you will make."
)

TASK_STATEMENT = (
    "You are given the full text of one review article. Your task is a high-resolution conceptual "
    "extraction, not a summary and not a screening decision. Two things are asked of you at once. "
    "First, grade: how deeply each domain is treated, and how each pair of domains and the triad are "
    "integrated, on explicit ladders. Second, and this is the larger half of the work, name: every "
    "biological, social, lifestyle, and existential factor the review carries, every psychological "
    "construct with what the review says it means, every relation it draws between constructs, every "
    "framework and instrument it uses, and every passage where the biopsychosocial label does work. "
    "Recording that a domain is present is not a coding; recording which factors carry it, in which "
    "role, on the strength of which sentence, is. These named items are assembled downstream into an "
    "ontology of how this literature reasons, so extraction completeness matters more than brevity: "
    "if the paper makes eight separate cross-domain claims, record eight integration items, not one."
)

CODING_PRINCIPLES = [
    "Code this paper only. Never import knowledge about the biopsychosocial model from the wider "
    "literature, from the models you know, or from other papers. If this paper does not say it, it is "
    "not coded.",
    "The word 'biopsychosocial' is never evidence of coverage. A review can use the label in every "
    "paragraph and still score a domain as absent. Coverage is judged on substantive domain content: "
    "named constructs, mechanisms, determinants, measures, or interventions belonging to that domain.",
    "Name the thing. Wherever a list asks for factors, constructs, frameworks, instruments, or "
    "relations, give the specific ones this paper names, one item each, in the paper's own wording. "
    "A generic entry such as 'psychological factors' or 'social context' is not an item; it is the "
    "absence of one.",
    "Quote, do not paraphrase. Every verbatim field must contain text copied exactly from the article, "
    "with the original wording and spelling. Copy a contiguous passage. Do not stitch fragments "
    "together, do not clean the language up, and never write a quote the paper does not contain. If "
    "you cannot find a passage that carries the point, leave the verbatim field empty and say what you "
    "saw in the accompanying note instead.",
    "Integration is a claim about a relation, not about co-occurrence. Two domains named in the same "
    "sentence with no relational verb is 'mentioned'. An association is 'descriptive'. A stated arrow "
    "of effect with no pathway is 'directional'. Only a stated pathway or process is 'mechanistic'.",
    "Serial listing of biological, then psychological, then social factors is 'none' for the triadic "
    "field, however long the lists are. Reserve 'mechanistic' for reasoning where removing any one "
    "domain would break the explanation.",
    "Separate absence of the thing from absence of evidence. 'absent' means the domain is genuinely "
    "not represented; 'unclear' on a nominal field means the paper does not let you tell. Never use a "
    "negative value as a default.",
    "Preferred labels are a spine, not a cage. Where a list of preferred labels is given, use the "
    "preferred label when it genuinely fits, and otherwise write the paper's own wording. A precise "
    "unusual label is worth more to this review than a controlled label that flattens it.",
    "A mapped label never replaces the paper's own words. Every item that carries a preferred label "
    "also carries the review's own wording next to it: subdomain_label sits beside factor_label, "
    "concept_family sits beside concept_label. Fill both. Never overwrite the specific term the paper "
    "used with the broader family it belongs to, and never drop a term because the family already "
    "covers it.",
    "Everything that does not fit is wanted, not discarded. When a factor, construct, framework, "
    "instrument, or relation falls outside the preferred lists, record it anyway with the paper's own "
    "label and leave the spine field empty or write your own best label there. Then also add the term "
    "to emergent_labels. The gaps between this literature and the project vocabularies are a finding "
    "the review is actively looking for, so err on the side of recording too much rather than too "
    "little.",
    "The text may contain '[... omitted ...]' markers where a paragraph without conceptual content was "
    "removed to fit the window. Code only what is present, and never quote across an omission marker.",
    "An empty list is a legitimate coding. The caps are ceilings, never targets: return as many items "
    "as the paper genuinely supports, and return none when it supports none.",
    "Do not judge whether the paper is good, and do not score its relevance. Eligibility, integration "
    "depth, conceptual yield, and synthesis priority are computed afterwards from the content you "
    "record.",
]

# --------------------------------------------------------------------------
# Operational anchors, one per coded field. These are the reliability layer.
# --------------------------------------------------------------------------
FIELD_INSTRUCTIONS: list[tuple[str, str]] = [
    # --- routing and context ---
    ("review_track",
     "Which of the two planned reviews this record belongs to, read from the pain condition the paper "
     "actually studies. 'musculoskeletal' covers low back, neck, osteoarthritis, fibromyalgia, "
     "shoulder, and similar; 'neuropathic' covers painful neuropathy, radicular pain, post-herpetic "
     "neuralgia, and similar; 'mixed_or_other' when several families are genuinely covered or the "
     "condition belongs to neither; 'unclear' when the paper does not say."),
    ("source_type",
     "What kind of evidence synthesis this is, read from how the paper describes itself in its "
     "abstract and methods. Prefer the most specific applicable value: 'meta-analysis' outranks "
     "'systematic review' when effect sizes are pooled, 'umbrella review' when the units are reviews."),
    ("icd11_pain_category",
     "The ICD-11 aligned pain category the paper is about, now read from the full text rather than "
     "from the abstract. Use 'mixed or unspecified chronic pain' when several categories are genuinely "
     "covered, and 'unclear' only when the paper never says."),
    ("population",
     "The population the reviewed evidence concerns. 'mixed ages' when adults and younger participants "
     "are both included, 'not applicable' for a purely theoretical paper with no population."),
    ("care_setting",
     "The care setting the paper is about, when it reports one. 'not reported' is the honest answer "
     "for most reviews and is preferred over a guess."),
    ("primary_discipline",
     "The disciplinary home of the paper, read from the journal, the framing, and the vocabulary "
     "rather than from author affiliations alone. 'multidisciplinary' describes the writing, not the "
     "author list."),
    ("pain_condition_detail",
     f"Free text, at most {MAX_NOTE_WORDS} words: the exact pain condition or conditions studied, in "
     "the paper's own words."),
    ("pain_conditions",
     "The specific pain conditions named, as a list of short labels. Preferred labels are given below; "
     "use the paper's own wording when it is more precise."),
    ("context_note",
     f"At most {MAX_NOTE_WORDS} words on the cultural, geographic, or healthcare-system context, when "
     "the paper states one. Empty when it does not."),
    ("quality_assessment_reported",
     "Whether the paper reports a formal quality or risk-of-bias assessment of the evidence it "
     "reviews. This is descriptive only; the review does not appraise the papers itself."),
    ("quality_assessment_tools",
     "The appraisal tools named (AMSTAR, AMSTAR-2, ROBIS, GRADE, Cochrane risk of bias, and others), "
     "as a list. Empty when none is named."),

    # --- how the BPS label is used and defined (RQ1) ---
    ("bps_label_used",
     "Which biopsychosocial vocabulary the paper actually uses. 'explicit_bps_term' when the words "
     "biopsychosocial or bio-psycho-social appear anywhere; 'variant_term_only' when only a neighbour "
     "term does (psychosocial, multidimensional, multifactorial, holistic); 'domain_language_only' "
     "when the domains are discussed with no model label at all; 'absent' when neither appears."),
    ("bps_primary_function",
     "The single dominant work the biopsychosocial label does in this paper, judged over the paper as "
     "a whole. 'explanatory framework' requires that BPS explicitly explains pain or pain-related "
     "disability. 'intervention rationale' is BPS mainly justifying multimodal or interdisciplinary "
     "treatment. 'organizing principle' is BPS structuring the scope or the categories without any "
     "integration mechanism. 'rhetorical label' is ceremonial, aspirational, or symbolic use with no "
     "analytic follow-through. 'operational definition' is reserved for a paper that turns the model "
     "into the variables it actually codes or measures. 'critique or problematization' is reserved for "
     "a paper that argues about the model itself."),
    ("bps_functions_present",
     "Every function the label performs anywhere in the paper, as a list drawn from the same "
     "vocabulary. A paper routinely does two or three of these, and which ones it combines is a "
     "finding, so do not collapse them into the primary function."),
    ("bps_definition_status",
     "How the paper handles the meaning of the model itself. 'formally_defined' when it states what "
     "the model means; 'described_informally' when the meaning is carried by description only; "
     "'cited_only' when a citation stands in for a definition; 'undefined' when the label is used with "
     "no meaning given anywhere. 'undefined' is a finding, not a coding failure."),
    ("bps_model_variants",
     "The model labels the paper actually uses, verbatim and de-duplicated (for example "
     "'biopsychosocial model', 'bio-psycho-social framework', 'sociopsychobiological model', "
     "'extended biopsychosocial model'). This is what makes terminological drift visible."),
    ("bps_usage_instances",
     "One item for every distinct passage where the biopsychosocial label does work. Give "
     "usage_verbatim, bps_function for that passage, is_definitional, attributed_source (who the model "
     "is credited to there, empty when nobody), section_located, and a short note. A paper that "
     "invokes the model in the introduction to justify the topic and again in the discussion to "
     "recommend multidisciplinary care yields two items, not one."),
    ("bps_definitions",
     "One item for every place where the paper says what the biopsychosocial model is. Give "
     "definition_verbatim, definition_type, attributed_source, elements_named (the components the "
     "definition lists, as short labels), and section_located. Return an empty list when the paper "
     "never says what the model is."),
    ("bps_operationalization_summary",
     f"At most {MAX_SUMMARY_WORDS} words, in your own words: what this paper actually does with the "
     "biopsychosocial model, as opposed to what it says about it. Name the mechanism of use, for "
     "example 'organizes the results section into three domain headings and never relates them'."),

    # --- coverage ---
    ("domain_coverage_bio",
     "Depth of biological content: anatomy, physiology, pathophysiology, nociception, central or "
     "peripheral sensitization, inflammation, imaging, genetics, pharmacology, tissue pathology."),
    ("domain_coverage_psych",
     "Depth of psychological content: cognition, affect, behaviour, beliefs, coping, catastrophizing, "
     "fear-avoidance, self-efficacy, depression, anxiety, acceptance, psychological treatment."),
    ("domain_coverage_social",
     "Depth of social content: work and occupational context, family and relationships, culture, "
     "socioeconomic position, healthcare system, social support, stigma, policy."),
    ("coverage_lifestyle",
     "Depth of lifestyle content on the same ladder: physical activity and exercise behaviour, sleep "
     "hygiene, diet and weight, smoking, alcohol. Lifestyle is registered as a domain in its own right "
     "and is not folded into the triad."),
    ("coverage_spiritual_existential",
     "Depth of spiritual or existential content on the same ladder: meaning, faith or religion, hope, "
     "existential suffering, acceptance of mortality. 'absent' is the expected value for most papers "
     "and is a finding in itself."),
    ("domain_evidence",
     "One item per core domain you did not score as 'absent', carrying the passage that justifies the "
     "coverage level you gave it. Give domain, coverage_level (identical to the field above), "
     "constructs_named (the domain-specific constructs the paper actually names), subdomains_named "
     "(the ontology subdomains the content belongs to, preferred labels below), evidence_verbatim, and "
     "section_located."),

    # --- the factor inventory (the ontology nodes) ---
    ("biological_factors",
     "Every biological factor the paper names, one item each. Give factor_label in the paper's own "
     "wording, always, and as specifically as the paper puts it; subdomain_label from the biological "
     "ontology below when one fits, and left empty or filled with your own short label when none does; "
     "mechanism_level; factor_role (what the factor does in this paper: determinant, mediator, "
     "moderator, outcome, treatment target, and so on); factor_verbatim; section_located; and "
     "evidence_basis. A factor with no home in the ontology is still recorded, and its label also goes "
     "into emergent_labels. Do not record psychological constructs here; they belong in "
     "psychological_concepts."),
    ("social_factors",
     "Every social factor the paper names, one item each, with the same structure and the same rule "
     "about labels: factor_label in the paper's own wording, subdomain_label from the social ontology "
     "below when one fits, social_level (the level of social organization the factor sits at), "
     "factor_role, factor_verbatim, section_located, evidence_basis. The social domain is the one this "
     "literature is thinnest on, so record every social factor it does name, however briefly."),
    ("other_domain_factors",
     "Factors outside the triad: lifestyle, spiritual or existential, and environmental. Give "
     "factor_label, domain, factor_role, factor_verbatim, and section_located. Return an empty list "
     "when the paper stays inside the three domains, and use 'other' freely when a factor belongs to "
     "none of the named domains but clearly matters to the paper's account."),
    ("psychological_concepts",
     "Every psychological construct the paper uses, one item each. Give concept_label (the paper's own "
     "term, always, at the resolution the paper uses it: 'fear of movement during lifting' is not the "
     "same item as 'kinesiophobia'), concept_family (the family from the taxonomy below when one fits, "
     "empty when none does), definitional_status "
     "('formally_defined' when the paper says what it means, 'operationalized_only' when the meaning "
     "is fixed only through a measure, 'described_informally' when meaning is carried by description "
     "alone, 'named_only' when it is used without any of these), definition_verbatim (the passage that "
     "defines it, empty when there is none), definition_source, measure_named (the instrument the "
     "paper operationalizes it with, empty when none), factor_role, and section_located."),
    ("concept_relations",
     "Every relation the paper draws between two constructs, one item each. This is the hierarchical "
     "and semantic structure the registration asks for, and it is what turns a list of concepts into a "
     "map. Give source_concept, target_concept, relation_type, explicitly_stated ('yes' when the paper "
     "states the relation, 'no' when it merely behaves as though it holds), relation_verbatim, and "
     "section_located. 'conflated_without_comment' is deliberately available: silent conflation of two "
     "constructs is one of the most informative things this review can find, and it is invisible "
     "unless a coder is allowed to record it."),

    # --- integration (the ontology edges) ---
    ("integration_bio_psych", "Integration between the biological and psychological domains."),
    ("integration_psych_social", "Integration between the psychological and social domains."),
    ("integration_bio_social", "Integration between the biological and social domains."),
    ("integration_triadic",
     "Three-domain integration. 'mechanistic' when biological, psychological, and social factors act "
     "on one another as one system; 'descriptive' when all three are genuinely related in one account "
     "but no pathway is specified; 'partial' when two domains are integrated and the third is present "
     "but only loosely attached; 'none' when the domains stand in parallel or one is absent."),
    ("integration_claims",
     "One item for every passage in which the paper relates two or three domains to each other. This "
     "is the evidence base for the four integration fields above, so a pairwise field coded above "
     "'mentioned' should have at least one item behind it. Give domains_linked, integration_level, "
     "source_factor_label and target_factor_label (the two specific factors that are linked, in the "
     "paper's own wording, which is what makes this an edge rather than a tally), direction, "
     "mediator_or_moderator (the named intermediate, empty when none), claim_verbatim, mechanism_note "
     "(the pathway in your words, empty when none is given), section_located, and evidence_basis."),
    ("integration_mechanism_summary",
     f"At most {MAX_SUMMARY_WORDS} words, in your own words: the cross-domain pathways this paper "
     "actually proposes. Write 'none proposed' when the paper proposes none."),

    # --- typology and balance ---
    ("overall_balance",
     "Relative emphasis across the three core domains, judged on how much of the paper each one "
     "occupies. 'balanced' means no domain clearly dominates; 'dyadic' means two domains carry the "
     "paper and the third is marginal; 'unclear' when the emphasis cannot be read off the text."),
    ("bps_typology",
     "What this review does with the biopsychosocial model at full-text depth. 'true_integrative': all "
     "three domains present and genuinely related, with at least a descriptive triadic account. "
     "'multifactorial': all three domains substantively present but treated in parallel. 'pseudo_bps': "
     "BPS language with one or more domains thin or absent. 'rhetorical_bps': the label is used "
     "ceremonially and does no analytic work. 'narrow_despite_label': the paper claims a BPS frame but "
     "is in practice a single-domain review. 'unclear' only when the text genuinely does not allow a "
     "judgement."),

    # --- concepts, frameworks, measures ---
    ("concept_definitions_present",
     "Whether the review defines the psychological constructs it uses. 'yes' when the main constructs "
     "are defined or clearly operationalized; 'partial' when some are and others are only named; 'no' "
     "when constructs are used without any meaning being given."),
    ("theoretical_frameworks",
     "Every theoretical model or framework the paper invokes, one item each. Give framework_label "
     "(preferred labels below), role, domains_covered (which of biological, psychological, and social "
     "the model actually spans, as a list), attributed_source, framework_verbatim, and section_located."),
    ("instruments",
     "Every measurement or appraisal instrument named, one item each. Give instrument_label (preferred "
     "labels below), abbreviation, domain_measured, construct_measured_as_stated (what the paper says "
     "the instrument captures, in its own wording), role, and instrument_verbatim. What a review "
     "measures is the most concrete form its operationalization of the model takes, so a paper that "
     "claims a biopsychosocial frame and measures only questionnaires from one domain should show that "
     "here."),

    # --- conceptual problems (SQ1) ---
    ("conceptual_problems",
     "Conceptual problems this paper names or displays, one item each. Give problem_type, "
     "problem_scope (what the problem is about), affected_labels (the constructs or terms it concerns, "
     "as a list), named_by_authors ('yes' when the paper itself points the problem out, 'no' when it "
     "merely displays it), problem_verbatim (the passage that shows it, which for a displayed problem "
     "may be the passage where the gap is visible), and a short note. Return an empty list when you "
     "see none."),

    # --- synthesis hooks ---
    ("key_quotes",
     "The most conceptually load-bearing passages in the paper: the ones a reviewer would want to read "
     "first when writing the synthesis. Prefer passages that stand on their own. Give claim_verbatim, "
     "claim_type, section_located, and why_it_matters (one short sentence)."),
    ("emergent_labels",
     "Every conceptually important term this paper uses that the preferred vocabularies above do not "
     "contain: a factor, construct, mechanism, framework, instrument, or population label that has no "
     "good home on the spine. Write it exactly as the paper writes it, one per item. This list is the "
     "review's own error signal: it is how the project ontology learns what it is missing, so use it "
     "generously whenever you had to leave a spine field empty or write your own label."),
    ("conceptual_tensions",
     "Contradictions, ambiguities, unresolved debates, and gaps the paper names or displays, including "
     "tensions visible inside the paper itself. One tension per item, in free text."),
    ("additional_observations",
     "Anything else conceptually relevant that no other field captures. One observation per item, as "
     "long as it needs to be. This field exists so that you never have to force an observation into a "
     "field where it does not belong, and so that the synthesis can see what the scheme did not "
     "anticipate."),
    ("synthesis_note",
     f"At most {MAX_SUMMARY_WORDS} words: what this paper contributes to the question of how the "
     "biopsychosocial model is operationalized, and what it does not. Write for a reviewer who has not "
     "read the paper."),
    ("coding_rationale",
     f"At most {MAX_NOTE_WORDS} words justifying the main judgements: the typology, the triadic "
     "integration level, and anything that was a close call."),
]

# The ladders, given to the coder as explicit rung-by-rung rules.
LADDERS: dict[str, dict[str, str]] = {
    "domain_coverage": {
        "elaborated": "The domain is developed as a substantive analytic thread: several distinct "
                      "constructs or mechanisms discussed, weighed, or synthesized rather than named.",
        "mentioned": "The domain is explicitly present with at least one named construct, but is not "
                     "developed into a sustained thread.",
        "minimal": "The domain appears only as an umbrella label or a single incidental reference, "
                   "with no concrete construct attached.",
        "absent": "The domain is not represented in the substantive content, even if a global BPS "
                  "label is used elsewhere.",
    },
    "pairwise_integration": {
        "mechanistic": "A pathway or process by which one domain acts on the other is specified: a "
                       "named mediator, moderator, or physiological or behavioural route.",
        "directional": "A directional or causal influence is asserted (X predicts, increases, worsens "
                       "Y) but no pathway is given.",
        "descriptive": "The two domains are linked as associated or correlated, without direction or "
                       "mechanism.",
        "mentioned": "Both domains appear near one another but the relationship is not characterized "
                     "at all.",
        "none": "No relationship between the two domains is articulated anywhere, or one of them is "
                "absent.",
    },
    "triadic_integration": {
        "mechanistic": "A genuinely three-domain mechanism in which biological, psychological, and "
                       "social factors act on one another as a system.",
        "descriptive": "All three domains are related to the outcome in one integrated narrative, but "
                       "as a joint description rather than a specified mechanism.",
        "partial": "Two domains are integrated with each other while the third is present but only "
                   "loosely attached.",
        "none": "No three-domain integration: the domains stand in parallel, or one or more is absent.",
    },
}

# Fields whose values are drawn from a closed list, with that list.
CONTROLLED_VALUES: dict[str, list[str]] = {
    "review_track": S.REVIEW_TRACK_OPTIONS,
    "source_type": S.SOURCE_TYPE_OPTIONS,
    "icd11_pain_category": S.ICD11_CATEGORY_OPTIONS,
    "population": S.POPULATION_OPTIONS,
    "care_setting": S.CARE_SETTING_OPTIONS,
    "primary_discipline": S.DISCIPLINE_OPTIONS,
    "quality_assessment_reported": S.TRISTATE_OPTIONS,
    "bps_label_used": S.BPS_LABEL_OPTIONS,
    "bps_primary_function": S.BPS_FUNCTION_OPTIONS,
    "bps_definition_status": S.BPS_DEFINITION_STATUS_OPTIONS,
    "domain_coverage_bio": S.COVERAGE_OPTIONS,
    "domain_coverage_psych": S.COVERAGE_OPTIONS,
    "domain_coverage_social": S.COVERAGE_OPTIONS,
    "coverage_lifestyle": S.COVERAGE_OPTIONS,
    "coverage_spiritual_existential": S.COVERAGE_OPTIONS,
    "integration_bio_psych": S.PAIRWISE_INTEGRATION_OPTIONS,
    "integration_psych_social": S.PAIRWISE_INTEGRATION_OPTIONS,
    "integration_bio_social": S.PAIRWISE_INTEGRATION_OPTIONS,
    "integration_triadic": S.TRIADIC_INTEGRATION_OPTIONS,
    "overall_balance": S.BALANCE_OPTIONS,
    "bps_typology": S.TYPOLOGY_OPTIONS,
    "concept_definitions_present": S.DEFINITIONS_PRESENT_OPTIONS,
}

# Record-level list fields whose entries come from a closed vocabulary.
CONTROLLED_LIST_VALUES: dict[str, list[str]] = {
    "bps_functions_present": S.BPS_FUNCTION_OPTIONS,
}

# The closed value lists inside the structured items.
ITEM_VALUE_LISTS: dict[str, dict[str, list[str]]] = {
    "bps_usage_instances": {
        "bps_function": S.BPS_FUNCTION_OPTIONS,
        "is_definitional": S.YES_NO_OPTIONS,
        "section_located": S.SECTION_OPTIONS,
    },
    "bps_definitions": {
        "definition_type": S.BPS_DEFINITION_TYPE_OPTIONS,
        "section_located": S.SECTION_OPTIONS,
    },
    "domain_evidence": {
        "domain": S.DOMAIN_OPTIONS,
        "coverage_level": S.COVERAGE_OPTIONS,
        "section_located": S.SECTION_OPTIONS,
    },
    "biological_factors": {
        "mechanism_level": S.BIO_MECHANISM_LEVEL_OPTIONS,
        "factor_role": S.FACTOR_ROLE_OPTIONS,
        "section_located": S.SECTION_OPTIONS,
        "evidence_basis": S.EVIDENCE_BASIS_OPTIONS,
    },
    "social_factors": {
        "social_level": S.SOCIAL_LEVEL_OPTIONS,
        "factor_role": S.FACTOR_ROLE_OPTIONS,
        "section_located": S.SECTION_OPTIONS,
        "evidence_basis": S.EVIDENCE_BASIS_OPTIONS,
    },
    "other_domain_factors": {
        "domain": S.OTHER_DOMAIN_OPTIONS,
        "factor_role": S.FACTOR_ROLE_OPTIONS,
        "section_located": S.SECTION_OPTIONS,
    },
    "psychological_concepts": {
        "definitional_status": S.DEFINITIONAL_STATUS_OPTIONS,
        "definition_source": S.DEFINITION_SOURCE_OPTIONS,
        "factor_role": S.FACTOR_ROLE_OPTIONS,
        "section_located": S.SECTION_OPTIONS,
    },
    "concept_relations": {
        "relation_type": S.CONCEPT_RELATION_OPTIONS,
        "explicitly_stated": S.YES_NO_OPTIONS,
        "section_located": S.SECTION_OPTIONS,
    },
    "integration_claims": {
        "domains_linked": S.DOMAIN_PAIR_OPTIONS,
        "integration_level": S.PAIRWISE_INTEGRATION_OPTIONS,
        "direction": S.LINK_DIRECTION_OPTIONS,
        "section_located": S.SECTION_OPTIONS,
        "evidence_basis": S.EVIDENCE_BASIS_OPTIONS,
    },
    "theoretical_frameworks": {
        "role": S.FRAMEWORK_ROLE_OPTIONS,
        "section_located": S.SECTION_OPTIONS,
    },
    "instruments": {
        "domain_measured": S.MEASURED_DOMAIN_OPTIONS,
        "role": S.INSTRUMENT_ROLE_OPTIONS,
    },
    "conceptual_problems": {
        "problem_type": S.CONCEPTUAL_PROBLEM_OPTIONS,
        "problem_scope": S.PROBLEM_SCOPE_OPTIONS,
        "named_by_authors": S.YES_NO_OPTIONS,
    },
    "key_quotes": {
        "claim_type": S.CLAIM_TYPE_OPTIONS,
        "section_located": S.SECTION_OPTIONS,
    },
}

# Which preferred-label list belongs with which field, shown to the coder as a
# spine rather than as a closed set.
PREFERRED_LABEL_FIELDS: dict[str, list[tuple[str, str]]] = {
    "pain_conditions": [("pain_conditions", "pain_condition")],
    "quality_assessment_tools": [("quality_assessment_tools", "instrument")],
    "domain_evidence": [("subdomains_named (biological)", "bio_subdomain"),
                        ("subdomains_named (psychological)", "psych_subdomain"),
                        ("subdomains_named (social)", "social_subdomain")],
    "biological_factors": [("subdomain_label", "bio_subdomain")],
    "social_factors": [("subdomain_label", "social_subdomain")],
    "psychological_concepts": [("concept_family", "concept_family"),
                               ("concept_label", "psych_concept")],
    "theoretical_frameworks": [("framework_label", "framework")],
    "instruments": [("instrument_label", "instrument")],
    "bps_usage_instances": [("attributed_source", "bps_source")],
    "bps_definitions": [("attributed_source", "bps_source")],
}


SYSTEM_PROMPT = (
    "You are a meticulous systematic-review coder working on how the biopsychosocial model is "
    "operationalized in chronic pain reviews. You read one full-text article at a time and return a "
    "single strict JSON object that follows the given schema exactly. You extract at high resolution: "
    "you name the specific factors, constructs, relations, frameworks, and instruments the article "
    "carries rather than reporting that content was present. You quote the article verbatim, you never "
    "invent content, you never treat the word biopsychosocial as evidence of coverage, and you prefer "
    "the lower rung of a ladder over a generous reading. You return JSON only, with no prose, no "
    "markdown, and no code fences."
)


def _item_spec(name: str) -> dict:
    model = S.ITEM_MODELS[name]
    fields: dict[str, object] = {}
    for field_name in model.model_fields:
        if field_name in ITEM_VALUE_LISTS.get(name, {}):
            fields[field_name] = {"values": ITEM_VALUE_LISTS[name][field_name]}
        elif field_name.endswith("verbatim"):
            fields[field_name] = {"type": f"verbatim quote, at most {MAX_QUOTE_WORDS} words, copied exactly"}
        elif field_name in ("note", "why_it_matters", "mechanism_note"):
            fields[field_name] = {"type": f"free text, at most {MAX_NOTE_WORDS} words"}
        elif model.model_fields[field_name].annotation == list[str]:
            fields[field_name] = {"type": f"list of short free-text labels, at most {ITEM_SUBLIST_CAP}"}
        else:
            fields[field_name] = {"type": "free text label"}
    spec: dict[str, object] = {"max_items": ITEM_CAPS.get(name), "item_fields": fields}
    preferred = PREFERRED_LABEL_FIELDS.get(name)
    if preferred:
        # Point at the shared vocabulary block rather than repeating it: the same
        # list is used by several fields, and the prompt is sent once per paper.
        spec["preferred_labels_for"] = {label: kind for label, kind in preferred}
    return spec


def build_schema_spec() -> dict:
    """The machine-readable half of the prompt: every field, its values, its caps."""
    spec: dict[str, object] = {}
    for field_name, instruction in FIELD_INSTRUCTIONS:
        entry: dict[str, object] = {"instruction": instruction}
        if field_name in CONTROLLED_VALUES:
            entry["values"] = CONTROLLED_VALUES[field_name]
        if field_name in CONTROLLED_LIST_VALUES:
            entry["values_for_each_item"] = CONTROLLED_LIST_VALUES[field_name]
        if field_name.startswith("domain_coverage_") or field_name.startswith("coverage_"):
            entry["ladder"] = LADDERS["domain_coverage"]
        if field_name in ("integration_bio_psych", "integration_psych_social", "integration_bio_social"):
            entry["ladder"] = LADDERS["pairwise_integration"]
        if field_name == "integration_triadic":
            entry["ladder"] = LADDERS["triadic_integration"]
        if field_name in S.OPEN_LIST_FIELDS and field_name not in S.ITEM_MODELS:
            entry["max_items"] = OPEN_LIST_CAPS.get(field_name)
            preferred = PREFERRED_LABEL_FIELDS.get(field_name)
            if preferred:
                entry["preferred_labels_for"] = {label: kind for label, kind in preferred}
        if field_name in S.ITEM_MODELS:
            entry.update(_item_spec(field_name))
        spec[field_name] = entry
    return spec


def preferred_label_block() -> dict[str, list[str]]:
    """The project vocabularies, given once and referenced by the fields that use them."""
    kinds = {kind for pairs in PREFERRED_LABEL_FIELDS.values() for _, kind in pairs}
    return {kind: V.controlled_labels(kind) for kind in sorted(kinds)}


def build_prompt(record: dict, coding_text: str) -> str:
    """Assemble the full user prompt for one paper."""
    instructions = {
        "review_context": REVIEW_CONTEXT,
        "task": TASK_STATEMENT,
        "coding_principles": CODING_PRINCIPLES,
        "preferred_label_vocabularies": preferred_label_block(),
        "fields": build_schema_spec(),
        "output_contract": [
            "Return exactly one JSON object with the keys listed under 'fields', plus 'record_id'.",
            f"'record_id' must be exactly '{record['record_id']}'.",
            "Every list field must be present, as an array; use an empty array when the paper offers "
            "nothing for it. Never omit a key.",
            "Controlled fields must use one of the listed values verbatim, in lowercase as given.",
            "'preferred_labels_for' names, per field, which list in "
            "'preferred_label_vocabularies' applies. Those lists are a spine and not a closed set: "
            "use the preferred label when it fits, and otherwise write the paper's own wording and "
            "add the term to emergent_labels.",
            f"Verbatim fields hold text copied from the article, at most {MAX_QUOTE_WORDS} words.",
            "Respect the max_items caps. They are ceilings, not targets: when a paper offers more than "
            "the cap, keep the most conceptually informative items; when it offers fewer, return fewer; "
            "when it offers none, return an empty array.",
            "Return JSON only. No markdown, no code fences, no commentary before or after.",
        ],
    }
    header = (
        "Code the following full-text review article according to the instructions.\n\n"
        f"INSTRUCTIONS:\n{json.dumps(instructions, ensure_ascii=False)}\n\n"
        f"RECORD_ID: {record['record_id']}\n"
        "ARTICLE:\n"
    )
    return header + coding_text


def prompt_overview() -> dict:
    """Compact description of the prompt, for the notebook and the dossier."""
    return {
        "n_coded_fields": len(FIELD_INSTRUCTIONS),
        "n_controlled_fields": len(CONTROLLED_VALUES),
        "n_extraction_lists": len(S.ITEM_MODELS),
        "n_open_lists": len([name for name in S.OPEN_LIST_FIELDS if name not in S.ITEM_MODELS]),
        "n_item_level_fields": sum(len(model.model_fields) for model in S.ITEM_MODELS.values()),
        "max_extracted_items": sum(ITEM_CAPS.values()),
        "max_quote_words": MAX_QUOTE_WORDS,
        "item_caps": ITEM_CAPS,
        "ladders": {name: list(rungs) for name, rungs in LADDERS.items()},
        "preferred_label_counts": V.vocabulary_overview(),
        "system_prompt": SYSTEM_PROMPT,
    }
