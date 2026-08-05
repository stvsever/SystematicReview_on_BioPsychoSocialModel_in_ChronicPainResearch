from __future__ import annotations

"""The high-resolution instruction set for the Stage 3 full-text coding scheme.

The prompt is assembled from the schema and the vocabularies rather than written
out by hand, so the instructions can never drift from what the validator accepts
or from what the dossier shows the experts.

It is built in five blocks:

1. **Role and review context.** What review this is, which protocol it follows,
   and what the coder is being asked to produce.
2. **Coding principles.** The evidence discipline: quote, do not paraphrase; code
   what the paper says, not what the field believes; never let the word
   biopsychosocial stand in for coverage; keep nuance rather than rounding it to
   the nearest controlled value.
3. **Field-by-field instructions.** Every field with its operational anchor, its
   controlled values, and its boundary against the adjacent value on the ladder.
4. **The ladders, spelled out.** Coverage, pairwise integration, and triadic
   integration, each with the rule that separates one rung from the next.
5. **The output contract.** One JSON object, exact keys, caps, and length
   allowances.
"""

import json

from bps_review.fulltext.coding import schema as S
from bps_review.fulltext.config import (
    ITEM_CAPS,
    MAX_NOTE_WORDS,
    MAX_QUOTE_WORDS,
    MAX_SUMMARY_WORDS,
)


REVIEW_CONTEXT = (
    "You are coding for an OSF-registered systematic review titled 'How the biopsychosocial model "
    "frames chronic pain research' (OSF DOI 10.17605/OSF.IO/T4FAM). The review asks how the "
    "biopsychosocial (BPS) model is actually operationalized in chronic pain review literature: how "
    "much biological, psychological, and social content each review carries, whether those domains "
    "are genuinely related to one another or merely listed side by side, which psychological concepts "
    "and theoretical frameworks recur, and which conceptual problems keep coming back when BPS is "
    "invoked. The central claim being tested is that BPS language is widespread while substantive "
    "three-domain integration is not, so the integration ladder is the most consequential judgement "
    "you will make."
)

TASK_STATEMENT = (
    "You are given the full text of one review article. Your task is a high-resolution conceptual "
    "coding: grade how deeply each of the three domains is treated, grade every pairwise and the "
    "triadic integration on an explicit ladder, and carry a verbatim quote for each of those "
    "judgements so a second coder can check it. This is not a screening task and not a summary task. "
    "Extraction completeness matters more than brevity: if the paper makes six separate cross-domain "
    "claims, record six integration items, not one."
)

CODING_PRINCIPLES = [
    "Code this paper only. Never import knowledge about the biopsychosocial model from the wider "
    "literature, from the models you know, or from other papers. If this paper does not say it, it is "
    "not coded.",
    "The word 'biopsychosocial' is never evidence of coverage. A review can use the label in every "
    "paragraph and still score a domain as absent. Coverage is judged on substantive domain content: "
    "named constructs, mechanisms, determinants, measures, or interventions belonging to that domain.",
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
    "Keep nuance rather than rounding it away. Concept and framework labels are free text: use the "
    "paper's own wording when no shorter label is faithful to it.",
    "The text may contain '[... omitted ...]' markers where a paragraph without conceptual content was "
    "removed to fit the window. Code only what is present, and never quote across an omission marker.",
    "An empty list is a legitimate coding. The caps are ceilings, never targets: return as many items "
    "as the paper genuinely supports, and return none when it supports none.",
    "Do not judge whether the paper is good, and do not score its relevance. Eligibility, integration "
    "depth, and synthesis priority are computed afterwards from the content you record.",
]

# --------------------------------------------------------------------------
# Operational anchors, one per coded field. These are the reliability layer.
# --------------------------------------------------------------------------
FIELD_INSTRUCTIONS: list[tuple[str, str]] = [
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
    ("pain_condition_detail",
     f"Free text, at most {MAX_NOTE_WORDS} words: the exact pain condition or conditions studied, in "
     "the paper's own words."),
    ("domain_coverage_bio",
     "Depth of biological content: anatomy, physiology, pathophysiology, nociception, central or "
     "peripheral sensitization, inflammation, imaging, genetics, pharmacology, tissue pathology."),
    ("domain_coverage_psych",
     "Depth of psychological content: cognition, affect, behaviour, beliefs, coping, catastrophizing, "
     "fear-avoidance, self-efficacy, depression, anxiety, acceptance, psychological treatment."),
    ("domain_coverage_social",
     "Depth of social content: work and occupational context, family and relationships, culture, "
     "socioeconomic position, healthcare system, social support, stigma, policy."),
    ("domain_evidence",
     "One item per domain you did not score as 'absent', carrying the passage that justifies the "
     "coverage level you gave it. Give domain, coverage_level (identical to the field above), "
     "constructs_named (the domain-specific constructs the paper actually names), evidence_verbatim, "
     "and section_located."),
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
     "claim_verbatim, mechanism_note (the pathway in your words, empty when none is given), "
     "section_located, and evidence_basis."),
    ("integration_mechanism_summary",
     f"At most {MAX_SUMMARY_WORDS} words, in your own words: the cross-domain pathways this paper "
     "actually proposes. Write 'none proposed' when the paper proposes none."),
    ("overall_balance",
     "Relative emphasis across the three domains, judged on how much of the paper each one occupies. "
     "'balanced' means no domain clearly dominates; 'dyadic' means two domains carry the paper and the "
     "third is marginal; 'unclear' when the emphasis cannot be read off the text."),
    ("bps_typology",
     "What this review does with the biopsychosocial model at full-text depth. 'true_integrative': all "
     "three domains present and genuinely related, with at least a descriptive triadic account. "
     "'multifactorial': all three domains substantively present but treated in parallel. 'pseudo_bps': "
     "BPS language with one or more domains thin or absent. 'rhetorical_bps': the label is used "
     "ceremonially and does no analytic work. 'narrow_despite_label': the paper claims a BPS frame but "
     "is in practice a single-domain review. 'unclear' only when the text genuinely does not allow a "
     "judgement."),
    ("concept_definitions_present",
     "Whether the review defines the psychological constructs it uses. 'yes' when the main constructs "
     "are defined or clearly operationalized; 'partial' when some are and others are only named; 'no' "
     "when constructs are used without any meaning being given."),
    ("psychological_concepts",
     "Every psychological construct the paper uses, as a list. For each give concept_label (the "
     "paper's own term), definitional_status ('formally_defined' when the paper says what it means, "
     "'operationalized_only' when the meaning is fixed only through a measure, 'named_only' when it is "
     "used without either), definition_verbatim (the passage that defines or operationalizes it, empty "
     "when there is none), and section_located."),
    ("theoretical_frameworks",
     "Every theoretical model or framework the paper invokes. For each give framework_label, role, and "
     "framework_verbatim."),
    ("conceptual_problems",
     "Conceptual problems this paper names or displays. Give problem_type, problem_verbatim (the "
     "passage that shows it, which for a displayed problem may be the passage where the gap is "
     "visible), and a short note. Return an empty list when you see none."),
    ("key_quotes",
     "The most conceptually load-bearing passages in the paper: the ones a reviewer would want to read "
     "first when writing the synthesis. Prefer passages that stand on their own. Give claim_verbatim, "
     "claim_type, section_located, and why_it_matters (one short sentence)."),
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
    "domain_coverage_bio": S.COVERAGE_OPTIONS,
    "domain_coverage_psych": S.COVERAGE_OPTIONS,
    "domain_coverage_social": S.COVERAGE_OPTIONS,
    "integration_bio_psych": S.PAIRWISE_INTEGRATION_OPTIONS,
    "integration_psych_social": S.PAIRWISE_INTEGRATION_OPTIONS,
    "integration_bio_social": S.PAIRWISE_INTEGRATION_OPTIONS,
    "integration_triadic": S.TRIADIC_INTEGRATION_OPTIONS,
    "overall_balance": S.BALANCE_OPTIONS,
    "bps_typology": S.TYPOLOGY_OPTIONS,
    "concept_definitions_present": S.DEFINITIONS_PRESENT_OPTIONS,
}

# The closed value lists inside the structured items.
ITEM_VALUE_LISTS: dict[str, dict[str, list[str]]] = {
    "integration_claims": {
        "domains_linked": S.DOMAIN_PAIR_OPTIONS,
        "integration_level": S.PAIRWISE_INTEGRATION_OPTIONS,
        "section_located": S.SECTION_OPTIONS,
        "evidence_basis": S.EVIDENCE_BASIS_OPTIONS,
    },
    "domain_evidence": {
        "domain": S.DOMAIN_OPTIONS,
        "coverage_level": S.COVERAGE_OPTIONS,
        "section_located": S.SECTION_OPTIONS,
    },
    "psychological_concepts": {
        "definitional_status": S.DEFINITIONAL_STATUS_OPTIONS,
        "section_located": S.SECTION_OPTIONS,
    },
    "theoretical_frameworks": {"role": S.FRAMEWORK_ROLE_OPTIONS},
    "conceptual_problems": {"problem_type": S.CONCEPTUAL_PROBLEM_OPTIONS},
    "key_quotes": {
        "claim_type": S.CLAIM_TYPE_OPTIONS,
        "section_located": S.SECTION_OPTIONS,
    },
}


SYSTEM_PROMPT = (
    "You are a meticulous systematic-review coder working on how the biopsychosocial model is "
    "operationalized in chronic pain reviews. You read one full-text article at a time and return a "
    "single strict JSON object that follows the given schema exactly. You quote the article verbatim, "
    "you never invent content, you never treat the word biopsychosocial as evidence of coverage, and "
    "you prefer the lower rung of a ladder over a generous reading. You return JSON only, with no "
    "prose, no markdown, and no code fences."
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
        elif field_name == "constructs_named":
            fields[field_name] = {"type": "list of short free-text labels"}
        else:
            fields[field_name] = {"type": "free text label"}
    return {"max_items": ITEM_CAPS.get(name), "item_fields": fields}


def build_schema_spec() -> dict:
    """The machine-readable half of the prompt: every field, its values, its caps."""
    spec: dict[str, object] = {}
    for field_name, instruction in FIELD_INSTRUCTIONS:
        entry: dict[str, object] = {"instruction": instruction}
        if field_name in CONTROLLED_VALUES:
            entry["values"] = CONTROLLED_VALUES[field_name]
        if field_name.startswith("domain_coverage_"):
            entry["ladder"] = LADDERS["domain_coverage"]
        if field_name in ("integration_bio_psych", "integration_psych_social", "integration_bio_social"):
            entry["ladder"] = LADDERS["pairwise_integration"]
        if field_name == "integration_triadic":
            entry["ladder"] = LADDERS["triadic_integration"]
        if field_name in S.ITEM_MODELS:
            entry.update(_item_spec(field_name))
        spec[field_name] = entry
    return spec


def build_prompt(record: dict, coding_text: str) -> str:
    """Assemble the full user prompt for one paper."""
    instructions = {
        "review_context": REVIEW_CONTEXT,
        "task": TASK_STATEMENT,
        "coding_principles": CODING_PRINCIPLES,
        "fields": build_schema_spec(),
        "output_contract": [
            "Return exactly one JSON object with the keys listed under 'fields', plus 'record_id'.",
            f"'record_id' must be exactly '{record['record_id']}'.",
            "Every list field must be present, as an array; use an empty array when the paper offers "
            "nothing for it. Never omit a key.",
            "Controlled fields must use one of the listed values verbatim, in lowercase as given.",
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
        "n_item_level_fields": sum(len(model.model_fields) for model in S.ITEM_MODELS.values()),
        "max_quote_words": MAX_QUOTE_WORDS,
        "item_caps": ITEM_CAPS,
        "ladders": {name: list(rungs) for name, rungs in LADDERS.items()},
        "system_prompt": SYSTEM_PROMPT,
    }
