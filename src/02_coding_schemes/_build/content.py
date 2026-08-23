# -*- coding: utf-8 -*-
"""
Single source of truth for the six BPS coding-scheme dossiers.

This module holds the enriched, expert-review-ready specification of each
coding scheme used in the systematic review "How the Biopsychosocial Model
Frames Chronic Pain Research" (OSF DOI 10.17605/OSF.IO/T4FAM).

The same content is rendered to three surfaces by build.py:
  - LaTeX  -> compiled PDF (formal dossier for sharing)
  - HTML   -> interactive evaluation surface (per scheme + aggregated)
  - README -> plain-text explanatory note (per scheme + directory index)

Design principles for the enrichment:
  1. Semantic quality. Every controlled value carries an operational anchor,
     positive indicators, negative indicators, and an explicit boundary rule
     against the adjacent value. This is what raises inter-rater reliability.
  2. Focus. Each dossier carries the purpose of the scheme and the scheme
     itself, and nothing else. Worked examples, reliability architecture, and
     documented divergences live in the pipeline documentation, not here,
     because an evaluator is being asked about the instrument.
  3. Provenance honesty. Everything described here is grounded in the code that
     runs and in the outputs it generates.

Status of every scheme in this release: DRAFT FOR EXPERT EVALUATION.
No em dashes are used anywhere in this content.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Project-level constants shared by every surface.
# --------------------------------------------------------------------------

PROJECT = {
    "title": "How the Biopsychosocial Model Frames Chronic Pain Research",
    "short": "BPS in Chronic Pain Research",
    "osf_doi": "10.17605/OSF.IO/T4FAM",
    "osf_url": "https://osf.io/t4fam",
    "osf_project": "https://osf.io/dwvru",
    "repo": "https://github.com/stvsever/SystematicReview_on_BioPsychoSocialModel_in_ChronicPainResearch",
    "status": "DRAFT FOR EXPERT EVALUATION",
    "status_long": (
        "These coding schemes are a working draft circulated for expert "
        "evaluation. They have not been applied to a final review corpus. "
        "The current manuscript is a test run that exercised an earlier, "
        "coarser generation of these schemes. The workflow itself has since "
        "been validated end to end in two cross-provider test runs, in which "
        "three large language models from three different providers applied "
        "the abstract-level and the full-text scheme independently and their "
        "agreement was quantified per coded field. The full run on the review "
        "corpus is deliberately held until this evaluation is complete."
    ),
    "version": "2.2-draft",
    "release_date": "2026-08-05",
    "review_scope": (
        "The team plans two parallel reviews: one on musculoskeletal chronic "
        "pain and one on neuropathic chronic pain. These coding schemes are a "
        "single uniform instrument for both. The pain-condition family "
        "(musculoskeletal or neuropathic) is the varying input that selects "
        "which records enter each review; the coding logic, value vocabularies, "
        "and anchors are shared so the two reviews stay directly comparable. "
        "Uniformity is kept wherever it is defensible. It is relaxed in exactly "
        "two places, where forcing it would distort the science: the routing "
        "flags that assign a record to a review (a musculoskeletal flag and a "
        "parallel neuropathic flag), and the biological subdomain ontology, "
        "which carries a shared core plus a musculoskeletal extension and a "
        "neuropathic extension because the biological mechanisms of the two "
        "pain families genuinely differ. The psychological and social layers, "
        "the integration ladder, the typology, and the concept taxonomy stay "
        "identical across both tracks."
    ),
    "test_run_model": "gemini-2.5-flash",
    "reviewers": [
        "Geert Crombez",
        "Christopher Eccleston",
        "Annick De Paepe",
        "Maya Braun",
        "Julie Dendauw",
        "Jose Luis Socorro Cumplido",
    ],
    "lead": "Stijn Van Severen",
}

RESEARCH_QUESTIONS = {
    "RQ1": "How is the BPS model operationalized in reviews of chronic pain?",
    "RQ2": "What is the scope, balance, and integration of biological, "
           "psychological, and social factors in musculoskeletal pain reviews?",
    "RQ3": "Which psychological concepts and theoretical frameworks are used "
           "in chronic pain literature?",
    "SQ1": "What conceptual problems (for example vague definitions or "
           "overlapping constructs) recur in reviews using the BPS model?",
}

# The three BPS domains carry a consistent accent colour across all surfaces.
DOMAIN_COLORS = {
    "biological": "#0E8F80",     # teal
    "psychological": "#6D5AE0",  # violet
    "social": "#D98016",         # amber
}


# --------------------------------------------------------------------------
# Small helpers to build value ladders without repeating structure.
# --------------------------------------------------------------------------

def v(value, anchor, pos=None, neg=None, boundary=None):
    """One controlled value with operational anchoring."""
    return {
        "value": value,
        "anchor": anchor,
        "pos": pos or [],
        "neg": neg or [],
        "boundary": boundary or "",
    }


def field(name, construct, values=None, notes=None, free_text=False,
          subfields=None, kind="", cap=None):
    """One coded field.

    ``subfields`` describes a structured extraction item (a list field whose
    entries are objects), ``kind`` labels the field for the reader (for example
    "extraction list" or "derived"), and ``cap`` is the maximum number of items
    the coder may return for a list field.
    """
    return {
        "name": name,
        "construct": construct,
        "values": values or [],
        "notes": notes or "",
        "free_text": free_text,
        "subfields": subfields or [],
        "kind": kind,
        "cap": cap,
    }


def sub(name, desc, values=None):
    """One field inside a structured extraction item."""
    return {"name": name, "desc": desc, "values": values or []}


# --------------------------------------------------------------------------
# Reusable value ladders (Stage 3 shares several of these).
# --------------------------------------------------------------------------

DOMAIN_COVERAGE_LADDER = [
    v("elaborated",
      "The domain is developed as a substantive analytic thread. Several "
      "distinct constructs or mechanisms are discussed, weighed, or "
      "synthesized rather than merely named.",
      pos=["A dedicated section or sustained multi-paragraph treatment",
           "Two or more distinct constructs named and related to one another",
           "Evidence is appraised, not only cited in passing"],
      neg=["A single passing sentence", "Only the domain umbrella word"],
      boundary="Prefer elaborated over mentioned when the domain carries its "
               "own argument rather than appearing as one item in a list."),
    v("mentioned",
      "The domain is explicitly present with at least one named construct, "
      "but it is not developed into a sustained analytic thread.",
      pos=["One or two named constructs", "A short paragraph or scattered sentences"],
      neg=["Only the umbrella word such as the bare token psychosocial"],
      boundary="Prefer mentioned over minimal when a concrete construct is "
               "named rather than only the domain umbrella term."),
    v("minimal",
      "The domain appears only as an umbrella label or a single incidental "
      "reference, with no concrete construct attached.",
      pos=["Bare token such as social factors with no elaboration",
           "One incidental clause"],
      neg=["Any named construct, which would qualify as mentioned"],
      boundary="Prefer minimal over absent when the domain word appears at all."),
    v("absent",
      "The domain is not represented in the substantive content of the "
      "review, even if a global BPS label is used elsewhere.",
      pos=["No domain-specific language beyond a global BPS label"],
      neg=["Any named construct or umbrella mention"],
      boundary="A review can carry a BPS label and still score a domain as "
               "absent. The label alone is never counted as coverage."),
]

PAIRWISE_INTEGRATION_LADDER = [
    v("mechanistic",
      "The review specifies a pathway or mechanism by which one domain acts "
      "on the other (a how, not only a that).",
      pos=["Named mediator, moderator, or physiological or behavioural pathway",
           "Language such as via, through, drives, sensitizes, amplifies"],
      neg=["Two domains listed as co-occurring with no linking process"],
      boundary="Mechanistic requires a stated process; if only a direction of "
               "effect is claimed without a pathway, use directional."),
    v("directional",
      "A directional or causal influence between the two domains is asserted, "
      "but no mechanism is given.",
      pos=["X predicts, increases, or worsens Y", "A stated arrow of influence"],
      neg=["A pathway is described, which would be mechanistic",
           "Mere co-mention with no direction"],
      boundary="Directional requires an arrow of effect; if the link is only "
               "an association or correlation, use descriptive."),
    v("descriptive",
      "The two domains are linked as associated or correlated, without "
      "direction or mechanism.",
      pos=["Associated with, related to, correlated with, linked to"],
      neg=["A stated direction of effect, which would be directional"],
      boundary="Descriptive requires an explicit relational claim; a bare "
               "co-occurrence in the same sentence is only mentioned."),
    v("mentioned",
      "Both domains appear near one another but the relationship is not "
      "characterized at all.",
      pos=["Two domains named in the same passage with no relational verb"],
      neg=["Any associational, directional, or mechanistic claim"],
      boundary="Prefer mentioned over none only when both domains are present "
               "in a shared context rather than in separate isolated sections."),
    v("none",
      "No relationship between the two domains is articulated anywhere in the "
      "review.",
      pos=["Domains handled in separate silos, or one domain absent"],
      neg=["Any co-located relational statement"],
      boundary="If either domain is absent, the pairwise code defaults to none."),
]

TRIADIC_INTEGRATION_LADDER = [
    v("mechanistic",
      "A genuinely three-domain mechanism is proposed in which biological, "
      "psychological, and social factors act on one another as a system.",
      pos=["A cross-domain loop or cascade involving all three domains",
           "All three domains appear as active parts of one explanation"],
      neg=["Two domains interact while the third is only listed"],
      boundary="Reserve for reasoning where removing any one domain breaks the "
               "proposed explanation."),
    v("descriptive",
      "All three domains are related to the outcome in an integrated narrative, "
      "but as a joint description rather than a specified mechanism.",
      pos=["A coherent three-domain account without a stated pathway"],
      neg=["A concrete cross-domain pathway, which would be mechanistic"],
      boundary="Descriptive triadic still requires all three domains to be in "
               "genuine relation, not merely co-present."),
    v("partial",
      "Two domains are integrated with each other while the third is present "
      "but only loosely attached.",
      pos=["A strong bio-psych link with social mentioned but not integrated"],
      neg=["All three integrated (descriptive or mechanistic)",
           "Only one domain present"],
      boundary="Partial marks a dyadic integration inside a nominally triadic "
               "frame."),
    v("none",
      "No three-domain integration is present; domains stand in parallel or "
      "one or more is absent.",
      pos=["Serial mention of separate domains", "Single-domain reasoning"],
      neg=["Any genuine multi-domain relational claim"],
      boundary="Serial listing of B, P, and S in turn is none, not descriptive."),
]


# --------------------------------------------------------------------------
# SCHEME 1 -- Stage 1 screening and eligibility decision scheme
# --------------------------------------------------------------------------

SCHEME_1 = {
    "id": "scheme_1",
    "num": 1,
    "title": "Stage 1 Screening and Eligibility Decision Scheme",
    "subtitle": "Title and abstract eligibility for the BPS chronic pain corpus",
    "tagline": "Rule-based provisional machine assist with mandatory human validation",
    "stage": "Stage 1",
    "stage_key": "screening",
    "meta": {
        "Workflow position": "Pre-extraction eligibility screen, run after "
            "search and deduplication and before Stage 2 abstract coding.",
        "Operational mode": "Deterministic rule set that emits a provisional "
            "decision, confidence, and reason. Every decision is validated by "
            "a human screener in Rayyan.",
        "Unit of analysis": "One bibliographic record (title, abstract, and "
            "publication metadata).",
        "Provenance basis": "Executable screening rules plus the OSF "
            "eligibility criteria.",
    },
    "rqs": ["Gatekeeper for RQ1, RQ2, RQ3 (defines the analysable corpus)"],
    "sources": [
        "src/01_protocol/decision_rules/screening_rules.md",
        "src/03_pipeline/bps_review/screening/rules.py",
        "src/09_review_stages/03_screening/README.md",
        "src/01_protocol/osf/OSF_registration_HTBMFCPR.md",
    ],
    "outputs": [
        "src/09_review_stages/03_screening/outputs/stage1_screening.csv",
        "src/09_review_stages/03_screening/audit/stage1_screening_summary.csv",
        "src/09_review_stages/03_screening/audit/reliability_report.csv",
    ],
    "sections": [
        {
            "kind": "prose",
            "id": "purpose",
            "title": "Purpose",
            "body": [
                "This scheme operationalizes title and abstract screening after "
                "search and deduplication. Its function is to decide whether a "
                "record enters the review corpus for downstream coding, using a "
                "human-validatable rule set centred on biopsychosocial language, "
                "chronic pain relevance, review design, and population eligibility.",
                "Because everything downstream inherits this decision, the scheme "
                "is deliberately conservative: it protects recall at the boundary "
                "and pushes genuinely ambiguous records into a borderline register "
                "rather than excluding them early.",
            ],
        },
        {
            "kind": "fields",
            "id": "decision-fields",
            "title": "Decision Fields and Controlled Values",
            "feedback": True,
            "field_feedback": True,
            "intro": "The scheme writes one decision bundle per record. The "
                     "controlled values below are the operational vocabulary.",
            "fields": [
                field("stage1_decision",
                      "Eligibility verdict for the record.",
                      values=[
                          v("include", "Meets every inclusion rule with no "
                            "triggered exclusion.",
                            pos=["Review design, BPS term, chronic pain, adult or "
                                 "mixed population, English, in window"],
                            neg=["Any hard exclusion trigger present"],
                            boundary="Any single hard exclusion overrides inclusion."),
                          v("exclude", "At least one hard exclusion rule fires "
                            "with high confidence.",
                            pos=["Clear primary study, wrong population, non-English"],
                            neg=["Only a soft or ambiguous signal, which is maybe"],
                            boundary="Use exclude only when the trigger is "
                                     "unambiguous; otherwise use maybe."),
                          v("maybe", "Borderline record retained for human "
                            "adjudication.",
                            pos=["Ambiguous review type, mixed acute and chronic, "
                                 "unclear age or duration"],
                            neg=["A clearly satisfied or clearly failed rule"],
                            boundary="maybe is the honest label for genuine "
                                     "uncertainty and must not be used to avoid a "
                                     "decision that the abstract actually supports."),
                          v("unclear", "Current executable fallback state for "
                            "ambiguous cases where maybe is not emitted.",
                            pos=["Record cannot be resolved from available metadata"],
                            neg=["A confident include or exclude"],
                            boundary="unclear is the state the executable rule set "
                                     "emits when a record cannot be resolved at all."),
                      ]),
                field("stage1_reason",
                      "Controlled reason attached to exclusions and unclear cases. "
                      "See the exclusion catalogue below."),
                field("stage1_confidence",
                      "Screener or rule confidence in the decision.",
                      values=[
                          v("high", "Decision rests on explicit, unambiguous "
                            "metadata (for example an explicit non-review "
                            "publication type)."),
                          v("medium", "Decision rests on strong but partly "
                            "inferential signals."),
                          v("low", "Decision rests on weak or conflicting signals; "
                            "always pair with a logged rationale."),
                      ]),
                field("stage1_screened_by",
                      "Provenance of the provisional decision. Current outputs "
                      "record codex_machine_assist before human validation."),
                field("stage1_screening_mode",
                      "Screening mode. Current outputs record "
                      "rule_based_provisional."),
            ],
        },
        {
            "kind": "fields",
            "id": "inclusion-logic",
            "title": "Inclusion Logic",
            "feedback": True,
            "field_feedback": True,
            "intro": "All of the following must hold for an include.",
            "fields": [
                field("Review design",
                      "The record is a review article or review-like evidence "
                      "synthesis (systematic, meta-analysis, scoping, umbrella, "
                      "narrative, realist, integrative, or expert review)."),
                field("BPS lexical trigger",
                      "The title or abstract explicitly contains a biopsychosocial "
                      "term: biopsychosocial, bio-psycho-social, or bio psycho "
                      "social."),
                field("Chronic pain focus",
                      "The focus concerns chronic pain, persistent pain, or a named "
                      "chronic pain condition."),
                field("Population",
                      "The population is adult or mixed-age rather than "
                      "pediatric-only."),
                field("Window and language",
                      "The record is within the operational search window and in "
                      "English."),
            ],
        },
        {
            "kind": "fields",
            "id": "exclusion-catalogue",
            "title": "Exclusion Catalogue and Implemented Reason Labels",
            "feedback": True,
            "field_feedback": True,
            "intro": "Each exclusion reason below is a controlled string with an "
                     "operational trigger.",
            "fields": [
                field("no biopsychosocial term in title/abstract",
                      "No explicit biopsychosocial term is present in the title or "
                      "abstract."),
                field("outside operational search window",
                      "Publication date falls before or after the operational "
                      "search dates configured in config/protocol.yaml."),
                field("protocol",
                      "Protocol papers without results are excluded."),
                field("commentary/editorial/letter",
                      "Commentary, editorial, and letter publication types are "
                      "excluded."),
                field("animal/non-human focus",
                      "Animal-only or non-human records are excluded unless a human "
                      "focus is explicit."),
                field("pediatric-only focus",
                      "Populations restricted to under 18 are excluded."),
                field("acute pain focus",
                      "Acute-only pain records are excluded when chronicity is not "
                      "also explicit."),
                field("chronic pain focus unclear",
                      "Pain is present but chronic pain relevance is insufficiently "
                      "clear."),
                field("non-English record",
                      "Records not in English are excluded."),
                field("review status unclear",
                      "The record cannot be confidently identified as a review or "
                      "evidence synthesis from title, abstract, or publication "
                      "metadata."),
            ],
        },
        {
            "kind": "prose",
            "id": "borderline",
            "title": "Borderline Handling",
            "feedback": True,
            "body": [
                "Ambiguous review type, mixed acute and chronic populations, "
                "unclear age group, or unclear pain duration should not be "
                "over-excluded. Borderline cases are logged with a rationale and "
                "lower confidence.",
                "Musculoskeletal ambiguity is intentionally carried forward. "
                "Stage 3 full-text adjudication resolves it rather than Stage 1 "
                "excluding too aggressively.",
            ],
        },
    ],
}


# --------------------------------------------------------------------------
# SCHEME 2 -- Stage 2 abstract-level structured coding scheme
# --------------------------------------------------------------------------

SCHEME_2 = {
    "id": "scheme_2",
    "num": 2,
    "title": "Stage 2 Abstract-Level Structured Coding Scheme",
    "subtitle": "Corpus-wide abstract coding of BPS usage, domains, and typology",
    "tagline": "Structured LLM-first coding with deterministic repair and "
               "rule-based fallback",
    "stage": "Stage 2",
    "stage_key": "abstract",
    "meta": {
        "Workflow position": "Abstract coding for every Stage 1 included record, "
            "before Stage 3 candidate selection.",
        "Operational mode": "Structured LLM coding with archived JSON batches, "
            "deterministic vocabulary normalization, and a rule-based fallback "
            "when model output is incomplete or unavailable.",
        "Unit of analysis": "One included review record, coded from title, "
            "abstract, publication types, and journal metadata only.",
        "Provenance basis": "The actual Stage 2 output schema in "
            "stage2_abstract_coding.csv.",
    },
    "rqs": ["RQ1 (BPS operationalization)", "RQ3 (concepts and frameworks)",
            "SQ1 (conceptual problems)", "Feeds the Stage 3 candidate gate"],
    "sources": [
        "src/01_protocol/codebooks/stage2_codebook.md",
        "src/09_review_stages/04_extraction/codebooks/stage2_codebook.csv",
        "src/03_pipeline/bps_review/extraction/stage2.py",
        "src/03_pipeline/bps_review/extraction/llm_stage2.py",
        "src/09_review_stages/04_extraction/outputs/stage2_abstract_coding.csv",
        "src/09_review_stages/04_extraction/outputs/stage2_llm_structured_coding.csv",
        "src/09_review_stages/04_extraction/outputs/llm_stage2_structured_batches.jsonl",
    ],
    "outputs": [
        "src/09_review_stages/04_extraction/outputs/stage2_abstract_coding.csv",
        "src/09_review_stages/04_extraction/forms/stage2_double_code_subset.csv",
        "src/09_review_stages/04_extraction/outputs/stage2_objective_llm_assist.csv",
        "src/09_review_stages/04_extraction/outputs/llm_objective_pilot.json",
    ],
    "sections": [
        {
            "kind": "prose",
            "id": "purpose",
            "title": "Purpose",
            "body": [
                "This scheme standardizes abstract-level extraction for all "
                "eligible chronic pain reviews. It is the main corpus-wide coding "
                "layer used to describe review characteristics, classify the "
                "function of biopsychosocial language, detect biological, "
                "psychological, and social content, flag conceptual problems, and "
                "generate a provisional biopsychosocial typology for downstream "
                "synthesis.",
                "It also sets the Stage 3 candidate gate: a record coded as "
                "musculoskeletal or as unspecified-but-not-excludable is carried "
                "forward for full-text work.",
            ],
        },
        {
            "kind": "list",
            "id": "metadata",
            "title": "Carried-Through Metadata Fields",
            "intro": "Descriptive fields retained from earlier stages for synthesis.",
            "items": [
                "record_id, database, pmid, pmcid, doi",
                "title, abstract (primary coding text)",
                "year, journal, authors, country_contact_author",
                "publication_types (source publication-type metadata)",
                "objective_text (objective sentence extracted from the abstract)",
                "screening_status, screening_reason (inherited from Stage 1)",
            ],
        },
        {
            "kind": "fields",
            "id": "coded-fields",
            "title": "Coded Fields and Controlled Values",
            "feedback": True,
            "field_feedback": True,
            "intro": "The operational Stage 2 vocabulary. Every value carries an "
                     "operational anchor, and adjacent values carry an explicit "
                     "boundary rule, because that is what raises agreement "
                     "between two coders.",
            "fields": [
                field("review_type",
                      "Evidence-synthesis design as stated in the abstract or "
                      "publication metadata.",
                      values=[
                          v("systematic review", "Explicit systematic methods "
                            "(protocol, search, screening)."),
                          v("meta-analysis", "Quantitative pooling of effect sizes."),
                          v("network meta-analysis", "Multiple-treatment "
                            "comparison with indirect evidence."),
                          v("umbrella review", "Review of reviews."),
                          v("scoping or mapping review", "Breadth-oriented mapping "
                            "of a field without pooled effects."),
                          v("rapid review", "Streamlined systematic methods."),
                          v("realist review", "Theory-driven mechanism review."),
                          v("integrative review", "Mixed evidence integration."),
                          v("narrative or expert review", "Non-systematic expert "
                            "synthesis."),
                          v("other evidence synthesis", "A synthesis type not "
                            "captured above."),
                          v("unclear", "Design cannot be determined from the "
                            "abstract."),
                      ],
                      notes="Boundary: prefer the most specific applicable label; "
                            "meta-analysis outranks systematic review when pooling "
                            "is explicit."),
                field("objective_category",
                      "Primary stated purpose of the review.",
                      values=[
                          v("conceptual", "Purpose is definitional, theoretical, "
                            "or about operationalizing a model.",
                            pos=["framework, concept, operationalize, model"],
                            neg=["A treatment-effect question, which is clinical"]),
                          v("clinical", "Purpose concerns treatment, management, "
                            "rehabilitation, or care.",
                            pos=["treatment, intervention, management, rehabilitation"]),
                          v("methodological", "Purpose concerns measurement, "
                            "psychometrics, or assessment tools.",
                            pos=["measurement, psychometric, instrument, assessment"]),
                          v("epidemiological", "Purpose concerns prevalence, "
                            "incidence, risk factors, or associations.",
                            pos=["prevalence, incidence, risk factor, association"]),
                          v("mixed", "Two or more purposes carry comparable weight."),
                          v("unclear", "Purpose cannot be determined."),
                      ],
                      notes="Boundary: choose the dominant purpose; use mixed only "
                            "when two purposes are genuinely co-primary."),
                field("objective_category_source",
                      "Whether the objective classification came from the "
                      "structured LLM, a repaired LLM batch, or the deterministic "
                      "fallback."),
                field("icd11_pain_category",
                      "ICD-11 aligned pain category inferred from the abstract.",
                      values=[
                          v("chronic secondary musculoskeletal pain", "Low back, "
                            "neck, osteoarthritis, whiplash, and similar."),
                          v("chronic neuropathic pain", "Neuropathic, CRPS, "
                            "radiculopathy."),
                          v("chronic cancer-related pain", "Cancer or oncology pain."),
                          v("chronic postsurgical or posttraumatic pain",
                            "Post-surgical or post-traumatic persistent pain."),
                          v("chronic secondary headache or orofacial pain",
                            "Headache, migraine, orofacial, TMD."),
                          v("chronic secondary visceral pain", "Visceral, "
                            "abdominal, pelvic pain."),
                          v("chronic primary pain", "Fibromyalgia and other "
                            "primary pain syndromes."),
                          v("mixed or unspecified chronic pain", "Chronic pain "
                            "with no single dominant site."),
                          v("unclear", "Category cannot be inferred."),
                      ],
                      notes="Abstract-based and allowed to remain unclear. "
                            "Musculoskeletal and mixed-or-unspecified both feed the "
                            "Stage 3 candidate gate."),
                field("musculoskeletal_flag",
                      "Whether the review concerns musculoskeletal pain. Routes "
                      "records into the musculoskeletal review.",
                      values=[v("yes", "Musculoskeletal focus is explicit."),
                              v("no", "Clearly non-musculoskeletal."),
                              v("unclear", "Cannot be determined; carried forward.")]),
                field("neuropathic_flag",
                      "Whether the review concerns neuropathic pain. Parallel to "
                      "musculoskeletal_flag; routes records into the neuropathic "
                      "review. A record may set both flags (for example a mixed "
                      "review) or neither.",
                      values=[v("yes", "Neuropathic focus is explicit (for "
                                "example neuropathy, radiculopathy, CRPS, "
                                "postherpetic or diabetic neuropathic pain)."),
                              v("no", "Clearly non-neuropathic."),
                              v("unclear", "Cannot be determined; carried forward.")]),
                field("stage3_track",
                      "Which review or reviews the record is routed to. Derived "
                      "from the two flags; a record can belong to both pools.",
                      values=[v("musculoskeletal", "Musculoskeletal review only."),
                              v("neuropathic", "Neuropathic review only."),
                              v("both", "Enters both pools (mixed or unresolved "
                                "pain family)."),
                              v("none", "Neither track; not a Stage 3 candidate.")]),
                field("bps_mention_location",
                      "Where the BPS term appears.",
                      values=[v("title only", "Only in the title."),
                              v("abstract only", "Only in the abstract."),
                              v("title and abstract", "In both."),
                              v("unclear", "Location cannot be resolved.")]),
                field("bps_function",
                      "The rhetorical and analytic work the BPS label performs. "
                      "This is the central RQ1 field and the finest-grained one.",
                      values=[
                          v("explanatory framework", "BPS is used as a model that "
                            "explains pain or pain-related disability.",
                            pos=["explains, accounts for, mechanism of"],
                            neg=["Only justifies multimodal treatment, which is "
                                 "intervention rationale"],
                            boundary="Requires explanatory intent, not only "
                                     "endorsement of a model's existence."),
                          v("intervention rationale", "BPS mainly justifies "
                            "multimodal or interdisciplinary treatment.",
                            pos=["supports multidisciplinary care"],
                            neg=["Explains pain aetiology, which is explanatory "
                                 "framework"]),
                          v("organizing principle", "BPS structures the scope or "
                            "categories of the review without specifying "
                            "integration mechanisms.",
                            pos=["We organize factors into bio, psycho, social"],
                            neg=["A stated cross-domain mechanism"]),
                          v("justification", "BPS justifies the relevance or "
                            "importance of the review topic."),
                          v("background framing", "BPS sets context in the "
                            "background without analytic use."),
                          v("conclusion", "BPS appears mainly in the concluding "
                            "claims."),
                          v("policy/practice implication", "BPS frames a policy or "
                            "practice recommendation."),
                          v("rhetorical label", "BPS is invoked ceremonially, "
                            "aspirationally, or symbolically with no substantive "
                            "analytic work.",
                            pos=["a biopsychosocial approach is needed, with no "
                                 "follow-through"],
                            neg=["Any substantive explanatory or organizing use"]),
                          v("unclear", "Function cannot be determined."),
                      ]),
                field("bio_mentioned / psych_mentioned / social_mentioned",
                      "Binary presence of each domain in the abstract.",
                      values=[v("yes", "The domain is present beyond the mere BPS token."),
                              v("no", "The domain is not present.")],
                      notes="The field is binary on purpose: a domain is either "
                            "substantively present or it is not, and the mere BPS "
                            "token never counts as presence."),
                field("quality_assessment_reported",
                      "Whether the abstract reports a risk-of-bias or quality "
                      "assessment.",
                      values=[v("yes", "AMSTAR, risk of bias, or quality "
                                "assessment named."),
                              v("no", "Not reported."),
                              v("unclear", "Cannot be determined.")]),
                field("psychological_concepts_detected",
                      "Normalized list of psychological concepts from title and "
                      "abstract. Feeds Scheme 5.", free_text=True),
                field("theoretical_frameworks_detected",
                      "Normalized list of named frameworks or model labels.",
                      free_text=True),
                field("conceptual_problem_flags",
                      "Provisional abstract-level conceptual problems.",
                      values=[
                          v("vague_definition", "BPS or a key construct is used "
                            "without a definition."),
                          v("tokenistic_bps", "BPS appears as a label with no "
                            "analytic follow-through."),
                          v("missing_social", "Social domain absent despite a BPS "
                            "claim."),
                          v("missing_biology", "Biological domain absent despite a "
                            "BPS claim."),
                          v("mechanistic_absence", "No cross-domain mechanism is "
                            "offered."),
                          v("construct_overlap", "Constructs are used "
                            "interchangeably or with blurred boundaries."),
                          v("parallel_listing_without_integration", "Domains are "
                            "listed side by side with no integration."),
                          v("none", "No inferable conceptual problem."),
                      ]),
                field("provisional_typology",
                      "Abstract-level BPS signal, refined later at Stage 3.",
                      values=[
                          v("potential integrative signal", "All three domains "
                            "substantively present with a cross-domain signal."),
                          v("multifactorial signal", "All three domains present "
                            "but mainly parallel."),
                          v("pseudo-bps or partial signal", "One or more core "
                            "domains thin or absent despite BPS language."),
                          v("rhetorical label signal", "BPS mainly symbolic or "
                            "confined to framing or conclusion."),
                      ]),
                field("stage3_candidate / stage3_priority",
                      "Whether and how urgently the record proceeds to Stage 3.",
                      values=[v("yes / no", "Candidate flag."),
                              v("high / medium / low", "Retrieval and coding "
                                "priority.")]),
                field("coding_rationale",
                      "One-sentence rationale supporting the coding bundle.",
                      free_text=True),
                field("coding_method / llm_model",
                      "Operational provenance of the row (llm_structured, "
                      "llm_batch_fallback, rule_based_fallback) and the model "
                      "identifier when the LLM stage was used."),
            ],
        },
        {
            "kind": "list",
            "id": "prompt-rules",
            "title": "Prompt Rules Used in the Structured LLM Stage",
            "feedback": True,
            "intro": "These constraints are embedded in the structured prompt and "
                     "are the operational meaning of the codes above.",
            "items": [
                "Code only from title, abstract, publication types, and journal "
                "metadata. Do not infer content that is absent from the record.",
                "Do not treat the single lexical token biopsychosocial as proof "
                "that all three domains are substantively present.",
                "Use explanatory framework only when BPS explicitly explains pain "
                "or pain-related disability.",
                "Use intervention rationale when BPS mainly justifies multimodal "
                "or interdisciplinary treatment.",
                "Use organizing principle when BPS structures scope or categories "
                "without integration mechanisms.",
                "Use rhetorical label when BPS is ceremonial, aspirational, or "
                "symbolic with no substantive analytic work.",
                "Set stage3_candidate to yes for musculoskeletal or neuropathic "
                "reviews, and for mixed or unspecified chronic pain reviews where "
                "either relevance cannot be ruled out. The two planned reviews "
                "(musculoskeletal and neuropathic) draw from this same candidate "
                "pool, routed by pain-condition family.",
                "Use potential integrative signal only when all three core domains "
                "are substantively present and the abstract signals cross-domain "
                "explanation or organization beyond simple listing.",
                "Return conceptual_problem_flags as none only when no inferable "
                "conceptual problem is present.",
            ],
        },
        {
            "kind": "list",
            "id": "fallback",
            "title": "Deterministic Repair and Fallback Logic",
            "intro": "How the pipeline guarantees a complete, in-vocabulary row.",
            "items": [
                "Missing or malformed model responses are repaired against the "
                "fixed vocabularies above.",
                "Aliases such as scoping review or narrative review are normalized "
                "to the implemented labels.",
                "stage3_candidate is forced to yes whenever musculoskeletal_flag "
                "or neuropathic_flag is yes or unclear, and stage3_track is set "
                "from whichever flags fire (musculoskeletal, neuropathic, or "
                "both).",
                "Conceptual problem flags are post-derived from typology, BPS "
                "function, and missing domains: rhetorical usage triggers "
                "tokenistic_bps and vague_definition; parallel non-mechanistic "
                "usage triggers parallel_listing_without_integration and "
                "mechanistic_absence; absent biological or social content triggers "
                "missing_biology or missing_social.",
                "When the LLM stage fails, the rule-based fallback still generates "
                "review type, objective category, ICD-11 category, BPS function, "
                "domain mention flags, concept detections, and provisional Stage 3 "
                "priority.",
            ],
        },
    ],
}


# --------------------------------------------------------------------------
# Shared value lists used inside the Stage 3 structured extraction items.
# --------------------------------------------------------------------------

SECTION_VALUES = [
    "abstract", "introduction", "methods", "results", "discussion", "conclusion",
    "table or figure", "other", "unclear",
]

EVIDENCE_BASIS_VALUES = [
    "asserted", "theorized", "empirically_supported", "empirically_contested",
    "cited_from_other_work", "clinical_observation", "other", "unclear",
]

FACTOR_ROLE_VALUES = [
    "determinant or risk factor", "protective factor", "mediator", "moderator",
    "outcome", "correlate", "treatment target", "intervention component",
    "contextual condition", "descriptive theme", "other", "unclear",
]

BPS_FUNCTION_VALUES = [
    "explanatory framework", "intervention rationale", "organizing principle",
    "justification", "background framing", "conclusion",
    "policy or practice implication", "rhetorical label",
    "critique or problematization", "operational definition", "unclear",
]


# --------------------------------------------------------------------------
# SCHEME 3 -- Stage 3 full-text deep coding scheme
# --------------------------------------------------------------------------

SCHEME_3 = {
    "id": "scheme_3",
    "num": 3,
    "title": "Stage 3 Full-Text Deep Coding Scheme",
    "subtitle": "High-resolution full-text extraction of biopsychosocial usage, "
                "factors, integration, and concepts, for the musculoskeletal and "
                "neuropathic reviews",
    "tagline": "Graded ladders with a quoted passage behind every rung, plus a "
               "named-item extraction layer that supplies the nodes and edges of "
               "the biopsychosocial ontology",
    "stage": "Stage 3",
    "stage_key": "fulltext",
    "meta": {
        "Workflow position": "Full-text coding after Stage 3 candidate "
            "identification and retrieval triage. It is the pass that feeds the "
            "synthesis: the domain-balance tables, the integration analysis, and "
            "the concept map.",
        "Operational mode": "One article per request, structured JSON output, "
            "deterministic vocabulary repair, item caps, and an explicit failure "
            "row when a model returns no usable coding. Nothing is fabricated to "
            "fill a gap. Human adjudication remains the final authority on "
            "eligibility, as the registration specifies.",
        "Unit of analysis": "One retrieved full-text review, coded against the "
            "complete text. Every extracted item carries a verbatim quote and the "
            "section it came from.",
        "Evidence rule": "Quotes are checked against the source text after the "
            "run. A quote that cannot be found in the article is reported as "
            "unverified rather than accepted.",
        "Resolution": "Thirteen structured extraction lists, seven open free-text "
            "lists, 82 fields inside the list items, and a ceiling of 116 "
            "extracted items per coding. Presence is never the answer; the answer "
            "is which ones.",
        "Provenance basis": "The implementation in bps_review/fulltext (schema, "
            "vocabulary, prompt, derivations, runner) and the human coding "
            "template generated by stage3_prep.py.",
    },
    "rqs": ["RQ1 (what the BPS label actually does, passage by passage)",
            "RQ2 (scope, balance, integration, with named factors on both ends of "
            "every link)",
            "RQ3 (concepts, definitions, hierarchical and semantic relations, "
            "frameworks, measures)",
            "SQ1 (conceptual problems, with the constructs they concern)"],
    "sources": [
        "src/03_pipeline/bps_review/fulltext/coding/schema.py",
        "src/03_pipeline/bps_review/fulltext/coding/vocabulary.py",
        "src/03_pipeline/bps_review/fulltext/coding/prompt.py",
        "src/03_pipeline/bps_review/fulltext/coding/derive.py",
        "src/03_pipeline/bps_review/fulltext/coding/runner.py",
        "src/03_pipeline/bps_review/fulltext/analysis/integrity.py",
        "src/01_protocol/codebooks/stage3_codebook.md",
        "src/09_review_stages/04_extraction/codebooks/stage3_codebook.csv",
        "src/09_review_stages/04_extraction/forms/stage3_fulltext_coding_template.csv",
    ],
    "outputs": [
        "src/05_test_runs/tests/02_pilot_fulltext/02_model_codings/01_codings/01_one_row_per_paper_and_provider.csv",
        "src/05_test_runs/tests/02_pilot_fulltext/02_model_codings/02_extracted_items/00_all_categories/01_one_row_per_item.csv",
        "src/05_test_runs/tests/02_pilot_fulltext/03_reliability/",
        "src/09_review_stages/04_extraction/forms/stage3_fulltext_coding_template.csv",
        "src/09_review_stages/04_extraction/outputs/stage3_candidate_manifest.csv",
    ],
    "sections": [
        {
            "kind": "prose",
            "id": "purpose",
            "title": "Purpose",
            "body": [
                "This scheme is the full-text deep coding framework for Stage 3 "
                "candidate reviews. It is applied as one uniform instrument to "
                "both planned reviews: the musculoskeletal chronic pain review "
                "and the neuropathic chronic pain review. The pain-condition "
                "family is the varying input that decides which records each "
                "review reads; the coding fields, value vocabularies, and anchors "
                "are identical across both tracks so the two reviews stay "
                "directly comparable.",
                "It does two things at once. It grades: how deeply each domain is "
                "treated, and how each pair of domains and the triad are "
                "integrated, on explicit ladders with a quoted passage behind "
                "every rung. And it extracts: the specific things a review names. "
                "The extraction half is the larger one. Recording that a domain is "
                "present is not a coding; recording which biological, "
                "psychological, social, lifestyle, and existential factors carry "
                "it, what role each plays, and on the strength of which sentence, "
                "is.",
                "Resolution is the point. A single article routinely yields "
                "between thirty and seventy extracted items: passages where the "
                "biopsychosocial label does work, factors per domain, "
                "psychological constructs with their definitional status, "
                "relations drawn between constructs, frameworks, instruments, "
                "conceptual problems, and quotable claims. Those items are the "
                "nodes and edges of the biopsychosocial ontology the synthesis "
                "builds, so they have to be nameable, countable, and traceable to "
                "a passage.",
                "Every extraction list carries a maximum number of items. The cap "
                "is a ceiling and never a target: a list is left empty when the "
                "article offers nothing of that kind, and an empty list is itself "
                "a coding, not a gap.",
                "Stage 3 is where the review's central claim is tested: does a "
                "BPS-labelled review actually integrate the three domains, and if "
                "so, how, between which factors, and by what mechanism.",
            ],
        },
        {
            "kind": "list",
            "id": "metadata",
            "title": "Carried-Through Metadata Fields",
            "intro": "Descriptive fields retained from earlier stages and from "
                     "retrieval, not coded by the model.",
            "items": [
                "record_id, abstract_record_id, pmid, pmcid, doi",
                "title, journal, year, authors",
                "n_sections, section_titles, body_chars (what was available to read)",
                "coder_id, coding_method, llm_model, adjudication_status "
                "(provenance of the row)",
            ],
        },
        {
            "kind": "fields",
            "id": "fields-context",
            "title": "A. Source, Context, and Routing",
            "feedback": True,
            "field_feedback": True,
            "intro": "What kind of source this is, which review it belongs to, "
                     "and which population, condition, and setting it concerns. "
                     "These are the registration's contextual variables, now read "
                     "from the full text rather than inferred from an abstract.",
            "fields": [
                field("review_track",
                      "Which review this coded record belongs to. The coding "
                      "fields are uniform across both; the track only tunes which "
                      "biological ontology extension the coder reads the "
                      "biological domain against.",
                      values=[v("musculoskeletal", "Low back, neck, "
                                "osteoarthritis, fibromyalgia, shoulder, and "
                                "similar."),
                              v("neuropathic", "Painful neuropathy, radicular "
                                "pain, post-herpetic neuralgia, and similar."),
                              v("mixed_or_other", "Several pain families are "
                                "genuinely covered, or the condition belongs to "
                                "neither."),
                              v("unclear", "The paper does not say.")]),
                field("source_type",
                      "The evidence-synthesis design, read from how the paper "
                      "describes itself in its abstract and methods.",
                      values=[
                          v("systematic review", "Explicit systematic methods."),
                          v("meta-analysis", "Quantitative pooling of effect sizes.",
                            boundary="Outranks systematic review when pooling is "
                                     "explicit."),
                          v("network meta-analysis", "Multiple-treatment comparison "
                            "with indirect evidence."),
                          v("umbrella review", "Review of reviews."),
                          v("scoping or mapping review", "Breadth-oriented mapping."),
                          v("rapid review", "Streamlined systematic methods."),
                          v("realist review", "Theory-driven mechanism review."),
                          v("integrative review", "Mixed evidence integration."),
                          v("narrative or expert review", "Non-systematic expert "
                            "synthesis."),
                          v("clinical guideline or consensus statement", "A guideline "
                            "or a formal consensus."),
                          v("other evidence synthesis", "A synthesis type none of "
                            "the values above describes."),
                          v("primary study", "Reads as a primary study rather than "
                            "a synthesis.",
                            boundary="An eligibility signal: the registration "
                                     "excludes primary studies, so this routes the "
                                     "record to human adjudication."),
                          v("unclear", "The design cannot be determined."),
                      ]),
                field("icd11_pain_category",
                      "The ICD-11 aligned pain category, now read from the full "
                      "text. Recording it again at Stage 3 makes the abstract-level "
                      "classification checkable rather than final.",
                      values=[
                          v("chronic secondary musculoskeletal pain", "Low back, "
                            "neck, osteoarthritis, whiplash, and similar."),
                          v("chronic neuropathic pain", "Neuropathic, CRPS, "
                            "radiculopathy."),
                          v("chronic cancer-related pain", "Cancer or oncology pain."),
                          v("chronic postsurgical or posttraumatic pain",
                            "Persistent pain after surgery or trauma."),
                          v("chronic secondary headache or orofacial pain",
                            "Headache, migraine, orofacial, TMD."),
                          v("chronic secondary visceral pain", "Visceral, abdominal, "
                            "pelvic pain."),
                          v("chronic primary pain", "Fibromyalgia and other primary "
                            "pain syndromes."),
                          v("mixed or unspecified chronic pain", "Several categories "
                            "genuinely covered."),
                          v("unclear", "The category cannot be inferred."),
                      ]),
                field("population",
                      "The population the reviewed evidence concerns.",
                      values=[v("adult", "Adults, the eligible population."),
                              v("older adult", "Explicitly an older population, "
                                "foregrounded by the paper."),
                              v("mixed ages", "Adults and younger participants both "
                                "included."),
                              v("pediatric", "Children or adolescents only.",
                                boundary="An exclusion signal under the "
                                         "registration."),
                              v("unclear", "The paper does not report it."),
                              v("not applicable", "A purely theoretical paper with "
                                "no population.")]),
                field("care_setting",
                      "The care setting the paper concerns, when it reports one. "
                      "The registration lists setting among the contextual "
                      "variables to extract.",
                      values=[v("primary care", "General practice and first-line care."),
                              v("secondary or tertiary specialist care", "Specialist "
                                "clinics, pain centres, hospital care."),
                              v("rehabilitation or multidisciplinary programme",
                                "Rehabilitation and multimodal programmes."),
                              v("occupational or workplace", "Workplace and "
                                "occupational health settings."),
                              v("community or population", "Community, population, "
                                "or public-health settings."),
                              v("mixed", "Several settings genuinely covered."),
                              v("not reported", "The paper does not say.",
                                boundary="The honest answer for most reviews, and "
                                         "preferred over a guess.")]),
                field("primary_discipline",
                      "The disciplinary home of the paper, read from the journal, "
                      "the framing, and the vocabulary rather than from author "
                      "affiliations alone.",
                      values=[v("physiotherapy or rehabilitation", ""),
                              v("clinical or health psychology", ""),
                              v("rheumatology or orthopaedics", ""),
                              v("pain medicine or anaesthesiology", ""),
                              v("neurology or neuroscience", ""),
                              v("nursing", ""),
                              v("general or family medicine", ""),
                              v("public health or epidemiology", ""),
                              v("multidisciplinary", "The paper is genuinely written "
                                "across disciplines.",
                                boundary="Multidisciplinary describes the writing, "
                                         "not the author list."),
                              v("other", "A discipline none of the values above "
                                "describes."),
                              v("unclear", "It cannot be read off the paper.")],
                      notes="Discipline is a covariate for the synthesis: whether "
                            "biopsychosocial usage differs by field is one of the "
                            "descriptive questions the corpus can answer."),
                field("pain_condition_detail",
                      "The exact pain condition or conditions studied, in the "
                      "paper's own words. At most 40 words.",
                      free_text=True),
                field("pain_conditions",
                      "The specific pain conditions named, as a list. Preferred "
                      "labels exist (chronic low back pain, knee osteoarthritis, "
                      "painful diabetic neuropathy, and others) and the paper's "
                      "own wording is used whenever it is more precise.",
                      free_text=True, kind="open list", cap=6),
                field("context_note",
                      "The cultural, geographic, or healthcare-system context, "
                      "when the paper states one. At most 40 words, empty when it "
                      "does not.",
                      free_text=True),
                field("quality_assessment_reported",
                      "Whether the paper reports a formal quality or risk-of-bias "
                      "assessment of the evidence it reviews.",
                      values=[v("yes", "An appraisal is reported."),
                              v("no", "No appraisal is reported."),
                              v("unclear", "It cannot be determined.")],
                      notes="Descriptive only. The registration is explicit that "
                            "this review does not appraise the methodological "
                            "quality of the reviews it studies."),
                field("quality_assessment_tools",
                      "The appraisal tools named (AMSTAR, AMSTAR-2, ROBIS, GRADE, "
                      "Cochrane risk of bias, and others). Empty when none is named.",
                      free_text=True, kind="open list", cap=4),
            ],
        },
        {
            "kind": "fields",
            "id": "fields-bps",
            "title": "B. What the Biopsychosocial Label Does (RQ1)",
            "feedback": True,
            "field_feedback": True,
            "intro": "The primary research question of the review, coded at "
                     "passage level. The registration asks for the location and "
                     "the function of the biopsychosocial mention; at full text "
                     "that becomes an inventory, because one paper routinely uses "
                     "the label for two or three different purposes in different "
                     "sections, and which purposes it combines is the finding.",
            "fields": [
                field("bps_label_used",
                      "Which biopsychosocial vocabulary the paper actually uses.",
                      values=[
                          v("explicit_bps_term", "The words biopsychosocial or "
                            "bio-psycho-social appear somewhere in the text."),
                          v("variant_term_only", "Only a neighbouring term appears "
                            "(psychosocial, multidimensional, multifactorial, "
                            "holistic).",
                            boundary="Prefer explicit_bps_term whenever the full "
                                     "term appears at all, even once."),
                          v("domain_language_only", "The domains are discussed with "
                            "no model label at all."),
                          v("absent", "Neither the label nor domain language appears."),
                      ]),
                field("bps_primary_function",
                      "The single dominant work the label does, judged over the "
                      "paper as a whole. Shares its vocabulary with the "
                      "abstract-level scheme so the two readings of a record are "
                      "directly comparable.",
                      values=[
                          v("explanatory framework", "BPS is used as a model that "
                            "explains pain or pain-related disability.",
                            pos=["explains, accounts for, mechanism of"],
                            neg=["Only justifies multimodal treatment, which is "
                                 "intervention rationale"],
                            boundary="Requires explanatory intent, not only "
                                     "endorsement of the model's existence."),
                          v("intervention rationale", "BPS mainly justifies "
                            "multimodal or interdisciplinary treatment."),
                          v("organizing principle", "BPS structures the scope or "
                            "the categories of the review without specifying "
                            "integration mechanisms.",
                            pos=["We organize factors into bio, psycho, social"],
                            neg=["A stated cross-domain mechanism"]),
                          v("justification", "BPS justifies the relevance or the "
                            "importance of the topic."),
                          v("background framing", "BPS sets context in the "
                            "background without analytic use."),
                          v("conclusion", "BPS appears mainly in the concluding "
                            "claims."),
                          v("policy or practice implication", "BPS frames a policy "
                            "or practice recommendation."),
                          v("rhetorical label", "BPS is invoked ceremonially, "
                            "aspirationally, or symbolically with no substantive "
                            "analytic work.",
                            pos=["a biopsychosocial approach is needed, with no "
                                 "follow-through"]),
                          v("critique or problematization", "The paper argues about "
                            "the model itself: what it leaves out, what it cannot "
                            "explain, how it is misused.",
                            boundary="New at full-text level. These papers are the "
                                     "most informative for the review and are "
                                     "almost invisible in an abstract."),
                          v("operational definition", "The paper turns the model "
                            "into the variables it actually codes or measures.",
                            boundary="New at full-text level. Reserve it for a "
                                     "paper that operationalizes the model, not "
                                     "one that merely organizes prose by domain."),
                          v("unclear", "The function cannot be determined."),
                      ]),
                field("bps_functions_present",
                      "Every function the label performs anywhere in the paper, "
                      "as a multi-label list from the same vocabulary. A paper "
                      "routinely does two or three of these at once.",
                      values=[v(value, "") for value in BPS_FUNCTION_VALUES],
                      kind="multi-label list", cap=6,
                      notes="This is the field that answers RQ1 as a distribution "
                            "rather than as a single label. A paper's usage "
                            "collapsed into one function loses the pattern the "
                            "review is looking for."),
                field("bps_definition_status",
                      "How the paper handles the meaning of the model itself.",
                      values=[
                          v("formally_defined", "The paper states what the model "
                            "means."),
                          v("described_informally", "The meaning is carried by "
                            "description rather than by a definition."),
                          v("cited_only", "A citation stands in for a definition."),
                          v("undefined", "The label is used with no meaning given "
                            "anywhere.",
                            boundary="A finding, not a coding failure. Papers that "
                                     "use the model without ever saying what it is "
                                     "are exactly what this review is about."),
                      ]),
                field("bps_model_variants",
                      "The model labels the paper actually uses, verbatim and "
                      "de-duplicated (biopsychosocial model, bio-psycho-social "
                      "framework, sociopsychobiological model, extended "
                      "biopsychosocial model, and others). This is what makes "
                      "terminological drift visible.",
                      free_text=True, kind="open list", cap=5),
                field("bps_usage_instances",
                      "One item for every distinct passage where the label does "
                      "work. A paper that invokes the model in the introduction to "
                      "justify the topic and again in the discussion to recommend "
                      "multidisciplinary care yields two items, not one.",
                      kind="extraction list", cap=8,
                      subfields=[
                          sub("usage_verbatim", "The exact passage, at most 60 words."),
                          sub("bps_function", "The function the label serves in "
                                              "this passage.",
                              values=BPS_FUNCTION_VALUES),
                          sub("is_definitional", "Whether this passage also says "
                                                 "what the model is.",
                              values=["yes", "no"]),
                          sub("attributed_source", "Who the model is credited to "
                                                   "here (Engel, Gatchel, Waddell, "
                                                   "IASP, a guideline, or nobody)."),
                          sub("section_located", "Where the passage appears.",
                              values=SECTION_VALUES),
                          sub("note", "Anything the fields above cannot hold."),
                      ]),
                field("bps_definitions",
                      "One item for every place where the paper says what the "
                      "biopsychosocial model is. Empty when it never does.",
                      kind="extraction list", cap=3,
                      subfields=[
                          sub("definition_verbatim", "The exact passage, at most 60 words."),
                          sub("definition_type", "What kind of definitional act it is.",
                              values=["explicit_formal", "operational",
                                      "implicit_description", "borrowed",
                                      "critique_of_definition", "other"]),
                          sub("attributed_source", "The citation the paper credits."),
                          sub("elements_named", "The components the definition "
                                                "lists, as short labels."),
                          sub("section_located", "Where the passage appears.",
                              values=SECTION_VALUES),
                      ]),
                field("bps_operationalization_summary",
                      "At most 90 words, in the coder's words: what this paper "
                      "actually does with the model, as opposed to what it says "
                      "about it. Written to name the mechanism of use, for example "
                      "organizes the results section into three domain headings "
                      "and never relates them.",
                      free_text=True),
            ],
        },
        {
            "kind": "fields",
            "id": "fields-coverage",
            "title": "C. Domain Coverage",
            "feedback": True,
            "field_feedback": True,
            "intro": "Coverage is coded at the level of substantive treatment, "
                     "not keyword presence. The same four-level ladder applies to "
                     "each domain and is identical across both reviews. Only the "
                     "biological reading is track-aware: for a musculoskeletal "
                     "record the biological content is judged against "
                     "musculoskeletal mechanisms, and for a neuropathic record "
                     "against neuropathic mechanisms (see Scheme 6).",
            "fields": [
                field("domain_coverage_bio", "Depth of biological content, read "
                      "against the track-appropriate biological mechanisms.",
                      values=DOMAIN_COVERAGE_LADDER),
                field("domain_coverage_psych", "Depth of psychological content.",
                      values=DOMAIN_COVERAGE_LADDER),
                field("domain_coverage_social", "Depth of social content.",
                      values=DOMAIN_COVERAGE_LADDER),
                field("coverage_lifestyle",
                      "Depth of lifestyle content on the same ladder: physical "
                      "activity and exercise behaviour, sleep hygiene, diet and "
                      "weight, smoking, alcohol.",
                      values=DOMAIN_COVERAGE_LADDER,
                      notes="The registration lists lifestyle factors alongside "
                            "the three core domains. Coding it separately keeps it "
                            "out of the triad, where it would inflate biological "
                            "or social coverage."),
                field("coverage_spiritual_existential",
                      "Depth of spiritual or existential content on the same "
                      "ladder: meaning, faith or religion, hope, existential "
                      "suffering.",
                      values=DOMAIN_COVERAGE_LADDER,
                      notes="Also registered as a domain of its own. Absent is the "
                            "expected value for most papers, and that distribution "
                            "is itself a result."),
                field("domain_evidence",
                      "One item per core domain not scored as absent, carrying the "
                      "passage that justifies the coverage level.",
                      kind="extraction list", cap=5,
                      subfields=[
                          sub("domain", "Which domain this passage carries.",
                              values=["biological", "psychological", "social"]),
                          sub("coverage_level", "The level given above, repeated "
                                                "here so the judgement and its "
                                                "evidence travel together.",
                              values=["elaborated", "mentioned", "minimal", "absent"]),
                          sub("constructs_named", "The domain-specific constructs "
                                                  "the paper actually names."),
                          sub("subdomains_named", "The Scheme 6 ontology "
                                                  "subdomains the content belongs "
                                                  "to, mapped when they fit."),
                          sub("evidence_verbatim", "The passage, at most 60 words."),
                          sub("section_located", "Where the passage appears.",
                              values=SECTION_VALUES),
                      ]),
            ],
        },
        {
            "kind": "fields",
            "id": "fields-factors",
            "title": "D. Which Factors Carry Each Domain (the Ontology Nodes)",
            "feedback": True,
            "field_feedback": True,
            "intro": "The core of the extraction layer. A coverage grade says how "
                     "much of a domain a review carries; these lists say what it "
                     "carries. "
                     "Each item holds the review's own label, an anchor onto the "
                     "project ontology where one fits, the role the factor plays "
                     "in the paper's account, and the passage behind it. "
                     "Psychological constructs are recorded in section G instead, "
                     "which carries the extra definitional fields RQ3 needs, so no "
                     "factor is ever written twice.",
            "fields": [
                field("biological_factors",
                      "Every biological factor the paper names.",
                      kind="extraction list", cap=12,
                      subfields=[
                          sub("factor_label", "The paper's own term, as specifically "
                                              "as the paper puts it."),
                          sub("subdomain_label", "The Scheme 6 biological subdomain "
                                                 "it belongs to, when one fits. "
                                                 "Empty when none does, and the "
                                                 "label then also goes into "
                                                 "emergent_labels."),
                          sub("mechanism_level", "Where the factor sits.",
                              values=["peripheral or tissue",
                                      "spinal or central nervous system",
                                      "systemic or whole body", "genetic or molecular",
                                      "structural or anatomical", "treatment related",
                                      "other", "unclear"]),
                          sub("factor_role", "What the factor does in this paper.",
                              values=FACTOR_ROLE_VALUES),
                          sub("factor_verbatim", "The passage, at most 60 words."),
                          sub("section_located", "Where the passage appears.",
                              values=SECTION_VALUES),
                          sub("evidence_basis", "What the claim rests on.",
                              values=EVIDENCE_BASIS_VALUES),
                      ],
                      notes="The role field is what separates a review that lists "
                            "biology from one that gives biology a job. A factor "
                            "coded as mediator or moderator is a different "
                            "contribution from the same factor coded as a "
                            "descriptive theme."),
                field("social_factors",
                      "Every social factor the paper names, with the level of "
                      "social organization it sits at.",
                      kind="extraction list", cap=12,
                      subfields=[
                          sub("factor_label", "The paper's own term."),
                          sub("subdomain_label", "The Scheme 6 social subdomain, "
                                                 "when one fits."),
                          sub("social_level", "The level of social organization.",
                              values=["interpersonal", "family or household",
                                      "workplace", "community", "healthcare system",
                                      "societal or policy", "cultural", "economic",
                                      "other", "unclear"]),
                          sub("factor_role", "What the factor does in this paper.",
                              values=FACTOR_ROLE_VALUES),
                          sub("factor_verbatim", "The passage, at most 60 words."),
                          sub("section_located", "Where the passage appears.",
                              values=SECTION_VALUES),
                          sub("evidence_basis", "What the claim rests on.",
                              values=EVIDENCE_BASIS_VALUES),
                      ],
                      notes="The social domain is the one this literature is "
                            "thinnest on, and the level field is what shows "
                            "whether social means an interpersonal relationship or "
                            "a structural condition. Both are coded as social by "
                            "the coverage ladder; they are not the same claim."),
                field("other_domain_factors",
                      "Factors outside the triad, kept visible rather than folded "
                      "into it.",
                      kind="extraction list", cap=6,
                      subfields=[
                          sub("factor_label", "The paper's own term."),
                          sub("domain", "Which non-triad domain it belongs to.",
                              values=["lifestyle", "spiritual or existential",
                                      "environmental", "other"]),
                          sub("factor_role", "What the factor does in this paper.",
                              values=FACTOR_ROLE_VALUES),
                          sub("factor_verbatim", "The passage, at most 60 words."),
                          sub("section_located", "Where the passage appears.",
                              values=SECTION_VALUES),
                      ]),
            ],
        },
        {
            "kind": "fields",
            "id": "fields-integration",
            "title": "E. Integration (the Core RQ2 Contribution, the Ontology Edges)",
            "feedback": True,
            "field_feedback": True,
            "intro": "Integration is the scheme's highest-resolution construct. "
                     "The pairwise ladder distinguishes a stated mechanism from a "
                     "mere direction, an association, or a bare co-mention, and "
                     "every graded link has to point at the sentence that carries "
                     "it. The claim items now name the two factors on either end "
                     "of the link, which is what turns a count of integration "
                     "statements into a map of what this literature says connects "
                     "to what.",
            "fields": [
                field("integration_bio_psych",
                      "Biological to psychological integration.",
                      values=PAIRWISE_INTEGRATION_LADDER),
                field("integration_psych_social",
                      "Psychological to social integration.",
                      values=PAIRWISE_INTEGRATION_LADDER),
                field("integration_bio_social",
                      "Biological to social integration.",
                      values=PAIRWISE_INTEGRATION_LADDER),
                field("integration_triadic",
                      "Three-domain integration.",
                      values=TRIADIC_INTEGRATION_LADDER),
                field("integration_claims",
                      "One item for every passage in which the paper relates two "
                      "or three domains to each other. A pairwise field graded "
                      "above mentioned should have at least one item behind it, "
                      "and the pipeline checks exactly that.",
                      kind="extraction list", cap=12,
                      subfields=[
                          sub("domains_linked", "Which domains the passage relates.",
                              values=["bio_psych", "psych_social", "bio_social",
                                      "triadic"]),
                          sub("integration_level", "The rung this passage supports.",
                              values=["mechanistic", "directional", "descriptive",
                                      "mentioned", "none"]),
                          sub("source_factor_label", "The factor doing the "
                                                     "influencing, in the paper's "
                                                     "own wording."),
                          sub("target_factor_label", "The factor being influenced."),
                          sub("direction", "Whether the influence runs one way or "
                                           "both.",
                              values=["unidirectional", "bidirectional or reciprocal",
                                      "unspecified"]),
                          sub("mediator_or_moderator", "The named intermediate, "
                                                       "when the paper gives one."),
                          sub("claim_verbatim", "The passage, at most 60 words."),
                          sub("mechanism_note", "The pathway in the coder's words, "
                                                "empty when none is given."),
                          sub("section_located", "Where the passage appears.",
                              values=SECTION_VALUES),
                          sub("evidence_basis", "What the claim rests on.",
                              values=EVIDENCE_BASIS_VALUES),
                      ],
                      notes="Naming both ends of the link is what makes these items "
                            "usable as an ontology. It lets the synthesis ask "
                            "which specific factors this literature connects, "
                            "rather than only how often it connects domains, and "
                            "it makes two coders' integration claims comparable as "
                            "edges rather than as counts."),
                field("integration_mechanism_summary",
                      "At most 90 words, in the coder's words: the cross-domain "
                      "pathways this paper actually proposes. Written as none "
                      "proposed when the paper proposes none.",
                      free_text=True),
            ],
        },
        {
            "kind": "fields",
            "id": "typology-balance",
            "title": "F. Typology and Balance",
            "feedback": True,
            "field_feedback": True,
            "intro": "The summary judgments that answer RQ1 at full-text depth. "
                     "Both are also derived independently by rule, and the two "
                     "readings are compared (see section J).",
            "fields": [
                field("overall_balance",
                      "Relative emphasis across the three core domains.",
                      values=[
                          v("balanced", "No domain dominates; the three are "
                            "weighted comparably."),
                          v("psych-dominant", "Psychological content dominates."),
                          v("bio-dominant", "Biological content dominates."),
                          v("social-dominant", "Social content dominates."),
                          v("dyadic", "Two domains dominate and one is marginal."),
                          v("unclear", "Balance cannot be determined."),
                      ]),
                field("bps_typology",
                      "Full-text BPS operationalization type.",
                      values=[
                          v("true_integrative", "Explicit cross-domain causal or "
                            "mechanistic interaction is central to the review's "
                            "logic.",
                            boundary="Requires at least descriptive triadic "
                                     "integration plus two elaborated domains."),
                          v("multifactorial", "Multiple domains covered "
                            "meaningfully but mostly in parallel.",
                            boundary="Distinguished from true_integrative by the "
                                     "absence of a genuine cross-domain link."),
                          v("pseudo_bps", "BPS language used but one or more core "
                            "domains are thin, absent, or tokenistic."),
                          v("rhetorical_bps", "BPS used mainly as framing or "
                            "justification without analytic substance."),
                          v("narrow_despite_label", "BPS claimed but substantive "
                            "scope is essentially single-domain."),
                          v("unclear", "Type cannot be determined."),
                      ],
                      notes="The cross-provider test run showed this to be the "
                            "loosest field in the scheme, both between coders and "
                            "against the rule-derived typology. It is the field "
                            "most in need of expert attention."),
            ],
        },
        {
            "kind": "fields",
            "id": "fields-concepts",
            "title": "G. Psychological Concepts, Their Relations, Frameworks, and Measures (RQ3)",
            "feedback": True,
            "field_feedback": True,
            "intro": "The registration asks for concept names, whether definitions "
                     "are provided and their text, the theoretical frameworks "
                     "invoked, and the hierarchical and semantic relationships "
                     "between concepts. All four are coded here as named items "
                     "rather than as delimited strings, so a concept, its definition, and "
                     "its relations to other concepts stay attached to one another.",
            "fields": [
                field("concept_definitions_present",
                      "Whether the review defines the psychological constructs it "
                      "uses.",
                      values=[
                          v("yes", "The main constructs are explicitly defined or "
                            "clearly operationalized."),
                          v("partial", "Some constructs defined, others named only."),
                          v("no", "Constructs used without any meaning given."),
                      ]),
                field("psychological_concepts",
                      "Every psychological construct the paper uses, one item "
                      "each, at the resolution the paper uses it.",
                      kind="extraction list", cap=16,
                      subfields=[
                          sub("concept_label", "The paper's own term, always."),
                          sub("concept_family", "The Scheme 5 family it belongs to, "
                                                "when one fits. Empty when none "
                                                "does."),
                          sub("definitional_status", "What kind of meaning the "
                                                     "paper gives it.",
                              values=["formally_defined", "operationalized_only",
                                      "described_informally", "named_only", "unclear"]),
                          sub("definition_verbatim", "The passage that defines it, "
                                                     "at most 60 words, empty when "
                                                     "there is none."),
                          sub("definition_source", "Whose definition it is.",
                              values=["own definition", "cited from other work",
                                      "taken from an instrument", "unattributed",
                                      "unclear"]),
                          sub("measure_named", "The instrument the paper "
                                               "operationalizes it with, when any."),
                          sub("factor_role", "What the construct does in this paper.",
                              values=FACTOR_ROLE_VALUES),
                          sub("section_located", "Where the passage appears.",
                              values=SECTION_VALUES),
                      ],
                      notes="concept_label and concept_family are both filled and "
                            "neither replaces the other: the family makes concepts "
                            "comparable across reviews, the label keeps the "
                            "distinction between fear of movement during lifting "
                            "and kinesiophobia."),
                field("concept_relations",
                      "Every relation the paper draws between two constructs. "
                      "These are the edges of the concept map the registration "
                      "asks for.",
                      kind="extraction list", cap=12,
                      subfields=[
                          sub("source_concept", "The construct the relation starts from."),
                          sub("target_concept", "The construct it relates to."),
                          sub("relation_type", "What kind of relation it is.",
                              values=["is_a_subtype_of", "part_of_or_component_of",
                                      "synonym_or_used_interchangeably",
                                      "overlapping_or_related",
                                      "antecedent_or_cause_of",
                                      "consequence_or_outcome_of", "mediates",
                                      "moderates", "measured_by",
                                      "contrasted_as_distinct_from",
                                      "conflated_without_comment", "other", "unclear"]),
                          sub("explicitly_stated", "Whether the paper states the "
                                                   "relation or merely behaves as "
                                                   "though it holds.",
                              values=["yes", "no"]),
                          sub("relation_verbatim", "The passage, at most 60 words."),
                          sub("section_located", "Where the passage appears.",
                              values=SECTION_VALUES),
                      ],
                      notes="conflated_without_comment is deliberately available. "
                            "Silent conflation of two constructs is one of the "
                            "main findings SQ1 can produce, and it is invisible "
                            "unless a coder is allowed to record it."),
                field("theoretical_frameworks",
                      "Every theoretical model or framework the paper invokes.",
                      kind="extraction list", cap=8,
                      subfields=[
                          sub("framework_label", "The model, in the paper's own "
                                                 "wording, with preferred labels "
                                                 "available."),
                          sub("role", "What the paper does with the model.",
                              values=["organizing framework", "tested or modelled",
                                      "extended or revised", "critiqued or rejected",
                                      "compared with another model",
                                      "mentioned in passing", "other", "unclear"]),
                          sub("domains_covered", "Which of biological, "
                                                 "psychological, and social the "
                                                 "model actually spans."),
                          sub("attributed_source", "The citation the paper credits."),
                          sub("framework_verbatim", "The passage, at most 60 words."),
                          sub("section_located", "Where the passage appears.",
                              values=SECTION_VALUES),
                      ]),
                field("instruments",
                      "Every measurement or appraisal instrument named.",
                      kind="extraction list", cap=8,
                      subfields=[
                          sub("instrument_label", "The instrument, with preferred "
                                                  "labels available."),
                          sub("abbreviation", "The abbreviation the paper uses."),
                          sub("domain_measured", "Which domain the instrument "
                                                 "measures in.",
                              values=["biological", "psychological", "social",
                                      "pain or symptom", "function or disability",
                                      "quality of life", "multiple domains",
                                      "methodological quality", "other", "unclear"]),
                          sub("construct_measured_as_stated", "What the paper says "
                                                              "the instrument "
                                                              "captures, in its own "
                                                              "wording."),
                          sub("role", "What the instrument does in this paper.",
                              values=["primary outcome", "secondary outcome",
                                      "predictor or covariate", "mediator or moderator",
                                      "screening or classification",
                                      "developed or validated here",
                                      "discussed conceptually", "critiqued",
                                      "referenced only", "other", "unclear"]),
                          sub("instrument_verbatim", "The passage, at most 60 words."),
                      ],
                      notes="What a review measures is the most concrete form its "
                            "operationalization of the model takes. A paper that "
                            "claims a biopsychosocial frame and measures only "
                            "psychological questionnaires has told you something "
                            "its prose did not."),
            ],
        },
        {
            "kind": "fields",
            "id": "fields-problems",
            "title": "H. Conceptual Problems (SQ1)",
            "feedback": True,
            "field_feedback": True,
            "intro": "The secondary question of the review. Coded as items rather "
                     "than as flags, so a problem carries what it is about, which "
                     "constructs it concerns, and whether the paper noticed it.",
            "fields": [
                field("conceptual_problems",
                      "One item per problem the paper names or displays. An empty "
                      "list is a legitimate coding.",
                      kind="extraction list", cap=8,
                      subfields=[
                          sub("problem_type", "The kind of problem.",
                              values=["vague_definition", "tokenistic_bps",
                                      "missing_social", "missing_biology",
                                      "missing_psychology", "mechanistic_absence",
                                      "construct_overlap",
                                      "parallel_listing_without_integration",
                                      "measurement_mismatch", "definitional_drift",
                                      "domain_reductionism",
                                      "unfalsifiable_or_untestable", "other"]),
                          sub("problem_scope", "What the problem is about.",
                              values=["the biopsychosocial model itself",
                                      "a psychological construct",
                                      "a biological construct", "a social construct",
                                      "integration between domains", "measurement",
                                      "terminology", "scope or coverage", "other"]),
                          sub("affected_labels", "The constructs or terms the "
                                                 "problem concerns."),
                          sub("named_by_authors", "Whether the paper points the "
                                                  "problem out itself, or merely "
                                                  "displays it.",
                              values=["yes", "no"]),
                          sub("problem_verbatim", "The passage that shows it, which "
                                                  "for a displayed problem may be "
                                                  "the passage where the gap is "
                                                  "visible."),
                          sub("note", "Anything the fields above cannot hold."),
                      ],
                      notes="The distinction between a problem the authors name and "
                            "one they display is the difference between a "
                            "literature that knows its difficulties and one that "
                            "does not. Both are coded; only the second is a "
                            "finding the reviews themselves cannot report."),
            ],
        },
        {
            "kind": "fields",
            "id": "fields-synthesis",
            "title": "I. Synthesis Hooks (Free Text)",
            "feedback": True,
            "field_feedback": True,
            "intro": "Deliberately unconstrained fields. They exist so that nuance "
                     "no controlled vocabulary can hold still reaches the "
                     "synthesis, and so that the coder never has to force an "
                     "observation into a field where it does not belong.",
            "fields": [
                field("key_quotes",
                      "The most conceptually load-bearing passages: the ones a "
                      "reviewer would read first when writing the synthesis.",
                      kind="extraction list", cap=6,
                      subfields=[
                          sub("claim_verbatim", "The passage, at most 60 words, "
                                                "quotable on its own."),
                          sub("claim_type", "What kind of claim it is.",
                              values=["definitional", "integrative", "operationalizing",
                                      "critical or problematizing", "measurement",
                                      "theoretical", "clinical or applied", "other"]),
                          sub("section_located", "Where it appears.",
                              values=SECTION_VALUES),
                          sub("why_it_matters", "One short sentence on why this "
                                                "passage was selected."),
                      ]),
                field("emergent_labels",
                      "Every conceptually important term the paper uses that the "
                      "project vocabularies do not contain: a factor, construct, "
                      "mechanism, framework, instrument, or population label with "
                      "no good home on the spine, written exactly as the paper "
                      "writes it.",
                      free_text=True, kind="open list", cap=12,
                      notes="The review's own error signal. It is how the project "
                            "ontology learns what it is missing, and the coder is "
                            "explicitly told to use it generously whenever a spine "
                            "field had to be left empty."),
                field("conceptual_tensions",
                      "Contradictions, ambiguities, unresolved debates, and gaps "
                      "the paper names or displays, including tensions visible "
                      "inside the paper itself.",
                      free_text=True, kind="open list", cap=5),
                field("additional_observations",
                      "Anything else conceptually relevant that no other field "
                      "captures. One observation per item, as long as it needs to "
                      "be.",
                      free_text=True, kind="open list", cap=6),
                field("synthesis_note",
                      "At most 90 words on what this paper contributes to the "
                      "question of how the biopsychosocial model is "
                      "operationalized, and what it does not, written for a "
                      "reviewer who has not read it.",
                      free_text=True),
                field("coding_rationale",
                      "At most 40 words justifying the main judgements: the "
                      "typology, the triadic integration level, and any close call.",
                      free_text=True),
            ],
        },
        {
            "kind": "fields",
            "id": "fields-derived",
            "title": "J. Derived Fields",
            "feedback": True,
            "field_feedback": True,
            "intro": "Never asked of the coder. Computed from the coded content by "
                     "fixed rules, so that the same coding always produces the "
                     "same verdict, in every model and in every run, and so that a "
                     "change to a rule takes effect without re-coding anything.",
            "fields": [
                field("presence flags",
                      "One yes or no per conceptual element, read off the coded "
                      "content rather than asked of the coder: BPS usage evidence, "
                      "a BPS definition, integration evidence, a triadic claim, a "
                      "named integration edge, biological factors, social factors, "
                      "other-domain factors, psychological concepts, defined "
                      "concepts, concept relations, a hierarchical relation, "
                      "frameworks, instruments, conceptual problems, and domain "
                      "evidence per domain.",
                      values=[
                          v("yes", "The coder returned at least one item of this "
                            "kind for this paper."),
                          v("no", "Nothing of this kind was returned.",
                            boundary="An observable fact about the coding, not a "
                                     "judgement: it says this coder recorded no "
                                     "material of this kind in this paper."),
                      ],
                      kind="derived",
                      notes="These are the variables part of the cross-provider "
                            "agreement is computed on. Whether two coders both "
                            "found a framework in a paper has one answer; which "
                            "label each wrote for it is a different question, "
                            "answered by the set-overlap metrics on the extraction "
                            "lists."),
                field("coverage and integration depth",
                      "coverage_depth per domain, coverage_total, domains_present, "
                      "pairwise_depth_total, pairwise_depth_max, triadic_depth, and "
                      "an integration_index between 0 and 1 that averages the "
                      "normalized pairwise mean with the normalized triadic rung.",
                      kind="derived"),
                field("ontology breadth",
                      "n_subdomains_bio, n_subdomains_psych, n_subdomains_social, "
                      "n_subdomains_named, n_named_integration_edges, "
                      "n_emergent_labels, and controlled_label_share.",
                      kind="derived",
                      notes="controlled_label_share measures the ontology against "
                            "the literature, not the coder against the ontology. A "
                            "low share says this literature is naming things the "
                            "project vocabularies do not yet carry, and the "
                            "off-spine labels are the candidate list for extending "
                            "them."),
                field("BPS usage profile",
                      "bps_function_set, n_bps_functions, "
                      "bps_has_substantive_function, and the sections the label "
                      "appears in, all read off the usage items.",
                      kind="derived"),
                field("item counts",
                      "One count per extraction list, plus n_triadic_claims, "
                      "n_defined_concepts, n_hierarchical_relations, "
                      "n_evidence_quotes, n_extracted_items, and "
                      "n_open_list_entries.",
                      kind="derived"),
                field("derived_typology",
                      "The typology recomputed from coverage and integration by a "
                      "fixed rule, alongside typology_matches_derived.",
                      kind="derived",
                      notes="The typology is the one judgement that is both coded "
                            "and derived. Comparing the two is the sharpest "
                            "available test of whether the typology is defined "
                            "tightly enough to be applied the same way twice. Where "
                            "they diverge, the codebook is under-specified, not the "
                            "coder."),
                field("conceptual_yield",
                      "How much conceptual material the paper actually yielded. A "
                      "measure of harvest, not of promise.",
                      values=[
                          v("high", "Three domains present with at least "
                            "descriptive triadic integration and three or more "
                            "integration claims; or four or more claims with two "
                            "defined concepts or two frameworks; or eight or more "
                            "distinct subdomains with two or more claims."),
                          v("moderate", "Two or more integration claims, or three "
                            "or more concept relations, or three concepts across "
                            "two substantive domains."),
                          v("low", "Some conceptual material, but thin."),
                          v("minimal", "Nothing extracted."),
                      ],
                      kind="derived"),
                field("fulltext_eligibility",
                      "The post-retrieval verdict. A recommendation for a human "
                      "adjudicator, not a final decision, and deliberately "
                      "recall-protecting.",
                      values=[
                          v("include", "Eligible and conceptually usable."),
                          v("uncertain", "Eligible on the formal criteria but "
                            "doubtful in substance.",
                            boundary="Everything doubtful lands here rather than "
                                     "in exclude. Uncertain is a request for human "
                                     "adjudication."),
                          v("exclude", "No biopsychosocial domain content at all, "
                            "or a single-domain review with no cross-domain claim."),
                      ],
                      kind="derived"),
                field("synthesis_priority",
                      "Reading order for the later synthesis: core, supporting, "
                      "background, or not_relevant.",
                      kind="derived"),
            ],
        },
        {
            "kind": "prose",
            "id": "spine-and-free-text",
            "title": "How the Preferred Labels and the Free Text Work Together",
            "feedback": True,
            "body": [
                "Two demands pull against each other in an extraction scheme this "
                "large. Comparability wants one label per thing, so that two "
                "reviews discussing central sensitization land on the same node "
                "whatever they call it. Resolution wants the review's own words, "
                "because a term the ontology cannot hold is a finding about the "
                "ontology.",
                "The scheme holds both, by never making them compete for the same "
                "field. Each item carries the paper's own label in one field and, "
                "where the project vocabularies apply, an anchor onto the ontology "
                "in another: subdomain_label sits beside factor_label, "
                "concept_family sits beside concept_label. Both are filled. A "
                "mapped label never replaces the specific term the paper used.",
                "The mapping itself is conservative. A label is snapped to a "
                "canonical one only on an exact match or a whole-token match on "
                "one of its lexical variants; anything else is kept as written. "
                "The vocabularies are the project's own: the Scheme 6 subdomain "
                "ontology for biological and social factors, the Scheme 5 concept "
                "taxonomy for psychological constructs, plus framework, "
                "instrument, pain-condition, and attributed-source lists.",
                "Whatever falls outside is wanted rather than discarded. The coder "
                "is told explicitly to record the item anyway, leave the anchor "
                "empty, and repeat the term in emergent_labels. After a run the "
                "pipeline reports, per extraction list, how much of what was "
                "extracted anchored to the spine and which off-spine labels recur, "
                "which is the working list for extending the vocabularies once the "
                "expert evaluation is in.",
            ],
        },
        {
            "kind": "prose",
            "id": "filter-logic",
            "title": "How the Eligibility Verdict Is Derived",
            "feedback": True,
            "body": [
                "The verdict is not asked of the coder. The coder codes what the "
                "article contains, and the pipeline then computes eligibility, "
                "yield, and priority by fixed rules. Given the same coded content "
                "the verdict is fixed, so two coders who read a paper the same way "
                "cannot disagree about the verdict, and a disagreement in the "
                "output always points back to a disagreement about the paper.",
                "The rule is recall-protecting by design. Only two conditions "
                "exclude: no biopsychosocial domain content anywhere in the full "
                "text, and a single-domain review with no cross-domain claim. "
                "Everything else that looks doubtful becomes uncertain, which is a "
                "request for human adjudication rather than a rejection: a source "
                "that reads as a primary study, fewer than two domains "
                "substantively covered, no biopsychosocial vocabulary with no "
                "readable typology, or a typology that cannot be read together "
                "with no triadic integration.",
                "This matters because the registration reserves eligibility "
                "decisions for human screeners and allows generative AI only as an "
                "assistant under human review. The derived verdict is therefore a "
                "triage recommendation that tells a reviewer where to look, and "
                "the adjudication_status field on the coding template is where the "
                "human decision is recorded.",
            ],
        },
    ],
}


# --------------------------------------------------------------------------
# SCHEME 4 -- Stage 3 retrieval and manual relevance triage scheme
# --------------------------------------------------------------------------

SCHEME_4 = {
    "id": "scheme_4",
    "num": 4,
    "title": "Stage 3 Retrieval and Manual Relevance Triage Scheme",
    "subtitle": "Full-text availability, retrieval need, and adjudication "
                "checklist",
    "tagline": "Automated candidate manifest plus human checklist completion",
    "stage": "Stage 3",
    "stage_key": "triage",
    "meta": {
        "Workflow position": "Bridge between Stage 2 coding and Stage 3 full-text "
            "coding.",
        "Operational mode": "Automated candidate manifest plus human checklist "
            "completion.",
        "Unit of analysis": "One Stage 3 candidate record and its retrieval "
            "state.",
        "Provenance basis": "Manifest-generation logic in stage3_prep.py and the "
            "generated manual relevance checklist.",
    },
    "rqs": ["Retrieval and adjudication gate that protects RQ2 and RQ3 corpus "
            "quality"],
    "sources": [
        "src/03_pipeline/bps_review/extraction/stage3_prep.py",
        "src/09_review_stages/04_extraction/forms/stage3_manual_relevance_checklist.csv",
        "src/09_review_stages/04_extraction/outputs/stage3_candidate_manifest.csv",
        "src/09_review_stages/04_extraction/outputs/stage3_manual_fulltext_queue.csv",
        "src/09_review_stages/04_extraction/outputs/stage3_retrieval_validation.csv",
    ],
    "outputs": [
        "src/09_review_stages/04_extraction/outputs/stage3_candidate_manifest.csv",
        "src/09_review_stages/04_extraction/forms/stage3_manual_relevance_checklist.csv",
        "src/09_review_stages/04_extraction/outputs/stage3_manual_fulltext_queue.csv",
        "src/09_review_stages/04_extraction/outputs/stage3_retrieval_validation.csv",
    ],
    "sections": [
        {
            "kind": "prose",
            "id": "purpose",
            "title": "Purpose",
            "body": [
                "This scheme governs the transition from Stage 2 abstract coding "
                "to Stage 3 full-text work. It standardizes which candidate "
                "reviews need manual retrieval, which records require manual "
                "relevance adjudication, and how retrieval status, risk signals, "
                "and reviewer decisions are recorded before deep coding begins.",
                "It is not a logistics file. It is a standardized adjudication "
                "framework that decides whether a review can enter Stage 3, "
                "whether the retrieved text is adequate, and how problematic or "
                "ambiguous records are escalated for human judgment.",
            ],
        },
        {
            "kind": "fields",
            "id": "manifest",
            "title": "Manifest Fields",
            "feedback": True,
            "field_feedback": True,
            "intro": "Automated fields written per candidate.",
            "fields": [
                field("fulltext_status",
                      "Operational retrieval status of the candidate.",
                      values=[
                          v("manual_retrieval_required", "No machine-available "
                            "full text was secured; human retrieval is needed."),
                          v("pmc_open_available_not_cached", "A PMCID exists but "
                            "the file was not yet cached."),
                          v("pmc_fulltext_cached", "A PMC full text is already "
                            "cached locally."),
                          v("pmc_fulltext_fetched", "A PMC full text was retrieved "
                            "during the current run."),
                          v("pmc_linked_fetch_failed", "A PMCID link existed but "
                            "retrieval or XML parsing failed."),
                          v("pmc_fulltext_low_content_manual_check", "Text was "
                            "fetched but appears too short and needs manual "
                            "inspection."),
                      ]),
                field("review_track",
                      "Which review pool the candidate belongs to, carried from "
                      "Scheme 2. Retrieval and triage logic itself is uniform "
                      "across both tracks.",
                      values=[v("musculoskeletal", "Musculoskeletal review pool."),
                              v("neuropathic", "Neuropathic review pool."),
                              v("both", "In both pools.")]),
                field("retrieval_source",
                      "Source of the PMCID or full-text link (existing metadata, "
                      "PubMed elink, or Europe PMC)."),
                field("fulltext_word_count",
                      "Word count of cached text, used to detect low-content full "
                      "texts.", free_text=True),
                field("manual_retrieval_needed",
                      "Whether a human must retrieve the text.",
                      values=[v("yes", "Retrieval required."),
                              v("no", "Machine text is available.")]),
                field("manual_relevance_priority",
                      "Adjudication urgency.",
                      values=[v("high", "Withdrawal or missing pain focus."),
                              v("medium", "Other concern flags, or retrieval "
                                "required with otherwise low signal."),
                              v("low", "No concern flags.")]),
                field("manual_relevance_flags",
                      "Pipe-delimited signal list describing why adjudication is "
                      "needed. See the signal logic below.", free_text=True),
                field("osf_manual_adjudication_required",
                      "Currently set to yes for the checklist workflow."),
                field("cached_xml_path / cached_text_path",
                      "Relative cache paths for machine-fetched PMC files."),
                field("pubmed_url",
                      "Direct URL to the source record when a PMID is available."),
            ],
        },
        {
            "kind": "fields",
            "id": "signals",
            "title": "Manual Relevance Signal Logic",
            "feedback": True,
            "field_feedback": True,
            "intro": "Each signal is derived from title and abstract and drives "
                     "priority.",
            "fields": [
                field("withdrawn_or_retracted_signal",
                      "Assigned when the title or abstract indicates a withdrawn "
                      "or retracted record. Forces high priority."),
                field("pain_focus_not_explicit",
                      "Assigned when pain is not explicit in title or abstract. "
                      "Forces high priority."),
                field("chronicity_not_explicit",
                      "Assigned when chronic, persistent, or long-term framing is "
                      "absent."),
                field("review_design_unclear",
                      "Assigned when review design is missing or too vague."),
            ],
        },
        {
            "kind": "fields",
            "id": "reviewer",
            "title": "Reviewer Checklist Fields",
            "feedback": True,
            "field_feedback": True,
            "intro": "Human adjudication fields completed against the manifest.",
            "fields": [
                field("reviewer_decision",
                      "Human reviewer decision after checking the candidate.",
                      free_text=True),
                field("reviewer_notes",
                      "Notes supporting the reviewer decision.", free_text=True),
                field("adjudication_decision",
                      "Final adjudicated status after disagreement resolution.",
                      free_text=True),
                field("adjudication_notes",
                      "Adjudication rationale.", free_text=True),
            ],
        },
    ],
}


# --------------------------------------------------------------------------
# SCHEME 5 -- Psychological concept clustering and framework mapping scheme
# --------------------------------------------------------------------------

SCHEME_5 = {
    "id": "scheme_5",
    "num": 5,
    "title": "Psychological Concept Clustering and Framework Mapping Scheme",
    "subtitle": "Second-order normalization of detected psychological concepts",
    "tagline": "Fixed pattern detection upstream, LLM clustering for higher-order "
               "normalization",
    "stage": "Cross-stage",
    "stage_key": "concepts",
    "meta": {
        "Workflow position": "Post-detection concept normalization over Stage 2 "
            "and Stage 3 concept strings.",
        "Operational mode": "Fixed pattern-based concept detection upstream, "
            "followed by LLM clustering for higher-order normalization.",
        "Unit of analysis": "The set of unique concept strings across the corpus, "
            "not whole records.",
        "Provenance basis": "coding.py concept and framework patterns plus the "
            "clustering prompt.",
    },
    "rqs": ["RQ3 (which psychological concepts and frameworks dominate)"],
    "sources": [
        "src/03_pipeline/bps_review/extraction/coding.py",
        "src/01_protocol/codebooks/stage2_codebook.md",
        "src/01_protocol/codebooks/stage3_codebook.md",
        "src/06_data/interim/extraction/llm_concept_clusters.json",
    ],
    "outputs": [
        "src/06_data/interim/extraction/llm_concept_clusters.json",
    ],
    "sections": [
        {
            "kind": "prose",
            "id": "purpose",
            "title": "Purpose",
            "body": [
                "This scheme standardizes higher-order concept mapping after "
                "concept detection. It groups extracted psychological concepts "
                "from chronic pain review records into interpretable families and "
                "links them to likely theoretical frameworks. The goal is "
                "cross-record comparability when the raw concepts are "
                "heterogeneous, overlapping, or variably named.",
                "It is a second-order coding scheme: it does not classify whole "
                "records, it normalizes the concept vocabulary itself.",
            ],
        },
        {
            "kind": "list",
            "id": "detection-concepts",
            "title": "Upstream Concept Detection Seeds (current)",
            "feedback": True,
            "intro": "The 16 fixed patterns used to detect concepts upstream. "
                     "The taxonomy below expands and structures these.",
            "items": [
                "fear-avoidance, catastrophizing, kinesiophobia, depression, "
                "anxiety, coping, self-efficacy, acceptance",
                "illness perceptions, pain beliefs, distress, expectations, "
                "emotion regulation, mindfulness, trauma, sleep",
            ],
        },
        {
            "kind": "list",
            "id": "detection-frameworks",
            "title": "Upstream Framework Seeds (current)",
            "intro": "The 7 fixed framework patterns.",
            "items": [
                "fear-avoidance model, cognitive behavioral therapy, acceptance "
                "and commitment therapy",
                "illness perception framework, operant learning, social cognitive "
                "theory, self-regulation",
            ],
        },
        {
            "kind": "fields",
            "id": "cluster-schema",
            "title": "Cluster Output Schema (current)",
            "feedback": True,
            "field_feedback": True,
            "intro": "The clustering LLM returns strict JSON with a top-level "
                     "clusters list.",
            "fields": [
                field("clusters", "Top-level list of normalized concept clusters."),
                field("family", "Higher-order concept family label for the "
                      "cluster."),
                field("members", "The original concept strings grouped into that "
                      "family.", free_text=True),
                field("possible_frameworks", "Likely theoretical or therapeutic "
                      "frameworks associated with the family.", free_text=True),
            ],
        },
        {
            "kind": "taxonomy",
            "id": "psych-taxonomy",
            "title": "Comprehensive Psychological Concept Taxonomy",
            "feedback": True,
            "intro": "The comprehensive concept space that the clustering step "
                     "should "
                     "normalize toward. Each family is aligned one to one with a "
                     "psychological subdomain of the Scheme 6 ontology, so the "
                     "concept map and the semantic ontology share a single "
                     "vocabulary. Members are representative constructs (not "
                     "exhaustive); frameworks are the theories most often invoked "
                     "for that family in chronic pain research. Experts are asked "
                     "to add, move, rename, or remove members and frameworks.",
            "families": [
                {"family": "Catastrophizing and negative cognitive appraisal",
                 "subdomain": "Catastrophizing and Negative Cognitive Appraisal",
                 "definition": "Exaggerated negative interpretation and mental "
                 "amplification of pain and its threat value.",
                 "members": ["pain catastrophizing", "rumination", "magnification",
                             "helplessness", "negative appraisal", "worry about pain"],
                 "frameworks": ["fear-avoidance model", "cognitive-behavioral model",
                                "communal coping model"]},
                {"family": "Fear, avoidance and pain-related fear",
                 "subdomain": "Fear Avoidance and Pain Related Fear",
                 "definition": "Fear of pain, movement, or reinjury and the "
                 "avoidance or escape behaviour it motivates.",
                 "members": ["kinesiophobia", "fear of movement or reinjury",
                             "pain-related fear", "avoidance behaviour",
                             "escape behaviour", "threat hypervigilance"],
                 "frameworks": ["fear-avoidance model", "avoidance-endurance model",
                                "operant learning"]},
                {"family": "Depression, low mood and negative affect",
                 "subdomain": "Depression Emotional Distress and Affect",
                 "definition": "Depressive symptoms and sustained negative affect "
                 "linked to living with pain.",
                 "members": ["depression", "depressive symptoms", "anhedonia",
                             "hopelessness", "demoralization", "negative affect"],
                 "frameworks": ["cognitive model of depression", "diathesis-stress",
                                "mutual maintenance model"]},
                {"family": "Anxiety and psychological reactivity",
                 "subdomain": "Anxiety and Psychological Reactivity",
                 "definition": "Anxious apprehension and heightened physiological "
                 "or cognitive reactivity to pain.",
                 "members": ["anxiety", "pain anxiety", "health anxiety",
                             "anxiety sensitivity", "physiological reactivity"],
                 "frameworks": ["anxiety sensitivity model", "fear-avoidance model"]},
                {"family": "Self-efficacy, control and mastery",
                 "subdomain": "Self Efficacy Control Beliefs and Perceived Mastery",
                 "definition": "Beliefs about ability to manage pain and exert "
                 "control over outcomes.",
                 "members": ["pain self-efficacy", "perceived control",
                             "locus of control", "mastery", "agency", "confidence"],
                 "frameworks": ["social cognitive theory", "self-efficacy theory"]},
                {"family": "Acceptance, psychological flexibility and mindfulness",
                 "subdomain": "Acceptance Psychological Flexibility and Mindfulness",
                 "definition": "Willingness to experience pain without avoidance "
                 "while pursuing valued activity.",
                 "members": ["pain acceptance", "psychological flexibility",
                             "values-based action", "committed action",
                             "present-moment awareness", "cognitive defusion",
                             "willingness"],
                 "frameworks": ["acceptance and commitment therapy",
                                "psychological flexibility model",
                                "mindfulness-based approaches"]},
                {"family": "Coping strategies and adjustment",
                 "subdomain": "Pain Coping Strategies and Adjustment",
                 "definition": "Cognitive and behavioural efforts to manage pain "
                 "and the resulting adjustment.",
                 "members": ["active coping", "passive coping",
                             "problem-focused coping", "emotion-focused coping",
                             "adaptive coping", "maladaptive coping", "adjustment"],
                 "frameworks": ["transactional model of stress and coping",
                                "self-regulation"]},
                {"family": "Attention, vigilance and pain processing",
                 "subdomain": "Attention Vigilance and Pain Processing",
                 "definition": "How attention is captured by or directed away from "
                 "pain and bodily threat.",
                 "members": ["attentional bias", "hypervigilance", "distraction",
                             "pain interruption", "attentional control",
                             "somatic focus"],
                 "frameworks": ["threat interpretation model",
                                "cognitive-affective model of interruption"]},
                {"family": "Illness beliefs, pain representations and meaning",
                 "subdomain": "Illness Beliefs Pain Representations and Meaning",
                 "definition": "Beliefs about the nature, cause, timeline, and "
                 "consequences of pain and its meaning.",
                 "members": ["illness perceptions", "pain beliefs",
                             "causal attributions", "timeline beliefs",
                             "consequence beliefs", "meaning of pain",
                             "illness identity"],
                 "frameworks": ["common-sense model of self-regulation",
                                "illness perception framework"]},
                {"family": "Cognitive-behavioural and psychotherapeutic approaches",
                 "subdomain": "Cognitive Behavioral and Psychotherapeutic Approaches",
                 "definition": "Structured psychological techniques targeting pain "
                 "cognition and behaviour.",
                 "members": ["cognitive restructuring", "behavioural activation",
                             "graded activity", "graded exposure", "relaxation",
                             "psychoeducation"],
                 "frameworks": ["cognitive-behavioral therapy",
                                "operant and respondent conditioning"]},
                {"family": "Third-wave, ACT and contextual approaches",
                 "subdomain": "Third Wave Therapies ACT and Contextual Approaches",
                 "definition": "Contextual and acceptance-based therapies that "
                 "target the function of experience.",
                 "members": ["acceptance", "defusion", "values clarification",
                             "self-as-context", "mindfulness practice",
                             "functional contextualism"],
                 "frameworks": ["acceptance and commitment therapy",
                                "contextual behavioural science",
                                "mindfulness-based stress reduction"]},
                {"family": "Resilience, positive psychology and growth",
                 "subdomain": "Resilience Positive Psychology and Post Traumatic Growth",
                 "definition": "Protective strengths and positive adaptation "
                 "despite persistent pain.",
                 "members": ["resilience", "optimism", "hope", "benefit-finding",
                             "post-traumatic growth", "positive affect", "gratitude"],
                 "frameworks": ["broaden-and-build theory", "resilience frameworks"]},
                {"family": "Identity, self-concept and pain biography",
                 "subdomain": "Identity Self Concept and Chronic Pain Biography",
                 "definition": "How pain reshapes self-concept, roles, and life "
                 "narrative.",
                 "members": ["pain identity", "self-discrepancy",
                             "biographical disruption", "role loss",
                             "self-concept", "loss of self"],
                 "frameworks": ["self-discrepancy theory",
                                "narrative and biographical approaches"]},
                {"family": "Trauma, adversity and life events",
                 "subdomain": "Trauma Adverse Childhood and Life Events",
                 "definition": "Traumatic and adverse experiences that predispose "
                 "to or maintain chronic pain.",
                 "members": ["post-traumatic stress", "trauma",
                             "adverse childhood experiences", "abuse history",
                             "life stressors", "victimization"],
                 "frameworks": ["mutual maintenance model", "diathesis-stress",
                                "shared vulnerability model"]},
                {"family": "Personality and individual differences",
                 "subdomain": "Personality Psychological Traits and Individual Differences",
                 "definition": "Stable traits and dispositions that shape pain "
                 "experience and response.",
                 "members": ["neuroticism", "negative affectivity", "harm avoidance",
                             "perfectionism", "alexithymia", "trait anxiety"],
                 "frameworks": ["trait vulnerability models",
                                "diathesis-stress models"]},
                {"family": "Cognitive function and executive processes",
                 "subdomain": "Cognitive Function Executive Processes and Brain Health",
                 "definition": "Cognitive resources and executive processes "
                 "affected by or affecting pain.",
                 "members": ["attention", "working memory", "executive function",
                             "cognitive load", "mental fatigue or brain fog",
                             "processing speed"],
                 "frameworks": ["cognitive resource models", "interruption models"]},
                {"family": "Motivation, goal pursuit and engagement",
                 "subdomain": "Motivational Processes Goal Pursuit and Engagement",
                 "definition": "Motivational dynamics of pursuing valued goals "
                 "alongside pain.",
                 "members": ["goal pursuit", "goal conflict",
                             "approach and avoidance motivation",
                             "activity engagement", "endurance", "valued goals"],
                 "frameworks": ["self-regulation", "motivational control theory",
                                "goal pursuit models"]},
                {"family": "Healthcare-seeking, adherence and engagement",
                 "subdomain": "Healthcare Seeking Treatment Adherence and Engagement",
                 "definition": "Behaviour around seeking, engaging with, and "
                 "adhering to care.",
                 "members": ["treatment adherence", "healthcare utilization",
                             "help-seeking", "therapeutic alliance",
                             "treatment expectations", "engagement"],
                 "frameworks": ["health belief model",
                                "common-sense model of self-regulation",
                                "expectancy models"]},
                {"family": "Emotion regulation and pain affect processing",
                 "subdomain": "Emotional Regulation and Pain Affect Processing",
                 "definition": "Strategies for regulating emotion and processing "
                 "the affective dimension of pain.",
                 "members": ["emotion regulation", "expressive suppression",
                             "cognitive reappraisal", "alexithymia",
                             "emotional awareness", "affect regulation"],
                 "frameworks": ["process model of emotion regulation",
                                "emotion regulation frameworks"]},
                {"family": "Mental health comorbidity and wellbeing",
                 "subdomain": "Mental Health Comorbidity and Psychological Wellbeing",
                 "definition": "Co-occurring mental health conditions and overall "
                 "psychological wellbeing.",
                 "members": ["comorbid depression or anxiety", "distress",
                             "quality of life", "wellbeing", "suicidality",
                             "life satisfaction"],
                 "frameworks": ["mutual maintenance model",
                                "biopsychosocial wellbeing models"]},
            ],
        },
        {
            "kind": "list",
            "id": "analytic-role",
            "title": "Analytic Role",
            "items": [
                "Preserves the original detected concepts while improving "
                "comparability across records.",
                "Supports synthesis questions about which psychological concepts "
                "and frameworks dominate the corpus.",
                "Allows concept families to be interpreted above raw string "
                "matching.",
            ],
        },
    ],
}


# --------------------------------------------------------------------------
# SCHEME 6 -- BPS ontology and semantic loading benchmark scheme
# --------------------------------------------------------------------------

SCHEME_6 = {
    "id": "scheme_6",
    "num": 6,
    "title": "BPS Ontology and Semantic Loading Benchmark Scheme",
    "subtitle": "Ontology prompts for benchmark-relative semantic quantification",
    "tagline": "Ontology-prompted embedding benchmark with TF-IDF fallback",
    "stage": "Synthesis",
    "stage_key": "ontology",
    "meta": {
        "Workflow position": "Semantic loading and synthesis after Stage 2 "
            "coding.",
        "Operational mode": "Ontology-prompted embedding benchmark, with "
            "OpenRouter embeddings when available and TF-IDF fallback otherwise.",
        "Unit of analysis": "One composed record string (title, abstract, "
            "objective) scored against domain and subdomain prompts.",
        "Provenance basis": "semantic_loading.py and the archived ontology terms "
            "JSON (openai/text-embedding-3-small).",
    },
    "rqs": ["RQ2 (scope and balance via benchmark-relative domain loading)"],
    "sources": [
        "src/03_pipeline/bps_review/reporting/semantic_loading.py",
        "src/07_semantic_space/semantic_loading/ontology/ontology_terms.json",
        "src/07_semantic_space/semantic_loading/analysis/domain_loading_summary.csv",
        "src/07_semantic_space/semantic_loading/analysis/subdomain_loading_summary.csv",
    ],
    "outputs": [
        "Record-level domain and subdomain loadings",
        "Domain summary and subdomain summary tables",
        "Pairwise loading analyses",
        "Dominance-by-review-type summaries",
    ],
    "sections": [
        {
            "kind": "prose",
            "id": "purpose",
            "title": "Purpose",
            "body": [
                "This scheme supplies the ontology scaffold used to quantify "
                "semantic emphasis across biological, psychological, and social "
                "axes. It is not a manual adjudication form, but it is an "
                "operational text-classification framework: it standardizes the "
                "domain and subdomain prompts against which review records are "
                "embedded and compared.",
                "Because loadings are benchmark-relative, the ontology prompts are "
                "the measuring instrument. Their wording and coverage directly "
                "determine the domain-balance results, so they warrant expert "
                "scrutiny.",
            ],
        },
        {
            "kind": "keyvals",
            "id": "scoring",
            "title": "Scoring Construction",
            "intro": "How a record is scored.",
            "items": [
                ("Record text", "Title, abstract, and extracted objective text "
                 "composed into one semantic analysis string."),
                ("Domain prompt", "{domain} chronic pain ontology. {joined "
                 "subdomains}."),
                ("Biological prompt assembly", "Shared biological core plus the "
                 "active track's extension: shared core plus musculoskeletal for "
                 "the musculoskeletal review, shared core plus neuropathic for "
                 "the neuropathic review. Psychological and social prompts are "
                 "identical across both tracks."),
                ("Subdomain prompt", "{domain} chronic pain subdomain. {term}."),
                ("Domain order", "Always biological, psychological, social."),
                ("Embedding", "OpenRouter openai/text-embedding-3-small when "
                 "available; TF-IDF cosine fallback otherwise."),
            ],
        },
        {
            "kind": "prose",
            "id": "bio-structure",
            "title": "Biological Ontology: Shared Core Plus Track Extensions",
            "body": [
                "The biological pole is the one place where a single uniform "
                "subdomain list would distort the science, because the biological "
                "mechanisms of musculoskeletal and neuropathic pain genuinely "
                "differ. The biological ontology is therefore built as a shared "
                "core that applies to both reviews, plus a musculoskeletal "
                "extension used only for the musculoskeletal review and a "
                "neuropathic extension used only for the neuropathic review.",
                "When a record is scored, the biological domain prompt is "
                "assembled from the shared core plus the active track's "
                "extension. The psychological and social ontologies below are "
                "identical for both reviews. This keeps the two reviews "
                "comparable on the psychological and social axes while measuring "
                "each one's biology against the mechanisms that actually matter "
                "for it.",
            ],
        },
        {
            "kind": "list",
            "id": "biological",
            "title": "Biological Subdomains: Shared Core (9)",
            "domain": "biological",
            "feedback": True,
            "intro": "Applied to both the musculoskeletal and neuropathic reviews.",
            "items": [
                "Central Sensitization and Neuroplasticity",
                "Nociceptive and Pain Signalling Pathways",
                "Immune Inflammatory and Neuroinflammatory Processes",
                "Neuroimaging Brain Structure and Function",
                "Genetic Epigenetic and Biological Vulnerability",
                "Sleep Disruption and Circadian Dysregulation",
                "Pharmacological and Biomedical Treatment",
                "Metabolic Nutritional and Hormonal Factors",
                "Physical Function Mobility and Deconditioning",
            ],
        },
        {
            "kind": "list",
            "id": "biological-msk",
            "title": "Biological Extension: Musculoskeletal (3)",
            "domain": "biological",
            "feedback": True,
            "intro": "Added only for the musculoskeletal review.",
            "items": [
                "Musculoskeletal and Structural Pathology",
                "Joint Degeneration and Osteoarthritic Change",
                "Muscle Tendon and Soft Tissue Pathology",
            ],
        },
        {
            "kind": "list",
            "id": "biological-neuro",
            "title": "Biological Extension: Neuropathic (5)",
            "domain": "biological",
            "feedback": True,
            "intro": "Added only for the neuropathic review.",
            "items": [
                "Peripheral Nerve Injury and Neuropathy",
                "Sensory Phenotype and Quantitative Sensory Testing",
                "Ectopic Firing Ion Channels and Neuronal Excitability",
                "Small Fiber and Nerve Conduction Pathology",
                "Deafferentation and Central Neuropathic Mechanisms",
            ],
        },
        {
            "kind": "list",
            "id": "psychological",
            "title": "Psychological Subdomains (20, uniform across both reviews)",
            "domain": "psychological",
            "feedback": True,
            "intro": "Identical for the musculoskeletal and neuropathic reviews.",
            "items": [
                "Catastrophizing and Negative Cognitive Appraisal",
                "Fear Avoidance and Pain Related Fear",
                "Depression Emotional Distress and Affect",
                "Anxiety and Psychological Reactivity",
                "Self Efficacy Control Beliefs and Perceived Mastery",
                "Acceptance Psychological Flexibility and Mindfulness",
                "Pain Coping Strategies and Adjustment",
                "Attention Vigilance and Pain Processing",
                "Illness Beliefs Pain Representations and Meaning",
                "Cognitive Behavioral and Psychotherapeutic Approaches",
                "Third Wave Therapies ACT and Contextual Approaches",
                "Resilience Positive Psychology and Post Traumatic Growth",
                "Identity Self Concept and Chronic Pain Biography",
                "Trauma Adverse Childhood and Life Events",
                "Personality Psychological Traits and Individual Differences",
                "Cognitive Function Executive Processes and Brain Health",
                "Motivational Processes Goal Pursuit and Engagement",
                "Healthcare Seeking Treatment Adherence and Engagement",
                "Emotional Regulation and Pain Affect Processing",
                "Mental Health Comorbidity and Psychological Wellbeing",
            ],
        },
        {
            "kind": "list",
            "id": "social",
            "title": "Social Subdomains (12, uniform across both reviews)",
            "domain": "social",
            "feedback": True,
            "intro": "Identical for the musculoskeletal and neuropathic reviews.",
            "items": [
                "Social Support Network and Interpersonal Resources",
                "Work Disability Occupational Function and Productivity",
                "Family Caregiver and Household Dynamics",
                "Socioeconomic Status and Health Inequity",
                "Healthcare Access Navigation and System Factors",
                "Cultural Ethnic and Demographic Context",
                "Community Participation and Social Role Functioning",
                "Legal Compensation and Medicolegal Systems",
                "Health Literacy Education and Patient Empowerment",
                "Stigma Social Isolation and Exclusion",
                "Return to Work Vocational Rehabilitation and Employment",
                "Social Determinants of Pain and Environment",
            ],
        },
        {
            "kind": "prose",
            "id": "interpretive-note",
            "title": "Interpretive Note",
            "body": [
                "This is an analytic coding scheme, not a human extraction "
                "codebook. It is included because the project uses ontology "
                "prompts as a standardized semantic coding scaffold, and because "
                "the instrument's design determines the domain-balance results "
                "that answer RQ2.",
            ],
        },
    ],
}


SCHEMES = [SCHEME_1, SCHEME_2, SCHEME_3, SCHEME_4, SCHEME_5, SCHEME_6]
