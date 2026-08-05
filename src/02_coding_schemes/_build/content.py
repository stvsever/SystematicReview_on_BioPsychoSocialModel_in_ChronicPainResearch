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


def field(name, construct, values=None, notes=None, free_text=False):
    """One coded field."""
    return {
        "name": name,
        "construct": construct,
        "values": values or [],
        "notes": notes or "",
        "free_text": free_text,
    }


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
        "src/06_review_stages/03_screening/README.md",
        "src/01_protocol/osf/OSF_registration_HTBMFCPR.md",
    ],
    "outputs": [
        "src/06_review_stages/03_screening/outputs/stage1_screening.csv",
        "src/06_review_stages/03_screening/audit/stage1_screening_summary.csv",
        "src/06_review_stages/03_screening/audit/reliability_report.csv",
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
        "src/06_review_stages/04_extraction/codebooks/stage2_codebook.csv",
        "src/03_pipeline/bps_review/extraction/stage2.py",
        "src/03_pipeline/bps_review/extraction/llm_stage2.py",
        "src/06_review_stages/04_extraction/outputs/stage2_abstract_coding.csv",
        "src/06_review_stages/04_extraction/outputs/stage2_llm_structured_coding.csv",
        "src/06_review_stages/04_extraction/outputs/llm_stage2_structured_batches.jsonl",
    ],
    "outputs": [
        "src/06_review_stages/04_extraction/outputs/stage2_abstract_coding.csv",
        "src/06_review_stages/04_extraction/forms/stage2_double_code_subset.csv",
        "src/06_review_stages/04_extraction/outputs/stage2_objective_llm_assist.csv",
        "src/06_review_stages/04_extraction/outputs/llm_objective_pilot.json",
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
# SCHEME 3 -- Stage 3 full-text deep coding scheme
# --------------------------------------------------------------------------

SCHEME_3 = {
    "id": "scheme_3",
    "num": 3,
    "title": "Stage 3 Full-Text Deep Coding Scheme",
    "subtitle": "Full-text adjudication and interpretive coding for the "
                "musculoskeletal and neuropathic reviews",
    "tagline": "One uniform instrument applied to both pain-condition tracks; "
               "human-coded with pilot and reliability subsamples",
    "stage": "Stage 3",
    "stage_key": "fulltext",
    "meta": {
        "Workflow position": "Full-text coding after Stage 3 candidate "
            "identification and retrieval triage.",
        "Operational mode": "Human-coded template with pilot and reliability "
            "subsamples. AI may assist concept mapping; final adjudication is "
            "human.",
        "Unit of analysis": "One retrieved full-text review, coded against the "
            "complete text.",
        "Provenance basis": "The generated full-text template and the prose "
            "Stage 3 codebook.",
    },
    "rqs": ["RQ2 (scope, balance, integration)",
            "RQ3 (concepts, frameworks, definitions)", "SQ1 (conceptual problems)"],
    "sources": [
        "src/01_protocol/codebooks/stage3_codebook.md",
        "src/06_review_stages/04_extraction/codebooks/stage3_codebook.csv",
        "src/03_pipeline/bps_review/extraction/stage3_prep.py",
        "src/06_review_stages/04_extraction/forms/stage3_fulltext_coding_template.csv",
        "src/06_review_stages/04_extraction/forms/stage3_pilot_sample.csv",
        "src/06_review_stages/04_extraction/forms/stage3_reliability_sample.csv",
    ],
    "outputs": [
        "src/06_review_stages/04_extraction/forms/stage3_fulltext_coding_template.csv",
        "src/06_review_stages/04_extraction/forms/stage3_pilot_sample.csv",
        "src/06_review_stages/04_extraction/forms/stage3_reliability_sample.csv",
        "src/06_review_stages/04_extraction/outputs/stage3_candidate_manifest.csv",
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
                "It captures conceptual depth that cannot be resolved reliably at "
                "the abstract level: coverage of each BPS domain, pairwise and "
                "triadic integration quality, biopsychosocial typology, "
                "psychological concepts, theoretical frameworks, conceptual "
                "problems, and evidential quotations.",
                "Stage 3 is where the review's central claim is tested: does a "
                "BPS-labelled review actually integrate the three domains, and if "
                "so, how.",
            ],
        },
        {
            "kind": "fields",
            "id": "coverage",
            "title": "Record Routing and Domain Coverage Fields",
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
                field("review_track",
                      "Which review this coded record belongs to. The coding "
                      "fields are uniform across both; the track only tunes which "
                      "biological ontology extension the coder reads the "
                      "biological domain against.",
                      values=[v("musculoskeletal", "Coded for the musculoskeletal "
                                "chronic pain review."),
                              v("neuropathic", "Coded for the neuropathic chronic "
                                "pain review.")]),
                field("pain_condition_detail",
                      "Free-text specification of the exact pain condition studied "
                      "(for example knee osteoarthritis, painful diabetic "
                      "neuropathy).", free_text=True),
                field("domain_coverage_bio", "Depth of biological content, read "
                      "against the track-appropriate biological mechanisms.",
                      values=DOMAIN_COVERAGE_LADDER),
                field("domain_coverage_psych", "Depth of psychological content.",
                      values=DOMAIN_COVERAGE_LADDER),
                field("domain_coverage_social", "Depth of social content.",
                      values=DOMAIN_COVERAGE_LADDER),
            ],
        },
        {
            "kind": "fields",
            "id": "integration",
            "title": "Integration Fields (the core RQ2 contribution)",
            "feedback": True,
            "field_feedback": True,
            "intro": "Integration is the scheme's highest-resolution construct. "
                     "The pairwise ladder distinguishes a stated mechanism from a "
                     "mere direction, an association, or a bare co-mention.",
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
                field("integration_mechanism_summary",
                      "Concise free-text summary of the proposed cross-domain "
                      "pathways.", free_text=True),
            ],
        },
        {
            "kind": "fields",
            "id": "typology-balance",
            "title": "Typology and Balance",
            "feedback": True,
            "field_feedback": True,
            "intro": "The summary judgments that answer RQ1 at full-text depth.",
            "fields": [
                field("overall_balance",
                      "Relative emphasis across domains.",
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
                      ]),
            ],
        },
        {
            "kind": "fields",
            "id": "concepts",
            "title": "Psychological Concepts and Evidence",
            "feedback": True,
            "field_feedback": True,
            "intro": "Concept-level fields that feed RQ3 and the concept map.",
            "fields": [
                field("concept_definitions_present",
                      "Whether the review defines the psychological constructs it "
                      "uses.",
                      values=[
                          v("yes", "Constructs are explicitly defined or "
                            "operationalized."),
                          v("partial", "Some constructs defined, others named "
                            "only."),
                          v("no", "Constructs named without definition."),
                      ]),
                field("psychological_concepts_fulltext",
                      "Normalized, semicolon-delimited full-text concept list.",
                      free_text=True),
                field("theoretical_frameworks_fulltext",
                      "Normalized, semicolon-delimited framework list.",
                      free_text=True),
                field("conceptual_problems_fulltext",
                      "Conceptual issues such as vague definitions, construct "
                      "overlap, tokenistic BPS use, missing social analysis, "
                      "missing biology, mechanistic absence, or unclear "
                      "boundaries.", free_text=True),
                field("integration_quotes_or_evidence",
                      "Supporting quotations, section references, or evidential "
                      "anchors from the full text.", free_text=True),
                field("coder_id / coder_notes / adjudication_status",
                      "Provenance and adjudication tracking fields."),
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
        "src/06_review_stages/04_extraction/forms/stage3_manual_relevance_checklist.csv",
        "src/06_review_stages/04_extraction/outputs/stage3_candidate_manifest.csv",
        "src/06_review_stages/04_extraction/outputs/stage3_manual_fulltext_queue.csv",
        "src/06_review_stages/04_extraction/outputs/stage3_retrieval_validation.csv",
    ],
    "outputs": [
        "src/06_review_stages/04_extraction/outputs/stage3_candidate_manifest.csv",
        "src/06_review_stages/04_extraction/forms/stage3_manual_relevance_checklist.csv",
        "src/06_review_stages/04_extraction/outputs/stage3_manual_fulltext_queue.csv",
        "src/06_review_stages/04_extraction/outputs/stage3_retrieval_validation.csv",
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
        "src/05_data/interim/extraction/llm_concept_clusters.json",
    ],
    "outputs": [
        "src/05_data/interim/extraction/llm_concept_clusters.json",
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
