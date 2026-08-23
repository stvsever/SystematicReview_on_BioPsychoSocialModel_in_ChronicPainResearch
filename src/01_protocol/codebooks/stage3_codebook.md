# Stage 3 Codebook

Stage 3 is full-text deep coding for the reviews carried forward by Stage 2. It is one uniform
instrument for both planned reviews, the musculoskeletal one and the neuropathic one: the
pain-condition family decides which records each review reads, the fields and value vocabularies
are identical.

The scheme does two things at once. It **grades**: how deeply each domain is treated, and how
each pair of domains and the triad are integrated, on explicit ladders with a quoted passage
behind every rung. And it **extracts**: the specific things a review names. Recording that a
domain is present is not a coding. Recording which factors carry it, in which role, on the
strength of which sentence, is. Those named items are the nodes and edges of the biopsychosocial
ontology the synthesis builds.

The authoritative, machine-readable version of this codebook is generated from the schema the
pipeline runs, at
[`src/09_review_stages/04_extraction/codebooks/stage3_codebook.csv`](../../09_review_stages/04_extraction/codebooks/stage3_codebook.csv),
and the expert-facing dossier is
[`src/02_coding_schemes/scheme_3/`](../../02_coding_schemes/scheme_3/). This file is the prose
summary of the same scheme.

## A. Source, context, and routing

- `record_id`, `full_text_available`: yes, no, partial.
- `review_track`: musculoskeletal, neuropathic, mixed_or_other, unclear.
- `source_type`: the evidence-synthesis design confirmed at full text. `primary study` is an
  eligibility signal and routes the record to human adjudication.
- `icd11_pain_category`: the ICD-11 category, re-read from the full text so the abstract-level
  classification is checkable rather than final.
- `population`: adult, older adult, mixed ages, pediatric, unclear, not applicable.
- `care_setting`: primary care, secondary or tertiary specialist care, rehabilitation or
  multidisciplinary programme, occupational or workplace, community or population, mixed, not
  reported.
- `primary_discipline`: the disciplinary home of the paper, read from the journal and the
  framing rather than from affiliations alone.
- `pain_condition_detail` (free text), `pain_conditions` (open list), `context_note` (cultural or
  healthcare context, when stated).
- `quality_assessment_reported`: yes, no, unclear, plus `quality_assessment_tools`. Descriptive
  only; this review does not appraise the methodological quality of the reviews it studies.

## B. What the biopsychosocial label does (RQ1)

- `bps_label_used`: explicit_bps_term, variant_term_only, domain_language_only, absent.
- `bps_primary_function`: the dominant work the label does over the paper as a whole, from the
  same vocabulary the abstract-level scheme uses, plus two values only a full text supports,
  `critique or problematization` and `operational definition`.
- `bps_functions_present`: every function the label performs anywhere in the paper. A paper
  routinely does two or three at once, and which ones it combines is the finding.
- `bps_definition_status`: formally_defined, described_informally, cited_only, undefined.
  `undefined` is a finding, not a coding failure.
- `bps_model_variants`: the model labels the paper actually uses, verbatim. This is what makes
  terminological drift visible.
- `bps_usage_instances`: one item per passage where the label does work, with its function, its
  section, whether it is definitional, who the model is credited to, and the quote.
- `bps_definitions`: one item per place where the paper says what the model is, with the kind of
  definitional act, the attributed source, and the components it names.
- `bps_operationalization_summary`: at most 90 words on what the paper does with the model, as
  opposed to what it says about it.

## C. Domain coverage

- `domain_coverage_bio`, `domain_coverage_psych`, `domain_coverage_social`: elaborated,
  mentioned, minimal, absent.
- `coverage_lifestyle`, `coverage_spiritual_existential`: the same ladder for the two domains the
  registration names alongside the triad, kept out of the triad so they cannot inflate it.
- `domain_evidence`: one item per core domain not scored absent, carrying the constructs named,
  the ontology subdomains touched, and the passage that justifies the level.

## D. Which factors carry each domain

- `biological_factors`: label in the paper's own wording, ontology subdomain when one fits,
  mechanism level, role in the paper, quote, section, evidence basis.
- `social_factors`: the same structure, with the level of social organization the factor sits at.
- `other_domain_factors`: lifestyle, spiritual or existential, and environmental factors, kept
  visible rather than folded into the triad.
- Psychological constructs are recorded in section F, which carries the extra definitional
  fields, so no factor is written twice.

## E. Integration

- `integration_bio_psych`, `integration_psych_social`, `integration_bio_social`: mechanistic,
  directional, descriptive, mentioned, none.
- `integration_triadic`: mechanistic, descriptive, partial, none.
- `integration_claims`: one item per cross-domain passage, naming the source factor and the
  target factor, the direction, any named mediator or moderator, the quote, and the section. A
  pairwise field graded above `mentioned` should have at least one claim behind it, and the
  pipeline checks exactly that.
- `integration_mechanism_summary`: the pathways the paper actually proposes, at most 90 words.

## F. Concepts, relations, frameworks, and measures

- `concept_definitions_present`: yes, partial, no.
- `psychological_concepts`: one item per construct, with the paper's own label, the concept
  family when one fits, the definitional status (formally_defined, operationalized_only,
  described_informally, named_only, unclear), the defining passage, whose definition it is, the
  instrument it is measured with, and the role it plays.
- `concept_relations`: the hierarchical and semantic relationships the registration asks for, as
  edges: source concept, relation type, target concept, whether the paper states the relation or
  merely behaves as though it holds, and the quote. `conflated_without_comment` is available on
  purpose.
- `theoretical_frameworks`: label, role, the domains the model spans, attributed source, quote.
- `instruments`: label, abbreviation, domain measured, what the paper says the instrument
  captures, and the role it plays. What a review measures is the most concrete form its
  operationalization of the model takes.

## G. Typology, balance, and conceptual problems

- `overall_balance`: balanced, psych-dominant, bio-dominant, social-dominant, dyadic, unclear.
- `bps_typology`: true_integrative, multifactorial, pseudo_bps, rhetorical_bps,
  narrow_despite_label, unclear.
- `conceptual_problems`: one item per problem the paper names or displays, with its type, what it
  is about, the constructs it concerns, whether the authors point it out themselves, and the
  passage that shows it.

## H. Synthesis hooks (free text)

- `key_quotes`: the conceptually load-bearing passages, with why each was selected.
- `emergent_labels`: every important term the project vocabularies do not carry, written exactly
  as the paper writes it. This is how the ontology learns what it is missing.
- `conceptual_tensions`, `additional_observations`: open lists for what no other field holds.
- `synthesis_note`, `coding_rationale`.

## Interpretation rules

- `true_integrative`: explicit cross-domain causal or mechanistic interaction is central to the
  review's logic. Requires at least descriptive triadic integration plus two elaborated domains.
- `multifactorial`: multiple domains covered meaningfully but mostly in parallel.
- `pseudo_bps`: BPS label used but one or more core domains are thin or absent.
- `rhetorical_bps`: BPS invoked mainly as framing or justification without analytic substance.
- `narrow_despite_label`: BPS claimed but the substantive scope is essentially single-domain.
- The word biopsychosocial is never evidence of coverage. A review can use the label in every
  paragraph and still score a domain as absent.
- Integration is a claim about a relation, not about co-occurrence. Serial listing of biological,
  then psychological, then social factors is `none` for the triadic field, however long the
  lists are.
- Preferred labels are a spine, not a cage. A label is mapped onto the project vocabularies only
  when it clearly matches; anything else is kept as the review wrote it and repeated in
  `emergent_labels`. A mapped label never replaces the paper's own wording: the ontology anchor
  and the free-text label are separate fields and both are filled.
- Eligibility, conceptual yield, and synthesis priority are derived from the coded content by
  fixed rule, never asked of the coder. The derived verdict is a triage recommendation; the human
  decision is recorded in `adjudication_status`, as the registration requires.

## Reliability

Inter-rater reliability on 20 percent of Stage 3 records, capped at 20 full texts, with percent
agreement and Cohen's kappa on the controlled fields, adjacent agreement on the ordered ladders,
and set overlap on the open extraction lists. Discrepancies are resolved by discussion, with the
principal investigator adjudicating unresolved cases.
