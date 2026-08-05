# Stage 2 Codebook

Stage 2 is abstract-level coding for all eligible chronic pain reviews.

## Core Variables

- `record_id`: internal pipeline identifier.
- `source_database`: `pubmed`, `wos`, `psycinfo`, or supplementary.
- `pmid`: PubMed identifier if available.
- `title`
- `abstract`
- `year`
- `journal`
- `authors`
- `country_contact_author`
- `review_type`: systematic review, meta-analysis, scoping review, narrative review, umbrella review, mixed, unclear.
- `objective_text`: verbatim objective from abstract where possible.
- `objective_category`: conceptual, clinical, methodological, epidemiological, treatment-oriented, mixed, unclear.
- `icd11_pain_category`: chronic primary pain, chronic secondary musculoskeletal pain, chronic neuropathic pain, chronic cancer-related pain, chronic postsurgical/posttraumatic pain, chronic headache/orofacial pain, visceral pain, unclear, mixed.
- `musculoskeletal_flag`: yes, no, unclear.
- `bps_mention_location`: title only, abstract only, title and abstract, unclear.
- `bps_function`: justification, organizing principle, background framing, intervention rationale, conclusion, policy/practice implication, rhetorical label, unclear.
- `bio_mentioned`: yes, no.
- `psych_mentioned`: yes, no.
- `social_mentioned`: yes, no.
- `reported_quality_assessment`: yes, no, unclear.
- `psychological_concepts_detected`: normalized pipe-delimited list.
- `theoretical_frameworks_detected`: normalized pipe-delimited list.
- `conceptual_problem_flags`: provisional abstract-level conceptual problems.
- `provisional_typology`: potential integrative signal, multifactorial signal, pseudo-BPS or partial signal, rhetorical label signal.
- `stage3_priority`: high, medium, low.
- `coding_rationale`: one-sentence abstract-based rationale supporting the coding bundle.
- `stage1_decision`: include, exclude, maybe.
- `stage1_exclusion_reason`: controlled reason string when excluded.
- `notes`

## Decision Notes

- Stage 2 follows successful Stage 1 inclusion.
- ICD-11 coding is abstract-based and may remain `unclear`.
- Reviews coded `musculoskeletal_flag=yes` or `unclear` are candidates for Stage 3 full-text retrieval.
- Stage 2 is now executed through structured LLM-first semantic coding with archived JSON batch outputs, fixed label vocabularies, and deterministic metadata fields for reproducibility.
- Domain coding should not treat the single lexical token `biopsychosocial` as sufficient evidence that all three domains are substantively present.
- LLM outputs remain reviewable and do not replace final human adjudication.
