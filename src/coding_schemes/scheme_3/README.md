# Scheme 3: Stage 3 Full-Text Deep Coding Scheme

> **Status: DRAFT FOR EXPERT EVALUATION.** These coding schemes are a working draft circulated for expert evaluation. They have not been applied to a final review corpus. The current manuscript is a test run that exercised an earlier, coarser generation of these schemes in the Python workflow with an LLM (gemini-2.5-flash); the full end-to-end run is deliberately held until after expert feedback, and the refinements proposed here are awaiting sign-off before any re-run.

*Full-text adjudication and interpretive coding for the musculoskeletal and neuropathic reviews*

One uniform instrument applied to both pain-condition tracks; human-coded with pilot and reliability subsamples.

## What this scheme does

This scheme is the full-text deep coding framework for Stage 3 candidate reviews. It is applied as one uniform instrument to both planned reviews: the musculoskeletal chronic pain review and the neuropathic chronic pain review. The pain-condition family is the varying input that decides which records each review reads; the coding fields, value vocabularies, and anchors are identical across both tracks so the two reviews stay directly comparable.

It captures conceptual depth that cannot be resolved reliably at the abstract level: coverage of each BPS domain, pairwise and triadic integration quality, biopsychosocial typology, psychological concepts, theoretical frameworks, conceptual problems, and evidential quotations.

Stage 3 is where the review's central claim is tested: does a BPS-labelled review actually integrate the three domains, and if so, how.

## At a glance

| Property | Value |
| --- | --- |
| Workflow position | Full-text coding after Stage 3 candidate identification and retrieval triage. |
| Operational mode | Human-coded template with pilot and reliability subsamples. AI may assist concept mapping; final adjudication is human. |
| Unit of analysis | One retrieved full-text review, coded against the complete text. |
| Provenance basis | The generated full-text template and the prose Stage 3 codebook. |
| Research questions | RQ2 (scope, balance, integration); RQ3 (concepts, frameworks, definitions); SQ1 (conceptual problems) |

## Files in this folder

- [`scheme_3.html`](scheme_3.html) is the interactive evaluation surface. Open it in a browser, record a verdict and comments per section, then export your feedback as JSON.
- [`scheme_3.pdf`](scheme_3.pdf) is the formal dossier for sharing and printing.
- [`scheme_3.tex`](scheme_3.tex) is the LaTeX source (generated from `_build/content.py`).

## Proposed refinements awaiting expert sign-off

These are the enhancements that raise semantic resolution. They are proposals only and are not yet applied to the pipeline:

- **Concept-Level Grid and Evidence Rule.** The current scheme stores concepts as a flat delimited string, which loses the structure that RQ3 needs. We propose a per-concept grid with one row per concept carrying: canonical concept name, verbatim label used in the review, definitional status (formally defined, operationalized only, named only), parent psychological subdomain (aligned to Scheme 6), associated framework, and a relation type to other concepts (is-a, part-of, or associative).

## Coded fields

### Domain Coverage Fields

- `domain_coverage_bio` (elaborated, mentioned, minimal, absent): Depth of biological content.
- `domain_coverage_psych` (elaborated, mentioned, minimal, absent): Depth of psychological content.
- `domain_coverage_social` (elaborated, mentioned, minimal, absent): Depth of social content.

### Integration Fields (the core RQ2 contribution)

- `integration_bio_psych` (mechanistic, directional, descriptive, mentioned, none): Biological to psychological integration.
- `integration_psych_social` (mechanistic, directional, descriptive, mentioned, none): Psychological to social integration.
- `integration_bio_social` (mechanistic, directional, descriptive, mentioned, none): Biological to social integration.
- `integration_triadic` (mechanistic, descriptive, partial, none): Three-domain integration.
- `integration_mechanism_summary` (free text): Concise free-text summary of the proposed cross-domain pathways.

### Typology and Balance

- `overall_balance` (balanced, psych-dominant, bio-dominant, social-dominant, dyadic, unclear): Relative emphasis across domains.
- `bps_typology` (true_integrative, multifactorial, pseudo_bps, rhetorical_bps, narrow_despite_label, unclear): Full-text BPS operationalization type.

### Psychological Concepts and Evidence

- `concept_definitions_present` (yes, partial, no): Whether the review defines the psychological constructs it uses.
- `psychological_concepts_fulltext` (free text): Normalized, semicolon-delimited full-text concept list.
- `theoretical_frameworks_fulltext` (free text): Normalized, semicolon-delimited framework list.
- `conceptual_problems_fulltext` (free text): Conceptual issues such as vague definitions, construct overlap, tokenistic BPS use, missing social analysis, missing biology, mechanistic absence, or unclear boundaries.
- `integration_quotes_or_evidence` (free text): Supporting quotations, section references, or evidential anchors from the full text.
- `coder_id / coder_notes / adjudication_status`: Provenance and adjudication tracking fields.

## Canonical source paths

- `src/protocol/codebooks/stage3_codebook.md`
- `src/review_stages/04_extraction/codebooks/stage3_codebook.csv`
- `src/bps_review/extraction/stage3_prep.py`
- `src/review_stages/04_extraction/forms/stage3_fulltext_coding_template.csv`
- `src/review_stages/04_extraction/forms/stage3_pilot_sample.csv`
- `src/review_stages/04_extraction/forms/stage3_reliability_sample.csv`

## Regenerating this dossier

All three surfaces (PDF, HTML, README) are generated from one source of truth:

```bash
cd src/coding_schemes/_build
python3 build.py
```

Edit the scheme content in `_build/content.py`, not the generated files.
