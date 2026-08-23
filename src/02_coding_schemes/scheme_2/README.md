# Scheme 2: Stage 2 Abstract-Level Structured Coding Scheme

> **Status: DRAFT FOR EXPERT EVALUATION.** These coding schemes are a working draft circulated for expert evaluation. They have not been applied to a final review corpus. The current manuscript is a test run that exercised an earlier, coarser generation of these schemes. The workflow itself has since been validated end to end in two cross-provider test runs, in which three large language models from three different providers applied the abstract-level and the full-text scheme independently and their agreement was quantified per coded field. The full run on the review corpus is deliberately held until this evaluation is complete.

*Corpus-wide abstract coding of BPS usage, domains, and typology*

Structured LLM-first coding with deterministic repair and rule-based fallback.

## What this scheme does

This scheme standardizes abstract-level extraction for all eligible chronic pain reviews. It is the main corpus-wide coding layer used to describe review characteristics, classify the function of biopsychosocial language, detect biological, psychological, and social content, flag conceptual problems, and generate a provisional biopsychosocial typology for downstream synthesis.

It also sets the Stage 3 candidate gate: a record coded as musculoskeletal or as unspecified-but-not-excludable is carried forward for full-text work.

## At a glance

| Property | Value |
| --- | --- |
| Workflow position | Abstract coding for every Stage 1 included record, before Stage 3 candidate selection. |
| Operational mode | Structured LLM coding with archived JSON batches, deterministic vocabulary normalization, and a rule-based fallback when model output is incomplete or unavailable. |
| Unit of analysis | One included review record, coded from title, abstract, publication types, and journal metadata only. |
| Provenance basis | The actual Stage 2 output schema in stage2_abstract_coding.csv. |
| Research questions | RQ1 (BPS operationalization); RQ3 (concepts and frameworks); SQ1 (conceptual problems); Feeds the Stage 3 candidate gate |

## Files in this folder

- [`scheme_2.html`](scheme_2.html) is the interactive evaluation surface. Open it in a browser, record a verdict and comments per section, then export your feedback as JSON.
- [`scheme_2.pdf`](scheme_2.pdf) is the formal dossier for sharing and printing.
- [`scheme_2.tex`](scheme_2.tex) is the LaTeX source (generated from `_build/content.py`).

## Coded fields

### Coded Fields and Controlled Values

- `review_type` (systematic review, meta-analysis, network meta-analysis, umbrella review, scoping or mapping review, rapid review, realist review, integrative review, narrative or expert review, other evidence synthesis, unclear): Evidence-synthesis design as stated in the abstract or publication metadata.
- `objective_category` (conceptual, clinical, methodological, epidemiological, mixed, unclear): Primary stated purpose of the review.
- `objective_category_source`: Whether the objective classification came from the structured LLM, a repaired LLM batch, or the deterministic fallback.
- `icd11_pain_category` (chronic secondary musculoskeletal pain, chronic neuropathic pain, chronic cancer-related pain, chronic postsurgical or posttraumatic pain, chronic secondary headache or orofacial pain, chronic secondary visceral pain, chronic primary pain, mixed or unspecified chronic pain, unclear): ICD-11 aligned pain category inferred from the abstract.
- `musculoskeletal_flag` (yes, no, unclear): Whether the review concerns musculoskeletal pain. Routes records into the musculoskeletal review.
- `neuropathic_flag` (yes, no, unclear): Whether the review concerns neuropathic pain. Parallel to musculoskeletal_flag; routes records into the neuropathic review. A record may set both flags (for example a mixed review) or neither.
- `stage3_track` (musculoskeletal, neuropathic, both, none): Which review or reviews the record is routed to. Derived from the two flags; a record can belong to both pools.
- `bps_mention_location` (title only, abstract only, title and abstract, unclear): Where the BPS term appears.
- `bps_function` (explanatory framework, intervention rationale, organizing principle, justification, background framing, conclusion, policy/practice implication, rhetorical label, unclear): The rhetorical and analytic work the BPS label performs. This is the central RQ1 field and the finest-grained one.
- `bio_mentioned / psych_mentioned / social_mentioned` (yes, no): Binary presence of each domain in the abstract.
- `quality_assessment_reported` (yes, no, unclear): Whether the abstract reports a risk-of-bias or quality assessment.
- `psychological_concepts_detected` (free text): Normalized list of psychological concepts from title and abstract. Feeds Scheme 5.
- `theoretical_frameworks_detected` (free text): Normalized list of named frameworks or model labels.
- `conceptual_problem_flags` (vague_definition, tokenistic_bps, missing_social, missing_biology, mechanistic_absence, construct_overlap, parallel_listing_without_integration, none): Provisional abstract-level conceptual problems.
- `provisional_typology` (potential integrative signal, multifactorial signal, pseudo-bps or partial signal, rhetorical label signal): Abstract-level BPS signal, refined later at Stage 3.
- `stage3_candidate / stage3_priority` (yes / no, high / medium / low): Whether and how urgently the record proceeds to Stage 3.
- `coding_rationale` (free text): One-sentence rationale supporting the coding bundle.
- `coding_method / llm_model`: Operational provenance of the row (llm_structured, llm_batch_fallback, rule_based_fallback) and the model identifier when the LLM stage was used.

## Canonical source paths

- `src/01_protocol/codebooks/stage2_codebook.md`
- `src/09_review_stages/04_extraction/codebooks/stage2_codebook.csv`
- `src/03_pipeline/bps_review/extraction/stage2.py`
- `src/03_pipeline/bps_review/extraction/llm_stage2.py`
- `src/09_review_stages/04_extraction/outputs/stage2_abstract_coding.csv`
- `src/09_review_stages/04_extraction/outputs/stage2_llm_structured_coding.csv`
- `src/09_review_stages/04_extraction/outputs/llm_stage2_structured_batches.jsonl`

## Regenerating this dossier

All three surfaces (PDF, HTML, README) are generated from one source of truth:

```bash
cd src/coding_schemes/_build
python3 build.py
```

Edit the scheme content in `_build/content.py`, not the generated files.
