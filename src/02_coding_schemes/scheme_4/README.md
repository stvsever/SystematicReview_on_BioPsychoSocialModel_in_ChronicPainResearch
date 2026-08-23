# Scheme 4: Stage 3 Retrieval and Manual Relevance Triage Scheme

> **Status: DRAFT FOR EXPERT EVALUATION.** These coding schemes are a working draft circulated for expert evaluation. They have not been applied to a final review corpus. The current manuscript is a test run that exercised an earlier, coarser generation of these schemes. The workflow itself has since been validated end to end in two cross-provider test runs, in which three large language models from three different providers applied the abstract-level and the full-text scheme independently and their agreement was quantified per coded field. The full run on the review corpus is deliberately held until this evaluation is complete.

*Full-text availability, retrieval need, and adjudication checklist*

Automated candidate manifest plus human checklist completion.

## What this scheme does

This scheme governs the transition from Stage 2 abstract coding to Stage 3 full-text work. It standardizes which candidate reviews need manual retrieval, which records require manual relevance adjudication, and how retrieval status, risk signals, and reviewer decisions are recorded before deep coding begins.

It is not a logistics file. It is a standardized adjudication framework that decides whether a review can enter Stage 3, whether the retrieved text is adequate, and how problematic or ambiguous records are escalated for human judgment.

## At a glance

| Property | Value |
| --- | --- |
| Workflow position | Bridge between Stage 2 coding and Stage 3 full-text coding. |
| Operational mode | Automated candidate manifest plus human checklist completion. |
| Unit of analysis | One Stage 3 candidate record and its retrieval state. |
| Provenance basis | Manifest-generation logic in stage3_prep.py and the generated manual relevance checklist. |
| Research questions | Retrieval and adjudication gate that protects RQ2 and RQ3 corpus quality |

## Files in this folder

- [`scheme_4.html`](scheme_4.html) is the interactive evaluation surface. Open it in a browser, record a verdict and comments per section, then export your feedback as JSON.
- [`scheme_4.pdf`](scheme_4.pdf) is the formal dossier for sharing and printing.
- [`scheme_4.tex`](scheme_4.tex) is the LaTeX source (generated from `_build/content.py`).

## Coded fields

### Manifest Fields

- `fulltext_status` (manual_retrieval_required, pmc_open_available_not_cached, pmc_fulltext_cached, pmc_fulltext_fetched, pmc_linked_fetch_failed, pmc_fulltext_low_content_manual_check): Operational retrieval status of the candidate.
- `review_track` (musculoskeletal, neuropathic, both): Which review pool the candidate belongs to, carried from Scheme 2. Retrieval and triage logic itself is uniform across both tracks.
- `retrieval_source`: Source of the PMCID or full-text link (existing metadata, PubMed elink, or Europe PMC).
- `fulltext_word_count` (free text): Word count of cached text, used to detect low-content full texts.
- `manual_retrieval_needed` (yes, no): Whether a human must retrieve the text.
- `manual_relevance_priority` (high, medium, low): Adjudication urgency.
- `manual_relevance_flags` (free text): Pipe-delimited signal list describing why adjudication is needed. See the signal logic below.
- `osf_manual_adjudication_required`: Currently set to yes for the checklist workflow.
- `cached_xml_path / cached_text_path`: Relative cache paths for machine-fetched PMC files.
- `pubmed_url`: Direct URL to the source record when a PMID is available.

### Manual Relevance Signal Logic

- `withdrawn_or_retracted_signal`: Assigned when the title or abstract indicates a withdrawn or retracted record. Forces high priority.
- `pain_focus_not_explicit`: Assigned when pain is not explicit in title or abstract. Forces high priority.
- `chronicity_not_explicit`: Assigned when chronic, persistent, or long-term framing is absent.
- `review_design_unclear`: Assigned when review design is missing or too vague.

### Reviewer Checklist Fields

- `reviewer_decision` (free text): Human reviewer decision after checking the candidate.
- `reviewer_notes` (free text): Notes supporting the reviewer decision.
- `adjudication_decision` (free text): Final adjudicated status after disagreement resolution.
- `adjudication_notes` (free text): Adjudication rationale.

## Canonical source paths

- `src/03_pipeline/bps_review/extraction/stage3_prep.py`
- `src/09_review_stages/04_extraction/forms/stage3_manual_relevance_checklist.csv`
- `src/09_review_stages/04_extraction/outputs/stage3_candidate_manifest.csv`
- `src/09_review_stages/04_extraction/outputs/stage3_manual_fulltext_queue.csv`
- `src/09_review_stages/04_extraction/outputs/stage3_retrieval_validation.csv`

## Regenerating this dossier

All three surfaces (PDF, HTML, README) are generated from one source of truth:

```bash
cd src/coding_schemes/_build
python3 build.py
```

Edit the scheme content in `_build/content.py`, not the generated files.
