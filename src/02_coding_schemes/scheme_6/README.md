# Scheme 6: BPS Ontology and Semantic Loading Benchmark Scheme

> **Status: DRAFT FOR EXPERT EVALUATION.** These coding schemes are a working draft circulated for expert evaluation. They have not been applied to a final review corpus. The current manuscript is a test run that exercised an earlier, coarser generation of these schemes. The workflow itself has since been validated end to end in two cross-provider test runs, in which three large language models from three different providers applied the abstract-level and the full-text scheme independently and their agreement was quantified per coded field. The full run on the review corpus is deliberately held until this evaluation is complete.

*Ontology prompts for benchmark-relative semantic quantification*

Ontology-prompted embedding benchmark with TF-IDF fallback.

## What this scheme does

This scheme supplies the ontology scaffold used to quantify semantic emphasis across biological, psychological, and social axes. It is not a manual adjudication form, but it is an operational text-classification framework: it standardizes the domain and subdomain prompts against which review records are embedded and compared.

Because loadings are benchmark-relative, the ontology prompts are the measuring instrument. Their wording and coverage directly determine the domain-balance results, so they warrant expert scrutiny.

## At a glance

| Property | Value |
| --- | --- |
| Workflow position | Semantic loading and synthesis after Stage 2 coding. |
| Operational mode | Ontology-prompted embedding benchmark, with OpenRouter embeddings when available and TF-IDF fallback otherwise. |
| Unit of analysis | One composed record string (title, abstract, objective) scored against domain and subdomain prompts. |
| Provenance basis | semantic_loading.py and the archived ontology terms JSON (openai/text-embedding-3-small). |
| Research questions | RQ2 (scope and balance via benchmark-relative domain loading) |

## Files in this folder

- [`scheme_6.html`](scheme_6.html) is the interactive evaluation surface. Open it in a browser, record a verdict and comments per section, then export your feedback as JSON.
- [`scheme_6.pdf`](scheme_6.pdf) is the formal dossier for sharing and printing.
- [`scheme_6.tex`](scheme_6.tex) is the LaTeX source (generated from `_build/content.py`).

## Coded fields

This scheme is specified through its prompts, seeds, and ontology rather than a single coded-field table. See the HTML or PDF for the full specification.

## Canonical source paths

- `src/03_pipeline/bps_review/reporting/semantic_loading.py`
- `src/07_semantic_space/semantic_loading/ontology/ontology_terms.json`
- `src/07_semantic_space/semantic_loading/analysis/domain_loading_summary.csv`
- `src/07_semantic_space/semantic_loading/analysis/subdomain_loading_summary.csv`

## Regenerating this dossier

All three surfaces (PDF, HTML, README) are generated from one source of truth:

```bash
cd src/coding_schemes/_build
python3 build.py
```

Edit the scheme content in `_build/content.py`, not the generated files.
