# Scheme 6: BPS Ontology and Semantic Loading Benchmark Scheme

> **Status: DRAFT FOR EXPERT EVALUATION.** These coding schemes are a working draft circulated for expert evaluation. They have not been applied to a final review corpus. The current manuscript is a test run built with an earlier, coarser generation of these schemes; the refinements proposed here are awaiting expert sign-off before any re-run.

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

## Proposed refinements awaiting expert sign-off

These are the enhancements that raise semantic resolution. They are proposals only and are not yet applied to the pipeline:

- **Seed-Term Expansion and Coverage Audit.** Each subdomain prompt is currently just the subdomain label, so the benchmark is only as rich as those short phrases. We propose attaching three to five expert-curated seed terms to every subdomain (for example Central Sensitization and Neuroplasticity would carry windup, temporal summation, descending modulation, cortical reorganization). Richer prompts sharpen the semantic contrast between neighbouring subdomains and reduce the chance that a record loads on a subdomain by generic vocabulary alone.

## Coded fields

This scheme is specified through its prompts, seeds, and ontology rather than a single coded-field table. See the HTML or PDF for the full specification.

## Canonical source paths

- `src/bps_review/reporting/semantic_loading.py`
- `src/vector_db/semantic_loading/ontology/ontology_terms.json`
- `src/vector_db/semantic_loading/analysis/domain_loading_summary.csv`
- `src/vector_db/semantic_loading/analysis/subdomain_loading_summary.csv`

## Regenerating this dossier

All three surfaces (PDF, HTML, README) are generated from one source of truth:

```bash
cd src/coding_schemes/_build
python3 build.py
```

Edit the scheme content in `_build/content.py`, not the generated files.
