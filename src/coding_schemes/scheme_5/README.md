# Scheme 5: Psychological Concept Clustering and Framework Mapping Scheme

> **Status: DRAFT FOR EXPERT EVALUATION.** These coding schemes are a working draft circulated for expert evaluation. They have not been applied to a final review corpus. The current manuscript is a test run built with an earlier, coarser generation of these schemes; the refinements proposed here are awaiting expert sign-off before any re-run.

*Second-order normalization of detected psychological concepts*

Fixed pattern detection upstream, LLM clustering for higher-order normalization.

## What this scheme does

This scheme standardizes higher-order concept mapping after concept detection. It groups extracted psychological concepts from chronic pain review records into interpretable families and links them to likely theoretical frameworks. The goal is cross-record comparability when the raw concepts are heterogeneous, overlapping, or variably named.

It is a second-order coding scheme: it does not classify whole records, it normalizes the concept vocabulary itself.

## At a glance

| Property | Value |
| --- | --- |
| Workflow position | Post-detection concept normalization over Stage 2 and Stage 3 concept strings. |
| Operational mode | Fixed pattern-based concept detection upstream, followed by LLM clustering for higher-order normalization. |
| Unit of analysis | The set of unique concept strings across the corpus, not whole records. |
| Provenance basis | coding.py concept and framework patterns plus the clustering prompt. |
| Research questions | RQ3 (which psychological concepts and frameworks dominate) |

## Files in this folder

- [`scheme_5.html`](scheme_5.html) is the interactive evaluation surface. Open it in a browser, record a verdict and comments per section, then export your feedback as JSON.
- [`scheme_5.pdf`](scheme_5.pdf) is the formal dossier for sharing and printing.
- [`scheme_5.tex`](scheme_5.tex) is the LaTeX source (generated from `_build/content.py`).

## Proposed refinements awaiting expert sign-off

These are the enhancements that raise semantic resolution. They are proposals only and are not yet applied to the pipeline:

- **Ontology-Aligned Lexicon and Typed Relations.** The upstream detector recognizes only 16 concepts, which caps the resolution of RQ3 and misses constructs that appear in the corpus (for example resilience, suffering, functioning, maladaptive beliefs, pain-related thoughts). We propose expanding the detection lexicon and aligning every concept family to one of the 20 psychological subdomains defined in Scheme 6, so the concept map and the semantic ontology share one vocabulary.

## Coded fields

### Cluster Output Schema (current)

- `clusters`: Top-level list of normalized concept clusters.
- `family`: Higher-order concept family label for the cluster.
- `members` (free text): The original concept strings grouped into that family.
- `possible_frameworks` (free text): Likely theoretical or therapeutic frameworks associated with the family.

## Canonical source paths

- `src/bps_review/extraction/coding.py`
- `src/protocol/codebooks/stage2_codebook.md`
- `src/protocol/codebooks/stage3_codebook.md`
- `src/data/interim/extraction/llm_concept_clusters.json`

## Regenerating this dossier

All three surfaces (PDF, HTML, README) are generated from one source of truth:

```bash
cd src/coding_schemes/_build
python3 build.py
```

Edit the scheme content in `_build/content.py`, not the generated files.
