# Full-text test run (scheme 3)

This package applies the Stage 3 full-text deep coding scheme to whole
open-access articles, once per model, with three cheap models from three
different providers, and reports how much they agree and whether their evidence
is real.

It is the second half of a chain. The abstract-level run
([`bps_review.pilot`](../pilot/)) codes a PubMed sample and produces a consensus
candidate set; this package retrieves the open-access full texts of those
candidates and codes them at full resolution.

## What the scheme asks for

The review's central claim is that biopsychosocial language is widespread while
substantive three-domain integration is not. The scheme is built to test exactly
that, so its most consequential graded fields are ladders rather than labels. It
is also the pass that harvests the ontology, so alongside the grades it asks for
the named things a review carries: presence is never the answer, the answer is
which ones.

| Layer | Fields |
| --- | --- |
| Coverage | `domain_coverage_bio`, `domain_coverage_psych`, `domain_coverage_social` on a four-rung ladder (elaborated, mentioned, minimal, absent), plus `coverage_lifestyle` and `coverage_spiritual_existential` on the same ladder |
| Integration | `integration_bio_psych`, `integration_psych_social`, `integration_bio_social` on a five-rung ladder (mechanistic, directional, descriptive, mentioned, none), plus `integration_triadic` on its own four-rung ladder |
| BPS usage (RQ1) | `bps_label_used`, `bps_primary_function`, `bps_functions_present`, `bps_definition_status`, `bps_model_variants`, plus `bps_usage_instances` and `bps_definitions` as quoted item lists |
| Typology | `overall_balance`, `bps_typology` |
| Ontology nodes | `biological_factors`, `social_factors`, `other_domain_factors`, `psychological_concepts`, each item carrying the review's own label, its ontology anchor, the role it plays, and a quote |
| Ontology edges | `integration_claims` (naming both linked factors, the direction, and any mediator) and `concept_relations` (hierarchical and semantic relations between constructs) |
| Concepts and measures | `concept_definitions_present`, `theoretical_frameworks`, `instruments` |
| Problems (SQ1) | `conceptual_problems`, with scope, affected constructs, and whether the authors noticed |
| Free text | `emergent_labels`, `conceptual_tensions`, `additional_observations`, plus the summaries `bps_operationalization_summary`, `integration_mechanism_summary`, `synthesis_note` |

Thirteen extraction lists, seven open free-text lists, and a ceiling of 116
extracted items per coding. The caps are ceilings and never targets.

### The spine and the free text

Labels are mapped onto the project vocabularies in
[`coding/vocabulary.py`](coding/vocabulary.py) (the Scheme 6 subdomain ontology,
the Scheme 5 concept families, frameworks, instruments, pain conditions) only
when they clearly match. Two rules protect resolution:

- a mapped label never replaces the review's own wording, because the anchor
  field (`subdomain_label`, `concept_family`) always sits next to a free-text
  label field (`factor_label`, `concept_label`), and the item table carries both;
- a term the vocabularies do not carry is recorded as written and repeated in
  `emergent_labels`, and the analysis reports the off-spine share per list. That
  share measures the ontology against the literature, not the other way round.

## How it runs

```python
from bps_review.fulltext import run_fulltext_testrun_pipeline

out = run_fulltext_testrun_pipeline()                   # reuse cached corpus and codings
out = run_fulltext_testrun_pipeline(force_coding=True)  # re-code every paper via the API
```

```bash
python -m bps_review build-fulltext-corpus     # retrieve the open-access full texts
python -m bps_review run-fulltext-testrun      # analyse the cached codings
python -m bps_review run-fulltext-testrun --force-coding
python -m bps_review build-fulltext-graph      # rebuild only the knowledge graph
```

`--no-semantic` and `--no-graph` switch off the two enrichment steps. Both are
built on top of an already complete result, so a semantic-overlap failure
(network, credentials, provider outage) is reported and skipped rather than
allowed to invalidate the run.

One paper is one request. Papers are coded concurrently inside a model and the
models run in parallel, every attempt is wrapped in a hard wall-clock timeout,
and a paper that never codes is written as an explicit `coding_failed` row rather
than as a fabricated one.

## Module map

| Module | Responsibility |
| --- | --- |
| `config.py` | models, per-model runtime, field groups, caps, ladder depths, paths |
| `corpus/pmc.py` | resolve PMC ids, fetch and parse JATS, build and log the corpus |
| `coding/schema.py` | the validated scheme: controlled decisions and structured items |
| `coding/vocabulary.py` | the preferred-label vocabularies and the conservative normalizer |
| `coding/prompt.py` | the instruction set, assembled from the schema, the ladders, and the vocabularies |
| `coding/condense.py` | fits a long article into the reading budget, dropping the least conceptual paragraphs first |
| `coding/derive.py` | repair, deterministic derivations, presence flags, serialization |
| `coding/runner.py` | one article per request, all models in parallel, with retries and usage accounting |
| `analysis/reliability.py` | categorical agreement, adjacent agreement on the ladders, presence agreement, lexical list overlap |
| `analysis/semantic.py` | the same lists compared by meaning: embedded labels, soft Jaccard, vocabulary collapse |
| `analysis/integrity.py` | completeness, quote verification, evidence discipline, extraction yield |
| `visualization/figures.py` | five multi-panel figures, none with a figure-level title |
| `graph_export.py` | rebuild the knowledge graph from the cached tables, without re-deriving anything |
| `report.py` | the standalone summary |
| `pipeline.py` | `run_fulltext_testrun_pipeline()` |

The knowledge graph itself lives one level up, in
[`bps_review/graph/`](../graph/README.md), because it is a rendering of a coded
run rather than a part of scheme 3.

## Design commitments

- **Verdicts are derived, never asked.** Eligibility, conceptual yield, synthesis
  priority, the integration index, and every presence flag are computed from the
  coded content by a fixed rule, so the filter is auditable and identical across
  providers. Derived columns are recomputed on load, so a cached run always
  reports the current rules.
- **The typology is checked against itself.** The coder codes `bps_typology`, and
  the pipeline independently derives `derived_typology` from coverage and
  integration. Their concordance is a direct test of whether the typology
  definition is tight enough to apply the same way twice.
- **Evidence is checkable.** Every graded judgement carries a verbatim quote, and
  every quote is matched back against the source article after the run.
  Unverified quotes are reported, not hidden.
- **A graded ladder needs a passage behind it.** For every domain pair graded
  above `mentioned`, the integrity check asks whether the coder returned a quoted
  claim for that pair. This is the measurement that decides whether the
  integration numbers mean anything.
- **Agreement is measured on the right variable.** Ladders get kappa plus an
  adjacent-agreement rate, conceptual elements get a derived binary presence, and
  open label lists get set overlap over vocabulary-normalized labels, with
  relations and integration claims compared as edges rather than as single labels.
- **Wording is not disagreement.** The open lists are measured twice: once
  lexically, and once over sentence embeddings, where two labels count as the same
  concept above a cosine threshold. `pain catastrophising` and `catastrophic
  thinking about pain` are one construct, and a metric that scores them as a
  disagreement is measuring the vocabulary rather than the reading. Both numbers
  are on the same 0 to 1 scale, the soft one reduces to the hard one at a
  threshold of 1.0, and the threshold sensitivity is written next to the result.
- **The ontology is measured, not assumed.** Every extracted item reports whether
  it anchored to the project vocabularies, so the run says how much of what this
  literature names the ontology can currently hold.
- **Nothing is fabricated to fill a gap.** A response that is not a coding of the
  given paper is rejected and retried; caps are ceilings and never targets, so an
  empty list is a coding rather than a hole.
