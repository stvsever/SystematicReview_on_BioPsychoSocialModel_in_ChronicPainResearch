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
that, so its most consequential fields are ladders rather than labels.

| Layer | Fields |
| --- | --- |
| Coverage | `domain_coverage_bio`, `domain_coverage_psych`, `domain_coverage_social` on a four-rung ladder (elaborated, mentioned, minimal, absent) |
| Integration | `integration_bio_psych`, `integration_psych_social`, `integration_bio_social` on a five-rung ladder (mechanistic, directional, descriptive, mentioned, none), plus `integration_triadic` on its own four-rung ladder |
| Typology | `overall_balance`, `bps_typology` |
| Concepts | `concept_definitions_present`, plus a per-concept list carrying definitional status |
| Evidence | `integration_claims`, `domain_evidence`, `psychological_concepts`, `theoretical_frameworks`, `conceptual_problems`, `key_quotes`, each item with a verbatim quote and its section |

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
```

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
| `coding/prompt.py` | the instruction set, assembled from the schema and the ladders |
| `coding/condense.py` | fits a long article into the reading budget, dropping the least conceptual paragraphs first |
| `coding/derive.py` | repair, deterministic derivations, presence flags, serialization |
| `coding/runner.py` | one article per request, all models in parallel, with retries and usage accounting |
| `analysis/reliability.py` | categorical agreement, adjacent agreement on the ladders, presence agreement, list overlap |
| `analysis/integrity.py` | completeness, quote verification, evidence discipline, extraction yield |
| `visualization/figures.py` | four multi-panel figures |
| `report.py` | the standalone summary |
| `pipeline.py` | `run_fulltext_testrun_pipeline()` |

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
  open label lists get set overlap.
- **Nothing is fabricated to fill a gap.** A response that is not a coding of the
  given paper is rejected and retried; caps are ceilings and never targets, so an
  empty list is a coding rather than a hole.
