# Abstract-level test run (scheme 2)

This package applies the Stage 2 abstract coding scheme to a fresh sample from
the review's operational PubMed query, once per model, with three cheap models
from three different providers, and reports how much they agree.

It is the first half of a chain. Its consensus output, the set of records the
models agree to carry forward, is what the full-text run
([`bps_review.fulltext`](../fulltext/)) retrieves and codes at full resolution.

## What it does

1. **Sample.** Run the operational PubMed query inside the protocol's date
   window, sorted by publication date, drop records without a usable abstract,
   and keep the most recent 100.
2. **Code.** Apply the Stage 2 scheme to every abstract with every model, one
   record per request, through exactly the same structured-JSON path the main
   pipeline uses. 100 records x 3 models is 300 codings from 300 API calls.
3. **Analyse.** Per-field agreement (Fleiss' kappa, Krippendorff's alpha,
   observed agreement, unanimous rate), model-by-model matrices, per-model
   behaviour, set overlap on the open lists, and a majority-vote consensus.
4. **Hand off.** Export the consensus candidate set for the full-text stage.

```python
from bps_review.pilot import run_abstract_testrun

out = run_abstract_testrun()                    # reuse cached sample and codings
out = run_abstract_testrun(force_coding=True)   # re-code every abstract via the API
```

```bash
python -m bps_review run-abstract-testrun
python -m bps_review run-abstract-testrun --force-coding
```

## Module map

| Module | Responsibility |
| --- | --- |
| `config.py` | the three models, run settings, field groups, paths |
| `coding/data.py` | build and load the PubMed sample, with a retrieval manifest |
| `coding/runner.py` | run every model over every abstract, with retries, hard timeouts, and usage accounting |
| `analysis/metrics.py` | the agreement primitives, implemented without external dependencies |
| `analysis/reliability.py` | per-field reliability, pairwise matrices, behaviour, consensus, list overlap |
| `visualization/figures.py` | four multi-panel figures |
| `report.py` | the standalone summary |
| `pipeline.py` | `run_abstract_testrun()` and the candidate-set export |

## What the numbers mean

- **Kappa-style coefficients** are computed on the categorical decisions, where
  the value itself carries the meaning and a disagreement is a real
  disagreement.
- **Set overlap (Jaccard)** is used for the open lists (psychological concepts,
  theoretical frameworks, conceptual problems), because two coders can both be
  right and still return different strings.
- **Three fields are derived, not asked**: `stage3_candidate` and
  `stage3_priority` are computed from the musculoskeletal routing flag, and
  `conceptual_problem_flags` is completed from the coded domain and typology
  values. Their agreement is the agreement of the judgements they read, not an
  independent judgement, and the reports say so.

## A note on the prompt

The first execution of this run exposed a defect in the main pipeline's Stage 2
prompt: it listed value vocabularies but never named the fields to return, so
models answered with the subset whose option lists they could see and omitted
`bio_mentioned`, `psych_mentioned`, `social_mentioned`, `musculoskeletal_flag`,
and `quality_assessment_reported`. The deterministic repair layer then filled
those five fields from the lexical rule-based coder, which made them identical
across providers and produced a perfect but meaningless kappa of 1.00. The
prompt now carries an explicit field specification and output contract, a test
asserts that the specification and the validated schema cannot drift apart, and
the reported agreement on those fields is now a real measurement.
