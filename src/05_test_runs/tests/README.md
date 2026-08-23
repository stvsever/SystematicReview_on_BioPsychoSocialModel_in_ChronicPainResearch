# Earlier stages of the workflow

[`01_pilot_abstract/`](01_pilot_abstract/) is the abstract-level run: the abstract coding scheme
(scheme 2) applied to 100 PubMed records by the same three providers as the full-text run.

It does two things. It quantifies how far three independent coders converge on the abstract scheme,
which came out at a mean Fleiss' kappa of 0.604 and 77.9% observed agreement, with all three models
agreeing on Stage 3 candidacy for 91 of the 100 abstracts. And it exports
[`01_pilot_abstract/05_fulltext_candidate_set.csv`](01_pilot_abstract/05_fulltext_candidate_set.csv),
the 88 records the full-text run then tried to retrieve.

| Folder | What is in it |
| --- | --- |
| `01_abstracts/` | The 100 PubMed records that were coded, and the manifest of how they were sampled |
| `02_model_codings/` | The 300 codings, combined and per model, with the API call trail |
| `03_reliability/` | Per-field agreement, pairwise agreement, consensus codings, list overlap |
| `04_figures/` | Four multi-panel figures of the run |
| `05_fulltext_candidate_set.csv` | The consensus candidate set handed to the full-text run |
| `TEST_RUN_SUMMARY.md` | The numeric summary, field by field |

The run it feeds is [`../official/`](../official/).

```bash
python -m bps_review run-abstract-testrun
```
