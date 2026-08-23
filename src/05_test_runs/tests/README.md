# The two pilots

One per coding level, run by the same three providers, and they are one chain: the abstract pilot
exports the candidate set, the full-text pilot retrieves the open-access members of it and codes
them.

| Folder | What it is | Scheme | Scale |
| --- | --- | --- | --- |
| [`01_pilot_abstract/`](01_pilot_abstract/) | Abstract-level coding of a fresh PubMed sample | [scheme 2](../../02_coding_schemes/scheme_2/) | 100 abstracts x 3 models = 300 codings |
| [`02_pilot_fulltext/`](02_pilot_fulltext/) | Full-text coding of the retrieved open-access corpus | [scheme 3](../../02_coding_schemes/scheme_3/) | 47 full texts x 3 models = 141 codings, 9400 extracted items |

Both are methods artifacts. They quantify how far independent coders converge on a scheme, which is
what the expert evaluation needs. They are neither findings about the literature nor manuscript
results. The review's official coding pass is still to come, into [`../official/`](../official/).

## `01_pilot_abstract/`

It does two things. It quantifies how far three independent coders converge on the abstract scheme,
which came out at a mean Fleiss' kappa of 0.604 and 77.9% observed agreement, with all three models
agreeing on Stage 3 candidacy for 91 of the 100 abstracts. And it exports
[`01_pilot_abstract/05_fulltext_candidate_set.csv`](01_pilot_abstract/05_fulltext_candidate_set.csv),
the 88 records the full-text pilot then tried to retrieve.

| Folder | What is in it |
| --- | --- |
| `01_abstracts/` | The 100 PubMed records that were coded, and the manifest of how they were sampled |
| `02_model_codings/` | The 300 codings, combined and per model, with the API call trail |
| `03_reliability/` | Per-field agreement, pairwise agreement, consensus codings, list overlap |
| `04_figures/` | Four multi-panel figures of the run |
| `05_fulltext_candidate_set.csv` | The consensus candidate set handed to the full-text pilot |
| `TEST_RUN_SUMMARY.md` | The numeric summary, field by field |

```bash
python -m bps_review run-abstract-testrun
```

## `02_pilot_fulltext/`

Of the 88 candidates, 53 have a PubMed Central record and 47 returned a body long enough to code.
Those 47 were coded by the same three providers, giving 141 codings and 9400 extracted items, each
carrying the verbatim sentence it was read from. 99.2% of the 8302 checked quotes were found in
their source article.

**Start at [`02_pilot_fulltext/02_model_codings/`](02_pilot_fulltext/02_model_codings/).** Every
table there begins with the citation, title, authors, and DOI rather than an internal id, every list
is split across one column per item, and the extracted items are filed per category with the
biological, psychological, and social factors first.

| Folder | What is in it |
| --- | --- |
| `01_corpus/` | The paper list with full citations, DOIs, and PubMed Central links, the retrieval log, and the manifest |
| `02_model_codings/` | Every coding of the run: long and wide, per paper, per provider, per extraction category |
| `03_reliability/` | Agreement between the three providers, consensus codings, extraction overlap, ontology coverage, quote verification |
| `04_figures/` | Five multi-panel figures of the run |
| `05_knowledge_graph/` | The whole run as a local interactive graph. Open `index.html` in a browser |
| `TEST_RUN_SUMMARY.md` | The numeric summary, field by field |

```bash
python -m bps_review run-fulltext-testrun
```

Both commands rebuild everything from what is already on disk and cost no API calls.
