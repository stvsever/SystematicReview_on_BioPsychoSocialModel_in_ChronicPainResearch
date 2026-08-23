# Full-text coding run

Three language models independently applied the full-text coding scheme (scheme 3) to the same 47
open-access review articles. Each model read every paper on its own and filled in the same
structured form. This folder holds everything they produced, plus the agreement analysis between
them.

This is a methods artifact. It is not a conceptual synthesis and not a manuscript result.

## Start here

The codings are in **[`02_model_codings/`](02_model_codings/)**. Every table carries the citation,
title, authors, and DOI next to the record id, and every list is split across one column per item
instead of being crammed into one cell. Long and wide forms are both written, so the same content
can be read per paper, per provider, or per extracted item.

| I want to | Open |
| --- | --- |
| Work in Excel instead of CSV | [`02_model_codings/00_workbook.xlsx`](02_model_codings/00_workbook.xlsx) |
| See the biological, psychological and social factors of a review side by side | [`02_model_codings/02_extracted_items/00_all_categories/02_one_row_per_named_factor_of_any_domain.csv`](02_model_codings/02_extracted_items/00_all_categories/02_one_row_per_named_factor_of_any_domain.csv) |
| Read one category on its own: biological factors, psychological concepts, social factors, integration claims | [`02_model_codings/02_extracted_items/`](02_model_codings/02_extracted_items/) |
| Read one provider's coding of one paper, with every quote | [`02_model_codings/01_codings/01_one_row_per_paper_and_provider.csv`](02_model_codings/01_codings/01_one_row_per_paper_and_provider.csv) |
| Compare the three providers on one paper | [`02_model_codings/01_codings/02_one_row_per_paper.csv`](02_model_codings/01_codings/02_one_row_per_paper.csv) |
| Look at only DeepSeek-V4-Flash, the primary provider | [`02_model_codings/03_by_provider/01_deepseek_v4_flash/`](02_model_codings/03_by_provider/01_deepseek_v4_flash/) |
| See which papers were coded | [`01_corpus/papers.csv`](01_corpus/papers.csv) |

[`02_model_codings/README.md`](02_model_codings/README.md) explains every column, in Dutch. The
coding scheme itself is [`src/02_coding_schemes/scheme_3/`](../../02_coding_schemes/scheme_3/).

## What the run produced

| | |
| --- | --- |
| Papers | 47 open-access reviews, retrieved from PubMed Central, all with a DOI |
| Coders | 3 language models from 3 providers, run independently |
| Codings | 141 of 141 completed, 0 failed |
| Extracted items | 9400, across 13 extraction categories |
| Evidence quotes verified against the source article | 99.2% of 8302 checked |
| Mean observed agreement on controlled fields | 69.2% |
| Coverage ladder, providers within one rung of each other | 92.5% |
| Integration ladder, providers within one rung of each other | 56.6% |
| Extraction-list overlap, semantic | 0.345 |
| API cost | $0.34 |

The three coders were DeepSeek-V4-Flash, Nex-N2-Mini, and Laguna-XS-2.1. DeepSeek-V4-Flash is the
primary model. They are the same three used by the abstract-level run, so the two stages are
directly comparable.

The corpus is the open-access subset of the candidate set the abstract-level run carried forward:
88 candidates, 53 with a PubMed Central record, 47 with a body long enough to code.

## Folder map

| Folder | What is in it |
| --- | --- |
| [`01_corpus/`](01_corpus/) | The paper list with full citations, DOIs, and PubMed Central links, plus the retrieval log and the corpus manifest |
| [`02_model_codings/`](02_model_codings/) | Every coding of the run: long and wide, the extracted items per category, and everything per provider |
| [`03_reliability/`](03_reliability/) | Agreement between the three providers, consensus codings, extraction overlap, ontology coverage, and quote verification |
| [`04_figures/`](04_figures/) | The five figures of the run |
| [`05_knowledge_graph/`](05_knowledge_graph/) | Interactive graph of the whole run, from the coding scheme down to the quoted sentence behind one extracted item. Open `index.html` in a browser |

[`TEST_RUN_SUMMARY.md`](TEST_RUN_SUMMARY.md) is the numeric summary of the run, field by field.

## What is not in Git

The article full texts are licensed material and stay local, together with the API call trail behind
the run. What is committed is the inventory, the identifiers, and everything the coding derived from
the text: the coded rows, the extracted items, the verbatim evidence quotes, and every aggregate
table. Retrieval and quote verification are repeatable locally with `make fulltext-corpus`.

## How to reproduce

Nothing here needs re-coding. All tables are rebuilt from the published ones, at no API cost:

```bash
python -m bps_review run-fulltext-testrun
```

The notebook walkthrough is
[`src/04_notebooks/02_fulltextlevel_testrun.ipynb`](../../04_notebooks/02_fulltextlevel_testrun.ipynb).

## How to read the agreement numbers

They describe agreement between three cheap models on a small open-access corpus. They are not a
finding about the biopsychosocial literature, and not a validation of the coding scheme against a
human standard. What they do show is which parts of the scheme are specified tightly enough that
independent coders converge, and which are not. That is the input the expert evaluation of the
scheme needs, and the reason the run exists.
