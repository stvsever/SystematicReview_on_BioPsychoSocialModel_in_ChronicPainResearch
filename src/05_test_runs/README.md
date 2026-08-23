# Test runs

Every run of the workflow, and everything it produced. Each run is self-contained: its own corpus,
codings, reliability analysis, and figures.

## The one to read

**[`tests/02_pilot_fulltext/`](tests/02_pilot_fulltext/)** is the full-text pilot: the full-text
coding scheme (scheme 3) applied to 47 open-access review articles by three independent language
models, producing 141 codings and 9400 extracted items. Start at
[`tests/02_pilot_fulltext/README.md`](tests/02_pilot_fulltext/README.md); the codings themselves are
in [`tests/02_pilot_fulltext/02_model_codings/`](tests/02_pilot_fulltext/02_model_codings/), where
every list is split into one column per item and every table starts with the citation and the DOI.

| Folder | What it is | Scheme |
| --- | --- | --- |
| [`tests/01_pilot_abstract/`](tests/01_pilot_abstract/) | 100 PubMed abstracts x 3 providers, and the candidate set it exported | [scheme 2](../02_coding_schemes/scheme_2/) |
| [`tests/02_pilot_fulltext/`](tests/02_pilot_fulltext/) | 47 open-access reviews x 3 providers, 9400 extracted items | [scheme 3](../02_coding_schemes/scheme_3/) |

The two are one chain: the abstract pilot exports its consensus candidate set, and the full-text
pilot retrieves the open-access members of that set and codes them.

## `official/` is reserved and empty

[`official/`](official/) is where the review's official coding pass will go. Nothing has been run
into it yet, so it holds a `.gitkeep` and nothing else.

Both runs under [`tests/`](tests/) are pilots. They exist to show that the schemes, the code, and
the reporting hold up on real records, and to quantify how far independent coders converge on each
scheme. That is what the expert evaluation of the schemes needs. They are not findings about the
biopsychosocial literature, and they are not manuscript results.

The coding schemes themselves are in [`../02_coding_schemes/`](../02_coding_schemes/); the code that
applies them is in [`../03_pipeline/`](../03_pipeline/).
