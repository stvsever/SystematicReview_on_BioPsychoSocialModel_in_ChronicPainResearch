# Test runs

Every run of the workflow, and everything it produced. Each run is self-contained: its own corpus,
codings, reliability analysis, and figures.

## The one to read

**[`official/`](official/)** is the full-text run: the full-text coding scheme (scheme 3) applied to
47 open-access review articles by three independent language models. Start at
[`official/README.md`](official/README.md); the codings themselves are in
[`official/02_model_codings/`](official/02_model_codings/), where every list is split into one
column per item and every table starts with the citation and the DOI.

## The run before it

[`tests/01_pilot_abstract/`](tests/01_pilot_abstract/) is the abstract-level run: the same three
providers applied the abstract coding scheme (scheme 2) to 100 PubMed records. It is not only a
development pilot, it is the upstream stage: its consensus candidate set of 88 records is what the
full-text run tried to retrieve, and the 47 papers it could retrieve are the corpus above.

| Folder | What it is | Scheme |
| --- | --- | --- |
| [`official/`](official/) | 47 open-access reviews x 3 providers, 9400 extracted items | [scheme 3](../02_coding_schemes/scheme_3/) |
| [`tests/01_pilot_abstract/`](tests/01_pilot_abstract/) | 100 PubMed abstracts x 3 providers, and the candidate set it exported | [scheme 2](../02_coding_schemes/scheme_2/) |

Both runs are methods artifacts. They quantify how far independent coders converge on a scheme, and
they are neither a synthesis of the literature nor a manuscript result.

The coding schemes themselves are in [`../02_coding_schemes/`](../02_coding_schemes/); the code that
applies them is in [`../03_pipeline/`](../03_pipeline/).
