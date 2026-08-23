# Corpus

The 47 open-access review articles this run coded, and how they were selected.

| File | One row is | Rows |
| --- | --- | --- |
| [`papers.csv`](papers.csv) | One paper: full citation, DOI, PubMed and PubMed Central identifiers, licence, how long the text was, and the abstract-level reading that put it here. **Start here to find a paper again.** | 47 |
| [`articles.csv`](articles.csv) | The same 47 papers as the coding pipeline reads them: the publisher metadata plus the abstract, the section titles, and the path to the retrieved text | 47 |
| [`retrieval_candidates.csv`](retrieval_candidates.csv) | One candidate handed over by the abstract-level run, with what was known about it before retrieval | 88 |
| [`retrieval_log.csv`](retrieval_log.csv) | One candidate and what happened to it: retrieved, no PubMed Central record, or a body too short to code | 88 |
| [`corpus_manifest.json`](corpus_manifest.json) | The retrieval itself: when it ran, from where, and why each dropped candidate was dropped | |
| `fulltext_txt/` | The retrieved article text, one file per paper. Local only, see below | 47 |

## How the corpus was built

The abstract-level run coded 100 PubMed abstracts with the same three providers and carried 88
records forward as its consensus candidate set. Of those, 53 have a record in the PubMed Central
open-access subset. Six of the 53 returned a body too short to code, which leaves the 47 papers
here. The reason for every drop is in `retrieval_log.csv` and counted in `corpus_manifest.json`.

Every one of the 47 has a registered DOI, so any paper in these tables resolves to its publisher
record from `papers.csv` alone.

## Why the text is not in Git

The articles are open access, which grants the review the right to read them, not the right to
redistribute them. So no article body is carried in Git. What is carried is the inventory, the
identifiers, and everything the coding derived from the text: the coded rows, the extracted items,
and the verbatim evidence quotes.

An authorized reader restores the texts locally, which repeats retrieval and makes quote
verification runnable again:

```bash
make fulltext-corpus
```
