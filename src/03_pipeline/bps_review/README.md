# `bps_review`: the pipeline package

Everything that runs in this review lives here. The package is organized by what
each part of the review actually does, not by technical layer, so a subpackage
name is a stage of the review rather than a category of code.

```text
bps_review/
├── search/          # database queries, deduplication, access checks
├── screening/       # title and abstract screening, screening reliability
├── extraction/      # Stage 2 abstract-level extraction and Stage 3 preparation
├── pilot/           # abstract-level cross-provider test run (scheme 2)
├── fulltext/        # full-text cross-provider test run (scheme 3)
├── graph/           # local interactive knowledge graph over a coded run
├── synthesis/       # cross-record analysis of the coded corpus
├── reporting/       # tables, figures, semantic loading, manuscript fragments
├── llm/             # OpenRouter chat and embedding client
├── utils/           # paths, environment, IO, metadata
├── settings.py      # configuration loaded from src/03_pipeline/config
└── cli.py           # every stage as a subcommand of `python -m bps_review`
```

## The two test runs

The review is coded twice, at two resolutions, by three cheap models from three
different providers. Both runs write to `src/05_data/pilot/`.

**`pilot/` is the abstract-level run (scheme 2).** It reads a title and an
abstract and acts mainly as a routing and relevance filter. Its purpose is to
decide which records deserve a full text.

**`fulltext/` is the full-text run (scheme 3).** It reads whole open-access
articles and answers the review's central question: does a biopsychosocially
labelled review actually integrate the three domains, and if so, how. It grades
the depth of each domain, grades every pairwise and the triadic integration on an
explicit ladder, carries a verbatim quote for each of those judgements, and
harvests the ontology: every biological, social, and lifestyle or existential
factor, every psychological construct with its definitional status, the relations
drawn between constructs, the frameworks and instruments in use, and every
passage where the biopsychosocial label does work.

## What a full-text run produces

`run_fulltext_testrun_pipeline()` chains the whole thing and writes to
`src/05_data/pilot/02_fulltext_level/`:

| Directory | Contents |
| --- | --- |
| `01_corpus/` | the retrieved corpus, its manifest, and the retrieval log. The article texts themselves stay local and are never pushed |
| `02_model_codings/` | every article by provider coding, the item-level extraction table, the raw audit trail, and the usage manifest |
| `03_reliability/` | agreement, consensus, lexical and semantic overlap, ontology coverage, and quote verification |
| `04_figures/` | the static review figures |
| `05_knowledge_graph/` | the self-contained interactive knowledge graph, opened by `index.html` |

## How agreement is quantified

Three kinds of variable, kept apart because they answer different questions.

* **Ordered ladders and nominal decisions** (`analysis/reliability.py`). Fleiss'
  kappa, Krippendorff's alpha, observed agreement, the unanimous rate, and on the
  ladders an adjacent-agreement rate, which separates a real disagreement about a
  paper from a one-rung difference in strictness.
* **Binary presence**, derived from the coded content rather than asked of the
  coder. Whether two coders both found a theoretical framework in a paper has one
  answer.
* **Open vocabularies**, measured twice, and there are far more of them than
  there are extraction lists. An item of this scheme is a record whose own fields
  carry open vocabularies, so `config.EXTRACTION_SPACES` declares 33 comparison
  spaces on three layers: identity (what was extracted), vocabulary (what it was
  called, read from a field or a sublist inside the items), and filtered (the
  subset that carries weight, such as the constructs a review actually defines).
  `analysis/spaces.py` reads a space; `analysis/reliability.py` scores the
  identity layer lexically; `analysis/semantic.py` scores every space both ways,
  embedding each label once and counting two labels as one concept above a cosine
  threshold. The distance between the two numbers is the part of the apparent
  disagreement that was only ever wording, and the spaces with a closed identity
  vocabulary sit in the same table as the control, where the two numbers are
  identical by construction. Vectors are cached on disk, so every rerun after the
  first is free and needs no network.

Quote verification (`analysis/integrity.py`) closes the loop: every verbatim
quote is matched back against the article it came from, so a graded judgement
with no passage behind it is visible as such.

## The knowledge graph

`graph/` turns a coded run into a local, desktop-first knowledge graph: plain
HTML, CSS, and JavaScript, no server and no network. The hierarchy is run, field
group, entity, coding field, provider, article coding, extracted item, so a
reviewer can walk from the scheme itself down to the sentence one judgement rests
on. The entity level is what the review is about: **Biopsychosocial entities**
holds the triad as three siblings and everything beyond it under a fourth heading
with children of its own, and the two lists that carry several entities at once
are split so the biological evidence sits under biology rather than in one
undifferentiated list. See [`graph/README.md`](graph/README.md).

## Running it

```bash
python -m bps_review run-fulltext-testrun          # reuse cached corpus and codings
python -m bps_review run-fulltext-testrun --force-coding   # re-code every paper via the API
python -m bps_review build-fulltext-graph          # rebuild only the knowledge graph
```

`--no-semantic` skips the embedding step, `--no-graph` skips the graph. Both are
enrichments of an otherwise complete run: a semantic-overlap failure is reported
and skipped rather than allowed to invalidate the lexical result.

Every subcommand is also a Makefile target; see the repository `Makefile`.

## Where things are on disk

Code never hard-codes a directory. `utils/paths.py` maps a section name
(`data`, `config`, `review_stages`, `semantic`, `docs`, `artifacts`) to its
numbered folder under `src/`, so the workspace layout can change without touching
a call site. `BPS_WORKSPACE_ROOT` overrides the root, which is what the Docker
image sets.
