# `src/` source map

Everything under `src/` is grouped into numbered sections that follow the order of the review:
what it committed to, the instruments it uses, the code that applies them, the notebooks that run
them, and the runs they produce. Sections 06 and up are working material and stay local.

**Looking for the results? They are in [`05_test_runs/official/`](05_test_runs/official/).**

```text
src/
├── 01_protocol/            # what the review committed to
│   ├── osf/OSF_registration_HTBMFCPR.md
│   ├── codebooks/
│   │   ├── stage2_codebook.md          # abstract-level fields
│   │   └── stage3_codebook.md          # full-text fields
│   └── decision_rules/screening_rules.md
│
├── 02_coding_schemes/      # the instruments, as expert-evaluation dossiers
│   ├── index.html              # aggregated dashboard and feedback console
│   ├── _build/                 # single source of truth (content.py) and generator
│   └── scheme_1/ ... scheme_6/ # one folder per scheme (HTML, PDF, TeX, README)
│
├── 03_pipeline/            # everything that runs
│   ├── config/                 # YAML that controls pipeline behavior
│   │   ├── pipeline.yaml           # stage toggles and runtime settings
│   │   ├── protocol.yaml           # eligibility and protocol constraints
│   │   └── search_queries.yaml     # search strings per database
│   ├── tests/                  # the test suite (pytest)
│   └── bps_review/             # the Python package
│       ├── README.md               # what each subpackage does and how a run flows
│       ├── cli.py                  # command routing (python -m bps_review ...)
│       ├── search/                 # PubMed, Web of Science, PsycINFO, deduplication
│       ├── screening/              # Stage 1 eligibility and reliability
│       ├── extraction/             # Stage 2 abstract coding and Stage 3 preparation
│       ├── synthesis/              # corpus-level synthesis helpers
│       ├── reporting/              # figures, tables, manuscript fragments, semantic loading
│       ├── graph/                  # local interactive knowledge graph over a coded run
│       ├── llm/                    # OpenRouter chat and embedding client
│       ├── utils/                  # paths, io, env, metadata helpers
│       │
│       ├── pilot/              # cross-provider run, abstract level (scheme 2)
│       │   ├── README.md               # what it does and what the numbers mean
│       │   ├── config.py               # the 3 models, fields, paths, run settings
│       │   ├── coding/                 # build the PubMed sample and run every model
│       │   ├── analysis/               # agreement primitives and per-field reliability
│       │   ├── visualization/          # four 2x2 multi-panel figures
│       │   ├── report.py               # the standalone summary
│       │   └── pipeline.py             # run_abstract_testrun()
│       │
│       └── fulltext/           # cross-provider run, full text (scheme 3)
│           ├── README.md               # how this pipeline works, in detail
│           ├── config.py               # models, per-model runtime, ladders, caps, paths
│           ├── corpus/pmc.py           # open-access retrieval for the candidate set
│           ├── coding/                 # schema, prompt, condenser, repair, derivations, runner
│           ├── analysis/               # reliability, semantic overlap, quote and evidence integrity
│           ├── visualization/figures.py
│           ├── publish.py              # the run written as the tables it is read in
│           ├── report.py
│           ├── graph_export.py         # rebuild the knowledge graph from the published tables
│           └── pipeline.py             # run_fulltext_testrun_pipeline()
│
├── 04_notebooks/           # the end-to-end notebooks
│   ├── 01_abstractlevel_testrun.ipynb  # scheme 2, 100 abstracts x 3 models
│   ├── 02_fulltextlevel_testrun.ipynb  # scheme 3, the retrieved full texts x 3 models
│   └── 03_synthesislevel_testrun.ipynb # corpus-level synthesis over a coded run
│
├── 05_test_runs/           # every run of the workflow, and all of its outputs
│   ├── README.md               # what each run is, and where to start
│   ├── official/               # THE run: 47 open-access reviews x 3 providers, scheme 3
│   │   ├── README.md               # start here
│   │   ├── 01_corpus/              # paper list with DOIs, retrieval log, manifest
│   │   ├── 02_model_codings/       # every output, long and wide, per provider, per category
│   │   ├── 03_reliability/         # agreement, consensus, overlap, quote verification
│   │   ├── 04_figures/
│   │   └── 05_knowledge_graph/     # open index.html in a browser
│   └── tests/
│       └── 01_pilot_abstract/      # 100 abstracts x 3 providers, scheme 2, and its candidate set
│
│   ---- everything below is local only, and absent from a fresh clone ----
│
├── 06_data/                # inputs and caches, never results
│   ├── interim/                # retrieval caches and the embedding vector store
│   ├── manual_imports/         # records imported by hand
│   └── raw/ processed/
│
├── 07_docs/                # project status and working notes
│
├── 08_artifacts/           # run logs and validation scratch space
│
└── 09_review_stages/       # stage-by-stage working files of the main pipeline
    ├── 01_protocol/ 02_search/ 03_screening/ 04_extraction/ 06_reporting/
    └── 05_synthesis/
        ├── outputs/                # PRISMA counts and the results summary
        └── semantic_space/         # ontology-aligned embeddings and domain loadings
```

## Where the section names live

Code refers to a section by its plain name (`data`, `config`, `test_runs`, `review_stages`,
`semantic`, ...) through `project_path()`. The mapping from that name to the numbered directory sits
in one place, `bps_review/utils/paths.py`, so the physical grouping can change without touching a
single call site.

Sections 01 through 05 are published. Sections 06 through 09 are local only: retrieval caches and
the embedding store, internal notes, scratch space, and the stage-by-stage working files of the main
pipeline. None of them is a result, and each is reproducible from the code.

## Where to look first

| I want to ...                                       | Go to                                                                 |
| --------------------------------------------------- | --------------------------------------------------------------------- |
| Read the full-text run outputs                       | `05_test_runs/official/README.md`                                     |
| Read one provider's codings and extracted items      | `05_test_runs/official/02_model_codings/`                             |
| See the biological, psychological, and social factors| `05_test_runs/official/02_model_codings/02_extracted_items/`          |
| Browse a coded run interactively                     | `05_test_runs/official/05_knowledge_graph/index.html`                 |
| Read the abstract-level run                          | `05_test_runs/tests/01_pilot_abstract/TEST_RUN_SUMMARY.md`            |
| Give feedback on a coding scheme                     | `02_coding_schemes/index.html`                                        |
| Understand the full-text coding scheme               | `03_pipeline/bps_review/fulltext/README.md`, then `coding/schema.py`   |
| See the exact instructions a coder receives          | `03_pipeline/bps_review/fulltext/coding/prompt.py`                    |
| See how eligibility and priority are derived         | `03_pipeline/bps_review/fulltext/coding/derive.py`                    |
| See how a run becomes the published tables           | `03_pipeline/bps_review/fulltext/publish.py`                          |
| Check whether the coding can be trusted              | `03_pipeline/bps_review/fulltext/analysis/integrity.py`               |
| Understand the abstract-level scheme                 | `03_pipeline/bps_review/extraction/llm_stage2.py`                     |
| Run the abstract-level run                           | `04_notebooks/01_abstractlevel_testrun.ipynb`                          |
| Run the full-text run                                | `04_notebooks/02_fulltextlevel_testrun.ipynb`                          |
| See how the knowledge graph is built                 | `03_pipeline/bps_review/graph/README.md`                              |
| Compare providers by meaning rather than by wording  | `03_pipeline/bps_review/fulltext/analysis/semantic.py`                |
| Change the models or the worker counts               | `03_pipeline/bps_review/pilot/config.py`                              |
| See the main-pipeline outputs stage by stage         | `09_review_stages/` (local only)                                      |

## The two pipelines in one line each

`bps_review.pilot` applies the Stage 2 abstract coding scheme to a fresh sample of 100 PubMed
records with three cheap models from three providers, quantifies how much they agree, and
exports the consensus candidate set.

`bps_review.fulltext` retrieves the open-access full texts of that candidate set, applies the
Stage 3 deep coding scheme with the same three models, verifies every extracted quote against
its source article, quantifies categorical agreement, adjacent agreement on the ordered
ladders, binary presence agreement, and extraction overlap both lexically and semantically,
writes every output as the tables it is read in, and builds the interactive knowledge graph over the
whole run.

```python
from bps_review.pilot import run_abstract_testrun
from bps_review.fulltext import run_fulltext_testrun_pipeline

abstract = run_abstract_testrun()                     # reuse cached sample and codings
fulltext = run_fulltext_testrun_pipeline()            # reuse the published tables
```
