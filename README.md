<div align="center">

# How the Biopsychosocial Model Frames Chronic Pain Research

### An OSF-registered systematic review with mixed-method synthesis, structured semantic coding, and ontology-aligned embeddings

[![Type](https://img.shields.io/badge/Type-systematic_review-7C3AED)](paper/report/main.tex)
[![OSF Registered](https://img.shields.io/badge/OSF-10.17605%2FOSF.IO%2FT4FAM-0F766E)](https://osf.io/t4fam)
[![Stage](https://img.shields.io/badge/Stage-coding_schemes_under_expert_review-B45309)](src/02_coding_schemes/index.html)
[![Dockerized](https://img.shields.io/badge/Docker-ready-2496ED)](docker/)
[![MIT License](https://img.shields.io/badge/License-MIT-16A34A)](LICENSE)

**Stijn Van Severen<sup>1,\*</sup> · Christopher Eccleston<sup>1,2</sup> · Annick De Paepe<sup>1</sup> · Maya Braun<sup>1</sup> · Julie Dendauw<sup>1</sup> · Jose Luis Socorro Cumplido<sup>3</sup> · Geert Crombez<sup>1</sup>**

<sup>1</sup> Ghent University, Ghent, Belgium<br>
<sup>2</sup> University of Bath, Bath, United Kingdom<br>
<sup>3</sup> Ramon Llull University, Barcelona, Spain<br>
<sup>\*</sup> Corresponding author

---

</div>

## 🚧 Project status: test run, coding schemes under expert review

**The current manuscript is a test run, not a final result.** It was produced with an earlier, coarser generation of the coding schemes. Everything in [paper/report/main.pdf](paper/report/main.pdf) (the 111 record counts, the typology distribution, the semantic-loading figures) should be read as a pipeline demonstration on provisional schemes, not as confirmed findings.

The coding schemes that define how biopsychosocial (BPS) reviews are classified and categorized have since been revised for higher semantic quality and resolution: operational anchors for every code value, positive and negative indicators, and explicit boundary rules between adjacent categories. They are built as one uniform instrument for **two planned reviews**, one on musculoskeletal chronic pain and one on neuropathic chronic pain, with uniformity relaxed only where the biology genuinely differs (the routing flags and the biological subdomain ontology). These revised schemes are now circulated for **expert evaluation** and are **awaiting sign-off before the pipeline is re-run**.

**Two of the six schemes carry the review's argument, and they are the ones that most need expert eyes.**

[**Scheme 2, abstract level**](src/02_coding_schemes/scheme_2/scheme_2.html) ([PDF](src/02_coding_schemes/scheme_2/scheme_2.pdf)) is the corpus-wide layer. It reads title, abstract, and publication metadata for every included record and codes review design, stated objective, ICD-11 pain family, where the BPS label appears and what work it does, which domains are substantively present, the psychological concepts and frameworks named, the conceptual problems visible at abstract level, and a provisional typology. It also routes each record to the musculoskeletal review, the neuropathic review, or both.

[**Scheme 3, full text**](src/02_coding_schemes/scheme_3/scheme_3.html) ([PDF](src/02_coding_schemes/scheme_3/scheme_3.pdf)) is the deep layer, and it does two things at once. It **grades**: how deeply each domain is treated, and how each pair of domains and the triad are integrated, on explicit ladders with a verbatim quote behind every rung. And it **extracts**, which is the larger half: *which* biological, social, lifestyle, and existential factors carry each domain, with the role each plays; every psychological construct with its definitional status, its measure, and its concept family; every hierarchical or semantic relation drawn between constructs; every framework and instrument; every passage where the BPS label does work and how the model is defined; and the conceptual problems, with the constructs they concern. Thirteen structured extraction lists, seven open free-text lists, 82 fields inside the list items, each item carrying the review's own wording, a quote, and the section it came from. Labels map onto the project vocabularies only where they clearly match, a mapped label never replaces the paper's own wording, and terms the vocabularies do not carry are kept verbatim and reported back as the working list for extending them. That extraction layer is what a high-resolution biopsychosocial ontology is built from.

- Interactive evaluation package, all six schemes (open in a browser): [src/02_coding_schemes/index.html](src/02_coding_schemes/index.html)
- Per-scheme dossiers (HTML, PDF, README): [src/02_coding_schemes/](src/02_coding_schemes/)

Each dossier carries the purpose of the scheme and the scheme itself, and nothing else. Every coded field has its own expert-feedback box, every section has a section-level box, and the overall assessment closes the page. Reviewers export one JSON file per scheme and the console on the index page consolidates them. A re-run with the finalized schemes, and an updated manuscript, will follow once evaluation is complete. Please do not cite the test-run numbers as conclusions in the meantime.

## 🧪 Cross-provider test runs

Before the schemes go back through the pipeline, the workflow itself was validated end to end. Both coding levels are stress-tested the same way: apply the scheme once per model, with **three deliberately cheap large language models from three different providers**, and quantify how strongly the providers agree. The three models act as three independent raters, so each run is a cross-provider inter-rater reliability check on the scheme and on the code that applies it.

> The three models (`deepseek/deepseek-v4-flash`, `nex-agi/nex-n2-mini`, `poolside/laguna-xs-2.1`) are low-cost stand-ins, chosen because they verified the highest share of their quoted evidence in an earlier five-model comparison. The real study will run the same workflow with state-of-the-art models. The test runs exist to validate the code, the metrics, and the reporting, so that step becomes a drop-in model swap in one config file.

The two runs are one chain, not two exercises:

```text
operational PubMed query
        |
        v
100 abstracts  ---->  Stage 2 coding, 3 models, 300 API calls
        |
        v
consensus filter (majority vote)  ---->  88 candidates
        |
        v
PubMed Central open access  ---->  47 full texts
        |
        v
Stage 3 deep coding, same 3 models, graded ladders with verbatim evidence
```

### Test run 1: abstract level (scheme 2)

100 records from the operational query x 3 models = 300 codings from 300 API calls, for about 13 US cents. Reliability per coded field with Fleiss' kappa, Krippendorff's alpha, observed agreement, and the unanimous rate, plus model-by-model matrices, set overlap on the open extraction lists, and a majority-vote consensus.

Mean Fleiss' kappa is 0.60 across the twelve coded fields, and the spread is the finding: substantial agreement on what a paper *is* (ICD-11 pain family 0.76, review type 0.71), fair agreement on what a paper *does* (BPS function 0.35, provisional typology 0.33). The two weakest fields are the ones carrying the review's own argument, which is exactly what the expert evaluation should concentrate on.

- Notebook: [`src/04_notebooks/01_abstractlevel_testrun.ipynb`](src/04_notebooks/01_abstractlevel_testrun.ipynb) or `python -m bps_review run-abstract-testrun`
- Code: [src/03_pipeline/bps_review/pilot/](src/03_pipeline/bps_review/pilot/README.md) · Results: [src/05_test_runs/tests/01_pilot_abstract/](src/05_test_runs/tests/01_pilot_abstract/)

### Test run 2: full text (scheme 3)

The open-access subset of the candidate set the abstract stage produced: 47 full texts x 3 models, coded on the current scheme 3 with all thirteen extraction lists. Everything from the first run is computed again, plus three things a full-text scheme needs:

- **Graded ladders with evidence.** Coverage per domain on a four-rung ladder, three pairwise and one triadic integration on their own ladders, and a verbatim quote behind every graded judgement. An adjacent-agreement rate is reported alongside kappa, because on an ordered ladder a one-rung difference and a total disagreement are not the same error.
- **Quote verification.** Every extracted quote is matched back against its source article. Unverified quotes are reported, not hidden, and the per-model spread is a concrete criterion for choosing the state-of-the-art model later.
- **Evidence discipline.** For every domain pair graded above `mentioned`, the run checks whether the coder returned a quoted claim for exactly that pair. A graded link with no passage behind it is a judgement the review cannot audit.
- **Ontology coverage.** Every extracted item reports whether its ontology anchor landed on the project vocabularies, so the run says how much of what this literature names the ontology can currently hold, and which off-spine labels recur. The off-spine list is the working list for extending the vocabularies.
- **Overlap on every vocabulary the extraction produces, measured twice.** A scheme 3 item is not a label but a small record, and several of its fields are open vocabularies in their own right: which constructs a coder says carry the biological domain, which measure a construct is tied to, which components a definition of the model lists. Comparing extraction lists by item identity, as reliability metrics normally do, leaves all of that unmeasured. The scheme therefore declares **33 comparison spaces** across three layers, identity, vocabulary, and filtered, and each is scored twice: lexically, and semantically, where every label is embedded once and two labels count as one concept above a cosine threshold. The gap between the two is the share of the apparent disagreement that was only ever wording.

  This run answers all 33 (13 identity, 16 vocabulary, 4 filtered) over 6,205 embedded labels, and the mean rises from 0.287 lexical to 0.345 semantic. Where the two columns diverge most is where the lexical reading was most misleading: the relations drawn between concepts score 0.026 by string and 0.284 by meaning, and the integration claims 0.020 against 0.208. Read lexically the providers look like they agree on almost nothing there; what they actually disagree about is how to word an edge, not whether it is in the paper. The largest vocabulary in the run, the 2,092 constructs the coders name as carrying the domains, rises from 0.327 to 0.402. The five spaces whose identity is a closed vocabulary act as the control: their two columns come out identical, which is the check that the method is not manufacturing agreement wherever it is pointed.
- **An interactive knowledge graph.** The whole run is also a browsable graph: run, field group, entity, coding field, provider, article coding, extracted item, with filters, full-text search over every label and quote, and an inspector that shows the verbatim passage and its verification verdict. The entity level carries the review's own subject: **Biopsychosocial entities** holds the triad as three siblings (biological, psychological, social) and everything the registration adds beyond it under a fourth heading, **Other factors**, with lifestyle and spiritual or existential as its own children. The depth is the argument: lifestyle is not a fourth domain sitting beside biology. The two lists that hold several entities at once (the domain evidence, and the factors beyond the triad) are split so the biological evidence sits under biology rather than inside one undifferentiated list. It is plain local HTML with no server. Open [`src/05_test_runs/official/05_knowledge_graph/index.html`](src/05_test_runs/official/05_knowledge_graph/index.html).

- Notebook: [`src/04_notebooks/02_fulltextlevel_testrun.ipynb`](src/04_notebooks/02_fulltextlevel_testrun.ipynb) or `python -m bps_review run-fulltext-testrun`
- Code and its own documentation: [src/03_pipeline/bps_review/fulltext/](src/03_pipeline/bps_review/fulltext/README.md) · Results: [src/05_test_runs/official/](src/05_test_runs/official/)

### What the test run found in the pipeline

The first execution produced perfect agreement, Fleiss' kappa of exactly 1.00, on five fields including all three domain-mention flags. Three different models do not agree perfectly on whether a review carries substantive social content. The cause was that the Stage 2 prompt listed value vocabularies but never named the fields to return: the models omitted those five, and the deterministic repair layer filled them from the lexical rule-based coder. The output looked like a complete structured coding and was in fact keyword matching for the review's core RQ2 variables.

The prompt now carries an explicit field specification and output contract, and a test asserts that the specification and the validated schema cannot drift apart again. The same defect was present in the pipeline that produced the current manuscript, which is one more reason the manuscript numbers are held as provisional.

The full-text run exposed two more, both in how the pipeline handled a provider that misbehaves. A failed request was silently re-asked without the JSON mode, the completion cap, and the reasoning settings it was made with, so a degraded answer could enter the table looking like a coding made under the run's stated settings; that fallback is gone, and the status code is raised instead. And retries backed off linearly and gave up inside fifteen seconds, which is far shorter than a provider outage: one pass lost 24 of 141 codings to a run of 503s. A congested provider is now told apart from a bad answer and waited out on a jittered exponential schedule, because the two failures need opposite responses. `--repair-coding` re-codes only the cells that failed and splices them in, so filling a gap never again costs the whole grid. The run on disk is complete: 141 of 141 codings, no failures.

## 📋 Table of Contents

- [🚧 Project Status](#-project-status-test-run-coding-schemes-under-expert-review)
- [🧪 Cross-provider test runs](#-cross-provider-test-runs)
- [📝 Abstract](#-abstract)
- [📌 Key Findings](#-key-findings)
- [📄 Full Paper](#-full-paper)
- [🗂️ Repository Structure](#-repository-structure)
- [🛠️ Setup and Installation](#-setup-and-installation)
- [🚀 Usage](#-usage)
- [🧬 Pipeline Overview](#-pipeline-overview)
- [📦 Outputs](#-outputs)
- [🔬 Methodological Notes](#-methodological-notes)
- [📚 Citation](#-citation)
- [⚖️ License](#-license)

## 📝 Abstract

This repository contains the end-to-end research pipeline for an OSF-registered systematic review of how the biopsychosocial (BPS) model is operationalized in chronic pain review literature. The project is not a static manuscript dump: it links protocol, search, deduplication, screening, abstract coding, full-text coding, ontology-aligned semantic loading, figure generation, and LaTeX compilation in one auditable workflow.

The current review asks four questions: how BPS is operationalized, how biological/psychological/social scope and integration are distributed in musculoskeletal pain reviews, which psychological concepts and frameworks dominate the literature, and which conceptual problems recur when BPS is invoked. To answer those questions, the repository combines OSF-anchored review methods with structured LLM-based coding, transformer embeddings, and benchmark-relative semantic analyses across a two-layer BPS ontology.

> Main manuscript result: BPS language is widespread, but substantive triadic integration is uncommon; the most stable corpus-level signal is a persistent social shortfall relative to biological and psychological loading.

## 📌 Key Findings

### Corpus scale

- The current manuscript analyzes 111 included chronic pain review records spanning 1990 to 2026.
- The operational search window extends from January 1, 1977 to March 31, 2026, with deviations logged against the original OSF registration.
- Musculoskeletal pain is the dominant ICD-11 category among included BPS-invoking reviews.

### Conceptual signal

- Most included reviews use BPS language more as framing, organization, or intervention rationale than as explicit cross-domain mechanism.
- The provisional typology is dominated by pseudo-BPS or partial-signal records rather than clear integrative ones.
- Psychological content is concentrated around depression, stress, and anxiety, while more theory-specific constructs appear much less often.

### Semantic signal

- The ontology-aligned embedding layer shows that most reviews cluster close to equal BPS loading, but with a repeated shift away from the social pole.
- Social language is often present lexically yet comparatively weak in the semantic centre of mass of the abstracts.
- Pairwise and triadic loading analyses show that the meaningful pattern is benchmark-relative redistribution, not large simplex dispersion.

## 📄 Full Paper

- PDF: [paper/report/main.pdf](paper/report/main.pdf)
- LaTeX source: [paper/report/main.tex](paper/report/main.tex)
- References: [paper/report/references.bib](paper/report/references.bib)
- Generated manuscript tables: [paper/report/generated](paper/report/generated)
- Generated figures: [paper/assets/figures](paper/assets/figures)

## 🗂️ Repository Structure

`src/` is grouped into numbered sections that follow the order of the review: what it committed to, the instruments, the code, the notebooks, the data, the stage outputs, the semantic layer, and the documentation. A full source map lives in [src/README.md](src/README.md).

```text
SystematicReview_on_BioPsychoSocialModel_in_ChronicPainResearch/   # project root
├── README.md                  # this file
├── LICENSE · Makefile · pyproject.toml · .env.example
├── docker/                    # containerized reproducible environment
├── paper/                     # manuscript, generated tables and figures
└── src/
    ├── README.md              # a clean source map of everything under src/
    ├── 01_protocol/           # OSF registration, codebooks, decision rules
    ├── 02_coding_schemes/     # expert-evaluation dossiers (HTML / PDF / README)
    │   ├── _build/                # single source of truth and generator
    │   └── scheme_1/ ... scheme_6/
    ├── 03_pipeline/           # everything that runs
    │   ├── bps_review/            # the Python package
    │   │   ├── pilot/                 # abstract-level test run (scheme 2)
    │   │   ├── fulltext/              # full-text test run (scheme 3)
    │   │   └── graph/                 # interactive knowledge graph over a coded run
    │   ├── config/                # YAML configs controlling pipeline behavior
    │   └── tests/                 # the test suite
    ├── 04_notebooks/
    │   ├── 01_abstractlevel_testrun.ipynb
    │   ├── 02_fulltextlevel_testrun.ipynb
    │   └── 03_synthesislevel_testrun.ipynb
    ├── 05_test_runs/          # every run of the workflow, and all of its outputs
    │   ├── official/              # 47 full texts x 3 models, scheme 3, plus the knowledge graph
    │   └── tests/01_pilot_abstract/   # 100 abstracts x 3 models, scheme 2
    │
    │   ---- everything below is local only, and absent from a fresh clone ----
    │
    ├── 06_data/               # raw, interim, and processed data areas, plus the embedding cache
    ├── 07_docs/               # project status
    ├── 08_artifacts/          # run logs and validation scratch space
    └── 09_review_stages/      # stage-by-stage working files of the main pipeline
        └── 05_synthesis/semantic_space/  # ontology-aligned embeddings and loadings
```

## 🛠️ Setup and Installation

### Option A. Local editable install

```bash
# 1. Clone the repository
git clone https://github.com/stvsever/SystematicReview_on_BioPsychoSocialModel_in_ChronicPainResearch.git
cd SystematicReview_on_BioPsychoSocialModel_in_ChronicPainResearch

# 2. Create and activate a Python environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install the project
python -m pip install --upgrade pip
python -m pip install -e .

# 4. Configure environment variables
cp .env.example .env
```

Recommended `.env` keys:

- `OPENROUTER_API_KEY` (required for both coding stages)
- `NCBI_EMAIL` and `NCBI_API_KEY` (optional but recommended for PubMed and PubMed Central)
- `CLARIVATE_API_KEY` if Web of Science Starter access is available
- `EDS_API_USER`, `EDS_API_PASSWORD`, `EDS_API_PROFILE`, `EDS_API_ORG` if PsycINFO EDS access is available

### Option B. 🐳 Docker

```bash
cp .env.example .env
docker compose -f docker/docker-compose.yml up --build
```

## 🚀 Usage

### Run the cross-provider test runs

```bash
python -m bps_review run-abstract-testrun                  # reuse cached sample and codings
python -m bps_review run-abstract-testrun --force-coding   # re-code all 100 abstracts (300 calls)
python -m bps_review build-fulltext-corpus                 # retrieve the open-access full texts
python -m bps_review run-fulltext-testrun --force-coding   # re-code every full text
python -m bps_review run-fulltext-testrun --repair-coding  # re-code only the codings that failed
python -m bps_review build-fulltext-graph                  # rebuild only the knowledge graph
```

### Run the full main pipeline

```bash
python -m bps_review run-all
```

### Rebuild the coding-scheme dossiers

```bash
cd src/02_coding_schemes/_build
python3 build.py            # HTML, README, and PDF for every scheme (needs tectonic)
python3 build.py --no-pdf   # text surfaces only
```

### Compile the paper

```bash
cd paper/report
tectonic --reruns 4 main.tex
```

### Common CLI commands

```bash
python -m bps_review check-api-access
python -m bps_review search-pubmed
python -m bps_review search-wos
python -m bps_review search-psycinfo
python -m bps_review dedupe
python -m bps_review prepare-screening
python -m bps_review screen-stage1
python -m bps_review extract-stage2
python -m bps_review prepare-stage3
python -m bps_review semantic-loading
python -m bps_review build-assets
python -m bps_review run-abstract-testrun
python -m bps_review build-fulltext-corpus
python -m bps_review run-fulltext-testrun
python -m bps_review build-fulltext-graph
```

### What `run-all` does

| Step | Purpose |
|------|---------|
| `search-pubmed` | Pull registered PubMed queries and normalize records |
| `search-wos` | Pull Web of Science Starter records when credentials are available |
| `search-psycinfo` | Pull PsycINFO via EDS when credentials are available |
| `dedupe` | Merge sources and remove duplicate records |
| `prepare-screening` | Generate pilot, Rayyan, and reliability materials |
| `screen-stage1` | Apply Stage 1 title/abstract eligibility logic |
| `extract-stage2` | Run structured LLM-first abstract coding with deterministic metadata fields |
| `prepare-stage3` | Build full-text manifest, retrieval queue, and coding templates |
| `semantic-loading` | Embed records and ontology anchors into a shared BPS semantic space |
| `build-assets` | Generate tables, figures, and manuscript fragments |

## 🧬 Pipeline Overview

```text
OSF protocol and codebooks
        |
        v
Database search (PubMed / Web of Science / PsycINFO)
        |
        v
Normalization + deduplication
        |
        v
Stage 1 screening
        |
        v
Stage 2 abstract coding                        <-- scheme 2, cross-provider test run
  - structured LLM judgements with an explicit field contract
  - deterministic repair and rule-derived candidacy
        |
        v
Stage 3 full-text coding                       <-- scheme 3, cross-provider test run
  - coverage and integration ladders
  - verbatim evidence per graded judgement
  - quote verification against the source text
        |
        v
Ontology-aligned semantic loading
  - record embeddings, 2-layer BPS ontology
  - benchmark-relative domain and pairwise analyses
        |
        v
Figures, tables, generated LaTeX, compiled manuscript
```

## 📦 Outputs

### Test-run outputs

- `src/05_test_runs/official/` (corpus, codings, extracted items, reliability, integrity, figures, summary)
- `src/05_test_runs/official/02_model_codings/` (**start here**: every coding long and wide, per paper, per provider, per extraction category, with every list split into one column per item)
- `src/05_test_runs/official/05_knowledge_graph/index.html` (the interactive graph over the coded run)
- `src/05_test_runs/tests/01_pilot_abstract/` (sample, codings, reliability tables, figures, candidate set, summary)

Article full texts are read but never redistributed, so no article body is carried in Git.
What is carried is the inventory, the identifiers, and everything the coding derived from the
text: the coded rows, the extracted items, the verbatim evidence quotes, and the aggregate
tables. `make fulltext-corpus` restores the texts locally and repeats quote verification.

### Core manuscript outputs

- `paper/report/main.pdf`
- `paper/report/generated/*.tex`
- `paper/assets/figures/*.png`
- `paper/assets/tables/*.csv`

### Review-stage outputs

These are working files of the main pipeline and stay local, so they are absent from a fresh clone.

- `src/09_review_stages/02_search/outputs/deduplicated_records.csv`
- `src/09_review_stages/03_screening/outputs/stage1_screening.csv`
- `src/09_review_stages/04_extraction/outputs/stage2_abstract_coding.csv`
- `src/09_review_stages/04_extraction/outputs/stage2_llm_structured_coding.csv`
- `src/09_review_stages/04_extraction/outputs/stage3_candidate_manifest.csv`
- `src/09_review_stages/04_extraction/forms/stage3_fulltext_coding_template.csv`

### Semantic outputs

- `src/09_review_stages/05_synthesis/semantic_space/records/semantic_corpus.jsonl`
- `src/09_review_stages/05_synthesis/semantic_space/records/record_embeddings.npy`
- `src/09_review_stages/05_synthesis/semantic_space/analysis/record_domain_loadings.csv`
- `src/09_review_stages/05_synthesis/semantic_space/analysis/pairwise_domain_loadings.csv`

### Coding-scheme dossiers

- `src/02_coding_schemes/index.html`
- `src/02_coding_schemes/scheme_N/scheme_N.html`, `scheme_N.pdf`, `README.md`

## 🔬 Methodological Notes

- **The OSF registration is the governing framework.** Deviations are logged rather than silently absorbed into the pipeline.
- **Verdicts are derived, never asked.** Stage 3 candidacy, full-text eligibility, conceptual yield, synthesis priority, and every binary presence flag are computed deterministically from the coded content, so the filter is auditable and identical across providers. Derived fields are recomputed on load, so a cached run always reports the current rules.
- **The word biopsychosocial is never evidence of coverage.** A domain counts only when domain-specific content is present, and the coding scheme says so in the anchor of every domain field.
- **Agreement is measured on the right variable.** Categorical decisions get kappa-style coefficients; ordered ladders additionally get an adjacent-agreement rate; conceptual elements are reduced to a derived binary presence; open extraction lists are compared with set overlap.
- **Evidence is checkable.** Every full-text judgement carries a verbatim quote, and every quote is matched back against the source article after the run. Unverified quotes are reported, not hidden.
- **Nothing is fabricated to fill a gap.** A response that is not a coding of the given paper is rejected and retried; a paper that never codes is written as an explicit failure row. Item caps are ceilings and never targets, so an empty list is a coding rather than a hole.
- **The typology is checked against itself.** The full-text scheme codes `bps_typology` and independently derives it from coverage and integration, and reports the concordance, which is a direct test of how tightly that definition is specified.
- The LLM stages use a high-concurrency `ThreadPoolExecutor`, structured JSON output, schema validation, deterministic repair, hard wall-clock timeouts, and per-model runtime settings where an endpoint requires them.
- Domain notation in outputs uses `B`, `P`, `S` for biological, psychological, and social; `S_lex` denotes lexical mention and `S_subst` denotes substantive mention after lexical-token exclusion.
- Web of Science and PsycINFO records are imported manually; the programmatic search covers MEDLINE (PubMed) and, for the full-text stage, PubMed Central open access.
- All coding schemes are validated by experts before the pipeline is re-run on the full corpus.
- No em dashes are used anywhere in the generated dossiers or the codebase.

## 📚 Citation

If you use this repository, manuscript, or outputs, cite the paper and OSF registration.

### OSF

> Van Severen, S., Eccleston, C., De Paepe, A., Braun, M., Dendauw, J., Socorro Cumplido, J. L., & Crombez, G. (2026). *How the biopsychosocial model frames chronic pain research* [Registration]. Open Science Framework. https://doi.org/10.17605/OSF.IO/T4FAM

### Repository citation

```bibtex
@misc{vanseveren2026bpsreview,
  title        = {How the Biopsychosocial Model Frames Chronic Pain Research},
  author       = {Van Severen, Stijn and Eccleston, Christopher and De Paepe, Annick and Braun, Maya and Dendauw, Julie and Socorro Cumplido, Jose Luis and Crombez, Geert},
  year         = {2026},
  howpublished = {\url{https://github.com/stvsever/SystematicReview_on_BioPsychoSocialModel_in_ChronicPainResearch}},
  note         = {OSF registration DOI: 10.17605/OSF.IO/T4FAM}
}
```

## ⚖️ License

This project is released under the MIT License. See [LICENSE](LICENSE).
