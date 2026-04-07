<div align="center">

# How the Biopsychosocial Model Frames Chronic Pain Research

### An OSF-registered systematic review with mixed-method synthesis, structured semantic coding, and ontology-aligned embeddings

[![Type](https://img.shields.io/badge/Type-systematic_review-7C3AED)](paper/report/main.tex)
[![OSF Registered](https://img.shields.io/badge/OSF-10.17605%2FOSF.IO%2FT4FAM-0F766E)](https://osf.io/t4fam)
[![Dockerized](https://img.shields.io/badge/Docker-ready-2496ED)](docker/)
[![MIT License](https://img.shields.io/badge/License-MIT-16A34A)](LICENSE)

**Stijn Van Severen<sup>1,\*</sup> · Christopher Eccleston<sup>1,2</sup> · Annick De Paepe<sup>1</sup> · Maya Braun<sup>1</sup> · Julie Dendauw<sup>1</sup> · Jose Luis Socorro Cumplido<sup>3</sup> · Geert Crombez<sup>1</sup>**

<sup>1</sup> Ghent University, Ghent, Belgium · <sup>2</sup> University of Bath, Bath, United Kingdom · <sup>3</sup> Ramon Llull University, Barcelona, Spain · <sup>\*</sup> Corresponding author

---

</div>

<a id="table-of-contents"></a>

## 🧭 Table of Contents

- [Abstract](#abstract)
- [Key Findings](#key-findings)
- [Full Paper](#full-paper)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Pipeline Overview](#pipeline-overview)
- [Outputs](#outputs)
- [Methodological Notes](#methodological-notes)
- [Citation](#citation)
- [License](#license)

## 📝 Abstract

This repository contains the end-to-end research pipeline for an OSF-registered systematic review of how the biopsychosocial (BPS) model is operationalized in chronic pain review literature. The project is not a static manuscript dump: it links protocol, search, deduplication, screening, abstract coding, Stage 3 full-text preparation, ontology-aligned semantic loading, figure generation, and LaTeX compilation in one auditable workflow.

The current review asks four questions: how BPS is operationalized, how biological/psychological/social scope and integration are distributed in musculoskeletal pain reviews, which psychological concepts and frameworks dominate the literature, and which conceptual problems recur when BPS is invoked. To answer those questions, the repository combines OSF-anchored review methods with structured LLM-based abstract coding, transformer embeddings, and benchmark-relative semantic analyses across a two-layer BPS ontology.

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

```text
SystematicReview_on_BioPsychoSocialModel_in_ChronicPainResearch/ # project root
├── README.md # project overview and usage guide
├── LICENSE # license terms
├── Makefile # shortcut commands for pipeline/report tasks
├── pyproject.toml # Python package metadata and dependencies
├── .env.example # example environment variables
├── config/ # YAML configs controlling pipeline behavior
│   ├── pipeline.yaml # stage toggles and runtime settings
│   ├── protocol.yaml # protocol constraints and coding rules
│   └── search_queries.yaml # search strings per database/source
├── docker/ # containerized reproducible environment
│   ├── Dockerfile # image definition
│   └── docker-compose.yml # multi-service/local orchestration
├── paper/ # manuscript and publication assets
│   ├── assets/ # generated figure/table inputs
│   │   ├── figures/ # PNG/PDF visual outputs
│   │   └── tables/ # CSV table outputs used in report
│   └── report/ # LaTeX manuscript sources and outputs
│       ├── generated/ # auto-generated .tex fragments
│       ├── main.tex # main manuscript file
│       └── main.pdf # compiled manuscript PDF
└── src/ # source code and stage artifacts
        ├── bps_review/ # main Python package
        │   ├── cli.py # CLI entry points and command routing
        │   ├── search/ # search/import and dedup logic
        │   ├── screening/ # stage 1/2 screening workflows
        │   ├── extraction/ # extraction and coding utilities
        │   ├── reporting/ # figure/table/report asset builders
        │   └── llm/ # LLM-assisted classification helpers
        ├── protocol/ # protocol support documents/codebooks
        │   ├── codebooks/ # coding dictionaries and labels
        │   └── osf/ # OSF registration materials
        ├── review_stages/ # organized outputs by review stage
        │   ├── 01_protocol/ # stage 1 protocol artifacts
        │   ├── 02_search/ # stage 2 search outputs
        │   ├── 03_screening/ # stage 3 screening decisions
        │   ├── 04_extraction/ # stage 4 extraction datasets
        │   └── 05_synthesis/ # stage 5 synthesis outputs
        └── vector_db/ # semantic embedding/index data
                └── semantic_loading/ # semantic loading vectors and exports
```

## 🛠️ Setup and Installation

### Option A. Local editable install

```bash
# 1. Clone the repository
git clone https://github.com/stvsever/SystematicReview_on_BioPsychoSocialModel_in_ChronicPainReseach.git
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

- `NCBI_EMAIL`
- `NCBI_API_KEY` (optional but recommended)
- `OPENROUTER_API_KEY`
- `CLARIVATE_API_KEY` if Web of Science Starter access is available
- `EDS_API_USER`, `EDS_API_PASSWORD`, `EDS_API_PROFILE`, `EDS_API_ORG` if PsycINFO EDS access is available

### Option B. 🐳 Docker

```bash
# 1. Clone the repository
git clone https://github.com/stvsever/SystematicReview_on_BioPsychoSocialModel_in_ChronicPainReseach.git
cd SystematicReview_on_BioPsychoSocialModel_in_ChronicPainResearch

# 2. Configure environment variables
cp .env.example .env

# 3. Build and run
docker compose -f docker/docker-compose.yml up --build
```

Manual Docker build:

```bash
docker build -f docker/Dockerfile -t bps-review .
docker run --env-file .env -v "$(pwd):/workspace" bps-review
```

## 🚀 Usage

### Run the full pipeline

```bash
python -m bps_review run-all
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
Stage 2 abstract coding
  - deterministic metadata fields
  - structured LLM semantic judgments
  - provisional typology
        |
        v
Stage 3 preparation
  - candidate manifest
  - full-text retrieval
  - coding template
        |
        v
Ontology-aligned semantic loading
  - record embeddings
  - 2-layer BPS ontology
  - benchmark-relative domain / pairwise analyses
        |
        v
Figures, tables, generated LaTeX, compiled manuscript
```

## 📦 Outputs

### Core manuscript outputs

- `paper/report/main.pdf`
- `paper/report/generated/*.tex`
- `paper/assets/figures/*.png`
- `paper/assets/tables/*.csv`

### Table 1 audit outputs

- `paper/report/generated/characteristics_table.tex`
- `paper/assets/tables/table1_description_audit.csv`
- `paper/assets/tables/included_review_full_references.csv`

The Table 1 Description column is generated as a concise past-tense one-sentence summary. The generator prioritizes cached full-text objective statements (Stage 3 open-access records), then `objective_text`, then abstract first sentence, with title fallback when needed. Description provenance, focus phrases, and full-text availability flags are exported for reproducible QA.

### Relevance + OSF alignment audits

- `paper/assets/tables/manual_relevance_audit.csv`
- `paper/assets/tables/osf_alignment_checklist.csv`
- `src/review_stages/04_extraction/forms/stage3_manual_relevance_checklist.csv`
- `src/review_stages/04_extraction/outputs/stage3_retrieval_validation.csv`

These outputs provide a human-adjudication queue for relevance and retrieval checks and document protocol-alignment checkpoints against the OSF registration.

### Domain-coding audit outputs

- `paper/assets/tables/domain_mentions.csv`
- `paper/assets/tables/musculoskeletal_scope.csv`
- `paper/assets/tables/domain_mention_recode_audit.csv`

The domain audit captures the transition from Stage 2 lexical/LLM mention flags to the substantive recode used in manuscript figures and tables.

### Review-stage outputs

- `src/review_stages/02_search/outputs/combined_records.csv`
- `src/review_stages/02_search/outputs/deduplicated_records.csv`
- `src/review_stages/03_screening/outputs/stage1_screening.csv`
- `src/review_stages/04_extraction/outputs/stage2_abstract_coding.csv`
- `src/review_stages/04_extraction/outputs/stage2_llm_structured_coding.csv`
- `src/review_stages/04_extraction/outputs/stage3_candidate_manifest.csv`
- `src/review_stages/04_extraction/forms/stage3_fulltext_coding_template.csv`

### Semantic outputs

- `src/vector_db/semantic_loading/records/semantic_corpus.jsonl`
- `src/vector_db/semantic_loading/records/record_embeddings.npy`
- `src/vector_db/semantic_loading/analysis/record_domain_loadings.csv`
- `src/vector_db/semantic_loading/analysis/pairwise_domain_loadings.csv`
- `paper/assets/tables/semantic_embedding_landscape_coordinates.csv`

## 🔬 Methodological Notes

- The OSF registration is the governing framework. Deviations are logged rather than silently absorbed into the pipeline.
- Stage 2 is no longer documented as a light optional LLM sidecar. It is a structured semantic coding layer with fixed output vocabularies and archived JSON batches.
- Domain coding is intentionally conservative: the mere presence of the word `biopsychosocial` does not automatically imply substantive biological, psychological, and social coverage.
- Domain notation in outputs uses `B`, `P`, `S` for biological, psychological, and social; `S_lex` denotes lexical mention and `S_subst` denotes substantive mention after lexical-token exclusion.
- Table 1 descriptions are evidence-bound to each included record and are generated from objective/abstract text before title fallback to reduce over-short and over-long entries.
- Stage 3 remains the final adjudication layer for mechanistic integration, framework architecture, and concept-definition evidence.
- The semantic ontology contains 42 subdomains distributed across biological, psychological, and social levels, and the manuscript keeps only the most RQ-relevant figures in the main body while pushing tables and audit material to supplementary outputs.

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
        howpublished = {\url{https://github.com/stvsever/SystematicReview_on_BioPsychoSocialModel_in_ChronicPainReseach}},
  note         = {OSF registration DOI: 10.17605/OSF.IO/T4FAM}
}
```

## ⚖️ License

This project is released under the MIT License. See [LICENSE](LICENSE).
