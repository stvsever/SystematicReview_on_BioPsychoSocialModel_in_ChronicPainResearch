# Project Status

Last updated: 2026-04-06

## Hierarchical goals

1. Preserve the OSF-registered review logic in machine-readable form.
2. Build a reproducible search to reporting pipeline that can be rerun and audited.
3. Keep automated assistance clearly separated from adjudicated review decisions.
4. Produce manuscript-ready assets and a compiled PDF from the current repository state.

## Current state

- Root repository skeleton created.
- Protocol metadata encoded in `config/protocol.yaml`.
- Search strings encoded in `config/search_queries.yaml`.
- Stage directories created.
- PubMed direct API access confirmed.
- Web of Science Starter and EDS API access checks implemented.
- Current API check status: PubMed `ok`; Web of Science Starter `missing_credentials`; EDS API `missing_credentials`.
- Manual import path reserved for Web of Science and PsycINFO exports.
- Executable Python CLI implemented and installed locally.
- PubMed operational and sensitivity queries executed through the March 31, 2026 operational window.
- Current combined normalized corpus size: 3874 records.
- Current deduplicated corpus size: 3372 records (502 duplicates removed).
- Current provisional Stage 1 includes: 109 records (3262 excludes; 1 unclear).
- Current Stage 2 abstract-coded set: 109 records.
- Current Stage 3 candidate set: 87 records.
- Current PMC full texts cached automatically: 34.
- Current manual full-text retrieval queue: 53.
- Rayyan-ready screening queue and reliability subsets generated.
- Full Stage 2 LLM assistance output generated for 109 included records.
- Tables and figures generated under `paper/assets/`.
- Tectonic manuscript compiled successfully to `paper/report/main.pdf`.
- Logged LLM pilot artifact written to `src/review_stages/04_extraction/outputs/llm_objective_pilot.json`.
- Logged full Stage 2 LLM assist written to `src/review_stages/04_extraction/outputs/stage2_objective_llm_assist.csv`.
- Current synthesis summary written to `src/review_stages/05_synthesis/outputs/results_summary.json`.
- Smoke tests currently passing (`3 passed`).

## Open constraints

- Full automated access to Web of Science and PsycINFO is not assumed in this environment.
- Stage 3 full-text coding will depend on accessible PDFs or manually supplied full texts for records that are not openly available.
- Any protocol deviations must be written to `src/review_stages/01_protocol/outputs/deviations.md`.
