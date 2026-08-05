# Stage 02: Search

Purpose:

- Execute protocol-consistent literature retrieval.
- Archive raw search results and normalized metadata.
- Preserve query strings, timestamps, and counts for PRISMA reporting.

Inputs:

- `config/search_queries.yaml`
- Manual exports placed in `data/manual_imports/`
- Optional API credentials in `.env` for PubMed, Clarivate, and EDS

Outputs:

- Raw PubMed JSON/XML exports
- Raw Web of Science or PsycINFO API exports where available
- Normalized CSV/JSONL search corpus
- Search log with exact query, date, count, and source
- Screening-ready Rayyan upload files
