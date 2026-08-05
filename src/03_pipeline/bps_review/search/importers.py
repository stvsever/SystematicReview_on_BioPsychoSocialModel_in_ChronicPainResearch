from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import rispy

from bps_review.settings import resolve_path


def _read_ris(path: Path, source_database: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        entries = rispy.load(handle)

    rows: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries, start=1):
        title = entry.get("title", "") or ""
        abstract = entry.get("abstract", "") or ""
        authors = entry.get("authors") or []
        year = str(entry.get("year", "") or "")
        doi = entry.get("doi", "") or ""
        row: dict[str, Any] = {
            "record_id": f"{source_database}:{path.stem}:{idx}",
            "source_database": source_database,
            "pmid": "",
            "pmcid": "",
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "journal": entry.get("journal_name", "") or entry.get("secondary_title", "") or "",
            "year": year,
            "language": entry.get("language", "") or "",
            "authors": "; ".join(authors) if isinstance(authors, list) else str(authors),
            "first_author_affiliation": "",
            "country_contact_author": "",
            "publication_types": entry.get("type_of_reference", "") or "",
            "mesh_terms": "",
            "keywords": "; ".join(entry.get("keywords", []) or []),
            "text_blob": f"{title} {abstract}".lower(),
        }
        rows.append(row)
    return rows


def load_manual_exports() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_key, source_name in [("raw_psycinfo", "psycinfo"), ("raw_wos", "wos")]:
        folder = resolve_path(source_key)
        for file_path in sorted(folder.glob("*.ris")):
            rows.extend(_read_ris(file_path, source_name))
    return pd.DataFrame(rows)
