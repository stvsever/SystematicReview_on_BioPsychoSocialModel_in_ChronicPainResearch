from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from bps_review.utils.env import get_env
from bps_review.utils.io import load_yaml, write_json
from bps_review.utils.paths import project_path


API_BASE = "https://api.clarivate.com/apis/wos-starter/v1/documents"


def _headers() -> dict[str, str]:
    api_key = get_env("CLARIVATE_API_KEY")
    if not api_key:
        raise EnvironmentError("CLARIVATE_API_KEY is not set.")
    return {"X-ApiKey": api_key}


def _query_text(query_key: str) -> str:
    config = load_yaml(project_path("config", "search_queries.yaml"))
    return config["queries"][query_key]["string"]


def search_wos_starter(query_key: str = "wos_starter_operational", page_size: int = 50) -> pd.DataFrame:
    query = _query_text(query_key)
    search_date = pd.Timestamp.utcnow().date().isoformat()
    page = 1
    records: list[dict[str, str]] = []

    while True:
        response = requests.get(
            API_BASE,
            headers=_headers(),
            params={"q": query, "limit": min(page_size, 50), "page": page, "db": "WOS"},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        write_json(project_path("review_stages", "02_search", "raw", f"wos_{query_key}_{search_date}_page_{page:03d}.json"), payload)
        hits = payload.get("hits", []) or payload.get("documents", []) or []
        for hit in hits:
            identifiers = hit.get("identifiers", {})
            source = hit.get("source", {}) if isinstance(hit.get("source"), dict) else {}
            names = hit.get("names", {})
            authors = names.get("authors", []) if isinstance(names, dict) else []
            author_names = []
            for author in authors:
                if isinstance(author, dict):
                    display = author.get("displayName") or author.get("wosStandard") or author.get("fullName")
                    if display:
                        author_names.append(display)
            title = hit.get("title")
            if isinstance(title, dict):
                title = title.get("value") or title.get("title")
            records.append(
                {
                    "record_id": hit.get("uid", f"wos_{page}_{len(records)+1}"),
                    "database": "Web of Science",
                    "interface": "Web of Science Starter API",
                    "query_key": query_key,
                    "query_label": query_key,
                    "search_date": search_date,
                    "retrieved_at_utc": pd.Timestamp.utcnow().isoformat(),
                    "pmid": "",
                    "pmcid": "",
                    "doi": identifiers.get("doi", ""),
                    "title": title or "",
                    "abstract": "",
                    "journal": source.get("sourceTitle", "") if isinstance(source, dict) else "",
                    "publication_date": hit.get("publishYear", ""),
                    "year": hit.get("publishYear", ""),
                    "authors": " | ".join(author_names),
                    "author_count": len(author_names),
                    "first_author": author_names[0] if author_names else "",
                    "affiliations": "",
                    "contact_author_country_guess": "",
                    "language": hit.get("language", ""),
                    "publication_types": hit.get("documentType", ""),
                    "keywords": "",
                    "mesh_terms": "",
                    "publication_status_flag": "",
                    "pubmed_url": "",
                }
            )
        metadata = payload.get("metadata", {})
        total_pages = metadata.get("totalPages") or payload.get("totalPages")
        if not hits or (total_pages and page >= int(total_pages)):
            break
        page += 1

    frame = pd.DataFrame(records)
    output_path = project_path("review_stages", "02_search", "outputs", f"wos_{query_key}_{search_date}_records.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame
