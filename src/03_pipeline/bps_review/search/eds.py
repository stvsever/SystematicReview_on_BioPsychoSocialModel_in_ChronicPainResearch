from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from bps_review.search.pubmed import operational_date_window
from bps_review.utils.env import get_env
from bps_review.utils.io import load_yaml, write_json
from bps_review.utils.paths import project_path


AUTH_URL = "https://eds-api.ebscohost.com/authservice/rest/uidauth"
SESSION_URL = "https://eds-api.ebscohost.com/edsapi/rest/CreateSession"
SEARCH_URL = "https://eds-api.ebscohost.com/edsapi/rest/Search"


def _query_text(query_key: str) -> str:
    config = load_yaml(project_path("config", "search_queries.yaml"))
    return config["queries"][query_key]["string"]


def _auth_token() -> str:
    user = get_env("EDS_API_USER")
    password = get_env("EDS_API_PASSWORD")
    interface_id = get_env("EDS_API_INTERFACE_ID") or get_env("EDS_API_INTERFACE")
    if not user or not password or not interface_id:
        raise EnvironmentError("EDS_API_USER, EDS_API_PASSWORD, and EDS_API_INTERFACE_ID (or EDS_API_INTERFACE) are required.")
    response = requests.post(
        AUTH_URL,
        json={"UserId": user, "Password": password, "InterfaceId": interface_id},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["AuthToken"]


def _session_token(auth_token: str) -> str:
    params = {
        "profile": get_env("EDS_API_PROFILE") or "",
        "guest": get_env("EDS_API_GUEST") or "n",
        "org": get_env("EDS_API_ORG") or "",
    }
    response = requests.get(SESSION_URL, headers={"x-authenticationToken": auth_token}, params=params, timeout=60)
    response.raise_for_status()
    return response.json()["SessionToken"]


def search_eds_psycinfo(query_key: str = "psycinfo_eds_operational", results_per_page: int = 100) -> pd.DataFrame:
    search_date = pd.Timestamp.utcnow().date().isoformat()
    auth_token = _auth_token()
    session_token = _session_token(auth_token)
    date_window = operational_date_window()
    query = _query_text(query_key)
    page = 1
    records: list[dict[str, str]] = []

    while True:
        params = {
            "query-1": query,
            "resultsperpage": results_per_page,
            "pagenumber": page,
            "publicationyearfrom": date_window["start"][:4],
            "publicationyearto": date_window["end"][:4],
        }
        headers = {"x-authenticationToken": auth_token, "x-sessionToken": session_token}
        api_key = get_env("EDS_API_KEY")
        if api_key:
            headers["x-apiKey"] = api_key
        response = requests.get(SEARCH_URL, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        write_json(project_path("review_stages", "02_search", "raw", f"psycinfo_{query_key}_{search_date}_page_{page:03d}.json"), payload)

        search_result = payload.get("SearchResult", {})
        data = search_result.get("Data", {})
        hits = data.get("Records", [])
        for hit in hits:
            record_info = hit.get("RecordInfo", {})
            bib = record_info.get("BibRecord", {}).get("BibEntity", {})
            titles = bib.get("Titles", []) if isinstance(bib.get("Titles", []), list) else []
            title = ""
            for item in titles:
                if item.get("Type") == "main":
                    title = item.get("TitleFull", "")
                    break
            if not title and titles:
                title = titles[0].get("TitleFull", "")
            authors = []
            for author in bib.get("Authors", []) if isinstance(bib.get("Authors", []), list) else []:
                name = author.get("Name", {}).get("NameFull") or author.get("Name", {}).get("Y1")
                if name:
                    authors.append(name)
            abstracts = hit.get("Items", [])
            abstract = ""
            for item in abstracts:
                if item.get("Name") == "Abstract":
                    abstract = item.get("Data", "")
                    break
            records.append(
                {
                    "record_id": record_info.get("RecordID", f"psycinfo_{page}_{len(records)+1}"),
                    "database": "PsycINFO",
                    "interface": "EBSCO EDS API",
                    "query_key": query_key,
                    "query_label": query_key,
                    "search_date": search_date,
                    "retrieved_at_utc": pd.Timestamp.utcnow().isoformat(),
                    "pmid": "",
                    "pmcid": "",
                    "doi": bib.get("DOI", ""),
                    "title": title,
                    "abstract": abstract,
                    "journal": bib.get("Source", {}).get("TitleFull", ""),
                    "publication_date": bib.get("PublicationDate", ""),
                    "year": bib.get("PublicationYear", ""),
                    "authors": " | ".join(authors),
                    "author_count": len(authors),
                    "first_author": authors[0] if authors else "",
                    "affiliations": "",
                    "contact_author_country_guess": "",
                    "language": bib.get("Languages", ""),
                    "publication_types": bib.get("DocumentType", ""),
                    "keywords": "",
                    "mesh_terms": "",
                    "publication_status_flag": "",
                    "pubmed_url": "",
                }
            )
        statistics = search_result.get("Statistics", {})
        total_hits = int(statistics.get("TotalHits", 0) or 0)
        if not hits or page * results_per_page >= total_hits:
            break
        page += 1

    frame = pd.DataFrame(records)
    output_path = project_path("review_stages", "02_search", "outputs", f"psycinfo_{query_key}_{search_date}_records.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame
