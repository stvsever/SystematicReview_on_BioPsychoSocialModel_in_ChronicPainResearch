from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from requests import RequestException

from bps_review.utils.io import load_yaml, slugify, utc_timestamp, write_json
from bps_review.utils.env import get_env, load_environment
from bps_review.utils.metadata import infer_country_from_text
from bps_review.utils.paths import project_path


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def operational_date_window() -> dict[str, str]:
    protocol = load_yaml(project_path("config", "protocol.yaml"))
    search_cfg = protocol["search"]
    return search_cfg.get("operational_date_window") or search_cfg.get("date_window") or search_cfg["registered_date_window"]


def _base_params() -> dict[str, str]:
    load_environment()
    params = {}
    if get_env("NCBI_EMAIL"):
        params["email"] = str(get_env("NCBI_EMAIL"))
    if get_env("NCBI_TOOL"):
        params["tool"] = str(get_env("NCBI_TOOL"))
    if get_env("NCBI_API_KEY"):
        params["api_key"] = str(get_env("NCBI_API_KEY"))
    return params


def _request(endpoint: str, params: dict[str, str | int]) -> requests.Response:
    merged = {**_base_params(), **params}
    retries = 4
    delay = 0.8
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(f"{EUTILS_BASE}/{endpoint}", params=merged, timeout=60)
            response.raise_for_status()
            return response
        except RequestException as error:
            last_error = error
            if attempt == retries:
                break
            time.sleep(delay * attempt)
    if last_error:
        raise last_error
    raise RuntimeError("Unexpected request failure without captured exception.")


def load_query(query_key: str) -> dict[str, str]:
    config = load_yaml(project_path("config", "search_queries.yaml"))
    query = config["queries"][query_key]
    return query


def _parse_pubdate(article: ET.Element) -> tuple[str, int | None]:
    pub_date = article.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate")
    if pub_date is None:
        return "", None
    year = pub_date.findtext("Year")
    medline_date = pub_date.findtext("MedlineDate")
    month = pub_date.findtext("Month") or ""
    day = pub_date.findtext("Day") or ""
    if year:
        date = "-".join(part for part in [year, month, day] if part)
        return date, int(year)
    if medline_date:
        match = re.search(r"(19|20)\d{2}", medline_date)
        if match:
            return medline_date, int(match.group(0))
        return medline_date, None
    return "", None


def _join_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return " ".join(text.strip() for text in node.itertext() if text and text.strip())


def _parse_article(article: ET.Element, query_key: str, query_label: str, search_date: str) -> dict[str, str]:
    pmid = article.findtext("./MedlineCitation/PMID", default="")
    article_node = article.find("./MedlineCitation/Article")
    title = _join_text(article_node.find("ArticleTitle") if article_node is not None else None)

    abstract_parts: list[str] = []
    if article_node is not None:
        for abstract_item in article_node.findall("./Abstract/AbstractText"):
            label = abstract_item.attrib.get("Label")
            text = _join_text(abstract_item)
            if not text:
                continue
            abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = "\n".join(abstract_parts)

    journal = article_node.findtext("./Journal/Title", default="") if article_node is not None else ""
    publication_date, year = _parse_pubdate(article)

    authors: list[str] = []
    affiliations: list[str] = []
    if article_node is not None:
        for author in article_node.findall("./AuthorList/Author"):
            last_name = author.findtext("LastName", default="")
            fore_name = author.findtext("ForeName", default="")
            collective = author.findtext("CollectiveName", default="")
            display = collective or " ".join(part for part in [fore_name, last_name] if part).strip()
            if display:
                authors.append(display)
            for affiliation in author.findall("./AffiliationInfo/Affiliation"):
                text = _join_text(affiliation)
                if text:
                    affiliations.append(text)

    publication_types = [
        _join_text(item)
        for item in article.findall("./MedlineCitation/Article/PublicationTypeList/PublicationType")
        if _join_text(item)
    ]
    languages = [
        _join_text(item)
        for item in article.findall("./MedlineCitation/Article/Language")
        if _join_text(item)
    ]
    keywords = [
        _join_text(item)
        for item in article.findall("./MedlineCitation/KeywordList/Keyword")
        if _join_text(item)
    ]
    mesh_terms = [
        _join_text(item)
        for item in article.findall("./MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName")
        if _join_text(item)
    ]

    doi = ""
    pmcid = ""
    for node in article.findall("./PubmedData/ArticleIdList/ArticleId"):
        if node.attrib.get("IdType") == "doi":
            doi = _join_text(node)
        elif node.attrib.get("IdType") == "pmc":
            pmcid = _join_text(node)

    first_affiliation = affiliations[0] if affiliations else ""
    publication_status = []
    if any("retracted" in item.lower() for item in publication_types):
        publication_status.append("retracted_or_retraction_notice")
    if any("corrected" in item.lower() or "erratum" in item.lower() for item in publication_types):
        publication_status.append("correction_or_erratum")

    record = {
        "record_id": f"pubmed_{pmid}" if pmid else f"pubmed_{slugify(title)[:40]}",
        "database": "PubMed",
        "interface": "MEDLINE via PubMed",
        "query_key": query_key,
        "query_label": query_label,
        "search_date": search_date,
        "retrieved_at_utc": utc_timestamp(),
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "publication_date": publication_date,
        "year": year or "",
        "authors": " | ".join(authors),
        "author_count": len(authors),
        "first_author": authors[0] if authors else "",
        "affiliations": " | ".join(dict.fromkeys(affiliations)),
        "contact_author_country_guess": infer_country_from_text(first_affiliation),
        "language": " | ".join(languages),
        "publication_types": " | ".join(publication_types),
        "keywords": " | ".join(keywords),
        "mesh_terms": " | ".join(mesh_terms),
        "publication_status_flag": " | ".join(publication_status) if publication_status else "none_detected",
        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
    }
    return record


def search_pubmed(query_key: str = "pubmed_operational_primary", batch_size: int = 200) -> pd.DataFrame:
    query = load_query(query_key)
    date_window = operational_date_window()
    search_date = utc_timestamp().split("T", 1)[0]

    esearch_response = _request(
        "esearch.fcgi",
        {
            "db": "pubmed",
            "term": query["string"],
            "retmode": "json",
            "retmax": 0,
            "usehistory": "y",
            "mindate": date_window["start"],
            "maxdate": date_window["end"],
            "datetype": "pdat",
            "sort": "pub date",
        },
    )
    esearch_payload = esearch_response.json()
    result = esearch_payload["esearchresult"]
    count = int(result["count"])
    webenv = result["webenv"]
    query_key_token = result["querykey"]

    raw_name = f"pubmed_{query_key}_{search_date}"
    write_json(project_path("review_stages", "02_search", "raw", f"{raw_name}_esearch.json"), esearch_payload)

    records: list[dict[str, str]] = []
    for start in range(0, count, batch_size):
        fetch_response = _request(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "query_key": query_key_token,
                "WebEnv": webenv,
                "retstart": start,
                "retmax": batch_size,
                "retmode": "xml",
            },
        )
        xml_text = fetch_response.text
        project_path("review_stages", "02_search", "raw").mkdir(parents=True, exist_ok=True)
        Path(project_path("review_stages", "02_search", "raw", f"{raw_name}_batch_{start:04d}.xml")).write_text(
            xml_text,
            encoding="utf-8",
        )
        root = ET.fromstring(xml_text)
        for article in root.findall("PubmedArticle"):
            records.append(_parse_article(article, query_key, query["label"], search_date))
        time.sleep(0.34)

    frame = pd.DataFrame(records).sort_values(["year", "title"], na_position="last", ascending=[False, True])
    output_path = project_path("review_stages", "02_search", "outputs", f"{raw_name}_records.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    manifest = pd.DataFrame(
        [
            {
                "database": "PubMed",
                "query_key": query_key,
                "query_label": query["label"],
                "search_date": search_date,
                "date_start": date_window["start"],
                "date_end": date_window["end"],
                "record_count": len(frame),
                "output_file": str(output_path.relative_to(project_path())),
            }
        ]
    )
    manifest_path = project_path("review_stages", "02_search", "outputs", "search_manifest.csv")
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        manifest = pd.concat([existing, manifest], ignore_index=True)
        manifest = manifest.drop_duplicates(subset=["database", "query_key", "search_date", "output_file"], keep="last")
    manifest.to_csv(manifest_path, index=False)
    return frame
