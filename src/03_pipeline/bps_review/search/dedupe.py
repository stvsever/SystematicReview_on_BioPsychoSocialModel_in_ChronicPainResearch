from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from bps_review.utils.paths import project_path


CORE_COLUMNS = [
    "record_id",
    "database",
    "interface",
    "query_key",
    "query_label",
    "search_date",
    "retrieved_at_utc",
    "pmid",
    "pmcid",
    "doi",
    "title",
    "abstract",
    "journal",
    "publication_date",
    "year",
    "authors",
    "author_count",
    "first_author",
    "affiliations",
    "contact_author_country_guess",
    "language",
    "publication_types",
    "keywords",
    "mesh_terms",
    "publication_status_flag",
    "pubmed_url",
]


def _normalize_title(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", (value or "").lower())
    return " ".join(text.split())


def _normalize_doi(value: str) -> str:
    text = (value or "").strip().lower()
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return text


def _parse_ris(path: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    current: dict[str, list[str]] = {}
    current_tag: str | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        if re.match(r"^[A-Z0-9]{2}  - ", line):
            tag, value = line[:2], line[6:]
            current_tag = tag
            current.setdefault(tag, []).append(value.strip())
            if tag == "ER":
                records.append(current)
                current = {}
                current_tag = None
        elif current_tag:
            current[current_tag][-1] += f" {line.strip()}"
    normalized = []
    for item in records:
        title = " ".join(item.get("TI", []) or item.get("T1", []))
        abstract = " ".join(item.get("AB", []) or item.get("N2", []))
        journal = " ".join(item.get("JO", []) or item.get("T2", []) or item.get("JF", []))
        year_source = " ".join(item.get("PY", []) or item.get("Y1", []))
        year_match = re.search(r"(19|20)\d{2}", year_source)
        normalized.append(
            {
                "record_id": f"manual_{path.stem}_{len(normalized)+1}",
                "database": path.stem,
                "interface": "manual_import",
                "query_key": "manual_import",
                "query_label": path.name,
                "search_date": "",
                "retrieved_at_utc": "",
                "pmid": "",
                "pmcid": "",
                "doi": " ".join(item.get("DO", [])),
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "publication_date": year_source,
                "year": year_match.group(0) if year_match else "",
                "authors": " | ".join(item.get("AU", [])),
                "author_count": len(item.get("AU", [])),
                "first_author": item.get("AU", [""])[0],
                "affiliations": "",
                "contact_author_country_guess": "",
                "language": " | ".join(item.get("LA", [])),
                "publication_types": " | ".join(item.get("TY", [])),
                "keywords": "",
                "mesh_terms": "",
                "publication_status_flag": "",
                "pubmed_url": "",
            }
        )
    return pd.DataFrame(normalized)


def _parse_nbib(path: Path) -> pd.DataFrame:
    records: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    current_tag: str | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if current:
                records.append(current)
                current = {}
                current_tag = None
            continue
        if re.match(r"^[A-Z]{2,4}\s*-\s", line):
            tag, value = line.split("-", 1)
            current_tag = tag.strip()
            current.setdefault(current_tag, []).append(value.strip())
        elif current_tag:
            current[current_tag][-1] += f" {line.strip()}"
    if current:
        records.append(current)

    normalized = []
    for idx, item in enumerate(records, start=1):
        title = " ".join(item.get("TI", []))
        abstract = " ".join(item.get("AB", []))
        authors = item.get("AU", [])
        affiliation = " | ".join(item.get("AD", []))
        year_source = " ".join(item.get("DP", []) or item.get("DEP", []))
        year_match = re.search(r"(19|20)\d{2}", year_source)
        normalized.append(
            {
                "record_id": f"manual_{path.stem}_{idx}",
                "database": "PubMed_manual",
                "interface": "manual_import",
                "query_key": "manual_import",
                "query_label": path.name,
                "search_date": "",
                "retrieved_at_utc": "",
                "pmid": " ".join(item.get("PMID", [])),
                "pmcid": " ".join(item.get("PMC", [])),
                "doi": " ".join(item.get("AID", [])),
                "title": title,
                "abstract": abstract,
                "journal": " ".join(item.get("JT", []) or item.get("TA", [])),
                "publication_date": year_source,
                "year": year_match.group(0) if year_match else "",
                "authors": " | ".join(authors),
                "author_count": len(authors),
                "first_author": authors[0] if authors else "",
                "affiliations": affiliation,
                "contact_author_country_guess": "",
                "language": " | ".join(item.get("LA", [])),
                "publication_types": " | ".join(item.get("PT", [])),
                "keywords": " | ".join(item.get("OT", [])),
                "mesh_terms": " | ".join(item.get("MH", [])),
                "publication_status_flag": "",
                "pubmed_url": f'https://pubmed.ncbi.nlm.nih.gov/{" ".join(item.get("PMID", []))}/' if item.get("PMID") else "",
            }
        )
    return pd.DataFrame(normalized)


def _normalize_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in CORE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
    return normalized[CORE_COLUMNS]


def _load_manual_imports() -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    manual_dir = project_path("data", "manual_imports")
    for path in sorted(manual_dir.iterdir()):
        if path.name.startswith("."):
            continue
        if path.suffix.lower() == ".csv":
            frames.append(_normalize_frame_columns(pd.read_csv(path)))
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            frames.append(_normalize_frame_columns(pd.read_excel(path)))
        elif path.suffix.lower() == ".ris":
            frames.append(_parse_ris(path))
        elif path.suffix.lower() == ".nbib":
            frames.append(_parse_nbib(path))
        else:
            skipped.append(path.name)
    if frames:
        return pd.concat(frames, ignore_index=True, sort=False), skipped
    return pd.DataFrame(), skipped


def _active_normalized_search_files(search_dir: Path) -> list[Path]:
    manifest_path = search_dir / "search_manifest.csv"
    if not manifest_path.exists():
        return sorted(search_dir.glob("*_records.csv"))

    manifest = pd.read_csv(manifest_path).fillna("")
    if manifest.empty:
        return sorted(search_dir.glob("*_records.csv"))

    latest = (
        manifest.sort_values(["database", "query_key", "search_date", "output_file"])
        .groupby(["database", "query_key"], as_index=False)
        .tail(1)
    )
    paths = []
    for output_file in latest["output_file"]:
        candidate = project_path(output_file)
        if candidate.exists():
            paths.append(candidate)
    return sorted(set(paths))


def deduplicate_search_corpus() -> pd.DataFrame:
    search_dir = project_path("review_stages", "02_search", "outputs")
    frames: list[pd.DataFrame] = []
    for path in _active_normalized_search_files(search_dir):
        frames.append(pd.read_csv(path))
    manual_frame, skipped = _load_manual_imports()
    if not manual_frame.empty:
        frames.append(manual_frame)
    if not frames:
        raise FileNotFoundError("No normalized search records found.")

    combined = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    combined["norm_doi"] = combined["doi"].map(_normalize_doi)
    combined["norm_title"] = combined["title"].map(_normalize_title)
    combined["year"] = combined["year"].astype(str)
    combined["dedupe_key"] = combined.apply(
        lambda row: row["norm_doi"] if row["norm_doi"] else f'{row["norm_title"]}::{row["year"]}',
        axis=1,
    )
    combined = combined.sort_values(["dedupe_key", "database", "query_key"]).reset_index(drop=True)
    combined["duplicate_rank"] = combined.groupby("dedupe_key").cumcount() + 1
    combined["is_duplicate"] = combined["duplicate_rank"] > 1

    combined_path = search_dir / "combined_records.csv"
    combined.to_csv(combined_path, index=False)

    deduped = combined.loc[~combined["is_duplicate"]].copy()
    deduped_path = search_dir / "deduplicated_records.csv"
    deduped.to_csv(deduped_path, index=False)

    audit = pd.DataFrame(
        [
            {
                "combined_records": len(combined),
                "deduplicated_records": len(deduped),
                "duplicates_removed": int(combined["is_duplicate"].sum()),
                "manual_import_files_skipped": " | ".join(skipped),
            }
        ]
    )
    audit.to_csv(search_dir / "deduplication_summary.csv", index=False)
    return deduped
