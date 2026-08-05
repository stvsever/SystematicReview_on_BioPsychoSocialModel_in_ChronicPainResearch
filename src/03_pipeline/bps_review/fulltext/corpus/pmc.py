from __future__ import annotations

"""Turn the abstract-level candidate set into a full-text corpus.

The full-text stage does not run its own search. It reads the candidate set the
abstract stage produced (the records the models agreed to carry forward) and
tries to retrieve the full text of each one from PubMed Central's open-access
subset, through the same NCBI E-utilities the review already uses for MEDLINE.

Retrieval runs in four steps.

1. **Resolve.** Every candidate carries a PMC id when PubMed has one. Candidates
   without one are looked up once through elink, in case the PMC id is missing
   from the MEDLINE record but the article is in PMC anyway.
2. **Fetch and parse.** JATS XML to structured text: front matter, abstract, and
   the body as an ordered list of sections. Reference lists, tables, and floating
   supplementary material are dropped, because they add tokens without adding
   conceptual content.
3. **Screen.** A paper is kept when a real body text came back. Everything else
   is logged with the reason it dropped out, so the funnel from candidates to
   coded papers is fully accounted for.
4. **Persist.** One CSV plus one plain-text file per paper, with a retrieval log
   of every candidate that was considered and a manifest of the whole step.

The result is an honest end-to-end funnel: PubMed query, abstract coding, filter,
open-access retrieval, full-text coding. Records that are not open access simply
do not reach the full-text stage, and the log says so for each one.
"""

import re
import time
import xml.etree.ElementTree as ET

import pandas as pd

from bps_review.fulltext.config import (
    corpus_candidates_csv,
    corpus_csv,
    corpus_manifest_json,
    corpus_selection_log_csv,
    corpus_text_dir,
)
from bps_review.pilot.config import candidate_set_csv
from bps_review.search.pubmed import _request as eutils_request
from bps_review.utils.io import ensure_parent, utc_timestamp, write_csv, write_json


FETCH_BATCH = 15
NCBI_PAUSE_SECONDS = 0.34
MIN_BODY_CHARS = 4000   # below this a "full text" is a stub, not an article

DROP_TAGS = {"table-wrap", "table", "supplementary-material", "fn-group", "ref-list", "back", "graphic", "media"}


def _text_of(node: ET.Element | None, skip: set[str] | None = None) -> str:
    """Flatten an element to text, skipping structural noise."""
    if node is None:
        return ""
    skip = skip or DROP_TAGS
    parts: list[str] = []
    for child in node.iter():
        if child.tag in skip:
            continue
        if child.text and child.text.strip():
            parts.append(child.text.strip())
        if child.tail and child.tail.strip():
            parts.append(child.tail.strip())
    return " ".join(parts)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _paragraph_text(node: ET.Element) -> str:
    """Text of one paragraph-like element, with citation markers removed."""
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    for child in node:
        if child.tag in DROP_TAGS or child.tag == "xref":
            if child.tail:
                parts.append(child.tail)
            continue
        parts.append(_text_of(child))
        if child.tail:
            parts.append(child.tail)
    return _clean(" ".join(parts))


def _parse_sections(body: ET.Element | None) -> list[dict[str, str]]:
    """Body to an ordered list of {title, text} sections, nested sections flattened."""
    if body is None:
        return []
    sections: list[dict[str, str]] = []

    def walk(node: ET.Element, inherited_title: str) -> None:
        title = _clean(_text_of(node.find("title"))) or inherited_title or "Body"
        paragraphs = []
        for child in node:
            if child.tag == "p":
                text = _paragraph_text(child)
                if text:
                    paragraphs.append(text)
            elif child.tag in ("list", "disp-quote"):
                text = _clean(_text_of(child))
                if text:
                    paragraphs.append(text)
        if paragraphs:
            sections.append({"title": title, "text": "\n\n".join(paragraphs)})
        for child in node.findall("sec"):
            walk(child, title)

    top_paragraphs = [_paragraph_text(child) for child in body.findall("p") if _paragraph_text(child)]
    if top_paragraphs:
        sections.append({"title": "Body", "text": "\n\n".join(top_paragraphs)})
    for section in body.findall("sec"):
        walk(section, "")
    return sections


def _parse_article(article: ET.Element) -> dict | None:
    meta = article.find(".//front/article-meta")
    if meta is None:
        return None

    ids = {node.get("pub-id-type"): _clean(_text_of(node)) for node in meta.findall("article-id")}
    pmcid = ids.get("pmcid") or ids.get("pmc") or ""
    pmid = ids.get("pmid", "")
    doi = ids.get("doi", "")

    title = _clean(_text_of(meta.find(".//title-group/article-title")))
    journal = _clean(_text_of(article.find(".//journal-meta//journal-title")))

    year = ""
    for pub_date in meta.findall(".//pub-date"):
        candidate = pub_date.findtext("year")
        if candidate:
            year = candidate
            break

    authors: list[str] = []
    for contrib in meta.findall(".//contrib-group/contrib"):
        surname = contrib.findtext(".//name/surname") or ""
        given = contrib.findtext(".//name/given-names") or ""
        display = " ".join(part for part in (given, surname) if part).strip()
        if display:
            authors.append(display)

    abstract_parts: list[str] = []
    for abstract_node in meta.findall("abstract"):
        if abstract_node.get("abstract-type") in ("graphical", "teaser"):
            continue
        for paragraph in abstract_node.iter("p"):
            text = _paragraph_text(paragraph)
            if text:
                abstract_parts.append(text)
    abstract = "\n".join(abstract_parts)

    keywords = [_clean(_text_of(node)) for node in meta.findall(".//kwd-group/kwd") if _clean(_text_of(node))]
    licence = ""
    licence_node = meta.find(".//permissions/license")
    if licence_node is not None:
        licence = licence_node.get("{http://www.w3.org/1999/xlink}href") or _clean(_text_of(licence_node))[:200]

    article_types = [article.get("article-type", "")] if article.get("article-type") else []
    for subject in meta.findall(".//article-categories//subject"):
        text = _clean(_text_of(subject))
        if text:
            article_types.append(text)

    sections = _parse_sections(article.find("body"))
    # Sections are delimited by a "## " heading so the text file round-trips back
    # into the same section structure when it is loaded for coding.
    body_text = "\n\n".join(f"## {section['title']}\n\n{section['text']}" for section in sections)

    return {
        "pmcid": pmcid if pmcid.startswith("PMC") else (f"PMC{pmcid}" if pmcid else ""),
        "pmid": pmid,
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year,
        "authors": " | ".join(authors),
        "keywords": " | ".join(keywords),
        "publication_types": " | ".join(dict.fromkeys(article_types)),
        "license": licence,
        "n_sections": len(sections),
        "section_titles": " | ".join(section["title"] for section in sections),
        "sections": sections,
        "body_text": body_text,
        "body_chars": len(body_text),
        "pmc_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else "",
    }


def resolve_pmc_ids(pmids: list[str]) -> dict[str, str]:
    """Look up PMC ids for PubMed records that do not carry one."""
    resolved: dict[str, str] = {}
    for start in range(0, len(pmids), 100):
        batch = [pmid for pmid in pmids[start : start + 100] if pmid]
        if not batch:
            continue
        response = eutils_request(
            "elink.fcgi",
            {"dbfrom": "pubmed", "db": "pmc", "id": ",".join(batch), "retmode": "json"},
        )
        payload = response.json()
        for linkset in payload.get("linksets", []):
            source = str(linkset.get("ids", [""])[0])
            for group in linkset.get("linksetdbs", []) or []:
                if group.get("linkname") == "pubmed_pmc":
                    links = group.get("links") or []
                    if links:
                        resolved[source] = f"PMC{links[0]}"
        time.sleep(NCBI_PAUSE_SECONDS)
    return resolved


def fetch_full_texts(pmc_ids: list[str], batch_size: int = FETCH_BATCH, verbose: bool = True) -> list[dict]:
    """Fetch and parse the JATS full text for each PMC id."""
    papers: list[dict] = []
    for start in range(0, len(pmc_ids), batch_size):
        batch = pmc_ids[start : start + batch_size]
        response = eutils_request("efetch.fcgi", {"db": "pmc", "id": ",".join(batch), "retmode": "xml"})
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            continue
        for article in root.iter("article"):
            parsed = _parse_article(article)
            if parsed is not None:
                papers.append(parsed)
        if verbose:
            print(f"  fetched {min(start + batch_size, len(pmc_ids))}/{len(pmc_ids)} open-access candidates", flush=True)
        time.sleep(NCBI_PAUSE_SECONDS)
    return papers


def build_corpus(candidates: pd.DataFrame | None = None, verbose: bool = True) -> pd.DataFrame:
    """Retrieve, screen, and persist the full-text corpus for the coded candidates."""
    if candidates is None:
        candidates = pd.read_csv(candidate_set_csv()).fillna("")
    candidates = candidates.copy()
    candidates["pmid"] = candidates["pmid"].astype(str).str.replace(r"\.0$", "", regex=True)
    candidates["pmcid"] = candidates.get("pmcid", "").astype(str).str.strip()

    missing = candidates.loc[candidates["pmcid"] == "", "pmid"].tolist()
    if verbose:
        print(f"{len(candidates)} abstract-level candidates; "
              f"{len(candidates) - len(missing)} already carry a PMC id", flush=True)
    if missing:
        resolved = resolve_pmc_ids(missing)
        candidates["pmcid"] = [
            row.pmcid or resolved.get(row.pmid, "") for row in candidates.itertuples()
        ]
        if verbose:
            print(f"  elink resolved {sum(1 for pmid in missing if pmid in resolved)} further PMC ids", flush=True)

    open_access = candidates[candidates["pmcid"] != ""].copy()
    papers = fetch_full_texts(open_access["pmcid"].tolist(), verbose=verbose) if len(open_access) else []
    by_pmcid = {paper["pmcid"]: paper for paper in papers}
    by_pmid = {paper["pmid"]: paper for paper in papers if paper["pmid"]}

    log_rows: list[dict] = []
    selected: list[dict] = []
    for row in candidates.itertuples():
        paper = by_pmcid.get(row.pmcid) or by_pmid.get(row.pmid)
        if not row.pmcid:
            status, reason = "dropped", "no PubMed Central record (not open access)"
        elif paper is None:
            status, reason = "dropped", "PMC record could not be retrieved or parsed"
        elif paper["body_chars"] < MIN_BODY_CHARS:
            status, reason = "dropped", f"retrieved body too short ({paper['body_chars']} chars)"
        else:
            status, reason = "retrieved", ""
            selected.append({"candidate": row, "paper": paper})
        log_rows.append(
            {
                "record_id": row.record_id,
                "pmid": row.pmid,
                "pmcid": row.pmcid,
                "title": getattr(row, "title", ""),
                "year": getattr(row, "year", ""),
                "stage3_priority": getattr(row, "stage3_priority", ""),
                "status": status,
                "reason": reason,
                "body_chars": paper["body_chars"] if paper else 0,
                "n_sections": paper["n_sections"] if paper else 0,
            }
        )

    ensure_parent(corpus_text_dir() / "placeholder")
    rows: list[dict] = []
    for index, entry in enumerate(selected):
        candidate, paper = entry["candidate"], entry["paper"]
        record_id = f"F{index + 1:03d}_{candidate.pmid}"
        text_path = corpus_text_dir() / f"{record_id}.txt"
        text_path.write_text(
            f"TITLE: {paper['title']}\n\nABSTRACT:\n{paper['abstract']}\n\nFULL TEXT:\n{paper['body_text']}\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "record_id": record_id,
                "abstract_record_id": candidate.record_id,
                "pmid": candidate.pmid,
                "pmcid": paper["pmcid"],
                "doi": paper["doi"] or getattr(candidate, "doi", ""),
                "title": paper["title"] or getattr(candidate, "title", ""),
                "journal": paper["journal"],
                "year": paper["year"] or getattr(candidate, "year", ""),
                "authors": paper["authors"],
                "keywords": paper["keywords"],
                "publication_types": paper["publication_types"],
                "license": paper["license"],
                "n_sections": paper["n_sections"],
                "section_titles": paper["section_titles"],
                "abstract_chars": len(paper["abstract"]),
                "body_chars": paper["body_chars"],
                "abstract_stage3_priority": getattr(candidate, "stage3_priority", ""),
                "abstract_typology": getattr(candidate, "provisional_typology", ""),
                "abstract_msk_flag": getattr(candidate, "musculoskeletal_flag", ""),
                "text_file": f"03_fulltext_txt/{record_id}.txt",
                "pmc_url": paper["pmc_url"],
                "abstract": paper["abstract"],
            }
        )

    frame = pd.DataFrame(rows)
    write_csv(corpus_csv(), frame)
    write_csv(corpus_candidates_csv(), candidates)
    write_csv(corpus_selection_log_csv(), pd.DataFrame(log_rows))

    manifest = {
        "built_at_utc": utc_timestamp(),
        "source": "PubMed Central open-access subset via NCBI E-utilities",
        "input": "abstract-level consensus candidate set",
        "n_candidates": int(len(candidates)),
        "n_with_pmcid": int((candidates["pmcid"] != "").sum()),
        "n_retrieved": int(len(frame)),
        "n_dropped": int(len(candidates) - len(frame)),
        "drop_reasons": pd.DataFrame(log_rows)["reason"].replace("", pd.NA).dropna().value_counts().to_dict(),
        "min_body_chars": MIN_BODY_CHARS,
        "median_body_chars": float(frame["body_chars"].median()) if not frame.empty else None,
        "max_body_chars": int(frame["body_chars"].max()) if not frame.empty else None,
        "outputs": {
            "corpus_csv": corpus_csv().name,
            "candidates_csv": corpus_candidates_csv().name,
            "retrieval_log_csv": corpus_selection_log_csv().name,
            "fulltext_dir": corpus_text_dir().name,
        },
    }
    write_json(corpus_manifest_json(), manifest)

    if verbose:
        print(f"  retrieved {len(frame)} full texts of {len(candidates)} candidates", flush=True)
        print(f"  corpus written to {corpus_csv().name}", flush=True)
    return frame


def load_corpus() -> pd.DataFrame:
    """Load the persisted corpus (metadata only, without the full text)."""
    return pd.read_csv(corpus_csv()).fillna("")


def load_corpus_records() -> list[dict]:
    """Load the corpus as coding records, each with its parsed sections."""
    frame = load_corpus()
    records: list[dict] = []
    for _, row in frame.iterrows():
        text_path = corpus_text_dir() / f"{row['record_id']}.txt"
        raw = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
        body = raw.split("FULL TEXT:\n", 1)[1] if "FULL TEXT:\n" in raw else raw
        sections: list[dict[str, str]] = []
        for block in re.split(r"\n##\s+", "\n" + body.strip()):
            if not block.strip():
                continue
            head, _, tail = block.partition("\n")
            sections.append({"title": head.strip() or "Body", "text": tail.strip()})
        records.append(
            {
                "record_id": row["record_id"],
                "abstract_record_id": row["abstract_record_id"],
                "pmcid": row["pmcid"],
                "pmid": row["pmid"],
                "doi": row["doi"],
                "title": row["title"],
                "journal": row["journal"],
                "year": row["year"],
                "abstract": row["abstract"],
                "publication_types": row["publication_types"],
                "sections": sections,
                "body_text": body,
            }
        )
    return records
