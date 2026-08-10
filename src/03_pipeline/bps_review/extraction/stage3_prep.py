from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests import RequestException

from bps_review.fulltext.coding.codebook import write_stage3_codebook
from bps_review.search.pubmed import _request as pubmed_request
from bps_review.utils.io import append_jsonl, write_csv, write_text
from bps_review.utils.paths import project_path


PMC_CACHE = project_path("review_stages", "04_extraction", "inputs", "fulltext_cache", "pmc")
MIN_FULLTEXT_WORDS = 250


def _pmc_link_for_pmid(pmid: str) -> str:
    response = pubmed_request(
        "elink.fcgi",
        {
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": pmid,
            "retmode": "json",
        },
    )
    payload = response.json()
    for linkset in payload.get("linksets", []):
        for linksetdb in linkset.get("linksetdbs", []):
            if linksetdb.get("linkname") == "pubmed_pmc" and linksetdb.get("links"):
                return f'PMC{linksetdb["links"][0]}'
    return ""


def _pmc_link_for_identifier_via_europepmc(pmid: str, doi: str) -> str:
    query_parts = []
    if pmid:
        query_parts.append(f"EXT_ID:{pmid} AND SRC:MED")
    if doi:
        query_parts.append(f'DOI:"{doi}"')
    if not query_parts:
        return ""

    response = requests.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={
            "query": " OR ".join(query_parts),
            "format": "json",
            "pageSize": 5,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    for result in payload.get("resultList", {}).get("result", []):
        pmcid = str(result.get("pmcid", "")).strip()
        if pmcid.startswith("PMC"):
            return pmcid
    return ""


def _join_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return " ".join(part.strip() for part in node.itertext() if part and part.strip())


def _fetch_pmc_fulltext(pmcid: str) -> tuple[Path, Path]:
    pmc_numeric = pmcid.replace("PMC", "")
    response = pubmed_request("efetch.fcgi", {"db": "pmc", "id": pmc_numeric, "retmode": "xml"})
    xml_text = response.text
    xml_path = PMC_CACHE / f"{pmcid}.xml"
    txt_path = PMC_CACHE / f"{pmcid}.txt"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(xml_text, encoding="utf-8")

    root = ET.fromstring(xml_text)
    body = root.find(".//body")
    article_title = _join_text(root.find(".//article-title"))
    text_parts = [article_title] if article_title else []
    for sec in body.findall(".//sec") if body is not None else []:
        title = _join_text(sec.find("title"))
        section_text = _join_text(sec)
        if title and section_text:
            text_parts.append(f"{title}\n{section_text}")
        elif section_text:
            text_parts.append(section_text)
    if not text_parts:
        text_parts.append(_join_text(body))
    txt_path.write_text("\n\n".join(part for part in text_parts if part), encoding="utf-8")
    return xml_path, txt_path


def _cache_paths_for_pmcid(pmcid: str) -> tuple[Path, Path]:
    return PMC_CACHE / f"{pmcid}.xml", PMC_CACHE / f"{pmcid}.txt"


def _cached_fulltext_available(pmcid: str) -> tuple[bool, Path, Path]:
    xml_path, txt_path = _cache_paths_for_pmcid(pmcid)
    exists = (
        xml_path.exists()
        and txt_path.exists()
        and txt_path.stat().st_size > 0
    )
    return exists, xml_path, txt_path


def _word_count(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    return len([token for token in text.split() if token.strip()])


def _relevance_signal(title: str, abstract: str, review_type: str) -> tuple[str, str]:
    blob = f"{title}\n{abstract}".lower()
    flags: list[str] = []
    if "withdrawn" in blob or "retracted" in blob:
        flags.append("withdrawn_or_retracted_signal")
    if "pain" not in blob:
        flags.append("pain_focus_not_explicit")
    if not any(term in blob for term in ("chronic", "persistent", "long-term")):
        flags.append("chronicity_not_explicit")
    if str(review_type).strip().lower() in {"", "other evidence synthesis"}:
        flags.append("review_design_unclear")

    if any(flag in flags for flag in {"withdrawn_or_retracted_signal", "pain_focus_not_explicit"}):
        return "high", " | ".join(flags)
    if flags:
        return "medium", " | ".join(flags)
    return "low", ""


# --------------------------------------------------------------------------
# The human Stage 3 coding surface.
#
# It mirrors the machine schema in
# ``bps_review.fulltext.coding.schema`` field for field, flattened for a
# spreadsheet: the single-value decisions get one column each, and every
# structured extraction list gets one column in which the coder writes one item
# per line, with the item's fields separated by a pipe. That keeps a human
# coding and a model coding of the same article directly comparable, which is
# what the registration's reliability checks and the AI-assist provision both
# require.
# --------------------------------------------------------------------------
CARRIED_FROM_MANIFEST = [
    "record_id",
    "title",
    "year",
    "review_type",
    "icd11_pain_category",
    "provisional_typology",
    "conceptual_problem_flags",
    "psychological_concepts_detected",
    "theoretical_frameworks_detected",
    "stage3_priority",
    "fulltext_status",
    "cached_text_path",
]

# (column, guidance shown in the codebook; empty for a plain controlled field)
FULLTEXT_TEMPLATE_COLUMNS: list[tuple[str, str]] = [
    ("full_text_available", "yes; no; partial"),
    # A. source, context, routing
    ("review_track", "musculoskeletal; neuropathic; mixed_or_other; unclear"),
    ("source_type_fulltext", "confirmed evidence-synthesis design"),
    ("icd11_pain_category_fulltext", "ICD-11 category confirmed at full text"),
    ("population", "adult; older adult; mixed ages; pediatric; unclear; not applicable"),
    ("care_setting", "primary care; specialist; rehabilitation; occupational; community; mixed; not reported"),
    ("primary_discipline", "disciplinary home of the paper"),
    ("pain_condition_detail", "free text"),
    ("pain_conditions", "one condition per line"),
    ("context_note", "cultural or healthcare context, free text"),
    ("quality_assessment_reported", "yes; no; unclear"),
    ("quality_assessment_tools", "one tool per line"),
    # B. what the BPS label does
    ("bps_label_used", "explicit_bps_term; variant_term_only; domain_language_only; absent"),
    ("bps_primary_function", "dominant function of the BPS label"),
    ("bps_functions_present", "one function per line"),
    ("bps_definition_status", "formally_defined; described_informally; cited_only; undefined"),
    ("bps_model_variants", "one model label per line, verbatim"),
    ("bps_usage_instances", "one per line: function | section | quote"),
    ("bps_definitions", "one per line: type | attributed source | quote"),
    ("bps_operationalization_summary", "free text, at most 90 words"),
    # C. coverage
    ("domain_coverage_bio", "elaborated; mentioned; minimal; absent"),
    ("domain_coverage_psych", "elaborated; mentioned; minimal; absent"),
    ("domain_coverage_social", "elaborated; mentioned; minimal; absent"),
    ("coverage_lifestyle", "elaborated; mentioned; minimal; absent"),
    ("coverage_spiritual_existential", "elaborated; mentioned; minimal; absent"),
    ("domain_evidence", "one per line: domain | level | subdomains | quote"),
    # D. the factor inventory
    ("biological_factors", "one per line: label | subdomain | level | role | quote"),
    ("social_factors", "one per line: label | subdomain | level | role | quote"),
    ("other_domain_factors", "one per line: label | lifestyle or spiritual or environmental | role | quote"),
    # E. integration
    ("integration_bio_psych", "mechanistic; directional; descriptive; mentioned; none"),
    ("integration_psych_social", "mechanistic; directional; descriptive; mentioned; none"),
    ("integration_bio_social", "mechanistic; directional; descriptive; mentioned; none"),
    ("integration_triadic", "mechanistic; descriptive; partial; none"),
    ("integration_claims", "one per line: pair | level | source factor -> target factor | quote"),
    ("integration_mechanism_summary", "free text, at most 90 words"),
    # F. typology and balance
    ("overall_balance", "balanced; psych-dominant; bio-dominant; social-dominant; dyadic; unclear"),
    ("bps_typology", "true_integrative; multifactorial; pseudo_bps; rhetorical_bps; narrow_despite_label; unclear"),
    # G. concepts, relations, frameworks, measures
    ("concept_definitions_present", "yes; partial; no"),
    ("psychological_concepts", "one per line: label | family | definitional status | measure | quote"),
    ("concept_relations", "one per line: source concept | relation | target concept | quote"),
    ("theoretical_frameworks", "one per line: label | role | domains covered | quote"),
    ("instruments", "one per line: label | domain measured | role | what it is said to measure"),
    # H. conceptual problems
    ("conceptual_problems", "one per line: type | scope | named by authors | quote"),
    # I. synthesis hooks
    ("key_quotes", "one per line: claim type | section | quote"),
    ("emergent_labels", "one term per line, verbatim, for terms the ontology does not carry"),
    ("conceptual_tensions", "one tension per line"),
    ("additional_observations", "one observation per line"),
    ("synthesis_note", "free text, at most 90 words"),
    ("coding_rationale", "free text, at most 40 words"),
    # provenance and adjudication
    ("coder_id", "initials of the coder"),
    ("coder_notes", "free text"),
    ("adjudication_status", "pending; agreed; adjudicated"),
]


def build_fulltext_coding_template(manifest_frame: pd.DataFrame) -> pd.DataFrame:
    """The Stage 3 human coding form, one empty row per candidate review."""
    carried = [column for column in CARRIED_FROM_MANIFEST if column in manifest_frame.columns]
    template = manifest_frame[carried].copy()
    for column, _ in FULLTEXT_TEMPLATE_COLUMNS:
        template[column] = ""
    return template


def prepare_stage3_candidates(fetch_fulltext: bool = True) -> dict[str, int]:
    stage2_path = project_path("review_stages", "04_extraction", "outputs", "stage2_abstract_coding.csv")
    frame = pd.read_csv(stage2_path).fillna("")

    candidates = frame.loc[frame["stage3_candidate"] == "yes"].copy()
    if candidates.empty:
        summary = {"stage3_candidates": 0, "pmc_open_fulltexts": 0, "manual_retrieval_required": 0}
        write_csv(project_path("review_stages", "04_extraction", "outputs", "stage3_candidate_summary.csv"), pd.DataFrame([summary]))
        return summary

    manifests = []
    pmc_open = 0
    checked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for _, row in candidates.iterrows():
        pmcid = str(row.get("pmcid", "")).strip()
        pmid = str(row.get("pmid", "")).strip()
        doi = str(row.get("doi", "")).strip()
        retrieval_source = "stage2_existing_pmcid" if pmcid else "none"
        if not pmcid and pmid:
            linked = _pmc_link_for_pmid(pmid)
            if linked:
                pmcid = linked
                retrieval_source = "pubmed_elink"
            time.sleep(0.34)
        if not pmcid:
            try:
                linked = _pmc_link_for_identifier_via_europepmc(pmid, doi)
                if linked:
                    pmcid = linked
                    retrieval_source = "europepmc"
            except RequestException:
                pmcid = ""
                retrieval_source = "none"
            time.sleep(0.2)

        status = "manual_retrieval_required"
        xml_path = ""
        txt_path = ""
        txt_word_count = 0
        if pmcid:
            status = "pmc_open_available_not_cached"
            if fetch_fulltext:
                cached, cached_xml, cached_txt = _cached_fulltext_available(pmcid)
                if cached:
                    xml_file, txt_file = cached_xml, cached_txt
                    status = "pmc_fulltext_cached"
                else:
                    try:
                        xml_file, txt_file = _fetch_pmc_fulltext(pmcid)
                        status = "pmc_fulltext_fetched"
                    except (requests.HTTPError, RequestException, ET.ParseError):
                        status = "pmc_linked_fetch_failed"
                        xml_file = None
                        txt_file = None

                if status in {"pmc_fulltext_cached", "pmc_fulltext_fetched"} and xml_file is not None and txt_file is not None:
                    txt_word_count = _word_count(txt_file)
                    xml_path = str(xml_file.relative_to(project_path()))
                    txt_path = str(txt_file.relative_to(project_path()))
                    if txt_word_count < MIN_FULLTEXT_WORDS:
                        status = "pmc_fulltext_low_content_manual_check"
                    else:
                        pmc_open += 1
            else:
                pmc_open += 1

        manual_retrieval_needed = "yes" if status in {
            "manual_retrieval_required",
            "pmc_linked_fetch_failed",
            "pmc_fulltext_low_content_manual_check",
        } else "no"
        manual_priority, manual_flags = _relevance_signal(
            str(row.get("title", "")),
            str(row.get("abstract", "")),
            str(row.get("review_type", "")),
        )
        if manual_retrieval_needed == "yes" and manual_priority == "low":
            manual_priority = "medium"

        manifests.append(
            {
                "record_id": row["record_id"],
                "pmid": pmid,
                "pmcid": pmcid,
                "doi": row.get("doi", ""),
                "title": row["title"],
                "journal": row["journal"],
                "year": row["year"],
                "icd11_pain_category": row["icd11_pain_category"],
                "musculoskeletal_flag": row["musculoskeletal_flag"],
                "review_type": row["review_type"],
                "provisional_typology": row.get("provisional_typology", ""),
                "conceptual_problem_flags": row.get("conceptual_problem_flags", ""),
                "psychological_concepts_detected": row.get("psychological_concepts_detected", ""),
                "theoretical_frameworks_detected": row.get("theoretical_frameworks_detected", ""),
                "stage3_priority": row.get("stage3_priority", "medium"),
                "coding_rationale": row.get("coding_rationale", ""),
                "fulltext_status": status,
                "retrieval_source": retrieval_source,
                "fulltext_word_count": txt_word_count,
                "fulltext_checked_utc": checked_at,
                "manual_retrieval_needed": manual_retrieval_needed,
                "manual_relevance_priority": manual_priority,
                "manual_relevance_flags": manual_flags,
                "osf_manual_adjudication_required": "yes",
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                "cached_xml_path": xml_path,
                "cached_text_path": txt_path,
            }
        )

    manifest_frame = pd.DataFrame(manifests).sort_values(
        ["stage3_priority", "musculoskeletal_flag", "year", "title"],
        ascending=[True, False, False, True],
    )
    write_csv(project_path("review_stages", "04_extraction", "outputs", "stage3_candidate_manifest.csv"), manifest_frame)
    write_csv(
        project_path("review_stages", "04_extraction", "outputs", "stage3_manual_fulltext_queue.csv"),
        manifest_frame.loc[manifest_frame["manual_retrieval_needed"] == "yes"].copy(),
    )

    retrieval_validation = (
        manifest_frame.groupby("fulltext_status", dropna=False)
        .agg(
            n=("record_id", "count"),
            mean_word_count=("fulltext_word_count", "mean"),
            manual_retrieval_n=("manual_retrieval_needed", lambda s: int((s == "yes").sum())),
        )
        .reset_index()
    )
    retrieval_validation["mean_word_count"] = retrieval_validation["mean_word_count"].fillna(0).round(1)
    write_csv(project_path("review_stages", "04_extraction", "outputs", "stage3_retrieval_validation.csv"), retrieval_validation)

    pilot_n = min(5, len(manifest_frame))
    reliability_n = min(20, max(1, int(len(manifest_frame) * 0.20))) if len(manifest_frame) else 0
    pilot = manifest_frame.sample(n=pilot_n, random_state=13) if pilot_n else manifest_frame.head(0)
    reliability = manifest_frame.sample(n=reliability_n, random_state=23) if reliability_n else manifest_frame.head(0)
    reliability = reliability.copy()
    reliability["reviewer_a_domain_coverage_bio"] = ""
    reliability["reviewer_a_domain_coverage_psych"] = ""
    reliability["reviewer_a_domain_coverage_social"] = ""
    reliability["reviewer_b_domain_coverage_bio"] = ""
    reliability["reviewer_b_domain_coverage_psych"] = ""
    reliability["reviewer_b_domain_coverage_social"] = ""
    reliability["reviewer_a_integration_triadic"] = ""
    reliability["reviewer_b_integration_triadic"] = ""
    reliability["reviewer_a_bps_typology"] = ""
    reliability["reviewer_b_bps_typology"] = ""
    reliability["reviewer_a_notes"] = ""
    reliability["reviewer_b_notes"] = ""
    reliability["adjudicated_integration_triadic"] = ""
    reliability["adjudicated_bps_typology"] = ""
    reliability["adjudication_notes"] = ""
    write_csv(project_path("review_stages", "04_extraction", "forms", "stage3_pilot_sample.csv"), pilot)
    write_csv(project_path("review_stages", "04_extraction", "forms", "stage3_reliability_sample.csv"), reliability)

    manual_relevance_form = manifest_frame[
        [
            "record_id",
            "title",
            "year",
            "review_type",
            "icd11_pain_category",
            "fulltext_status",
            "manual_retrieval_needed",
            "manual_relevance_priority",
            "manual_relevance_flags",
            "osf_manual_adjudication_required",
            "cached_text_path",
            "pubmed_url",
        ]
    ].copy()
    manual_relevance_form["reviewer_decision"] = ""
    manual_relevance_form["reviewer_notes"] = ""
    manual_relevance_form["adjudication_decision"] = ""
    manual_relevance_form["adjudication_notes"] = ""
    write_csv(project_path("review_stages", "04_extraction", "forms", "stage3_manual_relevance_checklist.csv"), manual_relevance_form)

    coding_template = build_fulltext_coding_template(manifest_frame)
    write_csv(project_path("review_stages", "04_extraction", "forms", "stage3_fulltext_coding_template.csv"), coding_template)

    # The operational codebook is rendered from the schema the pipeline runs, so
    # the form, the coder's instructions, and the codebook cannot drift apart.
    write_stage3_codebook()

    summary = {
        "stage3_candidates": len(manifest_frame),
        "pmc_open_fulltexts": pmc_open,
        "manual_retrieval_required": int((manifest_frame["manual_retrieval_needed"] == "yes").sum()),
    }
    write_csv(project_path("review_stages", "04_extraction", "outputs", "stage3_candidate_summary.csv"), pd.DataFrame([summary]))
    append_jsonl(project_path("review_stages", "04_extraction", "outputs", "stage3_prep_log.jsonl"), summary)
    return summary
