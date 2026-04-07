from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from bps_review.extraction.stage2 import (
    BIO_TERMS,
    ICD11_RULES,
    PSYCH_TERMS,
    SOCIAL_TERMS,
    _bps_function,
    _concepts,
    _contains_any,
    _musculoskeletal_flag,
    _objective_category,
    _quality_flag,
    _review_type,
)
from bps_review.llm.openrouter import chat_completion_json, resolve_default_model
from bps_review.utils.io import append_jsonl, write_csv
from bps_review.utils.paths import project_path


REVIEW_TYPE_OPTIONS = [
    "systematic review",
    "meta-analysis",
    "network meta-analysis",
    "umbrella review",
    "scoping or mapping review",
    "rapid review",
    "realist review",
    "integrative review",
    "narrative or expert review",
    "other evidence synthesis",
    "unclear",
]

OBJECTIVE_CATEGORY_OPTIONS = [
    "conceptual",
    "clinical",
    "methodological",
    "epidemiological",
    "mixed",
    "unclear",
]

ICD11_OPTIONS = [
    "chronic secondary musculoskeletal pain",
    "chronic neuropathic pain",
    "chronic cancer-related pain",
    "chronic postsurgical or posttraumatic pain",
    "chronic secondary headache or orofacial pain",
    "chronic secondary visceral pain",
    "chronic primary pain",
    "mixed or unspecified chronic pain",
    "unclear",
]

BPS_FUNCTION_OPTIONS = [
    "explanatory framework",
    "intervention rationale",
    "organizing principle",
    "justification",
    "background framing",
    "conclusion",
    "policy/practice implication",
    "rhetorical label",
    "unclear",
]

CONCEPTUAL_PROBLEM_OPTIONS = [
    "vague_definition",
    "tokenistic_bps",
    "missing_social",
    "missing_biology",
    "mechanistic_absence",
    "construct_overlap",
    "parallel_listing_without_integration",
    "none",
]

TYPOLOGY_OPTIONS = [
    "potential integrative signal",
    "multifactorial signal",
    "pseudo-bps or partial signal",
    "rhetorical label signal",
]

STAGE3_PRIORITY_OPTIONS = ["high", "medium", "low"]


class Stage2StructuredRecord(BaseModel):
    record_id: str
    review_type: Literal[
        "systematic review",
        "meta-analysis",
        "network meta-analysis",
        "umbrella review",
        "scoping or mapping review",
        "rapid review",
        "realist review",
        "integrative review",
        "narrative or expert review",
        "other evidence synthesis",
        "unclear",
    ]
    objective_category: Literal["conceptual", "clinical", "methodological", "epidemiological", "mixed", "unclear"]
    icd11_pain_category: Literal[
        "chronic secondary musculoskeletal pain",
        "chronic neuropathic pain",
        "chronic cancer-related pain",
        "chronic postsurgical or posttraumatic pain",
        "chronic secondary headache or orofacial pain",
        "chronic secondary visceral pain",
        "chronic primary pain",
        "mixed or unspecified chronic pain",
        "unclear",
    ]
    musculoskeletal_flag: Literal["yes", "no", "unclear"]
    bps_function: Literal[
        "explanatory framework",
        "intervention rationale",
        "organizing principle",
        "justification",
        "background framing",
        "conclusion",
        "policy/practice implication",
        "rhetorical label",
        "unclear",
    ]
    bio_mentioned: Literal["yes", "no"]
    psych_mentioned: Literal["yes", "no"]
    social_mentioned: Literal["yes", "no"]
    quality_assessment_reported: Literal["yes", "no", "unclear"] = "unclear"
    psychological_concepts_detected: list[str] = Field(default_factory=list)
    theoretical_frameworks_detected: list[str] = Field(default_factory=list)
    conceptual_problem_flags: list[
        Literal[
            "vague_definition",
            "tokenistic_bps",
            "missing_social",
            "missing_biology",
            "mechanistic_absence",
            "construct_overlap",
            "parallel_listing_without_integration",
            "none",
        ]
    ] = Field(default_factory=list)
    provisional_typology: Literal[
        "potential integrative signal",
        "multifactorial signal",
        "pseudo-bps or partial signal",
        "rhetorical label signal",
    ]
    stage3_candidate: Literal["yes", "no"]
    stage3_priority: Literal["high", "medium", "low"]
    coding_rationale: str


class Stage2StructuredBatch(BaseModel):
    records: list[Stage2StructuredRecord]


def _batch_prompt(batch: list[dict[str, str]]) -> str:
    instructions = {
        "review_type_options": REVIEW_TYPE_OPTIONS,
        "objective_category_options": OBJECTIVE_CATEGORY_OPTIONS,
        "icd11_options": ICD11_OPTIONS,
        "bps_function_options": BPS_FUNCTION_OPTIONS,
        "conceptual_problem_options": CONCEPTUAL_PROBLEM_OPTIONS,
        "provisional_typology_options": TYPOLOGY_OPTIONS,
        "stage3_priority_options": STAGE3_PRIORITY_OPTIONS,
        "coding_rules": [
            "Use title, abstract, publication types, and journal metadata only. Do not invent content absent from the record.",
            "Do not count the single lexical token 'biopsychosocial' as proof that biological, psychological, and social domains are all substantively covered. Mark a domain as mentioned only when the title or abstract contains domain-specific content, constructs, examples, mechanisms, determinants, or interventions from that domain.",
            "Use 'explanatory framework' only when the review explicitly treats BPS as a model/framework explaining pain or pain-related disability.",
            "Use 'intervention rationale' when BPS framing primarily justifies multimodal/interdisciplinary treatment or rehabilitation.",
            "Use 'organizing principle' when BPS is used to structure the review scope or categories without clearly specifying integration mechanisms.",
            "Use 'rhetorical label' when BPS is invoked ceremonially or aspirationally without substantive analytic work.",
            "Set stage3_candidate to 'yes' for musculoskeletal reviews and for mixed or unspecified chronic pain reviews where musculoskeletal relevance cannot be confidently ruled out.",
            "Use 'potential integrative signal' only when all three core domains are substantively present and the abstract signals cross-domain explanation or organization beyond simple listing.",
            "Use 'multifactorial signal' when all three domains are substantively present but treated mainly in parallel.",
            "Use 'pseudo-bps or partial signal' when one or more core domains are thin or absent despite BPS language.",
            "Use 'rhetorical label signal' when the BPS label is largely symbolic or concluding.",
            "Return concise normalized lists for psychological concepts and theoretical frameworks.",
            "Return conceptual_problem_flags=['none'] only when no clear conceptual problem is inferable from the abstract.",
        ],
        "records": batch,
    }
    return (
        "You are coding abstract-level Stage 2 data for an OSF-registered systematic review on how the "
        "biopsychosocial model is operationalized in chronic pain review literature. "
        "Return one JSON object with key 'records'. Each item must contain exactly the fields specified in the instructions. "
        f"{json.dumps(instructions, ensure_ascii=False)}"
    )


def _normalize_list(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).strip().lower().split())
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _normalize_choice(value: object, allowed: list[str], fallback: str) -> str:
    text = " ".join(str(value).strip().lower().replace("_", " ").split())
    allowed_map = {item.lower(): item for item in allowed}
    allowed_map.update({item.lower().replace("_", " "): item for item in allowed})
    if text in allowed_map:
        return allowed_map[text]
    alias_map = {
        "scoping review": "scoping or mapping review",
        "mapping review": "scoping or mapping review",
        "narrative review": "narrative or expert review",
        "expert review": "narrative or expert review",
        "systematic literature review": "systematic review",
        "mixed or unclear chronic pain": "mixed or unspecified chronic pain",
        "chronic postsurgical pain": "chronic postsurgical or posttraumatic pain",
        "chronic post-traumatic pain": "chronic postsurgical or posttraumatic pain",
        "background": "background framing",
        "organising principle": "organizing principle",
        "policy implication": "policy/practice implication",
        "practice implication": "policy/practice implication",
        "pseudo-bps or partial signal": "pseudo-bps or partial signal",
        "pseudo bps or partial signal": "pseudo-bps or partial signal",
    }
    if text in alias_map and alias_map[text] in allowed_map.values():
        return alias_map[text]
    return fallback


def _fallback_record_fields(record: dict[str, object]) -> dict[str, object]:
    text = f"{record.get('title', '')}\n{record.get('abstract', '')}".lower()
    icd11_label = "unclear"
    for label, terms in ICD11_RULES:
        if any(term in text for term in terms):
            icd11_label = label
            break
    if icd11_label == "unclear" and ("chronic pain" in text or "persistent pain" in text):
        icd11_label = "mixed or unspecified chronic pain"
    musculoskeletal_flag = _musculoskeletal_flag(icd11_label)
    rule_based_concepts = [token.strip() for token in _concepts(text).split("|") if token.strip()]
    return {
        "record_id": record["record_id"],
        "review_type": _review_type(text, str(record.get("publication_types", ""))),
        "objective_category": _objective_category(text),
        "icd11_pain_category": icd11_label,
        "musculoskeletal_flag": musculoskeletal_flag,
        "bps_function": _bps_function(text),
        "bio_mentioned": "yes" if _contains_any(text, BIO_TERMS) else "no",
        "psych_mentioned": "yes" if _contains_any(text, PSYCH_TERMS) else "no",
        "social_mentioned": "yes" if _contains_any(text, SOCIAL_TERMS) else "no",
        "quality_assessment_reported": _quality_flag(text),
        "psychological_concepts_detected": rule_based_concepts,
        "theoretical_frameworks_detected": [],
        "conceptual_problem_flags": ["none"],
        "provisional_typology": "multifactorial signal"
        if _contains_any(text, BIO_TERMS) and _contains_any(text, PSYCH_TERMS) and _contains_any(text, SOCIAL_TERMS)
        else "pseudo-bps or partial signal",
        "stage3_candidate": "yes" if musculoskeletal_flag in {"yes", "unclear"} else "no",
        "stage3_priority": "high" if musculoskeletal_flag == "yes" else "medium" if musculoskeletal_flag == "unclear" else "low",
        "coding_rationale": "Structured Stage 2 output required deterministic field repair for omitted items.",
    }


def _derive_conceptual_flags(merged: dict[str, object]) -> list[str]:
    flags = _normalize_list(list(merged.get("conceptual_problem_flags", []) or []))
    flags = [flag for flag in flags if flag != "none"]
    typology = str(merged.get("provisional_typology", "")).strip().lower()
    bps_function = str(merged.get("bps_function", "")).strip().lower()
    bio = str(merged.get("bio_mentioned", "")).strip().lower() == "yes"
    psych = str(merged.get("psych_mentioned", "")).strip().lower() == "yes"
    social = str(merged.get("social_mentioned", "")).strip().lower() == "yes"

    if "rhetorical" in typology or bps_function == "rhetorical label":
        flags.extend(["tokenistic_bps", "vague_definition"])
    if typology in {"multifactorial signal", "pseudo-bps or partial signal"} and bps_function in {
        "background framing",
        "intervention rationale",
        "organizing principle",
        "unclear",
    }:
        flags.append("parallel_listing_without_integration")
        flags.append("mechanistic_absence")
    if not bio:
        flags.append("missing_biology")
    if not social:
        flags.append("missing_social")
    return _normalize_list(flags) or ["none"]


def _repair_response_payload(batch: list[dict[str, str]], response_payload: object) -> dict[str, object]:
    if isinstance(response_payload, list):
        raw_records = response_payload
    elif isinstance(response_payload, dict):
        raw_records = response_payload.get("records", [])
    else:
        raw_records = []

    raw_map = {
        str(item.get("record_id", "")).strip(): item
        for item in raw_records
        if isinstance(item, dict) and str(item.get("record_id", "")).strip()
    }

    repaired_records: list[dict[str, object]] = []
    for record in batch:
        fallback = _fallback_record_fields(record)
        raw = dict(raw_map.get(record["record_id"], {}))
        if "icd11" in raw and "icd11_pain_category" not in raw:
            raw["icd11_pain_category"] = raw["icd11"]
        if "psychological_concepts" in raw and "psychological_concepts_detected" not in raw:
            raw["psychological_concepts_detected"] = raw["psychological_concepts"]
        if "theoretical_frameworks" in raw and "theoretical_frameworks_detected" not in raw:
            raw["theoretical_frameworks_detected"] = raw["theoretical_frameworks"]
        merged = {**fallback, **raw}
        merged["review_type"] = _normalize_choice(merged.get("review_type"), REVIEW_TYPE_OPTIONS, fallback["review_type"])
        merged["objective_category"] = _normalize_choice(merged.get("objective_category"), OBJECTIVE_CATEGORY_OPTIONS, fallback["objective_category"])
        merged["icd11_pain_category"] = _normalize_choice(merged.get("icd11_pain_category"), ICD11_OPTIONS, fallback["icd11_pain_category"])
        merged["bps_function"] = _normalize_choice(merged.get("bps_function"), BPS_FUNCTION_OPTIONS, fallback["bps_function"])
        merged["provisional_typology"] = _normalize_choice(merged.get("provisional_typology"), TYPOLOGY_OPTIONS, fallback["provisional_typology"])
        merged["stage3_priority"] = _normalize_choice(merged.get("stage3_priority"), STAGE3_PRIORITY_OPTIONS, fallback["stage3_priority"])
        merged["musculoskeletal_flag"] = _normalize_choice(merged.get("musculoskeletal_flag"), ["yes", "no", "unclear"], fallback["musculoskeletal_flag"])
        merged["bio_mentioned"] = _normalize_choice(merged.get("bio_mentioned"), ["yes", "no"], fallback["bio_mentioned"])
        merged["psych_mentioned"] = _normalize_choice(merged.get("psych_mentioned"), ["yes", "no"], fallback["psych_mentioned"])
        merged["social_mentioned"] = _normalize_choice(merged.get("social_mentioned"), ["yes", "no"], fallback["social_mentioned"])
        merged["quality_assessment_reported"] = _normalize_choice(
            merged.get("quality_assessment_reported"), ["yes", "no", "unclear"], fallback["quality_assessment_reported"]
        )
        normalized_stage3_candidate = _normalize_choice(merged.get("stage3_candidate"), ["yes", "no"], fallback["stage3_candidate"])
        if merged["musculoskeletal_flag"] in {"yes", "unclear"}:
            merged["stage3_candidate"] = "yes"
            merged["stage3_priority"] = "high" if merged["musculoskeletal_flag"] == "yes" else "medium"
        else:
            merged["stage3_candidate"] = normalized_stage3_candidate
            merged["stage3_priority"] = "low"
        merged["psychological_concepts_detected"] = _normalize_list(
            list(fallback.get("psychological_concepts_detected", []) or [])
            + list(merged.get("psychological_concepts_detected", []) or [])
        )
        merged["theoretical_frameworks_detected"] = _normalize_list(list(merged.get("theoretical_frameworks_detected", []) or []))
        merged["conceptual_problem_flags"] = [
            _normalize_choice(flag, CONCEPTUAL_PROBLEM_OPTIONS, "none")
            for flag in list(merged.get("conceptual_problem_flags", []) or ["none"])
        ]
        merged["conceptual_problem_flags"] = _derive_conceptual_flags(merged)
        merged["coding_rationale"] = " ".join(str(merged.get("coding_rationale", fallback["coding_rationale"])).split())
        repaired_records.append(merged)
    return {"records": repaired_records}


def _parse_batch(
    batch_index: int,
    batch: list[dict[str, str]],
    model: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    prompt = _batch_prompt(batch)
    response_payload = chat_completion_json(
        prompt,
        model=model,
        temperature=0.0,
        system_prompt=(
            "You are a high-rigor systematic review coding assistant. "
            "Return strict JSON only. Be conservative and evidence-bound."
        ),
    )
    repaired_payload = _repair_response_payload(batch, response_payload)
    parsed = Stage2StructuredBatch.model_validate(repaired_payload)

    expected_ids = [item["record_id"] for item in batch]
    received_ids = [item.record_id for item in parsed.records]
    if sorted(expected_ids) != sorted(received_ids):
        raise ValueError(f"Batch {batch_index} record_id mismatch: expected {expected_ids}, received {received_ids}")

    rows: list[dict[str, object]] = []
    for item in parsed.records:
        rows.append(
            {
                "record_id": item.record_id,
                "review_type": item.review_type,
                "objective_category": item.objective_category,
                "objective_category_source": "llm_structured",
                "icd11_pain_category": item.icd11_pain_category,
                "musculoskeletal_flag": item.musculoskeletal_flag,
                "bps_function": item.bps_function,
                "bio_mentioned": item.bio_mentioned,
                "psych_mentioned": item.psych_mentioned,
                "social_mentioned": item.social_mentioned,
                "quality_assessment_reported": item.quality_assessment_reported,
                "psychological_concepts_detected": " | ".join(_normalize_list(item.psychological_concepts_detected)),
                "theoretical_frameworks_detected": " | ".join(_normalize_list(item.theoretical_frameworks_detected)),
                "conceptual_problem_flags": " | ".join(_normalize_list(item.conceptual_problem_flags or ["none"])),
                "provisional_typology": item.provisional_typology,
                "stage3_candidate": item.stage3_candidate,
                "stage3_priority": item.stage3_priority,
                "coding_rationale": " ".join(item.coding_rationale.strip().split()),
                "coding_method": "llm_structured",
                "llm_model": model,
            }
        )

    audit = {
        "batch_index": batch_index,
        "model": model,
        "record_ids": expected_ids,
        "response_payload": response_payload,
        "repaired_payload": repaired_payload,
    }
    return rows, audit


def _batch_fallback_rows(batch: list[dict[str, str]], detail: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in batch:
        fallback = _fallback_record_fields(record)
        rows.append(
            {
                "record_id": fallback["record_id"],
                "review_type": fallback["review_type"],
                "objective_category": fallback["objective_category"],
                "objective_category_source": "llm_batch_fallback",
                "icd11_pain_category": fallback["icd11_pain_category"],
                "musculoskeletal_flag": fallback["musculoskeletal_flag"],
                "bps_function": fallback["bps_function"],
                "bio_mentioned": fallback["bio_mentioned"],
                "psych_mentioned": fallback["psych_mentioned"],
                "social_mentioned": fallback["social_mentioned"],
                "quality_assessment_reported": fallback["quality_assessment_reported"],
                "psychological_concepts_detected": " | ".join(_normalize_list(fallback["psychological_concepts_detected"])),
                "theoretical_frameworks_detected": "",
                "conceptual_problem_flags": "none",
                "provisional_typology": fallback["provisional_typology"],
                "stage3_candidate": fallback["stage3_candidate"],
                "stage3_priority": fallback["stage3_priority"],
                "coding_rationale": f"{fallback['coding_rationale']} Batch detail: {detail}",
                "coding_method": "llm_batch_fallback",
                "llm_model": "",
            }
        )
    return rows


def assist_stage2_objectives(
    frame: pd.DataFrame | None = None,
    batch_size: int = 6,
    max_workers: int = 4,
    model: str | None = None,
) -> pd.DataFrame:
    if frame is None:
        stage2_path = project_path("review_stages", "04_extraction", "outputs", "stage2_abstract_coding.csv")
        frame = pd.read_csv(stage2_path).fillna("")
    if frame.empty:
        return frame

    chosen_model = model or resolve_default_model()
    source = frame.copy().reset_index(drop=True)
    source["__order"] = source.index
    payload_records = source[
        ["__order", "record_id", "title", "abstract", "publication_types", "journal", "year"]
    ].to_dict(orient="records")

    batches = [payload_records[start : start + batch_size] for start in range(0, len(payload_records), batch_size)]
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_parse_batch, batch_index, batch, chosen_model): (batch_index, batch)
            for batch_index, batch in enumerate(batches)
        }
        for future in as_completed(future_map):
            batch_index, _ = future_map[future]
            try:
                batch_rows, audit = future.result()
                rows.extend(batch_rows)
                audits.append(audit)
            except Exception as exc:
                batch = future_map[future][1]
                rows.extend(_batch_fallback_rows(batch, str(exc)))
                audits.append(
                    {
                        "batch_index": batch_index,
                        "model": chosen_model,
                        "record_ids": [item["record_id"] for item in batch],
                        "status": "batch_fallback",
                        "detail": str(exc),
                    }
                )

    out = pd.DataFrame(rows)
    out = source[["__order", "record_id"]].merge(out, on="record_id", how="left").sort_values("__order").drop(columns="__order")

    llm_output_path = project_path("review_stages", "04_extraction", "outputs", "stage2_llm_structured_coding.csv")
    write_csv(llm_output_path, out)
    write_csv(
        project_path("review_stages", "04_extraction", "outputs", "stage2_objective_llm_assist.csv"),
        out[["record_id", "objective_category", "conceptual_problem_flags"]].rename(
            columns={"objective_category": "objective_category_llm"}
        ),
    )
    audit_path = project_path("review_stages", "04_extraction", "outputs", "llm_stage2_structured_batches.jsonl")
    for audit in sorted(audits, key=lambda item: item["batch_index"]):
        append_jsonl(audit_path, audit)
    return out
