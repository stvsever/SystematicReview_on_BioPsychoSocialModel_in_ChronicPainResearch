from __future__ import annotations

import re

import pandas as pd

from bps_review.utils.io import append_jsonl
from bps_review.utils.paths import project_path


BIO_TERMS = (
    "nociception",
    "inflammation",
    "biological",
    "biomedical",
    "neuro",
    "brain",
    "genetic",
    "physiological",
    "mechanism",
    "biomarker",
    "musculoskeletal",
    "pharmacological",
    "central sensitization",
)
PSYCH_TERMS = (
    "psychological",
    "catastroph",
    "fear",
    "avoidance",
    "depress",
    "anxiety",
    "coping",
    "self-efficacy",
    "kinesiophobia",
    "acceptance",
    "illness perception",
    "resilience",
    "stress",
    "cognitive",
    "emotion",
)
SOCIAL_TERMS = (
    "social",
    "support",
    "family",
    "work",
    "employment",
    "socioeconomic",
    "culture",
    "stigma",
    "interpersonal",
    "community",
    "occupational",
    "healthcare access",
)

PSYCHOLOGICAL_CONCEPTS = {
    "catastrophizing": ("catastroph",),
    "fear-avoidance": ("fear-avoidance", "fear avoidance", "kinesiophobia"),
    "coping": ("coping",),
    "depression": ("depress",),
    "anxiety": ("anxiety",),
    "acceptance": ("acceptance",),
    "illness perception": ("illness perception",),
    "self-efficacy": ("self-efficacy", "self efficacy"),
    "stress": ("stress",),
    "resilience": ("resilience",),
}

ICD11_RULES = [
    (
        "chronic secondary musculoskeletal pain",
        ("musculoskeletal", "low back pain", "neck pain", "osteoarthritis", "spinal", "back pain", "fibromyalgia"),
    ),
    ("chronic neuropathic pain", ("neuropathic", "neuropathy", "complex regional pain", "crps")),
    ("chronic cancer-related pain", ("cancer pain", "oncology")),
    ("chronic postsurgical or posttraumatic pain", ("postsurgical", "post-surgical", "postoperative", "post-operative", "posttraumatic", "post-traumatic")),
    ("chronic secondary headache or orofacial pain", ("headache", "migraine", "orofacial")),
    ("chronic secondary visceral pain", ("pelvic pain", "visceral", "abdominal pain", "endometriosis")),
    ("chronic primary pain", ("chronic primary pain",)),
]


def _blob(row: pd.Series) -> str:
    return f'{row.get("title", "")}\n{row.get("abstract", "")}'.lower()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _review_type(text: str, publication_types: str = "") -> str:
    combined = f"{text} {publication_types}".lower()
    if re.search(r"network\s+meta[- ]analysis|bayesian\s+meta[- ]analysis", combined):
        return "network meta-analysis"
    if re.search(r"meta[- ]analysis|meta[- ]analytic|meta\s+synthesis|quantitative\s+synthesis", combined):
        return "meta-analysis"
    if re.search(r"umbrella\s+review|overview\s+of\s+reviews", combined):
        return "umbrella review"
    if re.search(r"scoping\s+review|mapping\s+review|evidence\s+map", combined):
        return "scoping or mapping review"
    if re.search(r"rapid\s+review", combined):
        return "rapid review"
    if re.search(r"realist\s+review", combined):
        return "realist review"
    if re.search(r"integrative\s+review", combined):
        return "integrative review"
    if re.search(r"systematic\s+review", combined):
        return "systematic review"
    if "review" in combined or "clinical update" in combined or "state of the art" in combined:
        return "narrative or expert review"
    return "other evidence synthesis"


def _objective_category(text: str) -> str:
    if any(term in text for term in ("concept", "framework", "model", "theory", "construct")):
        return "conceptual"
    if any(term in text for term in ("treatment", "management", "intervention", "rehabilitation")):
        return "clinical"
    if any(term in text for term in ("method", "measurement", "assessment", "tool", "instrument")):
        return "methodological"
    if any(term in text for term in ("prevalence", "incidence", "risk factor", "predictor")):
        return "epidemiological"
    return "mixed"


def _bps_location(title: str, abstract: str) -> str:
    title_has = any(term in title.lower() for term in ("biopsychosocial", "bio-psycho-social", "bio psycho social"))
    abstract_has = any(term in abstract.lower() for term in ("biopsychosocial", "bio-psycho-social", "bio psycho social"))
    if title_has and abstract_has:
        return "title and abstract"
    if title_has:
        return "title only"
    if abstract_has:
        return "abstract only"
    return "unclear"


def _bps_function(text: str) -> str:
    if any(term in text for term in ("framework", "model", "conceptual model", "theoretical")):
        return "explanatory framework"
    if any(term in text for term in ("approach", "management", "rehabilitation", "treatment")):
        return "intervention rationale"
    if any(term in text for term in ("justify", "highlight", "important to consider")):
        return "justification"
    if any(term in text for term in ("organizing", "multidisciplinary", "multifactorial")):
        return "organizing principle"
    if any(term in text for term in ("conclude", "implication", "future research")):
        return "conclusion"
    return "unclear"


def _icd11(text: str) -> str:
    for label, terms in ICD11_RULES:
        if any(term in text for term in terms):
            return label
    if "chronic pain" in text or "persistent pain" in text:
        return "mixed or unspecified chronic pain"
    return "unclear"


def _concepts(text: str) -> str:
    found = [name for name, terms in PSYCHOLOGICAL_CONCEPTS.items() if any(term in text for term in terms)]
    return " | ".join(found)


def _quality_flag(text: str) -> str:
    if any(term in text for term in ("risk of bias", "quality assessment", "amstar", "critical appraisal", "quality of evidence")):
        return "yes"
    return "no"


def _musculoskeletal_flag(icd11_label: str) -> str:
    if icd11_label == "chronic secondary musculoskeletal pain":
        return "yes"
    if icd11_label == "mixed or unspecified chronic pain":
        return "unclear"
    return "no"


def _objective_text(abstract: str) -> str:
    text = str(abstract).strip()
    if not text:
        return ""
    patterns = [
        r"(objective[s]?:\s.*?)(?:\n[A-Z][A-Z ]+?:|\nCONCLUSION|\nRESULTS|\nMETHODS|$)",
        r"(aim[s]?\s.*?)(?:\n[A-Z][A-Z ]+?:|\nCONCLUSION|\nRESULTS|\nMETHODS|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return " ".join(match.group(1).split())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return sentences[0].strip() if sentences else text


def _base_stage2_frame(included: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in included.iterrows():
        rows.append(
            {
                "record_id": row["record_id"],
                "database": row["database"],
                "pmid": row.get("pmid", ""),
                "pmcid": row.get("pmcid", ""),
                "doi": row.get("doi", ""),
                "title": row["title"],
                "abstract": row["abstract"],
                "year": row["year"],
                "journal": row["journal"],
                "authors": row.get("authors", ""),
                "country_contact_author": row.get("contact_author_country_guess", ""),
                "publication_types": row.get("publication_types", ""),
                "objective_text": _objective_text(str(row.get("abstract", ""))),
                "bps_mention_location": _bps_location(str(row["title"]), str(row["abstract"])),
                "screening_status": row["stage1_decision"],
                "screening_reason": row["stage1_reason"],
            }
        )
    return pd.DataFrame(rows)


def extract_stage2_rule_based(base_frame: pd.DataFrame) -> pd.DataFrame:
    if base_frame.empty:
        return base_frame.copy()
    rows = []
    for _, row in base_frame.iterrows():
        text = f'{row.get("title", "")}\n{row.get("abstract", "")}'.lower()
        icd11_label = _icd11(text)
        musculoskeletal_flag = _musculoskeletal_flag(icd11_label)
        rows.append(
            {
                **row.to_dict(),
                "review_type": _review_type(text, str(row.get("publication_types", ""))),
                "objective_category": _objective_category(text),
                "objective_category_source": "rule_based_fallback",
                "icd11_pain_category": icd11_label,
                "musculoskeletal_flag": musculoskeletal_flag,
                "bps_function": _bps_function(text),
                "bio_mentioned": "yes" if _contains_any(text, BIO_TERMS) else "no",
                "psych_mentioned": "yes" if _contains_any(text, PSYCH_TERMS) else "no",
                "social_mentioned": "yes" if _contains_any(text, SOCIAL_TERMS) else "no",
                "quality_assessment_reported": _quality_flag(text),
                "psychological_concepts_detected": _concepts(text),
                "theoretical_frameworks_detected": "",
                "conceptual_problem_flags": "none",
                "provisional_typology": "",
                "stage3_candidate": "yes" if musculoskeletal_flag in {"yes", "unclear"} else "no",
                "stage3_priority": "medium" if musculoskeletal_flag in {"yes", "unclear"} else "low",
                "coding_rationale": "Fallback rule-based coding used because structured LLM extraction was unavailable.",
                "coding_method": "rule_based_fallback",
                "llm_model": "",
            }
        )
    return pd.DataFrame(rows)


def _write_stage2_outputs(frame: pd.DataFrame) -> pd.DataFrame:
    output_path = project_path("review_stages", "04_extraction", "outputs", "stage2_abstract_coding.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    if not frame.empty:
        reliability_n = min(50, max(1, int(len(frame) * 0.20)))
        reliability = frame.sample(n=reliability_n, random_state=29).copy()
    else:
        reliability = frame.copy()
    reliability["reviewer_a_objective_category"] = ""
    reliability["reviewer_b_objective_category"] = ""
    reliability["reviewer_a_provisional_typology"] = ""
    reliability["reviewer_b_provisional_typology"] = ""
    reliability["reviewer_a_notes"] = ""
    reliability["reviewer_b_notes"] = ""
    reliability["adjudicated_objective_category"] = ""
    reliability["adjudicated_typology"] = ""
    reliability.to_csv(project_path("review_stages", "04_extraction", "forms", "stage2_double_code_subset.csv"), index=False)
    return frame


def extract_stage2(
    use_llm: bool = True,
    model: str | None = None,
    batch_size: int = 6,
    max_workers: int = 4,
) -> pd.DataFrame:
    screening_path = project_path("review_stages", "03_screening", "outputs", "stage1_screening.csv")
    frame = pd.read_csv(screening_path).fillna("")
    included = frame.loc[frame["stage1_decision"] == "include"].copy()

    base_frame = _base_stage2_frame(included)
    if base_frame.empty:
        return _write_stage2_outputs(base_frame)

    if use_llm:
        try:
            from bps_review.extraction.llm_stage2 import assist_stage2_objectives

            llm_frame = assist_stage2_objectives(base_frame, batch_size=batch_size, max_workers=max_workers, model=model)
            out = base_frame.merge(llm_frame, on="record_id", how="left")
        except Exception as exc:  # pragma: no cover - resilience wrapper
            append_jsonl(
                project_path("review_stages", "04_extraction", "outputs", "stage2_fallback_log.jsonl"),
                {"status": "llm_failed_fallback_to_rules", "detail": str(exc)},
            )
            out = extract_stage2_rule_based(base_frame)
    else:
        out = extract_stage2_rule_based(base_frame)

    preferred_order = [
        "record_id",
        "database",
        "pmid",
        "pmcid",
        "doi",
        "title",
        "abstract",
        "year",
        "journal",
        "authors",
        "country_contact_author",
        "publication_types",
        "review_type",
        "objective_text",
        "objective_category",
        "objective_category_source",
        "icd11_pain_category",
        "musculoskeletal_flag",
        "bps_mention_location",
        "bps_function",
        "bio_mentioned",
        "psych_mentioned",
        "social_mentioned",
        "quality_assessment_reported",
        "psychological_concepts_detected",
        "theoretical_frameworks_detected",
        "conceptual_problem_flags",
        "provisional_typology",
        "stage3_candidate",
        "stage3_priority",
        "coding_rationale",
        "coding_method",
        "llm_model",
        "screening_status",
        "screening_reason",
    ]
    ordered_columns = [column for column in preferred_order if column in out.columns] + [
        column for column in out.columns if column not in preferred_order
    ]
    out = out[ordered_columns].copy()
    return _write_stage2_outputs(out)
