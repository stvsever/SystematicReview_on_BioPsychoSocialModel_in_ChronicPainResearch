from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from bps_review.llm.openrouter import OpenRouterClient
from bps_review.settings import resolve_path
from bps_review.utils.io import append_audit_log, write_csv, write_json


BIO_KEYWORDS = [
    "biological",
    "biomedical",
    "pathophysiology",
    "neuro",
    "nociception",
    "inflammation",
    "physiology",
    "genetic",
    "biomechanic",
    "tissue",
    "central sensitization",
    "peripheral sensitization",
]
PSYCH_KEYWORDS = [
    "psychological",
    "fear",
    "catastroph",
    "coping",
    "self-efficacy",
    "depression",
    "anxiety",
    "cognit",
    "emotion",
    "acceptance",
    "kinesiophobia",
    "stress",
    "belief",
    "expectation",
    "attention",
    "mindfulness",
]
SOCIAL_KEYWORDS = [
    "social",
    "family",
    "work",
    "occupational",
    "interpersonal",
    "socioeconomic",
    "culture",
    "cultural",
    "social support",
    "stigma",
    "healthcare access",
    "education",
]

PSYCHOLOGICAL_CONCEPT_PATTERNS = {
    "fear-avoidance": r"fear[- ]avoid",
    "catastrophizing": r"catastroph",
    "kinesiophobia": r"kinesiophobia",
    "depression": r"depress",
    "anxiety": r"anxiet",
    "coping": r"coping",
    "self-efficacy": r"self[- ]efficacy",
    "acceptance": r"acceptance",
    "illness perceptions": r"illness perception",
    "pain beliefs": r"pain belief",
    "distress": r"distress",
    "expectations": r"expectation",
    "emotion regulation": r"emotion regulation",
    "mindfulness": r"mindful",
    "trauma": r"trauma|ptsd",
    "sleep": r"sleep",
}

FRAMEWORK_PATTERNS = {
    "fear-avoidance model": r"fear[- ]avoidance",
    "cognitive behavioral therapy": r"\bcbt\b|cognitive behavioral",
    "acceptance and commitment therapy": r"\bact\b|acceptance and commitment",
    "illness perception framework": r"illness perception",
    "operant learning": r"operant",
    "social cognitive theory": r"social cognitive",
    "self-regulation": r"self-regulation",
}

ICD11_PATTERNS = {
    "chronic_secondary_musculoskeletal_pain": [
        "musculoskeletal",
        "low back pain",
        "back pain",
        "neck pain",
        "osteoarthritis",
        "fibromyalgia",
        "whiplash",
        "temporomandibular",
        "rheumatoid",
        "pelvic girdle",
    ],
    "chronic_neuropathic_pain": ["neuropathic", "complex regional pain", "crps", "radiculopathy"],
    "chronic_cancer_related_pain": ["cancer pain", "oncology pain", "tumor pain"],
    "chronic_secondary_headache_or_orofacial_pain": ["headache", "migraine", "orofacial"],
    "chronic_secondary_visceral_pain": ["visceral", "abdominal pain", "irritable bowel"],
    "chronic_postsurgical_or_posttraumatic_pain": ["postsurgical", "post-surgical", "posttraumatic", "post-traumatic"],
    "chronic_primary_pain": ["chronic primary pain"],
}


def _match_count(text: str, keywords: list[str]) -> int:
    return sum(text.count(keyword) for keyword in keywords)


def _coverage_from_count(count: int) -> str:
    if count >= 4:
        return "elaborated"
    if count >= 2:
        return "mentioned"
    if count >= 1:
        return "minimal"
    return "absent"


def _review_type(text: str) -> str:
    if "meta-analysis" in text or "meta analysis" in text:
        return "meta-analysis"
    if "scoping review" in text:
        return "scoping review"
    if "systematic review" in text:
        return "systematic review"
    if "narrative review" in text:
        return "narrative review"
    if "umbrella review" in text:
        return "umbrella review"
    if "review" in text:
        return "review"
    return "unclear"


def _objective_category(text: str) -> str:
    if any(term in text for term in ["framework", "concept", "operational", "model", "biopsychosocial"]):
        return "conceptual"
    if any(term in text for term in ["treatment", "management", "intervention", "rehabilitation"]):
        return "clinical"
    if any(term in text for term in ["measurement", "psychometric", "assessment", "instrument"]):
        return "methodological"
    if any(term in text for term in ["prevalence", "incidence", "risk factor", "association"]):
        return "epidemiological"
    return "unclear"


def _icd11_category(text: str) -> tuple[str, str]:
    for category, patterns in ICD11_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            musculoskeletal = "yes" if category == "chronic_secondary_musculoskeletal_pain" else "no"
            return category, musculoskeletal
    if "chronic pain" in text or "persistent pain" in text:
        return "mixed_or_unclear_chronic_pain", "unclear"
    return "unclear", "unclear"


def _bps_function(text: str) -> str:
    if "framework" in text or "model" in text or "approach" in text:
        return "organizing_principle"
    if "important for" in text or "implication" in text or "practice" in text:
        return "conclusion_or_implication"
    if "because" in text or "given the biopsychosocial" in text:
        return "justification"
    return "background_framing"


def _extract_concepts(text: str, patterns: dict[str, str]) -> list[str]:
    found: list[str] = []
    for label, pattern in patterns.items():
        if re.search(pattern, text):
            found.append(label)
    return found


def _semantic_projection(texts: list[str]) -> list[dict[str, float]]:
    anchors = {
        "biological": " ".join(BIO_KEYWORDS),
        "psychological": " ".join(PSYCH_KEYWORDS),
        "social": " ".join(SOCIAL_KEYWORDS),
    }
    corpus = texts + list(anchors.values())
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(corpus)
    doc_matrix = matrix[: len(texts)]
    anchor_matrix = matrix[len(texts) :]
    similarities = cosine_similarity(doc_matrix, anchor_matrix)
    rows: list[dict[str, float]] = []
    for scores in similarities:
        rows.append(
            {
                "semantic_biological": float(scores[0]),
                "semantic_psychological": float(scores[1]),
                "semantic_social": float(scores[2]),
            }
        )
    return rows


def _overall_balance(row: pd.Series) -> str:
    scores = {
        "bio": row["semantic_biological"],
        "psych": row["semantic_psychological"],
        "social": row["semantic_social"],
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_value = ordered[0]
    second_value = ordered[1][1]
    if top_value - second_value < 0.05:
        return "balanced"
    if top_label == "bio":
        return "bio-dominant"
    if top_label == "psych":
        return "psych-dominant"
    return "social-dominant"


def _bps_typology(row: pd.Series) -> str:
    domains = [row["domain_biological"], row["domain_psychological"], row["domain_social"]]
    elaborated = sum(value == "elaborated" for value in domains)
    present = sum(value != "absent" for value in domains)
    if present <= 1:
        return "narrow_despite_label"
    if elaborated >= 2 and row["semantic_social"] > 0.08 and row["semantic_biological"] > 0.08 and row["semantic_psychological"] > 0.08:
        return "multifactorial"
    if row["bps_function"] in {"background_framing", "justification"} and row["domain_social"] == "absent":
        return "rhetorical_bps"
    if row["domain_psychological"] != "absent" and (row["domain_biological"] == "absent" or row["domain_social"] == "absent"):
        return "pseudo_bps"
    return "multifactorial"


def code_stage2(screened: pd.DataFrame) -> pd.DataFrame:
    included = screened[screened["stage1_decision"].isin(["include", "maybe"])].copy()
    if included.empty:
        return included

    text_series = included["text_blob"].fillna("").astype(str)
    semantic_rows = _semantic_projection(text_series.tolist())

    coded_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(included.to_dict(orient="records")):
        text = record["text_blob"]
        icd11, musculoskeletal_flag = _icd11_category(text)
        concepts = _extract_concepts(text, PSYCHOLOGICAL_CONCEPT_PATTERNS)
        frameworks = _extract_concepts(text, FRAMEWORK_PATTERNS)
        semantic = semantic_rows[idx]
        row = {
            **record,
            "review_type": _review_type(text),
            "objective_category": _objective_category(text),
            "icd11_pain_category": icd11,
            "musculoskeletal_flag": musculoskeletal_flag,
            "bps_mention_in_title": "yes" if "biopsychosocial" in str(record["title"]).lower() else "no",
            "bps_mention_in_abstract": "yes" if "biopsychosocial" in str(record["abstract"]).lower() else "no",
            "bps_function": _bps_function(text),
            "domain_biological": _coverage_from_count(_match_count(text, BIO_KEYWORDS)),
            "domain_psychological": _coverage_from_count(_match_count(text, PSYCH_KEYWORDS)),
            "domain_social": _coverage_from_count(_match_count(text, SOCIAL_KEYWORDS)),
            "domain_spiritual_existential": "mentioned" if "spiritual" in text or "existential" in text else "absent",
            "domain_lifestyle": "mentioned" if "lifestyle" in text or "physical activity" in text else "absent",
            "reported_quality_assessment": "yes" if any(term in text for term in ["risk of bias", "quality assessment", "amstar"]) else "no",
            "psychological_concepts": "; ".join(concepts),
            "theoretical_frameworks": "; ".join(frameworks),
            **semantic,
        }
        row["overall_balance"] = _overall_balance(pd.Series(row))
        row["bps_typology"] = _bps_typology(pd.Series(row))
        coded_rows.append(row)

    coded = pd.DataFrame(coded_rows)
    write_csv(resolve_path("interim_extraction") / "stage2_coded.csv", coded)
    append_audit_log(
        resolve_path("audit_trail") / "extraction_log.jsonl",
        {
            "stage": "stage2_coding",
            "coded_records": int(coded.shape[0]),
            "musculoskeletal_candidates": int((coded["musculoskeletal_flag"] == "yes").sum()),
        },
    )
    return coded


def normalize_concepts_with_llm(coded: pd.DataFrame) -> dict[str, Any]:
    unique_concepts = sorted({concept.strip() for entry in coded["psychological_concepts"].fillna("") for concept in entry.split(";") if concept.strip()})
    if not unique_concepts:
        return {"status": "no_concepts_found", "clusters": []}

    client = OpenRouterClient()
    if not client.enabled:
        return {"status": "llm_not_configured", "clusters": []}

    system_prompt = (
        "You are assisting a systematic review. Cluster psychological chronic pain review concepts into higher-order families "
        "and note likely theoretical frameworks. Return strict JSON with a top-level 'clusters' list. "
        "Each cluster must contain 'family', 'members', and 'possible_frameworks'."
    )
    user_prompt = json.dumps({"concepts": unique_concepts}, indent=2)
    result = client.json_completion(system_prompt, user_prompt)
    write_json(resolve_path("interim_extraction") / "llm_concept_clusters.json", result)
    append_audit_log(
        resolve_path("audit_trail") / "extraction_log.jsonl",
        {
            "stage": "llm_concept_normalization",
            "status": "completed",
            "concept_count": len(unique_concepts),
        },
    )
    return result
