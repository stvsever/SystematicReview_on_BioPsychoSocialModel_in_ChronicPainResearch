from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from bps_review.llm.openrouter import embed_texts
from bps_review.utils.env import get_env
from bps_review.utils.io import ensure_parent
from bps_review.utils.paths import project_path


ONTOLOGY_TERMS: dict[str, list[str]] = {
    "biological": [
        "Central Sensitization and Neuroplasticity",
        "Musculoskeletal and Structural Pathology",
        "Pharmacological and Biomedical Treatment",
        "Sleep Disruption and Circadian Dysregulation",
        "Immune Inflammatory and Neuroinflammatory Processes",
        "Genetic Epigenetic and Biological Vulnerability",
        "Neuroimaging Brain Structure and Function",
        "Nociceptive Transmission and Pain Pathways",
        "Physical Function Mobility and Deconditioning",
        "Metabolic Nutritional and Hormonal Factors",
    ],
    "psychological": [
        "Catastrophizing and Negative Cognitive Appraisal",
        "Fear Avoidance and Pain Related Fear",
        "Depression Emotional Distress and Affect",
        "Anxiety and Psychological Reactivity",
        "Self Efficacy Control Beliefs and Perceived Mastery",
        "Acceptance Psychological Flexibility and Mindfulness",
        "Pain Coping Strategies and Adjustment",
        "Attention Vigilance and Pain Processing",
        "Illness Beliefs Pain Representations and Meaning",
        "Cognitive Behavioral and Psychotherapeutic Approaches",
        "Third Wave Therapies ACT and Contextual Approaches",
        "Resilience Positive Psychology and Post Traumatic Growth",
        "Identity Self Concept and Chronic Pain Biography",
        "Trauma Adverse Childhood and Life Events",
        "Personality Psychological Traits and Individual Differences",
        "Cognitive Function Executive Processes and Brain Health",
        "Motivational Processes Goal Pursuit and Engagement",
        "Healthcare Seeking Treatment Adherence and Engagement",
        "Emotional Regulation and Pain Affect Processing",
        "Mental Health Comorbidity and Psychological Wellbeing",
    ],
    "social": [
        "Social Support Network and Interpersonal Resources",
        "Work Disability Occupational Function and Productivity",
        "Family Caregiver and Household Dynamics",
        "Socioeconomic Status and Health Inequity",
        "Healthcare Access Navigation and System Factors",
        "Cultural Ethnic and Demographic Context",
        "Community Participation and Social Role Functioning",
        "Legal Compensation and Medicolegal Systems",
        "Health Literacy Education and Patient Empowerment",
        "Stigma Social Isolation and Exclusion",
        "Return to Work Vocational Rehabilitation and Employment",
        "Social Determinants of Pain and Environment",
    ],
}


@dataclass
class SemanticLoadingResult:
    status: str
    method: str
    model: str
    note: str
    record_loadings: pd.DataFrame
    record_subdomain_loadings: pd.DataFrame
    domain_summary: pd.DataFrame
    subdomain_summary: pd.DataFrame
    pairwise_loadings: pd.DataFrame
    pairwise_summary: pd.DataFrame
    dominance_by_review_type: pd.DataFrame


def _compose_text(frame: pd.DataFrame) -> pd.Series:
    parts = [
        frame.get("title", "").astype(str),
        frame.get("abstract", "").astype(str),
        frame.get("objective_text", "").astype(str),
    ]
    return (
        "Title. "
        + parts[0].str.strip()
        + "\n\nAbstract. "
        + parts[1].str.strip()
        + "\n\nObjective. "
        + parts[2].str.strip()
    ).str.strip()


def _ontology_prompts() -> dict[str, str]:
    prompts: dict[str, str] = {}
    for domain, terms in ONTOLOGY_TERMS.items():
        joined = "; ".join(terms)
        prompts[domain] = f"{domain} chronic pain ontology. {joined}."
    return prompts


def _subdomain_prompts(domain_order: list[str]) -> tuple[list[str], list[str], list[str]]:
    keys: list[str] = []
    domains: list[str] = []
    prompts: list[str] = []
    for domain in domain_order:
        for term in ONTOLOGY_TERMS.get(domain, []):
            key = f"{domain}::{term}"
            keys.append(key)
            domains.append(domain)
            prompts.append(f"{domain} chronic pain subdomain. {term}.")
    return keys, domains, prompts


def _safe_to_numpy(embeddings: list[list[float]]) -> np.ndarray:
    if not embeddings:
        return np.empty((0, 0))
    return np.array(embeddings, dtype=np.float32)


def _compute_loadings(record_vectors: np.ndarray, ontology_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cosine_matrix = cosine_similarity(record_vectors, ontology_vectors)
    stabilised = cosine_matrix - cosine_matrix.max(axis=1, keepdims=True)
    exp_values = np.exp(stabilised)
    loadings = exp_values / exp_values.sum(axis=1, keepdims=True)
    return cosine_matrix, loadings


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_tfidf_embeddings(texts: list[str], ontology_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts + ontology_texts)
    record_vectors = matrix[: len(texts)].toarray()
    ontology_vectors = matrix[len(texts) :].toarray()
    return record_vectors, ontology_vectors


def run_semantic_loading(stage2: pd.DataFrame) -> SemanticLoadingResult:
    columns = ["record_id", "year", "review_type", "title", "abstract", "objective_text"]
    if stage2.empty or "record_id" not in stage2.columns:
        return SemanticLoadingResult(
            status="no_data",
            method="none",
            model="none",
            note="Stage 2 file is empty or missing required columns.",
            record_loadings=pd.DataFrame(),
            record_subdomain_loadings=pd.DataFrame(),
            domain_summary=pd.DataFrame(),
            subdomain_summary=pd.DataFrame(),
            pairwise_loadings=pd.DataFrame(),
            pairwise_summary=pd.DataFrame(),
            dominance_by_review_type=pd.DataFrame(),
        )

    available = [name for name in columns if name in stage2.columns]
    working = stage2[available].copy()
    working["semantic_text"] = _compose_text(stage2)
    working = working.loc[working["semantic_text"].str.len() > 20].reset_index(drop=True)

    if working.empty:
        return SemanticLoadingResult(
            status="no_text",
            method="none",
            model="none",
            note="No records with usable semantic text were found.",
            record_loadings=pd.DataFrame(),
            record_subdomain_loadings=pd.DataFrame(),
            domain_summary=pd.DataFrame(),
            subdomain_summary=pd.DataFrame(),
            pairwise_loadings=pd.DataFrame(),
            pairwise_summary=pd.DataFrame(),
            dominance_by_review_type=pd.DataFrame(),
        )

    domain_order = ["biological", "psychological", "social"]
    ontology_prompts = _ontology_prompts()
    ontology_texts = [ontology_prompts[name] for name in domain_order]
    subdomain_keys, subdomain_domains, subdomain_texts = _subdomain_prompts(domain_order)
    record_texts = working["semantic_text"].tolist()

    method = "openrouter_embedding"
    model = get_env("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small") or "openai/text-embedding-3-small"
    note = "Ontology-aligned embeddings generated through OpenRouter."

    try:
        if not get_env("OPENROUTER_API_KEY"):
            raise EnvironmentError("OPENROUTER_API_KEY is not set.")
        record_embeddings = _safe_to_numpy(embed_texts(record_texts, model=model))
        ontology_embeddings = _safe_to_numpy(embed_texts(ontology_texts, model=model))
        subdomain_embeddings = _safe_to_numpy(embed_texts(subdomain_texts, model=model))
    except Exception as exc:
        method = "tfidf_fallback"
        model = "tfidf"
        note = f"OpenRouter embeddings unavailable. Falling back to TF-IDF semantics. Detail: {exc}"
        vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform(record_texts + ontology_texts + subdomain_texts)
        record_embeddings = matrix[: len(record_texts)].toarray()
        ontology_embeddings = matrix[len(record_texts) : len(record_texts) + len(ontology_texts)].toarray()
        subdomain_embeddings = matrix[len(record_texts) + len(ontology_texts) :].toarray()

    cosine_matrix, loadings = _compute_loadings(record_embeddings, ontology_embeddings)
    subdomain_cosine = cosine_similarity(record_embeddings, subdomain_embeddings)
    dominant_indices = np.argmax(loadings, axis=1)
    dominant_domain = [domain_order[index] for index in dominant_indices]

    loadings_frame = pd.DataFrame(
        {
            "record_id": working["record_id"],
            "year": pd.to_numeric(working.get("year", pd.Series(dtype=float)), errors="coerce"),
            "review_type": working.get("review_type", pd.Series(["unclear"] * len(working))).astype(str),
            "cosine_biological": np.round(cosine_matrix[:, 0], 4),
            "cosine_psychological": np.round(cosine_matrix[:, 1], 4),
            "cosine_social": np.round(cosine_matrix[:, 2], 4),
            "loading_biological": np.round(loadings[:, 0], 4),
            "loading_psychological": np.round(loadings[:, 1], 4),
            "loading_social": np.round(loadings[:, 2], 4),
            "dominant_domain": dominant_domain,
        }
    )

    summary_rows: list[dict[str, object]] = []
    for domain in domain_order:
        summary_rows.append(
            {
                "domain": domain,
                "mean_cosine": round(float(loadings_frame[f"cosine_{domain}"] .mean()), 4),
                "mean_loading": round(float(loadings_frame[f"loading_{domain}"].mean()), 4),
                "dominance_n": int((loadings_frame["dominant_domain"] == domain).sum()),
            }
        )
    summary_frame = pd.DataFrame(summary_rows)

    dominance_by_review_type = pd.crosstab(loadings_frame["review_type"], loadings_frame["dominant_domain"]).reset_index()

    subdomain_raw = pd.DataFrame(subdomain_cosine, columns=subdomain_keys)
    subdomain_norm = pd.DataFrame(index=subdomain_raw.index)
    subdomain_weighted = pd.DataFrame(index=subdomain_raw.index)
    for domain in domain_order:
        domain_columns = [key for key, key_domain in zip(subdomain_keys, subdomain_domains) if key_domain == domain]
        if not domain_columns:
            continue
        domain_slice = subdomain_raw[domain_columns]
        stabilised = domain_slice.sub(domain_slice.max(axis=1), axis=0)
        exp_values = np.exp(stabilised)
        normalized = exp_values.div(exp_values.sum(axis=1), axis=0)
        subdomain_norm[domain_columns] = normalized
        subdomain_weighted[domain_columns] = normalized.mul(loadings_frame[f"loading_{domain}"], axis=0)

    readable_cols = {
        key: key.split("::", 1)[1] for key in subdomain_weighted.columns
    }
    record_subdomain_loadings = pd.concat(
        [
            loadings_frame[["record_id", "year", "review_type"]].reset_index(drop=True),
            subdomain_weighted.rename(columns=readable_cols).reset_index(drop=True),
        ],
        axis=1,
    )

    subdomain_rows: list[dict[str, object]] = []
    for key, key_domain in zip(subdomain_keys, subdomain_domains):
        readable = key.split("::", 1)[1]
        if readable not in record_subdomain_loadings.columns:
            continue
        values = pd.to_numeric(record_subdomain_loadings[readable], errors="coerce").fillna(0.0)
        subdomain_rows.append(
            {
                "domain": key_domain,
                "subdomain": readable,
                "mean_loading": round(float(values.mean()), 4),
                "median_loading": round(float(values.median()), 4),
            }
        )
    subdomain_summary = pd.DataFrame(subdomain_rows).sort_values(["domain", "mean_loading"], ascending=[True, False])

    pairwise_loadings = pd.DataFrame(
        {
            "record_id": loadings_frame["record_id"],
            "year": loadings_frame["year"],
            "review_type": loadings_frame["review_type"],
            "bio_psych": np.round(loadings_frame["loading_biological"] * loadings_frame["loading_psychological"], 4),
            "bio_social": np.round(loadings_frame["loading_biological"] * loadings_frame["loading_social"], 4),
            "psych_social": np.round(loadings_frame["loading_psychological"] * loadings_frame["loading_social"], 4),
            "triadic_product": np.round(
                loadings_frame["loading_biological"]
                * loadings_frame["loading_psychological"]
                * loadings_frame["loading_social"],
                4,
            ),
        }
    )
    pairwise_summary = pd.DataFrame(
        [
            {
                "bio_psych_mean": round(float(pairwise_loadings["bio_psych"].mean()), 4),
                "bio_social_mean": round(float(pairwise_loadings["bio_social"].mean()), 4),
                "psych_social_mean": round(float(pairwise_loadings["psych_social"].mean()), 4),
                "triadic_product_mean": round(float(pairwise_loadings["triadic_product"].mean()), 4),
            }
        ]
    )

    vector_root = project_path("src", "vector_db", "semantic_loading")
    records_dir = vector_root / "records"
    ontology_dir = vector_root / "ontology"
    analysis_dir = vector_root / "analysis"

    corpus_rows = []
    for _, row in working.iterrows():
        corpus_rows.append(
            {
                "record_id": row.get("record_id", ""),
                "year": row.get("year", ""),
                "review_type": row.get("review_type", ""),
                "semantic_text": row.get("semantic_text", ""),
            }
        )

    _write_jsonl(records_dir / "semantic_corpus.jsonl", corpus_rows)
    ensure_parent(records_dir / "record_embeddings.npy")
    np.save(records_dir / "record_embeddings.npy", record_embeddings)

    ontology_payload = {
        "domains": ONTOLOGY_TERMS,
        "prompts": ontology_prompts,
        "domain_order": domain_order,
        "method": method,
        "model": model,
    }
    ensure_parent(ontology_dir / "ontology_terms.json").write_text(json.dumps(ontology_payload, indent=2), encoding="utf-8")
    ensure_parent(ontology_dir / "ontology_embeddings.npy")
    np.save(ontology_dir / "ontology_embeddings.npy", ontology_embeddings)
    ensure_parent(ontology_dir / "subdomain_embeddings.npy")
    np.save(ontology_dir / "subdomain_embeddings.npy", subdomain_embeddings)

    loadings_frame.to_csv(ensure_parent(analysis_dir / "record_domain_loadings.csv"), index=False)
    record_subdomain_loadings.to_csv(ensure_parent(analysis_dir / "record_subdomain_loadings.csv"), index=False)
    summary_frame.to_csv(ensure_parent(analysis_dir / "domain_loading_summary.csv"), index=False)
    subdomain_summary.to_csv(ensure_parent(analysis_dir / "subdomain_loading_summary.csv"), index=False)
    pairwise_loadings.to_csv(ensure_parent(analysis_dir / "pairwise_domain_loadings.csv"), index=False)
    pairwise_summary.to_csv(ensure_parent(analysis_dir / "pairwise_domain_summary.csv"), index=False)
    dominance_by_review_type.to_csv(ensure_parent(analysis_dir / "review_type_domain_dominance.csv"), index=False)

    return SemanticLoadingResult(
        status="ok",
        method=method,
        model=model,
        note=note,
        record_loadings=loadings_frame,
        record_subdomain_loadings=record_subdomain_loadings,
        domain_summary=summary_frame,
        subdomain_summary=subdomain_summary,
        pairwise_loadings=pairwise_loadings,
        pairwise_summary=pairwise_summary,
        dominance_by_review_type=dominance_by_review_type,
    )
