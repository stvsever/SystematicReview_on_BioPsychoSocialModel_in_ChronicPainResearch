from __future__ import annotations

"""Cross-provider reliability for the full-text coding scheme (scheme 3).

Agreement is quantified on three kinds of variable, and they are kept apart
because they answer different questions.

* **Categorical decisions**, including the coverage and integration ladders.
  Fleiss' kappa, Krippendorff's alpha, observed agreement, and the unanimous
  rate, per field. On the ordered ladders an additional adjacent-agreement rate
  is reported: the share of papers on which two coders were at most one rung
  apart, which separates a genuine disagreement about a paper from a
  one-rung difference in strictness.
* **Binary presence**, derived from the coded content. Whether two coders both
  found integration evidence in a paper has one answer.
* **Open lists**, compared with Jaccard set overlap over normalized labels,
  because two coders can both be right and still write different strings.

Produces, and persists under ``src/05_data/pilot/02_fulltext_level/03_reliability``:
``01_field_reliability.csv``, ``02_pairwise_percent_agreement.csv``,
``03_pairwise_cohen_kappa.csv``, ``04_per_model_behavior.csv``,
``05_consensus_codings.csv``, ``06_eligibility_depth_distribution.csv``,
``07_list_overlap.csv``, ``08_typology_concordance.csv``, and
``reliability_summary.json``.
"""

import itertools
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from bps_review.fulltext.coding.vocabulary import normalize_label
from bps_review.fulltext.config import (
    AUXILIARY_COVERAGE_FIELDS,
    BALANCE_ORDER,
    BPS_DEFINITION_STATUS_ORDER,
    BPS_LABEL_ORDER,
    CATEGORICAL_FIELDS,
    COVERAGE_FIELDS,
    COVERAGE_ORDER,
    DEFINITIONS_ORDER,
    ELIGIBILITY_ORDER,
    FIELD_LABELS,
    INTEGRATION_FIELDS,
    LIST_FIELDS,
    LIST_LABEL_KEY,
    LIST_LABEL_VOCAB,
    MODEL_LABELS,
    NOMINAL_FIELDS,
    PAIRWISE_ORDER,
    PRESENCE_FIELDS,
    PRESENCE_ORDER,
    PRIORITY_ORDER,
    RELIABILITY_FIELDS,
    TRACK_ORDER,
    TRIADIC_ORDER,
    TRISTATE_ORDER,
    TYPOLOGY_ORDER,
    reliability_dir,
)
from bps_review.pilot.analysis.metrics import (
    cohen_kappa,
    fleiss_kappa,
    krippendorff_alpha,
    landis_koch_label,
    mean_pairwise_percent_agreement,
    percent_agreement,
    unanimous_rate,
)
from bps_review.utils.io import write_csv, write_json


ORDER_HINTS: dict[str, list[str]] = {
    **{field: COVERAGE_ORDER for field in COVERAGE_FIELDS + AUXILIARY_COVERAGE_FIELDS},
    "integration_bio_psych": PAIRWISE_ORDER,
    "integration_psych_social": PAIRWISE_ORDER,
    "integration_bio_social": PAIRWISE_ORDER,
    "integration_triadic": TRIADIC_ORDER,
    "overall_balance": BALANCE_ORDER,
    "bps_typology": TYPOLOGY_ORDER,
    "concept_definitions_present": DEFINITIONS_ORDER,
    "fulltext_eligibility": ELIGIBILITY_ORDER,
    "synthesis_priority": PRIORITY_ORDER,
    "review_track": TRACK_ORDER,
    "bps_label_used": BPS_LABEL_ORDER,
    "bps_definition_status": BPS_DEFINITION_STATUS_ORDER,
    "quality_assessment_reported": TRISTATE_ORDER,
    **{field: PRESENCE_ORDER for field in PRESENCE_FIELDS},
}

# The ordered ladders, best rung first, used for the adjacent-agreement rate.
LADDER_ORDERS: dict[str, list[str]] = {
    **{field: COVERAGE_ORDER for field in COVERAGE_FIELDS + AUXILIARY_COVERAGE_FIELDS},
    "integration_bio_psych": PAIRWISE_ORDER,
    "integration_psych_social": PAIRWISE_ORDER,
    "integration_bio_social": PAIRWISE_ORDER,
    "integration_triadic": TRIADIC_ORDER,
    "concept_definitions_present": DEFINITIONS_ORDER,
    "bps_definition_status": BPS_DEFINITION_STATUS_ORDER,
}


def _field_group(field: str) -> str:
    if field in COVERAGE_FIELDS + AUXILIARY_COVERAGE_FIELDS:
        return "coverage"
    if field in INTEGRATION_FIELDS:
        return "integration"
    if field in PRESENCE_FIELDS:
        return "presence"
    return "nominal"


def aligned_columns(long_df: pd.DataFrame, field: str, models: list[str] = MODEL_LABELS):
    """Return (columns, record_ids): one label column per model, aligned by item."""
    pivot = long_df.pivot(index="record_id", columns="model_label", values=field)
    pivot = pivot.reindex(columns=models)
    record_ids = list(pivot.index)
    columns = [pivot[model].astype(str).fillna("").replace("nan", "").tolist() for model in models]
    return columns, record_ids


def adjacent_agreement(columns: list[list[str]], order: list[str]) -> float:
    """Share of item-pairs where two coders sit at most one rung apart."""
    rank = {value: index for index, value in enumerate(order)}
    hits, total = 0, 0
    for first, second in itertools.combinations(columns, 2):
        for a, b in zip(first, second):
            if a not in rank or b not in rank:
                continue
            total += 1
            if abs(rank[a] - rank[b]) <= 1:
                hits += 1
    return hits / total if total else float("nan")


# --------------------------------------------------------------------------
# Per-field reliability
# --------------------------------------------------------------------------
def compute_field_reliability(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in RELIABILITY_FIELDS:
        columns, _ = aligned_columns(long_df, field)
        categories = sorted({label for column in columns for label in column if label})
        fleiss = fleiss_kappa(columns)
        ladder = LADDER_ORDERS.get(field)
        rows.append(
            {
                "field": field,
                "field_label": FIELD_LABELS.get(field, field),
                "group": _field_group(field),
                "n_categories": len(categories),
                "mean_pairwise_agreement": mean_pairwise_percent_agreement(columns),
                "adjacent_agreement": adjacent_agreement(columns, ladder) if ladder else float("nan"),
                "unanimous_rate": unanimous_rate(columns),
                "fleiss_kappa": fleiss,
                "krippendorff_alpha": krippendorff_alpha(columns),
                "landis_koch": landis_koch_label(fleiss),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Pairwise model-by-model agreement (averaged across fields)
# --------------------------------------------------------------------------
def compute_pairwise_matrices(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = MODEL_LABELS
    agreement = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    kappa = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)

    per_field = {field: dict(zip(models, aligned_columns(long_df, field)[0])) for field in RELIABILITY_FIELDS}

    for first, second in itertools.combinations(models, 2):
        agreements, kappas = [], []
        for field in RELIABILITY_FIELDS:
            a, b = per_field[field][first], per_field[field][second]
            observed = percent_agreement(a, b)
            if not np.isnan(observed):
                agreements.append(observed)
            value = cohen_kappa(a, b)
            if not np.isnan(value):
                kappas.append(value)
        agreement.loc[first, second] = agreement.loc[second, first] = (
            float(np.mean(agreements)) if agreements else np.nan)
        kappa.loc[first, second] = kappa.loc[second, first] = float(np.mean(kappas)) if kappas else np.nan
    for model in models:
        agreement.loc[model, model] = 1.0
        kappa.loc[model, model] = 1.0
    return agreement, kappa


# --------------------------------------------------------------------------
# Set overlap on the open extraction lists
# --------------------------------------------------------------------------
def _normalize_label(value: str) -> str:
    cleaned = " ".join(str(value or "").strip().lower().split())
    return re.sub(r"[^a-z0-9 \-]+", "", cleaned)


def _labels_of(value: object, field: str) -> set[str]:
    """Normalized label set from one JSON-serialized extraction list cell.

    An item is identified by one key or by several joined together, because a
    concept relation and an integration claim are edges rather than labels. The
    first key is additionally mapped onto the project vocabulary when one applies,
    so two coders who wrote the same thing in different words are counted as
    agreeing, and a label the vocabulary does not carry survives as written.
    """
    keys = LIST_LABEL_KEY.get(field, ())
    kind = LIST_LABEL_VOCAB.get(field, "")
    labels: set[str] = set()
    try:
        items = json.loads(value) if isinstance(value, str) and value.strip() else []
    except json.JSONDecodeError:
        return labels
    if not isinstance(items, list):
        return labels
    for item in items:
        if not isinstance(item, dict):
            continue
        parts = []
        for index, key in enumerate(keys):
            raw = item.get(key, "")
            if index == 0 and kind:
                raw = normalize_label(raw, kind)
            part = _normalize_label(raw)
            if part:
                parts.append(part)
        if parts:
            labels.add(" | ".join(parts))
    return labels


def _jaccard(first: set[str], second: set[str]) -> float:
    if not first and not second:
        return float("nan")
    union = first | second
    return len(first & second) / len(union) if union else float("nan")


def compute_list_overlap(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per open list: mean pairwise Jaccard overlap and how much each model returned."""
    rows = []
    for field in LIST_FIELDS:
        columns, _ = aligned_columns(long_df, field)
        as_sets = [[_labels_of(value, field) for value in column] for column in columns]
        scores: list[float] = []
        for first, second in itertools.combinations(range(len(MODEL_LABELS)), 2):
            for index in range(len(as_sets[first])):
                value = _jaccard(as_sets[first][index], as_sets[second][index])
                if not np.isnan(value):
                    scores.append(value)
        shared_by_all = 0
        for index in range(len(as_sets[0])):
            per_model = [as_sets[model][index] for model in range(len(MODEL_LABELS))]
            if all(per_model) and set.intersection(*per_model):
                shared_by_all += 1
        distinct = {label for model_sets in as_sets for labels in model_sets for label in labels}
        rows.append(
            {
                "field": field,
                "field_label": FIELD_LABELS.get(field, field),
                "mean_pairwise_jaccard": float(np.mean(scores)) if scores else float("nan"),
                "n_comparable_pairs": len(scores),
                "distinct_labels_total": len(distinct),
                "papers_with_a_shared_label": shared_by_all,
                **{f"mean_items_{MODEL_LABELS[index]}": round(
                    float(np.mean([len(labels) for labels in as_sets[index]])), 2)
                   for index in range(len(MODEL_LABELS))},
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Per-model coding behaviour
# --------------------------------------------------------------------------
def _rate(series: pd.Series, value: str) -> float:
    if len(series) == 0:
        return float("nan")
    return float((series.astype(str) == value).mean())


def per_model_behavior(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_LABELS:
        subset = long_df[long_df["model_label"] == model]
        if subset.empty:
            continue
        numeric = lambda column: pd.to_numeric(subset[column], errors="coerce")  # noqa: E731
        rows.append(
            {
                "model_label": model,
                "provider": subset["provider"].iloc[0],
                "include_rate": _rate(subset["fulltext_eligibility"], "include"),
                "core_priority_rate": _rate(subset["synthesis_priority"], "core"),
                "true_integrative_rate": _rate(subset["bps_typology"], "true_integrative"),
                "triadic_any_rate": float((subset["integration_triadic"].astype(str) != "none").mean()),
                "bio_elaborated_rate": _rate(subset["domain_coverage_bio"], "elaborated"),
                "psych_elaborated_rate": _rate(subset["domain_coverage_psych"], "elaborated"),
                "social_elaborated_rate": _rate(subset["domain_coverage_social"], "elaborated"),
                "mean_integration_index": float(numeric("integration_index").mean()),
                "mean_integration_claims": float(numeric("n_integration_claims").mean()),
                "mean_named_integration_edges": float(numeric("n_named_integration_edges").mean()),
                "mean_bio_factors": float(numeric("n_biological_factors").mean()),
                "mean_social_factors": float(numeric("n_social_factors").mean()),
                "mean_concepts": float(numeric("n_psychological_concepts").mean()),
                "mean_defined_concepts": float(numeric("n_defined_concepts").mean()),
                "mean_concept_relations": float(numeric("n_concept_relations").mean()),
                "mean_frameworks": float(numeric("n_theoretical_frameworks").mean()),
                "mean_instruments": float(numeric("n_instruments").mean()),
                "mean_bps_usage_instances": float(numeric("n_bps_usage_instances").mean()),
                "mean_subdomains_named": float(numeric("n_subdomains_named").mean()),
                "mean_extracted_items": float(numeric("n_extracted_items").mean()),
                "mean_controlled_label_share": float(numeric("controlled_label_share").mean()),
                "typology_matches_derived": _rate(subset["typology_matches_derived"], "yes"),
                "structured_share": float((subset["coding_method"] == "llm_structured").mean()),
            }
        )
    return pd.DataFrame(rows)


def typology_concordance(long_df: pd.DataFrame) -> pd.DataFrame:
    """Coded typology against the typology the rule derives from coverage and integration."""
    table = (
        long_df.groupby(["bps_typology", "derived_typology"]).size().reset_index(name="n_codings")
        .sort_values("n_codings", ascending=False)
    )
    return table


# --------------------------------------------------------------------------
# Consensus (majority vote) coding
# --------------------------------------------------------------------------
def _majority(labels: list[str], order: list[str] | None = None) -> tuple[str, int]:
    labels = [label for label in labels if label]
    if not labels:
        return "", 0
    counts = Counter(labels)
    top = max(counts.values())
    tied = [label for label, count in counts.items() if count == top]
    if len(tied) == 1:
        return tied[0], top
    if order:
        for candidate in order:
            if candidate in tied:
                return candidate, top
    return sorted(tied)[0], top


def consensus_codings(long_df: pd.DataFrame) -> pd.DataFrame:
    record_ids = sorted(long_df["record_id"].unique())
    rows = []
    for record_id in record_ids:
        subset = long_df[long_df["record_id"] == record_id].sort_values("model_order")
        row: dict[str, object] = {"record_id": record_id}
        for field in RELIABILITY_FIELDS:
            labels = subset[field].astype(str).tolist()
            label, depth = _majority(labels, ORDER_HINTS.get(field))
            row[field] = label
            if field == "fulltext_eligibility":
                row["eligibility_consensus_depth"] = depth
            if field == "bps_typology":
                row["typology_consensus_depth"] = depth
        row["mean_integration_index"] = float(
            pd.to_numeric(subset["integration_index"], errors="coerce").mean())
        row["mean_extracted_items"] = float(
            pd.to_numeric(subset["n_extracted_items"], errors="coerce").mean())
        for field in LIST_FIELDS:
            union: set[str] = set()
            for value in subset[field].tolist():
                union |= _labels_of(value, field)
            row[field] = " | ".join(sorted(union))
        rows.append(row)
    return pd.DataFrame(rows)


def eligibility_depth_distribution(consensus: pd.DataFrame) -> pd.DataFrame:
    dist = consensus["eligibility_consensus_depth"].value_counts().sort_index()
    frame = dist.reset_index()
    frame.columns = ["models_backing_majority", "n_items"]
    return frame


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _clean_for_json(value):
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def build_reliability(long_df: pd.DataFrame, write: bool = True) -> dict[str, object]:
    field_rel = compute_field_reliability(long_df)
    agreement, kappa = compute_pairwise_matrices(long_df)
    behavior = per_model_behavior(long_df)
    consensus = consensus_codings(long_df)
    depth = eligibility_depth_distribution(consensus)
    overlap = compute_list_overlap(long_df)
    concordance = typology_concordance(long_df)

    if write:
        out = reliability_dir()
        write_csv(out / "01_field_reliability.csv", field_rel)
        write_csv(out / "02_pairwise_percent_agreement.csv", agreement.reset_index(names="model_label"))
        write_csv(out / "03_pairwise_cohen_kappa.csv", kappa.reset_index(names="model_label"))
        write_csv(out / "04_per_model_behavior.csv", behavior)
        write_csv(out / "05_consensus_codings.csv", consensus)
        write_csv(out / "06_eligibility_depth_distribution.csv", depth)
        write_csv(out / "07_list_overlap.csv", overlap)
        write_csv(out / "08_typology_concordance.csv", concordance)

    def _group_mean(group: str, column: str) -> float:
        subset = field_rel[field_rel["group"] == group]
        return float(subset[column].mean()) if len(subset) else float("nan")

    def _field_value(field: str, column: str):
        match = field_rel.loc[field_rel.field == field, column]
        return _clean_for_json(float(match.iloc[0])) if len(match) else None

    key_fields = ["integration_triadic", "bps_typology", "domain_coverage_social",
                  "fulltext_eligibility", "present_integration_evidence"]
    key_rel = {
        field: {
            "fleiss_kappa": _field_value(field, "fleiss_kappa"),
            "krippendorff_alpha": _field_value(field, "krippendorff_alpha"),
            "mean_pairwise_agreement": _field_value(field, "mean_pairwise_agreement"),
            "adjacent_agreement": _field_value(field, "adjacent_agreement"),
        }
        for field in key_fields
    }

    off_diagonal = agreement.where(~np.eye(len(agreement), dtype=bool)).stack()
    n_models = len(MODEL_LABELS)

    summary = {
        "n_papers": int(long_df["record_id"].nunique()),
        "n_models": int(long_df["model_label"].nunique()),
        "n_codings": int(len(long_df)),
        "mean_fleiss_kappa": float(field_rel["fleiss_kappa"].mean()),
        "mean_krippendorff_alpha": float(field_rel["krippendorff_alpha"].mean()),
        "mean_pairwise_agreement": float(field_rel["mean_pairwise_agreement"].mean()),
        "mean_unanimous_rate": float(field_rel["unanimous_rate"].mean()),
        "categorical_mean_fleiss_kappa": float(
            field_rel[field_rel["field"].isin(CATEGORICAL_FIELDS)]["fleiss_kappa"].mean()),
        "presence_mean_fleiss_kappa": _group_mean("presence", "fleiss_kappa"),
        "coverage_mean_adjacent_agreement": _group_mean("coverage", "adjacent_agreement"),
        "integration_mean_adjacent_agreement": _group_mean("integration", "adjacent_agreement"),
        "key_field_reliability": key_rel,
        "most_agreeing_pair": {"models": list(off_diagonal.idxmax()), "agreement": float(off_diagonal.max())},
        "least_agreeing_pair": {"models": list(off_diagonal.idxmin()), "agreement": float(off_diagonal.min())},
        "eligibility_unanimous_items": int((consensus["eligibility_consensus_depth"] == n_models).sum()),
        "consensus_include": int((consensus["fulltext_eligibility"] == "include").sum()),
        "consensus_typology_counts": consensus["bps_typology"].value_counts().to_dict(),
        "consensus_triadic_counts": consensus["integration_triadic"].value_counts().to_dict(),
        "list_overlap": {
            row["field"]: _clean_for_json(round(float(row["mean_pairwise_jaccard"]), 3))
            for _, row in overlap.iterrows()
        },
        "typology_coded_matches_derived": float(
            (long_df["typology_matches_derived"].astype(str) == "yes").mean()),
    }
    if write:
        write_json(reliability_dir() / "reliability_summary.json", summary)

    return {
        "field_reliability": field_rel,
        "pairwise_agreement": agreement,
        "pairwise_kappa": kappa,
        "per_model_behavior": behavior,
        "consensus": consensus,
        "eligibility_depth": depth,
        "list_overlap": overlap,
        "typology_concordance": concordance,
        "summary": summary,
    }
