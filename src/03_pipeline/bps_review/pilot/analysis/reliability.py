from __future__ import annotations

"""Turn the long model-coding table into cross-provider reliability results.

Produces, and persists under ``src/05_data/pilot/01_abstract_level/03_reliability``:

* ``01_field_reliability.csv``   - per field: observed agreement, unanimous rate,
  Fleiss' kappa, Krippendorff's alpha, and a Landis-Koch label;
* ``02_pairwise_percent_agreement.csv`` and ``03_pairwise_cohen_kappa.csv`` -
  model-by-model matrices averaged across fields;
* ``04_per_model_behavior.csv``  - how lenient and how conceptual each model is;
* ``05_consensus_codings.csv``   - the majority-vote coding per abstract;
* ``06_candidate_depth_distribution.csv`` - how many models back the majority
  Stage 3 candidacy call;
* ``07_list_overlap.csv``        - Jaccard overlap on the open extraction lists;
* ``reliability_summary.json``   - headline numbers for the notebook and README.

Two kinds of variable are kept apart on purpose. Categorical decisions are
quantified with kappa-style coefficients, because there the value itself carries
the meaning and a disagreement is a real disagreement. The open lists (concepts,
frameworks, conceptual problems) are quantified with set overlap instead: two
coders can both be right and still return different strings, so a chance-corrected
coefficient over an unbounded label space would be the wrong instrument.
"""

import itertools
import re
from collections import Counter

import numpy as np
import pandas as pd

from bps_review.pilot.analysis.metrics import (
    cohen_kappa,
    fleiss_kappa,
    krippendorff_alpha,
    landis_koch_label,
    mean_pairwise_percent_agreement,
    percent_agreement,
    unanimous_rate,
)
from bps_review.pilot.config import (
    BINARY_ORDER,
    DOMAIN_FIELDS,
    FIELD_LABELS,
    LIST_FIELDS,
    MODEL_LABELS,
    MSK_ORDER,
    NOMINAL_FIELDS,
    ORDINAL_FIELDS,
    PRIORITY_ORDER,
    RELIABILITY_FIELDS,
    TYPOLOGY_ORDER,
    reliability_dir,
)
from bps_review.utils.io import write_csv, write_json


ORDER_HINTS: dict[str, list[str]] = {
    "provisional_typology": TYPOLOGY_ORDER,
    "stage3_priority": PRIORITY_ORDER,
    "musculoskeletal_flag": MSK_ORDER,
    "stage3_candidate": BINARY_ORDER,
    **{field: BINARY_ORDER for field in DOMAIN_FIELDS},
}


def _field_group(field: str) -> str:
    if field in DOMAIN_FIELDS:
        return "domain"
    if field in ORDINAL_FIELDS:
        return "ordinal"
    return "nominal"


def aligned_columns(long_df: pd.DataFrame, field: str, models: list[str] = MODEL_LABELS):
    """Return (columns, record_ids): one label column per model, aligned by item."""
    pivot = long_df.pivot(index="record_id", columns="model_label", values=field)
    pivot = pivot.reindex(columns=models)
    record_ids = list(pivot.index)
    columns = [pivot[model].astype(str).fillna("").replace("nan", "").tolist() for model in models]
    return columns, record_ids


# --------------------------------------------------------------------------
# Per-field reliability
# --------------------------------------------------------------------------
def compute_field_reliability(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in RELIABILITY_FIELDS:
        columns, _ = aligned_columns(long_df, field)
        categories = sorted({label for col in columns for label in col if label})
        fleiss = fleiss_kappa(columns)
        rows.append(
            {
                "field": field,
                "field_label": FIELD_LABELS.get(field, field),
                "group": _field_group(field),
                "n_categories": len(categories),
                "mean_pairwise_agreement": mean_pairwise_percent_agreement(columns),
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
    agr = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    kap = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)

    per_field = {field: dict(zip(models, aligned_columns(long_df, field)[0])) for field in RELIABILITY_FIELDS}

    for first, second in itertools.combinations(models, 2):
        agreements, kappas = [], []
        for field in RELIABILITY_FIELDS:
            a, b = per_field[field][first], per_field[field][second]
            observed = percent_agreement(a, b)
            if not np.isnan(observed):
                agreements.append(observed)
            kappa = cohen_kappa(a, b)
            if not np.isnan(kappa):
                kappas.append(kappa)
        agr.loc[first, second] = agr.loc[second, first] = float(np.mean(agreements)) if agreements else np.nan
        kap.loc[first, second] = kap.loc[second, first] = float(np.mean(kappas)) if kappas else np.nan
    for model in models:
        agr.loc[model, model] = 1.0
        kap.loc[model, model] = 1.0
    return agr, kap


# --------------------------------------------------------------------------
# Set overlap on the open extraction lists
# --------------------------------------------------------------------------
def _labels_of(value: object) -> set[str]:
    """Normalize one pipe-delimited list cell into a comparable label set."""
    text = str(value or "")
    labels = set()
    for part in re.split(r"\||;", text):
        cleaned = " ".join(part.strip().lower().split())
        cleaned = re.sub(r"[^a-z0-9 \-]+", "", cleaned)
        if cleaned and cleaned != "none":
            labels.add(cleaned)
    return labels


def _jaccard(first: set[str], second: set[str]) -> float:
    if not first and not second:
        return float("nan")   # neither coder found anything: nothing to compare
    union = first | second
    if not union:
        return float("nan")
    return len(first & second) / len(union)


def compute_list_overlap(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per open list: mean pairwise Jaccard overlap, plus how much each model returned."""
    rows = []
    for field in LIST_FIELDS:
        columns, _ = aligned_columns(long_df, field)
        as_sets = [[_labels_of(value) for value in column] for column in columns]
        pair_scores: list[float] = []
        for first, second in itertools.combinations(range(len(MODEL_LABELS)), 2):
            for index in range(len(as_sets[first])):
                score = _jaccard(as_sets[first][index], as_sets[second][index])
                if not np.isnan(score):
                    pair_scores.append(score)
        sizes = {MODEL_LABELS[index]: float(np.mean([len(s) for s in as_sets[index]]))
                 for index in range(len(MODEL_LABELS))}
        all_labels = {label for model_sets in as_sets for labels in model_sets for label in labels}
        shared_by_all = 0
        n_items = len(as_sets[0])
        for index in range(n_items):
            per_model = [as_sets[model][index] for model in range(len(MODEL_LABELS))]
            if all(per_model) and set.intersection(*per_model):
                shared_by_all += 1
        rows.append(
            {
                "field": field,
                "field_label": FIELD_LABELS.get(field, field),
                "mean_pairwise_jaccard": float(np.mean(pair_scores)) if pair_scores else float("nan"),
                "n_comparable_pairs": len(pair_scores),
                "distinct_labels_total": len(all_labels),
                "items_with_a_shared_label": shared_by_all,
                **{f"mean_items_{MODEL_LABELS[index]}": round(sizes[MODEL_LABELS[index]], 2)
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


def _domain_count(row: pd.Series) -> int:
    return int(sum(1 for field in DOMAIN_FIELDS if str(row.get(field, "")) == "yes"))


def _list_size(value: object) -> int:
    return len(_labels_of(value))


def per_model_behavior(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_LABELS:
        subset = long_df[long_df["model_label"] == model]
        typology = subset["provisional_typology"].astype(str)
        rows.append(
            {
                "model_label": model,
                "provider": subset["provider"].iloc[0] if len(subset) else "",
                "candidate_yes_rate": _rate(subset["stage3_candidate"], "yes"),
                "msk_yes_rate": _rate(subset["musculoskeletal_flag"], "yes"),
                "integrative_signal_rate": _rate(typology, "potential integrative signal"),
                "multifactorial_rate": _rate(typology, "multifactorial signal"),
                "bio_rate": _rate(subset["bio_mentioned"], "yes"),
                "psych_rate": _rate(subset["psych_mentioned"], "yes"),
                "social_rate": _rate(subset["social_mentioned"], "yes"),
                "mean_domain_count": float(subset.apply(_domain_count, axis=1).mean()) if len(subset) else float("nan"),
                "mean_concepts": float(subset["psychological_concepts_detected"].map(_list_size).mean())
                if len(subset) else float("nan"),
                "mean_frameworks": float(subset["theoretical_frameworks_detected"].map(_list_size).mean())
                if len(subset) else float("nan"),
                "structured_share": float((subset["coding_method"] == "llm_structured").mean())
                if len(subset) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


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
            if field == "stage3_candidate":
                row["candidate_consensus_depth"] = depth
            if field == "provisional_typology":
                row["typology_consensus_depth"] = depth
        # The union of the open lists across models, which is what a human
        # adjudicator would look at rather than any single model's list.
        for field in LIST_FIELDS:
            union: set[str] = set()
            for value in subset[field].tolist():
                union |= _labels_of(value)
            row[field] = " | ".join(sorted(union))
        row["n_domains_consensus"] = sum(1 for field in DOMAIN_FIELDS if row.get(field) == "yes")
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_depth_distribution(consensus: pd.DataFrame) -> pd.DataFrame:
    """How many models back the majority Stage 3 candidacy decision per item."""
    dist = consensus["candidate_consensus_depth"].value_counts().sort_index()
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
    depth = candidate_depth_distribution(consensus)
    overlap = compute_list_overlap(long_df)

    if write:
        out = reliability_dir()
        write_csv(out / "01_field_reliability.csv", field_rel)
        write_csv(out / "02_pairwise_percent_agreement.csv", agreement.reset_index(names="model_label"))
        write_csv(out / "03_pairwise_cohen_kappa.csv", kappa.reset_index(names="model_label"))
        write_csv(out / "04_per_model_behavior.csv", behavior)
        write_csv(out / "05_consensus_codings.csv", consensus)
        write_csv(out / "06_candidate_depth_distribution.csv", depth)
        write_csv(out / "07_list_overlap.csv", overlap)

    def _mean(column: str) -> float:
        return float(field_rel[column].mean())

    def _field_value(field: str, column: str) -> float | None:
        match = field_rel.loc[field_rel.field == field, column]
        return _clean_for_json(float(match.iloc[0])) if len(match) else None

    key_fields = ["provisional_typology", "musculoskeletal_flag", "bio_mentioned",
                  "psych_mentioned", "social_mentioned", "stage3_candidate"]
    key_rel = {
        field: {
            "fleiss_kappa": _field_value(field, "fleiss_kappa"),
            "krippendorff_alpha": _field_value(field, "krippendorff_alpha"),
            "mean_pairwise_agreement": _field_value(field, "mean_pairwise_agreement"),
        }
        for field in key_fields
    }

    off_diagonal = agreement.where(~np.eye(len(agreement), dtype=bool)).stack()
    n_models = len(MODEL_LABELS)

    summary = {
        "n_abstracts": int(long_df["record_id"].nunique()),
        "n_models": int(long_df["model_label"].nunique()),
        "n_codings": int(len(long_df)),
        "mean_fleiss_kappa": _mean("fleiss_kappa"),
        "mean_krippendorff_alpha": _mean("krippendorff_alpha"),
        "mean_pairwise_agreement": _mean("mean_pairwise_agreement"),
        "mean_unanimous_rate": _mean("unanimous_rate"),
        "key_field_reliability": key_rel,
        "most_agreeing_pair": {"models": list(off_diagonal.idxmax()), "agreement": float(off_diagonal.max())},
        "least_agreeing_pair": {"models": list(off_diagonal.idxmin()), "agreement": float(off_diagonal.min())},
        "candidate_unanimous_items": int((consensus["candidate_consensus_depth"] == n_models).sum()),
        "consensus_candidate_yes": int((consensus["stage3_candidate"] == "yes").sum()),
        "per_model_candidate_yes_rate": {
            row["model_label"]: _clean_for_json(round(float(row["candidate_yes_rate"]), 3))
            for _, row in behavior.iterrows()
        },
        "list_overlap": {
            row["field"]: _clean_for_json(round(float(row["mean_pairwise_jaccard"]), 3))
            for _, row in overlap.iterrows()
        },
    }
    if write:
        write_json(reliability_dir() / "reliability_summary.json", summary)

    return {
        "field_reliability": field_rel,
        "pairwise_agreement": agreement,
        "pairwise_kappa": kappa,
        "per_model_behavior": behavior,
        "consensus": consensus,
        "candidate_depth": depth,
        "list_overlap": overlap,
        "summary": summary,
    }
