from __future__ import annotations

"""Write the standalone summary of the abstract-level test run.

The summary is the artefact a reader opens without running anything: what was
coded, by whom, how much the providers agreed, where they diverged, and what the
run costs. It is written next to the data it describes.
"""

import json

import pandas as pd

from bps_review.pilot.config import (
    DERIVED_FIELDS,
    DOMAIN_FIELDS,
    FIELD_LABELS,
    LIST_FIELDS,
    MODEL_LABELS,
    TESTRUN_MODELS,
    codings_dir,
    summary_md,
)
from bps_review.utils.io import write_text


def _pct(value: float) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value * 100:.1f}%"


def _num(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def _run_manifest() -> dict:
    path = codings_dir() / "run_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(long_df: pd.DataFrame, results: dict, corpus: pd.DataFrame) -> str:
    summary = results["summary"]
    field_rel = results["field_reliability"].sort_values("fleiss_kappa", ascending=False, na_position="last")
    behavior = results["per_model_behavior"]
    consensus = results["consensus"]
    overlap = results["list_overlap"]
    manifest = _run_manifest()
    usage = manifest.get("token_usage_total", {})

    lines: list[str] = []
    lines.append("# Abstract-level test run: cross-provider reliability")
    lines.append("")
    lines.append(
        f"The Stage 2 abstract coding scheme was applied to {summary['n_abstracts']} PubMed records "
        f"by {summary['n_models']} large language models from {summary['n_models']} different providers, "
        f"which gives {summary['n_codings']} independent codings from "
        f"{manifest.get('n_api_calls', 'n/a')} API calls. The models act as independent raters, so the run "
        "is a cross-provider inter-rater reliability check on the coding scheme and on the code that "
        "applies it."
    )
    lines.append("")
    lines.append("## What was coded")
    lines.append("")
    lines.append(f"- Corpus: {len(corpus)} records from the operational PubMed query, most recent first, "
                 "each with a usable abstract.")
    lines.append("- Models: " + ", ".join(f"`{model.openrouter_id}` ({model.provider})" for model in TESTRUN_MODELS))
    lines.append(f"- Coding method counts: {manifest.get('coding_method_counts', {})}")
    if usage:
        lines.append(f"- Token usage: {usage.get('total_tokens', 0):,} tokens for "
                     f"${usage.get('cost_usd', 0):.3f}")
    lines.append("")
    lines.append("## Headline agreement")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Mean Fleiss' kappa over all coded fields | {_num(summary['mean_fleiss_kappa'])} |")
    lines.append(f"| Mean Krippendorff's alpha | {_num(summary['mean_krippendorff_alpha'])} |")
    lines.append(f"| Mean observed agreement | {_pct(summary['mean_pairwise_agreement'])} |")
    lines.append(f"| Mean unanimous rate | {_pct(summary['mean_unanimous_rate'])} |")
    lines.append(f"| Abstracts where all models agree on Stage 3 candidacy | "
                 f"{summary['candidate_unanimous_items']} of {summary['n_abstracts']} |")
    lines.append("")
    lines.append("## Per field")
    lines.append("")
    lines.append("| Field | Fleiss' kappa | Krippendorff alpha | Observed agreement | Unanimous | Interpretation |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for _, row in field_rel.iterrows():
        lines.append(
            f"| {row['field_label']} | {_num(row['fleiss_kappa'])} | {_num(row['krippendorff_alpha'])} | "
            f"{_pct(row['mean_pairwise_agreement'])} | {_pct(row['unanimous_rate'])} | {row['landis_koch']} |"
        )
    lines.append("")
    lines.append(
        "Three of these fields are derived rather than asked: "
        + ", ".join(f"`{field}`" for field in DERIVED_FIELDS)
        + ". They are computed from the coded content by a fixed rule, so their agreement is the agreement "
        "of the judgements they read, not an independent judgement."
    )
    lines.append("")
    lines.append("## Open extraction lists")
    lines.append("")
    lines.append("Agreement on an open list is measured with set overlap, because two coders can both be "
                 "right and still return different strings.")
    lines.append("")
    lines.append("| List | Mean pairwise Jaccard | Distinct labels | Items with a label all models share |")
    lines.append("| --- | --- | --- | --- |")
    for _, row in overlap.iterrows():
        lines.append(
            f"| {row['field_label']} | {_num(row['mean_pairwise_jaccard'])} | "
            f"{int(row['distinct_labels_total'])} | {int(row['items_with_a_shared_label'])} |"
        )
    lines.append("")
    lines.append("## How the models behave")
    lines.append("")
    lines.append("| Model | Provider | Stage 3 candidate | Musculoskeletal yes | Integrative signal | "
                 "Mean domains | Mean concepts |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, row in behavior.iterrows():
        lines.append(
            f"| {row['model_label']} | {row['provider']} | {_pct(row['candidate_yes_rate'])} | "
            f"{_pct(row['msk_yes_rate'])} | {_pct(row['integrative_signal_rate'])} | "
            f"{_num(row['mean_domain_count'], 2)} | {_num(row['mean_concepts'], 2)} |"
        )
    lines.append("")
    most = summary["most_agreeing_pair"]
    least = summary["least_agreeing_pair"]
    lines.append(f"The closest pair is {most['models'][0]} and {most['models'][1]} "
                 f"({_pct(most['agreement'])} observed agreement); the most distant pair is "
                 f"{least['models'][0]} and {least['models'][1]} ({_pct(least['agreement'])}).")
    lines.append("")
    lines.append("## Consensus picture of the corpus")
    lines.append("")
    for field in DOMAIN_FIELDS:
        present = int((consensus[field] == "yes").sum())
        lines.append(f"- {FIELD_LABELS[field]}: {present} of {len(consensus)} abstracts "
                     f"({_pct(present / len(consensus))}).")
    typology_counts = consensus["provisional_typology"].value_counts()
    lines.append("- Provisional typology: " + ", ".join(
        f"{label} {count}" for label, count in typology_counts.items()))
    lines.append(f"- Stage 3 candidates by majority vote: {summary['consensus_candidate_yes']} "
                 f"of {summary['n_abstracts']}.")
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "These numbers describe agreement between three cheap models on a test corpus. They are not a "
        "finding about the biopsychosocial literature and they are not a validation of the coding scheme "
        "against a human standard. What they do show is where the scheme is specified tightly enough that "
        "independent coders converge, and where it is not, which is exactly the input the expert "
        "evaluation of the coding schemes needs."
    )
    lines.append("")
    return "\n".join(lines)


def write_summary(long_df: pd.DataFrame, results: dict, corpus: pd.DataFrame) -> str:
    text = build_summary(long_df, results, corpus)
    write_text(summary_md(), text)
    return text
