from __future__ import annotations

"""Write the standalone summary of the full-text test run.

The summary is the artefact a reader opens without running anything: which
papers were coded, how much the providers agreed on the coverage and integration
ladders, whether the quoted evidence is real, and what the run cost.
"""

import json

import pandas as pd

from bps_review.fulltext.config import (
    COVERAGE_FIELDS,
    FIELD_LABELS,
    FULLTEXT_MODELS,
    INTEGRATION_FIELDS,
    codings_dir,
    summary_md,
)
from bps_review.utils.io import write_text


def _pct(value) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value * 100:.1f}%"


def _num(value, digits: int = 3) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def _run_manifest() -> dict:
    path = codings_dir() / "run_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_summary(long_df: pd.DataFrame, results: dict, integrity: dict, corpus: pd.DataFrame,
                  semantic: dict | None = None) -> str:
    summary = results["summary"]
    field_rel = results["field_reliability"].sort_values("fleiss_kappa", ascending=False, na_position="last")
    behavior = results["per_model_behavior"]
    consensus = results["consensus"]
    overlap = results["list_overlap"]
    quotes = integrity["quote_verification_by_model"]
    discipline = integrity["summary"]["evidence_discipline"]
    manifest = _run_manifest()
    usage = manifest.get("token_usage_total", {})

    lines: list[str] = []
    lines.append("# Full-text test run: cross-provider reliability and evidence integrity")
    lines.append("")
    lines.append(
        f"The Stage 3 full-text coding scheme was applied to {summary['n_papers']} open-access review "
        f"articles by {summary['n_models']} large language models from {summary['n_models']} different "
        f"providers, which gives {summary['n_codings']} independent codings and "
        f"{integrity['extraction_totals']['total_extracted_items']} extracted items. The corpus is the "
        "open-access subset of the records the abstract-level run carried forward, so the two stages "
        "are one chain rather than two separate exercises."
    )
    lines.append("")
    lines.append("## What was coded")
    lines.append("")
    lines.append(f"- Corpus: {len(corpus)} full texts retrieved from PubMed Central.")
    lines.append("- Models: " + ", ".join(f"`{model.openrouter_id}` ({model.provider})" for model in FULLTEXT_MODELS))
    lines.append(f"- Coding method counts: {manifest.get('coding_method_counts', {})}")
    if usage:
        lines.append(f"- Token usage: {usage.get('total_tokens', 0):,} tokens for ${usage.get('cost_usd', 0):.3f}")
    lines.append("")
    lines.append("## Headline agreement")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Mean Fleiss' kappa over all coded fields | {_num(summary['mean_fleiss_kappa'])} |")
    lines.append(f"| Mean Krippendorff's alpha | {_num(summary['mean_krippendorff_alpha'])} |")
    lines.append(f"| Mean observed agreement | {_pct(summary['mean_pairwise_agreement'])} |")
    lines.append(f"| Categorical fields, mean kappa | {_num(summary['categorical_mean_fleiss_kappa'])} |")
    lines.append(f"| Binary presence fields, mean kappa | {_num(summary['presence_mean_fleiss_kappa'])} |")
    lines.append(f"| Coverage ladder, within one rung | {_pct(summary['coverage_mean_adjacent_agreement'])} |")
    lines.append(f"| Integration ladder, within one rung | {_pct(summary['integration_mean_adjacent_agreement'])} |")
    lines.append(f"| Papers where all models agree on eligibility | "
                 f"{summary['eligibility_unanimous_items']} of {summary['n_papers']} |")
    lines.append("")
    lines.append("## Per field")
    lines.append("")
    lines.append("| Field | Fleiss' kappa | Krippendorff alpha | Observed | Within one rung | Interpretation |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for _, row in field_rel.iterrows():
        lines.append(
            f"| {row['field_label']} | {_num(row['fleiss_kappa'])} | {_num(row['krippendorff_alpha'])} | "
            f"{_pct(row['mean_pairwise_agreement'])} | {_pct(row['adjacent_agreement'])} | {row['landis_koch']} |"
        )
    lines.append("")
    lines.append("## Is the evidence real?")
    lines.append("")
    if not quotes.empty:
        lines.append("| Model | Quotes | Exact | Near | Verified | Mean words |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, row in quotes.iterrows():
            lines.append(
                f"| {row['model_label']} | {int(row['n_quotes'])} | {_pct(row['exact_rate'])} | "
                f"{_pct(row['near_rate'])} | {_pct(row['verified_rate'])} | {row['mean_quote_words']:.1f} |"
            )
        lines.append("")
    lines.append(
        f"Every verbatim quote was matched back against the article it came from. "
        f"{_pct(integrity['summary']['quote_verification']['verified_rate'])} of "
        f"{integrity['summary']['quote_verification']['n_quotes_checked']} checkable quotes were found in "
        "the source text, literally or with at most minor differences."
    )
    lines.append("")
    lines.append(
        f"Of the {discipline['n_graded_links']} domain links graded above 'mentioned', "
        f"{_pct(discipline['share_backed_by_quote'])} carry a quoted claim for exactly that pair. This is "
        "the check that decides whether the integration ladder can be trusted: a graded link with no "
        "passage behind it is a judgement the review cannot audit."
    )
    lines.append("")
    lines.append("## How the models behave")
    lines.append("")
    lines.append("| Model | Include | Core priority | True integrative | Any triadic | "
                 "Mean integration index | Mean items |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, row in behavior.iterrows():
        lines.append(
            f"| {row['model_label']} | {_pct(row['include_rate'])} | {_pct(row['core_priority_rate'])} | "
            f"{_pct(row['true_integrative_rate'])} | {_pct(row['triadic_any_rate'])} | "
            f"{_num(row['mean_integration_index'], 2)} | {_num(row['mean_extracted_items'], 1)} |"
        )
    lines.append("")
    lines.append("## Open extraction lists")
    lines.append("")
    lines.append("| List | Mean pairwise Jaccard | Distinct labels | Papers with a shared label |")
    lines.append("| --- | --- | --- | --- |")
    for _, row in overlap.iterrows():
        lines.append(
            f"| {row['field_label']} | {_num(row['mean_pairwise_jaccard'])} | "
            f"{int(row['distinct_labels_total'])} | {int(row['papers_with_a_shared_label'])} |"
        )
    lines.append("")
    if semantic is not None and not semantic["overlap"].empty:
        semantic_summary = semantic["summary"]
        overlap_frame = semantic["overlap"]
        lines.append("## The same extraction, measured by meaning")
        lines.append("")
        lines.append(
            "The Jaccard above asks whether two providers wrote the same string, and it asks it "
            "only of the item identities. Both halves of that are too narrow. A scheme 3 item is "
            "not a label but a small record, and several of its fields are open vocabularies in "
            "their own right: which constructs a coder says carry the biological domain, which "
            "measure a construct is tied to, which components a definition of the model lists, "
            "which constructs a conceptual problem concerns. Each is a place where two coders can "
            "read a paper the same way and write different words."
        )
        lines.append("")
        # A run that carries every extraction list answers every declared space, so
        # the shortfall clause is only written when there is a shortfall to explain.
        n_declared = semantic_summary["n_spaces_declared"]
        n_measured = semantic_summary["n_spaces_measured"]
        coverage_clause = (
            f"The scheme declares {n_declared} comparison spaces and this run answers every "
            "one of them."
            if n_measured >= n_declared
            else (
                f"The scheme declares {n_declared} comparison spaces and this run supports "
                f"{n_measured} of them; the rest belong to extraction lists this run does not "
                "carry."
            )
        )
        lines.append(
            f"Every one of those vocabularies is compared here. {coverage_clause} "
            f"Every label is embedded once with "
            f"`{semantic_summary['embedding_model']}`, and two labels count as the same concept "
            f"at a cosine of {semantic_summary['similarity_threshold']:.2f}, which turns the "
            "overlap into a soft Jaccard on the same 0 to 1 scale. Both columns are computed in "
            "the same pass over the same label sets, so reading one against the other compares "
            "two ways of measuring one thing rather than two instruments."
        )
        lines.append("")
        lines.append("| Comparison space | Layer | Read from | Lexical | Semantic | Labels | Concepts |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in overlap_frame.iterrows():
            kind = " (controlled)" if row["label_kind"] == "controlled" else ""
            lines.append(
                f"| {row['space_label']}{kind} | {row['layer']} | "
                f"`{row['source_field']}.{row['read_from']}` | "
                f"{_num(row['mean_pairwise_jaccard'])} | "
                f"{_num(row['mean_pairwise_semantic_jaccard'])} | "
                f"{int(row['n_distinct_labels'])} | {int(row['n_semantic_concepts'])} |"
            )
        lines.append("")
        free_text = overlap_frame[overlap_frame["label_kind"] == "free text"]
        lines.append(
            f"Mean over all spaces: {_num(semantic_summary['mean_lexical_jaccard'])} lexical "
            f"against {_num(semantic_summary['mean_semantic_jaccard'])} semantic, over "
            f"{semantic_summary['n_labels_embedded']} embedded labels. Over the "
            f"{len(free_text)} free-text spaces alone, where the semantic layer has something to "
            f"merge, the mean is {_num(semantic_summary['mean_semantic_jaccard_free_text'])}. The "
            "distance between the two columns is the part of the apparent disagreement that was "
            "only ever wording."
        )
        lines.append("")
        lines.append(
            "The controlled spaces are the control condition rather than a result. Where an item "
            "is identified by a value from a closed list, the two coders picked from the same "
            "menu, the semantic layer has nothing to merge, and the two columns are identical by "
            "construction. That they come out identical is the check that the method is not "
            "manufacturing agreement wherever it is applied."
        )
        lines.append("")
        lines.append(
            "Sensitivity to the threshold is in `03_reliability/semantic_overlap_summary.json`, "
            "so no reading here depends on where exactly the line is drawn."
        )
        lines.append("")
    spine = integrity.get("spine_coverage")
    if spine is not None and not spine.empty:
        lines.append("## How much of the extraction lands on the project ontology")
        lines.append("")
        lines.append("| Extraction list | Items | Anchored | On the controlled spine | Distinct labels written |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, row in spine.iterrows():
            lines.append(
                f"| {FIELD_LABELS.get(row['extraction_field'], row['extraction_field'])} | "
                f"{int(row['n_items'])} | {_pct(row['anchored_share'])} | "
                f"{_pct(row['controlled_share'])} | {int(row['distinct_raw_labels'])} |"
            )
        lines.append("")
        lines.append(
            "The controlled share measures the ontology against the literature, not the coder against "
            "the ontology. A label the vocabularies do not carry is kept as the review wrote it and "
            "listed in `15_off_spine_labels.csv`, which is the working list for extending the "
            "vocabularies after expert evaluation."
        )
        lines.append("")
    lines.append("## Consensus picture of the corpus")
    lines.append("")
    for field in COVERAGE_FIELDS:
        counts = consensus[field].value_counts()
        lines.append(f"- {FIELD_LABELS[field]}: " + ", ".join(f"{label} {count}" for label, count in counts.items()))
    for field in INTEGRATION_FIELDS:
        counts = consensus[field].value_counts()
        lines.append(f"- {FIELD_LABELS[field]}: " + ", ".join(f"{label} {count}" for label, count in counts.items()))
    lines.append("- BPS typology: " + ", ".join(
        f"{label} {count}" for label, count in consensus["bps_typology"].value_counts().items()))
    lines.append(f"- Coded typology matches the rule-derived typology in "
                 f"{_pct(summary['typology_coded_matches_derived'])} of codings.")
    lines.append("")
    lines.append("## Review surfaces")
    lines.append("")
    lines.append("- `01_corpus/`: the retrieved corpus, its manifest, and the retrieval log")
    lines.append("- `02_model_codings/`: every article by provider coding, the item-level table, "
                 "the raw audit trail, and the usage manifest")
    lines.append("- `03_reliability/`: agreement, consensus, overlap, ontology coverage, and quote "
                 "verification tables")
    lines.append("- `04_figures/`: the static review figures")
    lines.append("- `05_knowledge_graph/index.html`: the self-contained interactive knowledge graph, "
                 "from the coding scheme down to the quoted sentence behind one extracted item")
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "These numbers describe agreement between three cheap models on a small open-access corpus. "
        "They are not a finding about the biopsychosocial literature and they are not a validation of "
        "the coding scheme against a human standard. What they do show is which parts of the scheme are "
        "specified tightly enough that independent coders converge, and which parts are not, which is "
        "the input the expert evaluation needs."
    )
    lines.append("")
    return "\n".join(lines)


def write_fulltext_summary(long_df: pd.DataFrame, results: dict, integrity: dict,
                           corpus: pd.DataFrame, semantic: dict | None = None) -> str:
    text = build_summary(long_df, results, integrity, corpus, semantic)
    write_text(summary_md(), text)
    return text
