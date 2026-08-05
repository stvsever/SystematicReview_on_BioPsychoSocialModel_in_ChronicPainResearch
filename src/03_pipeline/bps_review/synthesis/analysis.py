from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from bps_review.settings import resolve_path
from bps_review.utils.io import append_audit_log, write_csv, write_json


def _ensure_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def _save_barplot(series: pd.Series, title: str, path: Path, color: str = "#2b6777") -> None:
    _ensure_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    series.plot(kind="bar", ax=ax, color=color)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_radar(summary: dict[str, float], path: Path) -> None:
    labels = ["Biological", "Psychological", "Social"]
    values = [summary["semantic_biological"], summary["semantic_psychological"], summary["semantic_social"]]
    values += values[:1]
    angles = [0, 2.09439510239, 4.18879020479]
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, color="#d1495b", linewidth=2)
    ax.fill(angles, values, color="#edae49", alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Mean Semantic Projection onto BPS Axes")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_prisma(counts: dict[str, int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    boxes = [
        (0.5, 0.88, f"Records identified\n(n = {counts['identified']})"),
        (0.5, 0.66, f"Records after deduplication\n(n = {counts['deduped']})"),
        (0.5, 0.44, f"Stage 1 included or maybe\n(n = {counts['stage1_included']})"),
        (0.5, 0.22, f"Stage 2 coded reviews\n(n = {counts['stage2_coded']})"),
    ]
    for x, y, label in boxes:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=14,
            bbox={"boxstyle": "round,pad=0.6", "fc": "#f7f3e9", "ec": "#2b6777", "lw": 2},
        )
    for start, end in [(0.82, 0.72), (0.60, 0.50), (0.38, 0.28)]:
        ax.annotate("", xy=(0.5, end), xytext=(0.5, start), arrowprops={"arrowstyle": "->", "lw": 2})
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_concept_network(coded: pd.DataFrame, path: Path) -> None:
    graph = nx.Graph()
    for concepts in coded["psychological_concepts"].fillna(""):
        items = [item.strip() for item in concepts.split(";") if item.strip()]
        for item in items:
            graph.add_node(item)
        for i, source in enumerate(items):
            for target in items[i + 1 :]:
                if graph.has_edge(source, target):
                    graph[source][target]["weight"] += 1
                else:
                    graph.add_edge(source, target, weight=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No concept network available", ha="center", va="center")
    else:
        position = nx.spring_layout(graph, seed=7, k=0.8)
        weights = [graph[u][v]["weight"] for u, v in graph.edges()]
        nx.draw_networkx(
            graph,
            position,
            ax=ax,
            node_color="#669bbc",
            edge_color="#b08968",
            width=[0.5 + weight for weight in weights],
            font_size=10,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def synthesize(
    raw_pubmed: pd.DataFrame,
    deduped: pd.DataFrame,
    screened: pd.DataFrame,
    coded: pd.DataFrame,
    llm_clusters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tables_dir = resolve_path("processed_tables")
    figures_dir = resolve_path("processed_figures")

    counts = {
        "identified": int(raw_pubmed.shape[0]),
        "deduped": int(deduped.shape[0]),
        "stage1_included": int(screened["stage1_decision"].isin(["include", "maybe"]).sum()),
        "stage2_coded": int(coded.shape[0]),
    }

    review_type_counts = coded["review_type"].value_counts().rename_axis("review_type").reset_index(name="count")
    icd11_counts = coded["icd11_pain_category"].value_counts().rename_axis("icd11_pain_category").reset_index(name="count")
    bps_function_counts = coded["bps_function"].value_counts().rename_axis("bps_function").reset_index(name="count")
    typology_counts = coded["bps_typology"].value_counts().rename_axis("bps_typology").reset_index(name="count")
    balance_counts = coded["overall_balance"].value_counts().rename_axis("overall_balance").reset_index(name="count")

    concept_counter: Counter[str] = Counter()
    for entry in coded["psychological_concepts"].fillna(""):
        for concept in [item.strip() for item in entry.split(";") if item.strip()]:
            concept_counter[concept] += 1
    top_concepts = pd.DataFrame(concept_counter.most_common(15), columns=["concept", "count"])

    domain_means = {
        "semantic_biological": float(coded["semantic_biological"].mean()) if not coded.empty else 0.0,
        "semantic_psychological": float(coded["semantic_psychological"].mean()) if not coded.empty else 0.0,
        "semantic_social": float(coded["semantic_social"].mean()) if not coded.empty else 0.0,
    }

    write_csv(tables_dir / "review_type_counts.csv", review_type_counts)
    write_csv(tables_dir / "icd11_counts.csv", icd11_counts)
    write_csv(tables_dir / "bps_function_counts.csv", bps_function_counts)
    write_csv(tables_dir / "typology_counts.csv", typology_counts)
    write_csv(tables_dir / "balance_counts.csv", balance_counts)
    write_csv(tables_dir / "top_concepts.csv", top_concepts)

    _save_prisma(counts, figures_dir / "prisma_flow.png")
    _save_barplot(review_type_counts.set_index("review_type")["count"], "Review Types", figures_dir / "review_types.png")
    _save_barplot(typology_counts.set_index("bps_typology")["count"], "BPS Operationalization Typology", figures_dir / "typology.png", color="#ef476f")
    _save_radar(domain_means, figures_dir / "semantic_radar.png")
    _save_concept_network(coded, figures_dir / "concept_network.png")

    summary = {
        "counts": counts,
        "domain_means": domain_means,
        "top_review_type": review_type_counts.iloc[0].to_dict() if not review_type_counts.empty else {},
        "top_typology": typology_counts.iloc[0].to_dict() if not typology_counts.empty else {},
        "top_balance": balance_counts.iloc[0].to_dict() if not balance_counts.empty else {},
        "top_concepts": top_concepts.head(10).to_dict(orient="records"),
        "llm_clusters": llm_clusters or {},
    }
    write_json(resolve_path("processed_manuscript") / "summary_metrics.json", summary)
    append_audit_log(
        resolve_path("audit_trail") / "synthesis_log.jsonl",
        {
            "stage": "synthesis",
            "counts": counts,
            "top_typology": summary["top_typology"],
        },
    )
    return summary
