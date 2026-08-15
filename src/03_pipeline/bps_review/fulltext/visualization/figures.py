from __future__ import annotations

"""Multi-panel figures for the cross-provider full-text test run.

Clustered figures in the same restrained academic style as the abstract-level
run. No figure carries a figure-level title: each panel states what it shows, and
a title above them all would only repeat the caption.

* ``01_field_reliability_2x2.png``  - kappa, alpha, observed agreement, and the
  adjacent-agreement rate on the ordered ladders;
* ``02_pairwise_and_consensus.png`` - model-by-model agreement, provider
  centrality, and the consensus depth of the eligibility call;
* ``03_integration_profile.png``    - the coverage and integration ladders per
  model, and the corpus-level integration index;
* ``04_evidence_and_yield.png``     - extraction volume, quote verification,
  evidence discipline, and item-identity overlap. Both the volume panel and the
  overlap panel cover every extraction list the scheme defines, and skip the ones
  a given run does not carry, so neither goes stale when the scheme grows;
* ``05_semantic_overlap.png``       - every comparable vocabulary of the scheme
  measured lexically and semantically, and how far each label vocabulary
  collapses once wording variants are merged. Written only when the semantic
  layer ran.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bps_review.fulltext.config import (
    COVERAGE_FIELDS,
    COVERAGE_ORDER,
    FIELD_LABELS,
    INTEGRATION_FIELDS,
    MODEL_LABELS,
    PAIRWISE_ORDER,
    TRIADIC_ORDER,
    TYPOLOGY_ORDER,
    figures_dir,
)
from bps_review.utils.io import ensure_parent


INK = "#1f2733"
MUTED = "#8a94a6"
GRID = "#e4e7ec"
PALETTE = {
    "primary": "#2f4b7c",
    "accent": "#d1495b",
    "teal": "#0f8f80",
    "amber": "#e0a33a",
    "violet": "#6d5ae0",
    "slate": "#8a94a6",
}
DOMAIN_COLORS = {
    "domain_coverage_bio": "#0E8F80",
    "domain_coverage_psych": "#6D5AE0",
    "domain_coverage_social": "#D98016",
}
MODEL_COLORS = {
    "DeepSeek-V4-Flash": "#2f4b7c",
    "Nex-N2-Mini": "#0f8f80",
    "Laguna-XS-2.1": "#6d5ae0",
}
KAPPA_BAND_COLORS = [
    (0.20, "#c85b6b"),
    (0.40, "#e0a33a"),
    (0.60, "#4fb0a3"),
    (0.80, "#3f79b0"),
    (1.01, "#2f4b7c"),
]
# The integration ladder, dark for a real mechanism down to pale for nothing.
LADDER_COLORS = ["#1f3d63", "#2f4b7c", "#5f7bb0", "#9fb0cf", "#dfe3e8"]

# Every extraction list of the scheme, in the order the extraction-volume panel
# reads best, with labels short enough for a legend. A list a run does not carry
# is skipped at plot time rather than left out here, so the panel keeps up with
# the scheme on its own.
EXTRACTION_YIELD_ORDER = [
    "biological_factors",
    "psychological_concepts",
    "social_factors",
    "other_domain_factors",
    "domain_evidence",
    "integration_claims",
    "concept_relations",
    "theoretical_frameworks",
    "instruments",
    "conceptual_problems",
    "bps_usage_instances",
    "bps_definitions",
    "key_quotes",
]
# One colour per extraction list, so a list keeps its colour between runs and no
# two series in the legend look alike. Cycling a six-colour palette over thirteen
# lists made "bio factors" and "concept relations" the same blue.
EXTRACTION_COLORS = {
    "biological_factors": "#0e8f80",
    "psychological_concepts": "#6d5ae0",
    "social_factors": "#d98016",
    "other_domain_factors": "#7fae4a",
    "domain_evidence": "#2f4b7c",
    "integration_claims": "#d1495b",
    "concept_relations": "#a8809f",
    "theoretical_frameworks": "#e0a33a",
    "instruments": "#4fb0a3",
    "conceptual_problems": "#b0424f",
    "bps_usage_instances": "#5f7bb0",
    "bps_definitions": "#8a94a6",
    "key_quotes": "#3f79b0",
}
EXTRACTION_SHORT_LABELS = {
    "biological_factors": "bio factors",
    "psychological_concepts": "concepts",
    "social_factors": "social factors",
    "other_domain_factors": "beyond-triad factors",
    "domain_evidence": "domain evidence",
    "integration_claims": "integration claims",
    "concept_relations": "concept relations",
    "theoretical_frameworks": "frameworks",
    "instruments": "instruments",
    "conceptual_problems": "problems",
    "bps_usage_instances": "BPS usages",
    "bps_definitions": "BPS definitions",
    "key_quotes": "key quotes",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.labelsize": 9.5,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.titlesize": 13,
            "figure.titleweight": "bold",
        }
    )


def _despine(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def _panel_label(ax, letter: str) -> None:
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="right", color=INK)


def _kappa_color(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return GRID
    if value < 0:
        return "#b0424f"
    for upper, color in KAPPA_BAND_COLORS:
        if value < upper:
            return color
    return "#2f4b7c"


def _metric_hbar(ax, labels, values, colors, title, xmax=1.0, refs=None, xlabel="Coefficient") -> None:
    y = list(range(len(labels)))
    plot_values = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in values]
    bars = ax.barh(y, plot_values, color=colors, edgecolor="white", height=0.74)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(min(0, min(plot_values) - 0.02), xmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    _despine(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if refs:
        for ref in refs:
            ax.axvline(ref, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), alpha=0.7)
    for bar, raw in zip(bars, values):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            ax.text(0.01, bar.get_y() + bar.get_height() / 2, "n/a", va="center", ha="left",
                    fontsize=7.5, color=MUTED)
        else:
            ax.text(bar.get_width() + 0.012 * xmax, bar.get_y() + bar.get_height() / 2,
                    f"{raw:.2f}", va="center", ha="left", fontsize=7.8, color=INK)


def _heatmap(ax, matrix: pd.DataFrame, title, cmap, vmin, vmax) -> None:
    data = matrix.values.astype(float)
    image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=25, ha="right", fontsize=8.2)
    ax.set_yticklabels(matrix.index, fontsize=8.2)
    ax.set_title(title)
    threshold = vmin + 0.6 * (vmax - vmin)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if value >= threshold else INK)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.ax.tick_params(labelsize=7)


def _stacked_by_model(ax, long_df, field, order, colors, title, xlabel="Proportion of papers") -> None:
    proportions = []
    for model in MODEL_LABELS:
        subset = long_df[long_df["model_label"] == model][field].astype(str)
        counts = subset.value_counts(normalize=True)
        proportions.append([counts.get(category, 0.0) for category in order])
    proportions = np.array(proportions)
    y = list(range(len(MODEL_LABELS)))
    left = np.zeros(len(MODEL_LABELS))
    for index, category in enumerate(order):
        ax.barh(y, proportions[:, index], left=left, color=colors[index % len(colors)], height=0.7,
                label=category, edgecolor="white")
        left += proportions[:, index]
    ax.set_yticks(y)
    ax.set_yticklabels(MODEL_LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    _despine(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(len(order), 3),
              frameon=False, fontsize=7.2)


# --------------------------------------------------------------------------
# Figure 1: field reliability
# --------------------------------------------------------------------------
def fig_field_reliability(field_rel: pd.DataFrame, out_path: Path) -> None:
    order = field_rel.sort_values("fleiss_kappa", ascending=False, na_position="last")
    labels = order["field_label"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))

    _metric_hbar(axes[0, 0], labels, order["fleiss_kappa"].tolist(),
                 [_kappa_color(value) for value in order["fleiss_kappa"]],
                 f"Fleiss' kappa (chance-corrected, {len(MODEL_LABELS)} raters)", refs=[0.2, 0.4, 0.6, 0.8])
    _panel_label(axes[0, 0], "A")

    _metric_hbar(axes[0, 1], labels, order["krippendorff_alpha"].tolist(),
                 [_kappa_color(value) for value in order["krippendorff_alpha"]],
                 "Krippendorff's alpha (nominal)", refs=[0.667, 0.8])
    _panel_label(axes[0, 1], "B")

    _metric_hbar(axes[1, 0], labels, order["mean_pairwise_agreement"].tolist(),
                 [PALETTE["teal"]] * len(labels),
                 "Observed agreement (mean over model pairs)", refs=[0.5], xlabel="Proportion")
    _panel_label(axes[1, 0], "C")

    ladder = order[order["adjacent_agreement"].notna()]
    _metric_hbar(axes[1, 1], ladder["field_label"].tolist(), ladder["adjacent_agreement"].tolist(),
                 [PALETTE["amber"]] * len(ladder),
                 "Within one rung (ordered ladders only)", xlabel="Proportion of coder pairs")
    _panel_label(axes[1, 1], "D")

    fig.tight_layout()
    ensure_parent(out_path)
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2: pairwise agreement and consensus depth
# --------------------------------------------------------------------------
def fig_pairwise(agreement: pd.DataFrame, kappa: pd.DataFrame, depth: pd.DataFrame,
                 consensus: pd.DataFrame, out_path: Path) -> None:
    n_models = len(agreement)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))

    off_values = agreement.where(~np.eye(n_models, dtype=bool)).stack()
    floor = min(0.5, float(np.floor(off_values.min() * 20) / 20))
    _heatmap(axes[0, 0], agreement, "Observed agreement (mean over fields)", "GnBu", vmin=floor, vmax=1.0)
    _panel_label(axes[0, 0], "A")

    _heatmap(axes[0, 1], kappa, "Cohen's kappa (mean over fields)", "YlGnBu", vmin=0.0, vmax=1.0)
    _panel_label(axes[0, 1], "B")

    ax = axes[1, 0]
    off = agreement.where(~np.eye(n_models, dtype=bool))
    centrality = off.mean(axis=1).sort_values(ascending=False)
    y = list(range(len(centrality)))
    ax.barh(y, centrality.values, color=[MODEL_COLORS.get(m, PALETTE["primary"]) for m in centrality.index],
            edgecolor="white", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(centrality.index)
    ax.invert_yaxis()
    ax.set_xlim(max(0.0, min(0.5, float(np.nanmin(centrality.values)) - 0.05)), 1.0)
    ax.set_xlabel("Mean agreement with the other models")
    ax.set_title("Provider centrality (higher = more typical)")
    _despine(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for position, value in zip(y, centrality.values):
        ax.text(value + 0.005, position, f"{value:.2f}", va="center", ha="left", fontsize=8, color=INK)
    _panel_label(ax, "C")

    ax = axes[1, 1]
    counts = consensus["bps_typology"].value_counts().reindex(TYPOLOGY_ORDER).fillna(0)
    x = list(range(len(counts)))
    colors = LADDER_COLORS + [MUTED]
    bars = ax.bar(x, counts.values, color=colors[: len(counts)], edgecolor="white", width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", " ") for label in counts.index], fontsize=8, rotation=18, ha="right")
    ax.set_ylabel("Papers")
    ax.set_title("Consensus BPS typology")
    _despine(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, counts.values):
        if value:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, str(int(value)),
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    _panel_label(ax, "D")

    fig.tight_layout()
    ensure_parent(out_path)
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3: coverage and integration
# --------------------------------------------------------------------------
def fig_integration(long_df: pd.DataFrame, consensus: pd.DataFrame, behavior: pd.DataFrame,
                    out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))

    # Panel A: consensus coverage ladder per domain.
    ax = axes[0, 0]
    y = np.arange(len(COVERAGE_FIELDS))
    left = np.zeros(len(COVERAGE_FIELDS))
    shades = ["#1f3d63", "#5f7bb0", "#9fb0cf", "#dfe3e8"]
    for index, level in enumerate(COVERAGE_ORDER):
        values = [float((consensus[field] == level).mean()) for field in COVERAGE_FIELDS]
        ax.barh(y, values, left=left, color=shades[index], height=0.62, label=level, edgecolor="white")
        left += np.array(values)
    ax.set_yticks(y)
    ax.set_yticklabels([FIELD_LABELS[field].replace(" coverage", "") for field in COVERAGE_FIELDS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of papers (consensus)")
    ax.set_title("Consensus domain coverage")
    _despine(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False, fontsize=7.4)
    _panel_label(ax, "A")

    # Panel B: consensus integration ladder per pair, plus the triadic field.
    ax = axes[0, 1]
    fields = list(INTEGRATION_FIELDS)
    y = np.arange(len(fields))
    left = np.zeros(len(fields))
    for index, level in enumerate(PAIRWISE_ORDER):
        values = [float((consensus[field] == level).mean()) for field in fields]
        ax.barh(y, values, left=left, color=LADDER_COLORS[index], height=0.62, label=level, edgecolor="white")
        left += np.array(values)
    ax.set_yticks(y)
    ax.set_yticklabels([FIELD_LABELS[field].replace(" integration", "") for field in fields])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of papers (consensus)")
    ax.set_title("Consensus integration ladder")
    _despine(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, frameon=False, fontsize=7.2)
    _panel_label(ax, "B")

    # Panel C: triadic integration profile per model.
    _stacked_by_model(axes[1, 0], long_df, "integration_triadic", TRIADIC_ORDER,
                      LADDER_COLORS, "Triadic integration by model")
    _panel_label(axes[1, 0], "C")

    # Panel D: the integration index, per paper, per model.
    ax = axes[1, 1]
    data = [
        pd.to_numeric(long_df[long_df["model_label"] == model]["integration_index"], errors="coerce").dropna()
        for model in MODEL_LABELS
    ]
    parts = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": INK, "linewidth": 1.4})
    for patch, model in zip(parts["boxes"], MODEL_LABELS):
        patch.set_facecolor(MODEL_COLORS.get(model, PALETTE["primary"]))
        patch.set_alpha(0.75)
        patch.set_edgecolor("white")
    ax.set_xticks(range(1, len(MODEL_LABELS) + 1))
    ax.set_xticklabels(MODEL_LABELS, fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Integration index (0 to 1)")
    ax.set_title("How deeply each model reads the integration")
    _despine(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _panel_label(ax, "D")

    fig.tight_layout()
    ensure_parent(out_path)
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 4: evidence and yield
# --------------------------------------------------------------------------
def fig_evidence(yield_table: pd.DataFrame, quotes_by_model: pd.DataFrame,
                 discipline_by_model: pd.DataFrame, overlap: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))

    # Panel A: extraction volume per model, over every list the scheme defines.
    # The columns a run does not carry are skipped, so an older run shows what it
    # holds rather than an empty slot.
    ax = axes[0, 0]
    categories = [(f"mean_{name}", EXTRACTION_SHORT_LABELS.get(name, name.replace("_", " ")))
                  for name in EXTRACTION_YIELD_ORDER]
    available = [(column, label) for column, label in categories
                 if column in yield_table.columns and float(yield_table[column].sum()) > 0]
    x = np.arange(len(yield_table))
    width = 0.86 / max(1, len(available))
    for index, (column, label) in enumerate(available):
        ax.bar(x + (index - (len(available) - 1) / 2) * width, yield_table[column].values, width,
               color=EXTRACTION_COLORS.get(column[len("mean_"):], PALETTE["slate"]),
               edgecolor="white", linewidth=0.4, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(yield_table["model_label"], fontsize=8.5)
    ax.set_ylabel("Mean items per paper")
    ax.set_title(f"Extraction volume ({len(available)} lists)")
    _despine(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=min(4, max(1, len(available))), frameon=False, fontsize=6.4)
    _panel_label(ax, "A")

    # Panel B: quote verification per model.
    ax = axes[0, 1]
    if not quotes_by_model.empty:
        order = quotes_by_model.sort_values("verified_rate", ascending=False)
        y = list(range(len(order)))
        ax.barh(y, order["exact_rate"].values, color=PALETTE["primary"], height=0.62, label="exact")
        ax.barh(y, order["near_rate"].values, left=order["exact_rate"].values, color=PALETTE["teal"],
                height=0.62, label="near")
        remainder = 1 - order["exact_rate"].values - order["near_rate"].values
        ax.barh(y, remainder, left=order["exact_rate"].values + order["near_rate"].values,
                color="#dfe3e8", height=0.62, label="unverified")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{row.model_label} ({int(row.n_quotes)})" for row in order.itertuples()])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of quotes")
        ax.legend(loc="lower left", frameon=False, fontsize=7.4)
        for position, value in zip(y, order["verified_rate"].values):
            ax.text(0.02, position, f"{value:.1%} verified", va="center", ha="left", fontsize=8, color="white")
    ax.set_title("Are the quotes really in the paper?")
    _despine(ax)
    _panel_label(ax, "B")

    # Panel C: evidence discipline.
    ax = axes[1, 0]
    if not discipline_by_model.empty:
        order = discipline_by_model.sort_values("share_backed_by_quote", ascending=False)
        y = list(range(len(order)))
        bars = ax.barh(y, order["share_backed_by_quote"].values,
                       color=[MODEL_COLORS.get(m, PALETTE["primary"]) for m in order["model_label"]],
                       edgecolor="white", height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{row.model_label} ({int(row.n_graded_links)})" for row in order.itertuples()])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of graded links with a quoted claim")
        for bar, value in zip(bars, order["share_backed_by_quote"].values):
            ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.0%}",
                    va="center", ha="left", fontsize=8, color=INK)
    ax.set_title("Is a graded integration backed by evidence?")
    _despine(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _panel_label(ax, "C")

    # Panel D: every extraction list the run carries, by item identity. The lists
    # identified by a value off a closed vocabulary are drawn in a muted colour,
    # because "did two coders pick the same menu entry" is a different question
    # from "did two coders write the same label".
    ax = axes[1, 1]
    order = overlap.sort_values("mean_pairwise_jaccard", ascending=False)
    kinds = (order["label_kind"] if "label_kind" in order.columns
             else pd.Series(["free text"] * len(order), index=order.index))
    y = list(range(len(order)))
    height = 0.82 if len(order) > 8 else 0.6
    bars = ax.barh(y, order["mean_pairwise_jaccard"].values,
                   color=[PALETTE["amber"] if kind == "free text" else "#c7ccd4" for kind in kinds],
                   edgecolor="white", height=height)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{FIELD_LABELS.get(field, field)}{'' if kind == 'free text' else ' (controlled)'}"
         for field, kind in zip(order["field"], kinds)],
        fontsize=7.6 if len(order) > 8 else 8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean pairwise Jaccard overlap")
    ax.set_title(f"Extraction lists by item identity ({len(order)} lists)")
    _despine(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, order["mean_pairwise_jaccard"].values):
        ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
                va="center", ha="left", fontsize=7.4, color=INK)
    _panel_label(ax, "D")

    fig.tight_layout()
    ensure_parent(out_path)
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 5: lexical against semantic overlap
# --------------------------------------------------------------------------
def fig_semantic_overlap(semantic_overlap: pd.DataFrame, summary: dict, out_path: Path) -> None:
    """Every comparable vocabulary of the scheme, measured lexically and by meaning.

    One row per comparison space, not per extraction list. Comparing lists by item
    identity is one question out of several the extraction supports, and the rows
    below the identity layer are the ones that carry the resolution: which
    constructs a coder says carry a domain, which measure a construct is tied to,
    which constructs a conceptual problem concerns.
    """
    frame = semantic_overlap.dropna(subset=["mean_pairwise_semantic_jaccard"]).copy()
    n_rows = max(len(frame), 1)
    height = max(5.4, 0.42 * n_rows + 2.2)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, height))
    threshold = summary.get("similarity_threshold")
    n_models = len(MODEL_LABELS)

    def tick_labels(subset: pd.DataFrame) -> list[str]:
        return [
            f"{row.space_label} (controlled)" if row.label_kind == "controlled" else row.space_label
            for row in subset.itertuples()
        ]

    # Panel A: the same label sets, measured twice.
    ax = axes[0]
    order = frame.sort_values("mean_pairwise_semantic_jaccard", ascending=False)
    y = np.arange(len(order))
    bar = 0.38
    ax.barh(y - bar / 2, order["mean_pairwise_semantic_jaccard"], height=bar,
            color=PALETTE["teal"], edgecolor="white", label="semantic (embeddings)")
    ax.barh(y + bar / 2, order["mean_pairwise_jaccard"], height=bar,
            color=PALETTE["violet"], edgecolor="white", label="lexical (exact string)")
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels(order), fontsize=8)
    ax.invert_yaxis()
    upper = float(order["mean_pairwise_semantic_jaccard"].max()) if len(order) else 0.6
    ax.set_xlim(0, min(1.0, max(0.7, upper + 0.16)))
    rule = f"same concept at cosine >= {threshold:.2f}" if threshold is not None else "same concept"
    ax.set_xlabel(f"Mean pairwise overlap between provider pairs ({rule}, "
                  f"{summary.get('embedding_model', '')})")
    ax.set_title("Do the providers extract the same things?")
    for position, row in zip(y, order.itertuples()):
        ax.text(row.mean_pairwise_semantic_jaccard + 0.008, position - bar / 2,
                f"{row.mean_pairwise_semantic_jaccard:.2f}", va="center", ha="left",
                fontsize=7, color=INK)
        ax.text(row.mean_pairwise_jaccard + 0.008, position + bar / 2,
                f"{row.mean_pairwise_jaccard:.2f}", va="center", ha="left",
                fontsize=7, color=MUTED)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    _despine(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _panel_label(ax, "A")

    # Panel B: how much of each vocabulary is wording rather than concept.
    # A share rather than two counts, because the spaces differ by two orders of
    # magnitude in size and raw bars would render the small ones invisible.
    ax = axes[1]
    collapse = frame.copy()
    collapse["wording_share"] = 1 - (collapse["n_semantic_concepts"] / collapse["n_distinct_labels"])
    collapse = collapse.sort_values("wording_share", ascending=False)
    y = np.arange(len(collapse))
    ax.barh(y, collapse["wording_share"], height=0.62,
            color=[PALETTE["teal"] if kind == "free text" else "#c7ccd4"
                   for kind in collapse["label_kind"]], edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels(collapse), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.45, float(collapse["wording_share"].max()) * 1.35) if len(collapse) else 1)
    ax.set_xlabel(f"Share of distinct labels that are another wording of a label already there\n"
                  f"(all {n_models} providers, all papers)")
    ax.set_title("How much of the vocabulary is wording rather than concept")
    for position, row in zip(y, collapse.itertuples()):
        ax.text(row.wording_share + 0.006, position,
                f"{int(row.n_distinct_labels)} to {int(row.n_semantic_concepts)}",
                va="center", ha="left", fontsize=7, color=INK)
    _despine(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _panel_label(ax, "B")

    # No figure-level title. The matching rule belongs with the axis it applies
    # to, so it is carried by the x-axis label of panel A instead.
    fig.tight_layout()
    ensure_parent(out_path)
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def build_figures(long_df: pd.DataFrame, results: dict, integrity: dict,
                  out_dir: Path | None = None, semantic: dict | None = None) -> list[Path]:
    _apply_style()
    out_dir = out_dir or figures_dir()
    builders = {
        "01_field_reliability_2x2.png": lambda path: fig_field_reliability(results["field_reliability"], path),
        "02_pairwise_and_consensus.png": lambda path: fig_pairwise(
            results["pairwise_agreement"], results["pairwise_kappa"],
            results["eligibility_depth"], results["consensus"], path),
        "03_integration_profile.png": lambda path: fig_integration(
            long_df, results["consensus"], results["per_model_behavior"], path),
        "04_evidence_and_yield.png": lambda path: fig_evidence(
            integrity["extraction_yield"], integrity["quote_verification_by_model"],
            integrity["evidence_discipline_by_model"], results["list_overlap"], path),
    }
    if semantic and not semantic["overlap"].empty:
        builders["05_semantic_overlap.png"] = lambda path: fig_semantic_overlap(
            semantic["overlap"], semantic["summary"], path)
    written = []
    for name, builder in builders.items():
        path = out_dir / name
        builder(path)
        written.append(path)
    return written
