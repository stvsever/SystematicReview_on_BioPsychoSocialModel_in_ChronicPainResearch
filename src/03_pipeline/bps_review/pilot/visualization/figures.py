from __future__ import annotations

"""Multi-panel figures for the cross-provider abstract-level test run.

Four clustered figures, each a 2x2 panel, in the restrained academic style of
the main pipeline's reporting module:

* ``01_field_reliability_2x2.png`` - Fleiss' kappa, Krippendorff's alpha,
  observed agreement, and unanimous rate, per coded field (the headline 2x2);
* ``02_pairwise_agreement.png``    - model-by-model agreement heatmaps, provider
  centrality, and the consensus depth of the Stage 3 candidacy call;
* ``03_per_model_behavior.png``    - how lenient and how conceptual each model is;
* ``04_consensus_profile.png``     - the majority-vote BPS profile of the corpus
  and a compact reliability overview.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bps_review.pilot.config import (
    BPS_FUNCTION_ORDER,
    DOMAIN_FIELDS,
    FIELD_LABELS,
    MODEL_LABELS,
    MSK_ORDER,
    TYPOLOGY_ORDER,
    figures_dir,
)
from bps_review.utils.io import ensure_parent


# --------------------------------------------------------------------------
# Shared academic style
# --------------------------------------------------------------------------
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
# The three BPS domains keep the accent colours used across the dossiers.
DOMAIN_COLORS = {"bio_mentioned": "#0E8F80", "psych_mentioned": "#6D5AE0", "social_mentioned": "#D98016"}
MODEL_COLORS = {
    "DeepSeek-V4-Flash": "#2f4b7c",
    "Nex-N2-Mini": "#0f8f80",
    "Laguna-XS-2.1": "#6d5ae0",
}

KAPPA_BAND_COLORS = [
    (0.20, "#c85b6b"),   # poor / slight
    (0.40, "#e0a33a"),   # fair
    (0.60, "#4fb0a3"),   # moderate
    (0.80, "#3f79b0"),   # substantial
    (1.01, "#2f4b7c"),   # almost perfect
]


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
            offset = 0.012 * xmax if raw >= 0 else -0.012 * xmax
            ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                    f"{raw:.2f}", va="center", ha="left" if raw >= 0 else "right",
                    fontsize=7.8, color=INK)


def _heatmap(ax, matrix: pd.DataFrame, title, cmap, vmin, vmax, fmt="{:.2f}") -> None:
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
            color = "white" if value >= threshold else INK
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.ax.tick_params(labelsize=7)


def _stacked_by_model(ax, long_df, field, order, colors, title, xlabel="Proportion of abstracts") -> None:
    proportions = []
    for model in MODEL_LABELS:
        subset = long_df[long_df["model_label"] == model][field].astype(str)
        counts = subset.value_counts(normalize=True)
        proportions.append([counts.get(category, 0.0) for category in order])
    proportions = np.array(proportions)
    y = list(range(len(MODEL_LABELS)))
    left = np.zeros(len(MODEL_LABELS))
    for index, category in enumerate(order):
        ax.barh(y, proportions[:, index], left=left, color=colors[index], height=0.7,
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
# Figure 1: field reliability 2x2 (headline)
# --------------------------------------------------------------------------
def fig_field_reliability(field_rel: pd.DataFrame, out_path: Path) -> None:
    order = field_rel.sort_values("fleiss_kappa", ascending=False, na_position="last")
    labels = order["field_label"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    _metric_hbar(axes[0, 0], labels, order["fleiss_kappa"].tolist(),
                 [_kappa_color(v) for v in order["fleiss_kappa"]],
                 f"Fleiss' kappa (chance-corrected, {len(MODEL_LABELS)} raters)", refs=[0.2, 0.4, 0.6, 0.8])
    _panel_label(axes[0, 0], "A")

    _metric_hbar(axes[0, 1], labels, order["krippendorff_alpha"].tolist(),
                 [_kappa_color(v) for v in order["krippendorff_alpha"]],
                 "Krippendorff's alpha (nominal)", refs=[0.667, 0.8])
    _panel_label(axes[0, 1], "B")

    _metric_hbar(axes[1, 0], labels, order["mean_pairwise_agreement"].tolist(),
                 [PALETTE["teal"]] * len(labels),
                 "Observed agreement (mean over model pairs)", refs=[0.5], xlabel="Proportion")
    _panel_label(axes[1, 0], "C")

    _metric_hbar(axes[1, 1], labels, order["unanimous_rate"].tolist(),
                 [PALETTE["primary"]] * len(labels),
                 f"Unanimous rate (all {len(MODEL_LABELS)} models agree)", xlabel="Proportion")
    _panel_label(axes[1, 1], "D")

    fig.tight_layout()
    ensure_parent(out_path)
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2: pairwise agreement
# --------------------------------------------------------------------------
def fig_pairwise(agreement: pd.DataFrame, kappa: pd.DataFrame, depth: pd.DataFrame, out_path: Path) -> None:
    n_models = len(agreement)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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
    lower = min(0.5, float(np.nanmin(centrality.values)) - 0.05)
    ax.set_xlim(max(0.0, lower), 1.0)
    ax.set_xlabel("Mean agreement with the other models")
    ax.set_title("Provider centrality (higher = more typical)")
    _despine(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for position, value in zip(y, centrality.values):
        ax.text(value + 0.005, position, f"{value:.2f}", va="center", ha="left", fontsize=8, color=INK)
    _panel_label(ax, "C")

    ax = axes[1, 1]

    def _depth_label(backing: int) -> str:
        other = n_models - backing
        tag = " (unanimous)" if other == 0 else ""
        return f"{backing} vs {other}{tag}"

    labels = [_depth_label(int(value)) for value in depth["models_backing_majority"]]
    palette_cycle = [PALETTE["accent"], PALETTE["amber"], PALETTE["teal"], PALETTE["primary"]]
    ordered = sorted(int(value) for value in depth["models_backing_majority"])
    depth_color = {value: palette_cycle[index % len(palette_cycle)] for index, value in enumerate(ordered)}
    colors = [depth_color[int(value)] for value in depth["models_backing_majority"]]
    x = list(range(len(labels)))
    bars = ax.bar(x, depth["n_items"].values, color=colors, edgecolor="white", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Abstracts")
    ax.set_title("Agreement on the Stage 3 candidacy call")
    _despine(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, depth["n_items"].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, str(int(value)),
                ha="center", va="bottom", fontsize=8.5, color=INK)
    _panel_label(ax, "D")

    fig.tight_layout()
    ensure_parent(out_path)
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3: per-model behaviour
# --------------------------------------------------------------------------
def fig_per_model(long_df: pd.DataFrame, behavior: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    # Panel A: domain-mention rate per model, one bar group per domain.
    ax = axes[0, 0]
    width = 0.25
    x = np.arange(len(MODEL_LABELS))
    for index, field in enumerate(DOMAIN_FIELDS):
        rates = [
            float((long_df[long_df["model_label"] == model][field].astype(str) == "yes").mean())
            for model in MODEL_LABELS
        ]
        ax.bar(x + (index - 1) * width, rates, width, color=DOMAIN_COLORS[field], edgecolor="white",
               label=FIELD_LABELS[field].replace(" mention", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of abstracts")
    ax.set_title("Domain-mention rate by model")
    _despine(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=7.6)
    _panel_label(ax, "A")

    # Panel B: provisional typology distribution.
    typology_colors = ["#2f4b7c", "#5f7bb0", "#9fb0cf", "#c7ccd4"]
    _stacked_by_model(axes[0, 1], long_df, "provisional_typology", TYPOLOGY_ORDER, typology_colors,
                      "Provisional typology profile")
    _panel_label(axes[0, 1], "B")

    # Panel C: musculoskeletal routing flag.
    msk_colors = [PALETTE["teal"], MUTED, "#dfe3e8"]
    _stacked_by_model(axes[1, 0], long_df, "musculoskeletal_flag", MSK_ORDER, msk_colors,
                      "Musculoskeletal routing flag")
    _panel_label(axes[1, 0], "C")

    # Panel D: how many distinct psychological concepts each model returns.
    ax = axes[1, 1]
    order = behavior.sort_values("mean_concepts", ascending=False)
    y = list(range(len(order)))
    ax.barh(y, order["mean_concepts"].values,
            color=[MODEL_COLORS.get(m, PALETTE["primary"]) for m in order["model_label"]],
            edgecolor="white", height=0.6, label="psychological concepts")
    ax.barh([position + 0.32 for position in y], order["mean_frameworks"].values,
            color=MUTED, edgecolor="white", height=0.28, label="theoretical frameworks")
    ax.set_yticks(y)
    ax.set_yticklabels(order["model_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean distinct items per abstract")
    ax.set_title("Extraction volume on the open lists")
    _despine(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=7.6)
    for position, value in zip(y, order["mean_concepts"].values):
        ax.text(value + 0.05, position, f"{value:.1f}", va="center", ha="left", fontsize=8, color=INK)
    _panel_label(ax, "D")

    fig.tight_layout()
    ensure_parent(out_path)
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 4: consensus profile and reliability overview
# --------------------------------------------------------------------------
def fig_consensus(long_df: pd.DataFrame, consensus: pd.DataFrame, field_rel: pd.DataFrame,
                  overlap: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    # Panel A: consensus typology distribution.
    ax = axes[0, 0]
    counts = consensus["provisional_typology"].value_counts().reindex(TYPOLOGY_ORDER).fillna(0)
    colors = {"potential integrative signal": "#2f4b7c", "multifactorial signal": "#5f7bb0",
              "pseudo-bps or partial signal": "#9fb0cf", "rhetorical label signal": "#c7ccd4"}
    x = list(range(len(counts)))
    bars = ax.bar(x, counts.values, color=[colors[c] for c in counts.index], edgecolor="white", width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace(" signal", "").replace("potential ", "") for label in counts.index],
                       fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("Abstracts")
    ax.set_title("Consensus provisional typology")
    _despine(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, counts.values):
        if value:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, str(int(value)),
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    _panel_label(ax, "A")

    # Panel B: consensus domain coverage, the corpus-level BPS signal.
    ax = axes[0, 1]
    labels = [FIELD_LABELS[field].replace(" mention", "") for field in DOMAIN_FIELDS]
    present = [int((consensus[field] == "yes").sum()) for field in DOMAIN_FIELDS]
    absent = [len(consensus) - value for value in present]
    y = list(range(len(labels)))
    ax.barh(y, present, color=[DOMAIN_COLORS[field] for field in DOMAIN_FIELDS], height=0.66, label="mentioned")
    ax.barh(y, absent, left=present, color="#dfe3e8", height=0.66, label="not mentioned")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Abstracts (consensus)")
    ax.set_title("Consensus domain coverage")
    _despine(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=7.6)
    for position, value in zip(y, present):
        ax.text(value + 0.6, position, f"{value}", va="center", ha="left", fontsize=8, color=INK)
    _panel_label(ax, "B")

    # Panel C: reliability by field group.
    ax = axes[1, 0]
    grouped = field_rel.groupby("group")[["fleiss_kappa", "krippendorff_alpha", "mean_pairwise_agreement"]].mean()
    grouped = grouped.reindex(["nominal", "domain", "ordinal"]).dropna(how="all")
    metrics = ["fleiss_kappa", "krippendorff_alpha", "mean_pairwise_agreement"]
    metric_labels = ["Fleiss' kappa", "Krippendorff alpha", "Observed agreement"]
    metric_colors = [PALETTE["primary"], PALETTE["violet"], PALETTE["teal"]]
    x = np.arange(len(grouped))
    width = 0.26
    for index, (metric, color) in enumerate(zip(metrics, metric_colors)):
        ax.bar(x + (index - 1) * width, grouped[metric].values, width, color=color, edgecolor="white",
               label=metric_labels[index])
    ax.set_xticks(x)
    ax.set_xticklabels([name.capitalize() for name in grouped.index])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Coefficient")
    ax.set_title("Reliability by field group")
    _despine(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=7.4)
    _panel_label(ax, "C")

    # Panel D: set overlap on the open lists.
    ax = axes[1, 1]
    order = overlap.sort_values("mean_pairwise_jaccard", ascending=False)
    y = list(range(len(order)))
    bars = ax.barh(y, order["mean_pairwise_jaccard"].values, color=PALETTE["amber"],
                   edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([FIELD_LABELS.get(field, field) for field in order["field"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean pairwise Jaccard overlap")
    ax.set_title("Open extraction lists (set overlap, not kappa)")
    _despine(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, order["mean_pairwise_jaccard"].values):
        ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
                va="center", ha="left", fontsize=8, color=INK)
    _panel_label(ax, "D")

    fig.tight_layout()
    ensure_parent(out_path)
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def build_figures(long_df: pd.DataFrame, results: dict, out_dir: Path | None = None) -> list[Path]:
    _apply_style()
    out_dir = out_dir or figures_dir()
    builders = {
        "01_field_reliability_2x2.png": lambda path: fig_field_reliability(results["field_reliability"], path),
        "02_pairwise_agreement.png": lambda path: fig_pairwise(
            results["pairwise_agreement"], results["pairwise_kappa"], results["candidate_depth"], path),
        "03_per_model_behavior.png": lambda path: fig_per_model(long_df, results["per_model_behavior"], path),
        "04_consensus_profile.png": lambda path: fig_consensus(
            long_df, results["consensus"], results["field_reliability"], results["list_overlap"], path),
    }
    written = []
    for name, builder in builders.items():
        path = out_dir / name
        builder(path)
        written.append(path)
    return written
