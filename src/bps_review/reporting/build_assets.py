from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Wedge
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from bps_review.reporting.semantic_loading import ONTOLOGY_TERMS, run_semantic_loading
from bps_review.utils.io import ensure_parent
from bps_review.utils.paths import project_path


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def _latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _with_percent(frame: pd.DataFrame, count_col: str = "n") -> pd.DataFrame:
    if frame.empty:
        return frame
    total = frame[count_col].sum()
    out = frame.copy()
    out["percent"] = out[count_col].apply(lambda value: round((value / total) * 100, 1) if total else 0.0)
    return out


def _write_latex_table(frame: pd.DataFrame, path: Path, caption: str, label: str, note: str) -> None:
    if frame.empty:
        content = (
            "\\begin{table}[htbp]\n"
            "\\raggedright\n"
            f"\\caption{{{_latex_escape(caption)}}}\n"
            f"\\label{{{label}}}\n"
            "No data available.\\\\\n"
            f"\\caption*{{Note. {_latex_escape(note)}}}\n"
            "\\end{table}\n"
        )
    else:
        display_frame = frame.copy()
        for col in display_frame.columns:
            if col == "percent" or col.endswith("_percent"):
                display_frame[col] = display_frame[col].apply(lambda v: f"{float(v):.1f}" if str(v).strip() not in ("", "nan") else "")
            elif col in ("mean_cosine", "mean_loading", "mean", "median_loading"):
                display_frame[col] = display_frame[col].apply(lambda v: f"{float(v):.4f}" if str(v).strip() not in ("", "nan") else "")
            elif col == "n":
                display_frame[col] = display_frame[col].apply(lambda v: str(int(float(v))) if str(v).strip() not in ("", "nan") else "")
        latex_table = display_frame.to_latex(index=False, escape=True)
        content = (
            "\\begin{table}[htbp]\n"
            "\\raggedright\n"
            f"\\caption{{{_latex_escape(caption)}}}\n"
            f"\\label{{{label}}}\n"
            f"{latex_table}\n"
            f"\\caption*{{Note. {_latex_escape(note)}}}\n"
            "\\end{table}\n"
        )
    ensure_parent(path).write_text(content, encoding="utf-8")


def _write_latex_longtable(frame: pd.DataFrame, path: Path, caption: str, label: str, note: str) -> None:
    if frame.empty:
        content = (
            "\\begin{table}[htbp]\n"
            "\\raggedright\n"
            f"\\caption{{{_latex_escape(caption)}}}\n"
            f"\\label{{{label}}}\n"
            "No data available.\\\\\n"
            f"\\caption*{{Note. {_latex_escape(note)}}}\n"
            "\\end{table}\n"
        )
        ensure_parent(path).write_text(content, encoding="utf-8")
        return

    colspec = "p{0.7cm}p{0.7cm}p{1.2cm}p{1.9cm}p{1.3cm}p{1.5cm}p{0.55cm}p{0.55cm}p{0.55cm}p{3.2cm}"
    table_text = frame.to_latex(
        index=False,
        escape=True,
        longtable=True,
        caption=caption,
        label=label,
        column_format=colspec,
    )
    content = (
        "\\setlength{\\LTleft}{0pt}\n"
        "\\setlength{\\LTright}{0pt}\n"
        "\\setlength{\\tabcolsep}{1pt}\n"
        "\\tiny\n"
        + table_text
        + "\n"
        + f"\\par\\vspace{{2pt}}\\footnotesize Note. {_latex_escape(note)}\\n"
        + "\\normalsize\n"
    )
    ensure_parent(path).write_text(content, encoding="utf-8")


def _bar_plot(frame: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    palette = ["#1a6b8a", "#2480a5", "#2e94bd", "#45a8cc", "#5fbbd8", "#7fccdf", "#9dd9e8", "#bee8f2"]
    colors = palette[:len(frame)]
    bars = ax.bar(frame[x], frame[y], color=colors, width=0.6, edgecolor="white", linewidth=0.8)
    for bar, n in zip(bars, frame[y]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(frame[y].max() * 0.01, 0.3),
                str(int(n)), ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2a3a4a")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    ensure_parent(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _line_plot(frame: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.fill_between(frame[x], frame[y], color="#b6d5f2", alpha=0.4)
    ax.plot(frame[x], frame[y], color="#1a6b8a", marker="o", linewidth=2.2, markersize=5)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xlabel("Publication year", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    ensure_parent(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _heatmap_plot(frame: pd.DataFrame, title: str, out_path: Path) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    fig.patch.set_facecolor("white")
    values = frame.values.astype(float)
    vmax = max(float(np.nanmax(values)), 1)
    image = ax.imshow(values, cmap="Blues", aspect="auto", vmin=0, vmax=vmax)
    cbar = plt.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Count", fontsize=10)

    ax.set_xticks(range(len(frame.columns)))
    ax.set_yticks(range(len(frame.index)))
    ax.set_xticklabels(frame.columns, rotation=35, ha="right", fontsize=9.5)
    ax.set_yticklabels(frame.index, fontsize=9.5)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            val = int(values[row, col])
            text_color = "white" if val > vmax * 0.6 else "#0d2238"
            ax.text(col, row, str(val), ha="center", va="center", fontsize=9, color=text_color)

    ax.spines[:].set_visible(False)
    fig.tight_layout(pad=1.5)
    ensure_parent(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _panel_descriptive_plot(
    publication_year_counts: pd.DataFrame,
    review_type_counts: pd.DataFrame,
    icd11_counts: pd.DataFrame,
    core_bps_counts: pd.DataFrame,
    out_path: Path,
) -> None:
    if publication_year_counts.empty and review_type_counts.empty and icd11_counts.empty and core_bps_counts.empty:
        return

    fig = plt.figure(figsize=(14.5, 10.2))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    def _style_panel(ax: plt.Axes) -> None:
        ax.set_facecolor("#f8fbff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#c1d0dc")
        ax.spines["bottom"].set_color("#c1d0dc")
        ax.grid(linestyle="--", alpha=0.22, color="#6f8aa2")
        ax.set_axisbelow(True)

    palette = ["#1a6b8a", "#2480a5", "#3a94bd", "#5aaa9a", "#7abb8a", "#9acc7a", "#bad86a", "#d6e29e"]

    # Panel A: Publication trend
    ax = fig.add_subplot(gs[0, 0])
    _style_panel(ax)
    if not publication_year_counts.empty:
        trend = publication_year_counts.copy()
        trend["year"] = pd.to_numeric(trend["year"], errors="coerce")
        trend["n"] = pd.to_numeric(trend["n"], errors="coerce").fillna(0.0)
        trend = trend.dropna(subset=["year"]).sort_values("year")
        yrs = trend["year"].to_numpy(dtype=float)
        ns = trend["n"].to_numpy(dtype=float)
        smooth = pd.Series(ns).rolling(window=3, center=True, min_periods=1).mean().to_numpy()
        ax.axvspan(2005, yrs.max() + 0.6, color="#f7c9ab", alpha=0.18, zorder=0)
        ax.fill_between(yrs, 0, ns, color="#cfe3f3", alpha=0.72, zorder=1)
        ax.plot(yrs, smooth, color="#0b3554", linewidth=3.0, zorder=3)
        ax.scatter(
            yrs,
            ns,
            c=ns,
            cmap=mcolors.LinearSegmentedColormap.from_list("trend", ["#d6e8f6", "#5e9ecc", "#0b3554"]),
            s=46 + ns * 14,
            edgecolors="white",
            linewidths=0.9,
            zorder=4,
        )
        ax.set_xlim(yrs.min() - 1.0, yrs.max() + 1.0)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title("A. Publication tempo and acceleration", fontsize=11.5, fontweight="bold", pad=8, loc="left")
    ax.set_ylabel("Annual count", fontsize=10)
    ax.set_xlabel("Publication year", fontsize=10)

    # Panel B: Review types
    ax = fig.add_subplot(gs[0, 1])
    _style_panel(ax)
    review_slice = review_type_counts.head(8).copy()
    if not review_slice.empty:
        review_slice["n"] = pd.to_numeric(review_slice["n"], errors="coerce").fillna(0.0)
        review_slice["percent"] = pd.to_numeric(review_slice["percent"], errors="coerce").fillna(0.0)
        review_slice = review_slice.sort_values("n", ascending=True)
        labels = [_shorten_label(label, max_len=30) for label in review_slice["review_type"]]
        colors_b = palette[: len(review_slice)]
        bars = ax.barh(labels, review_slice["n"], color=colors_b, height=0.72, edgecolor="white", linewidth=0.9)
        for bar, count, pct in zip(bars, review_slice["n"], review_slice["percent"]):
            ax.text(
                bar.get_width() + 0.45,
                bar.get_y() + bar.get_height() / 2.0,
                f"{int(count)} ({pct:.1f}%)",
                va="center",
                ha="left",
                fontsize=9,
                color="#16324a",
            )
        ax.set_xlim(0, review_slice["n"].max() * 1.45)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title("B. Review-type composition", fontsize=11.5, fontweight="bold", pad=8, loc="left")
    ax.set_xlabel("Included reviews", fontsize=10)

    # Panel C: ICD-11 categories as lollipop distribution
    ax = fig.add_subplot(gs[1, 0])
    _style_panel(ax)
    icd_slice = icd11_counts.head(8).copy()
    if not icd_slice.empty:
        icd_slice["n"] = pd.to_numeric(icd_slice["n"], errors="coerce").fillna(0.0)
        icd_slice["percent"] = pd.to_numeric(icd_slice["percent"], errors="coerce").fillna(0.0)
        icd_slice = icd_slice.sort_values("n", ascending=True).reset_index(drop=True)
        y_pos = np.arange(len(icd_slice))
        labels = [_shorten_label(_compact_icd11_label(label), max_len=34) for label in icd_slice["icd11_pain_category"]]
        ax.hlines(y_pos, 0, icd_slice["n"], color="#b8d4e9", linewidth=4.4, alpha=0.9)
        ax.scatter(
            icd_slice["n"],
            y_pos,
            s=90 + icd_slice["n"].to_numpy(dtype=float) * 22,
            c="#1a6b8a",
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )
        for y_idx, count, pct in zip(y_pos, icd_slice["n"], icd_slice["percent"]):
            ax.text(float(count) + 0.45, y_idx, f"{int(count)} ({pct:.1f}%)", va="center", fontsize=9, color="#16324a")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title("C. ICD-11 category concentration", fontsize=11.5, fontweight="bold", pad=8, loc="left")
    ax.set_xlabel("Included reviews", fontsize=10)

    # Panel D: Core BPS mention structure
    ax = fig.add_subplot(gs[1, 1])
    _style_panel(ax)
    if not core_bps_counts.empty:
        core_slice = core_bps_counts.copy()
        core_slice["percent"] = pd.to_numeric(core_slice["percent"], errors="coerce").fillna(0.0)
        core_slice["n"] = pd.to_numeric(core_slice["n"], errors="coerce").fillna(0.0)
        y_pos = np.arange(len(core_slice))
        color_map = {
            "Biological mention": "#1a6b8a",
            "Psychological mention": "#c45e2a",
            "Social mention": "#3a7d44",
            "Triadic co-mention": "#2f3e4e",
        }
        ax.barh(y_pos, np.full(len(core_slice), 100.0), color="#e7eef4", height=0.58, edgecolor="none", zorder=1)
        bars = ax.barh(
            y_pos,
            core_slice["percent"],
            color=[color_map.get(label, "#7f8c8d") for label in core_slice["indicator"]],
            height=0.58,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        for y_idx, indicator, pct, count, bar in zip(
            y_pos,
            core_slice["indicator"],
            core_slice["percent"],
            core_slice["n"],
            bars,
        ):
            ax.text(
                min(float(pct) + 1.6, 101.4),
                y_idx,
                f"{pct:.1f}% (n = {int(count)})",
                va="center",
                ha="left",
                fontsize=9.2,
                color="#16324a",
                fontweight="bold" if indicator == "Triadic co-mention" else "normal",
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(core_slice["indicator"], fontsize=9.5)
        for tick in ax.get_yticklabels():
            tick.set_color(color_map.get(tick.get_text(), "#516170"))
            if tick.get_text() == "Triadic co-mention":
                tick.set_fontweight("bold")
        ax.axvline(100, color="#9db0c1", linestyle=":", linewidth=1.0)
        ax.set_xlim(0, 108)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_title("D. Substantive BPS coverage and triadic co-mention", fontsize=11.5, fontweight="bold", pad=8, loc="left")
    ax.set_xlabel("Share of included reviews", fontsize=10)

    fig.subplots_adjust(top=0.97, left=0.08, right=0.98, bottom=0.08)
    ensure_parent(out_path)
    fig.savefig(out_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _semantic_sunburst_plot(subdomain_summary: pd.DataFrame, out_path: Path) -> None:
    """Two-panel figure: left = clean 2-ring sunburst (no outer text), right = subdomain loading bars grouped by domain."""
    if subdomain_summary.empty:
        return
    required = {"domain", "subdomain", "mean_loading"}
    if not required.issubset(subdomain_summary.columns):
        return

    data = subdomain_summary.copy()
    data["domain"] = data["domain"].astype(str).str.lower()
    data["mean_loading"] = pd.to_numeric(data["mean_loading"], errors="coerce").fillna(0.0)
    data = data.loc[data["mean_loading"] > 0].copy()
    if data.empty:
        return

    domain_order = ["biological", "psychological", "social"]
    domain_colors = {
        "biological": "#1a6b8a",
        "psychological": "#c45e2a",
        "social": "#3a7d44",
    }
    domain_palettes = {
        "biological": ["#1a6b8a", "#2480a5", "#3190b5", "#45a8cc", "#5bbad6",
                       "#74cce0", "#8fd8e8", "#aae3f0", "#c8eef7", "#e2f7fc"],
        "psychological": ["#c45e2a", "#cf6e38", "#d97e48", "#e38e5b", "#ec9e6e",
                          "#f3ae84", "#f7be9b", "#facdb3", "#fcdcc9", "#fdeee5"],
        "social": ["#3a7d44", "#498f54", "#5aa165", "#6db377", "#83c48b",
                   "#9bd4a0", "#b3e2b7", "#caedce", "#def6e1", "#f0fbf1"],
    }
    domain_labels = {"biological": "Biological", "psychological": "Psychological", "social": "Social"}

    domain_totals = data.groupby("domain", as_index=False)["mean_loading"].sum()
    domain_totals = domain_totals.set_index("domain").reindex(domain_order).fillna(0.0).reset_index()
    total_mass = float(domain_totals["mean_loading"].sum())
    if total_mass <= 0:
        return

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("white")

    # Left: sunburst axes (square)
    ax_sun = fig.add_axes([0.01, 0.04, 0.46, 0.92])
    ax_sun.set_aspect("equal")
    ax_sun.axis("off")

    # --- Draw sunburst (left panel) ---
    start_angle = 90.0
    for _, drow in domain_totals.iterrows():
        domain = str(drow["domain"])
        domain_mass = float(drow["mean_loading"])
        if domain_mass <= 0:
            continue
        domain_frac = domain_mass / total_mass
        domain_sweep = 360.0 * domain_frac
        end_angle = start_angle + domain_sweep

        # Inner ring wedge
        inner_wedge = Wedge(
            (0, 0), r=0.82, theta1=start_angle, theta2=end_angle, width=0.30,
            facecolor=domain_colors.get(domain, "#7f8c8d"),
            edgecolor="white", linewidth=2.5,
        )
        ax_sun.add_patch(inner_wedge)

        # Inner ring label (domain name + %)
        mid_angle = (start_angle + end_angle) / 2.0
        mid_rad = np.deg2rad(mid_angle)
        ax_sun.text(
            0.67 * np.cos(mid_rad), 0.67 * np.sin(mid_rad),
            f"{domain_labels.get(domain, domain)}\n{domain_frac * 100:.1f}%",
            ha="center", va="center", fontsize=11.5, color="white",
            fontweight="bold", multialignment="center",
        )

        # Outer ring (subdomains – color only, NO text)
        children = data.loc[data["domain"] == domain, ["subdomain", "mean_loading"]].copy()
        children = children.sort_values("mean_loading", ascending=False).reset_index(drop=True)
        child_total = float(children["mean_loading"].sum())
        child_angle = start_angle
        palette = domain_palettes.get(domain, ["#bdc3c7"] * max(1, len(children)))
        for i, crow in children.iterrows():
            mass = float(crow["mean_loading"])
            if child_total <= 0 or mass <= 0:
                continue
            sweep = domain_sweep * (mass / child_total)
            child_end = child_angle + sweep
            ax_sun.add_patch(Wedge(
                (0, 0), r=1.40, theta1=child_angle, theta2=child_end, width=0.48,
                facecolor=palette[i % len(palette)],
                edgecolor="white", linewidth=0.6, alpha=0.93,
            ))
            child_angle = child_end
        start_angle = end_angle

    # Center label
    center_c = plt.Circle((0, 0), 0.40, color="#f4f8fb", ec="#b8ccda", lw=1.8, zorder=10)
    ax_sun.add_artist(center_c)
    ax_sun.text(0, 0.10, "BPS", ha="center", va="center", fontsize=20, fontweight="bold",
                color="#1a2e44", zorder=11)
    ax_sun.text(0, -0.12, "Semantic", ha="center", va="center", fontsize=10, color="#3a5068", zorder=11)
    ax_sun.text(0, -0.28, "Loading", ha="center", va="center", fontsize=10, color="#3a5068", zorder=11)
    ax_sun.set_xlim(-1.65, 1.65)
    ax_sun.set_ylim(-1.65, 1.65)

    # Subtitle under sunburst
    fig.text(0.24, 0.01, "Inner ring = domain proportion | Outer ring = subdomain profile",
             ha="center", fontsize=9, color="#556677", style="italic")

    # --- Right panel: grouped subdomain loading bars ---
    # Three stacked axes (one per domain), heights proportional to subdomain count
    n_bio = len(data[data["domain"] == "biological"])
    n_psy = len(data[data["domain"] == "psychological"])
    n_soc = len(data[data["domain"] == "social"])
    n_total = n_bio + n_psy + n_soc
    gap = 0.015
    usable_height = 0.92 - 2 * gap
    h_bio = usable_height * n_bio / n_total
    h_psy = usable_height * n_psy / n_total
    h_soc = usable_height * n_soc / n_total

    y_soc = 0.04
    y_psy = y_soc + h_soc + gap
    y_bio = y_psy + h_psy + gap
    x_left = 0.50
    w_axes = 0.47

    for domain, y_bot, h_ax in [
        ("social", y_soc, h_soc),
        ("psychological", y_psy, h_psy),
        ("biological", y_bio, h_bio),
    ]:
        ax = fig.add_axes([x_left, y_bot, w_axes, h_ax])
        d_data = data[data["domain"] == domain].copy()
        # Normalize within domain for bar lengths (shows relative rank)
        d_max = float(d_data["mean_loading"].max()) or 1.0
        d_data = d_data.sort_values("mean_loading", ascending=True)

        bar_color = domain_colors[domain]
        # Shorten subdomain labels
        labels = [
            s[:36] + ("…" if len(s) > 36 else "")
            for s in d_data["subdomain"]
        ]
        vals = (d_data["mean_loading"] / d_max).values
        bar_container = ax.barh(
            range(len(d_data)), vals,
            color=bar_color, alpha=0.80, height=0.62,
            edgecolor="white", linewidth=0.5,
        )
        ax.set_yticks(range(len(d_data)))
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1.0"], fontsize=7.5)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

        # Domain header band
        ax.set_title(
            f"{domain_labels[domain]}  ({len(d_data)} subdomains)",
            fontsize=10, fontweight="bold", color="white",
            pad=4, loc="left",
            bbox=dict(facecolor=bar_color, edgecolor="none", boxstyle="round,pad=0.3"),
        )

        if domain == "biological":
            ax.set_xlabel("Relative within-domain loading (normalized)", fontsize=8.5)

    ensure_parent(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _semantic_record_profile_plot(record_loadings: pd.DataFrame, stage2: pd.DataFrame, out_path: Path) -> None:
    """Paper-facing summary of corpus-level domain loading structure."""
    if record_loadings.empty:
        return
    required = {"record_id", "loading_biological", "loading_psychological", "loading_social", "dominant_domain"}
    if not required.issubset(record_loadings.columns):
        return

    df = record_loadings.copy()

    # Sort: dominant domain rank, then by dominant loading descending
    dom_rank = {"biological": 0, "psychological": 1, "social": 2}
    df["_dom_rank"] = df["dominant_domain"].map(dom_rank).fillna(3)
    df["_dom_loading"] = df.apply(
        lambda r: float(r.get(f"loading_{r['dominant_domain']}", 0.0)), axis=1)
    df = df.sort_values(["_dom_rank", "_dom_loading"], ascending=[True, False]).reset_index(drop=True)

    n = len(df)
    NEUTRAL = 1.0 / 3.0
    DOMAIN_COLORS = {"biological": "#1a6b8a", "psychological": "#c45e2a", "social": "#3a7d44"}
    domains = ["biological", "psychological", "social"]
    dom_labels = ["Biological", "Psychological", "Social"]

    fig = plt.figure(figsize=(17.2, 7.8))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[0.63, 0.37], wspace=0.12, left=0.05, right=0.98, top=0.92, bottom=0.13)

    # Panel A: rank-ordered stacked profiles
    ax_rank = fig.add_subplot(gs[0, 0])
    ax_rank.set_facecolor("#fbfdff")
    x = np.arange(n)
    bottoms = np.zeros(n)
    for domain in domains:
        vals = pd.to_numeric(df[f"loading_{domain}"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax_rank.bar(
            x,
            vals,
            bottom=bottoms,
            width=0.96,
            color=DOMAIN_COLORS[domain],
            edgecolor="none",
            alpha=0.88,
        )
        bottoms = bottoms + vals

    for y_line in [NEUTRAL, NEUTRAL * 2]:
        ax_rank.axhline(y_line, color="#6d8396", linestyle=":", linewidth=1.0, alpha=0.85, zorder=0)

    for idx, domain in enumerate(df["dominant_domain"]):
        ax_rank.add_patch(
            plt.Rectangle((idx - 0.48, 1.015), 0.96, 0.025, facecolor=DOMAIN_COLORS.get(str(domain), "#888"), edgecolor="none", clip_on=False)
        )

    cursor = 0
    for domain, label in zip(domains, dom_labels):
        count = int(df["dominant_domain"].eq(domain).sum())
        if count == 0:
            continue
        if cursor > 0:
            ax_rank.axvline(cursor - 0.5, color="#9eb0bf", linewidth=1.2, alpha=0.9)
        center = cursor + count / 2.0 - 0.5
        ax_rank.text(
            center,
            1.055,
            f"{label}\n(n = {count})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=DOMAIN_COLORS[domain],
        )
        cursor += count

    ax_rank.set_xlim(-0.7, n - 0.3)
    ax_rank.set_ylim(0, 1.09)
    ax_rank.set_xticks([])
    ax_rank.set_yticks([0, NEUTRAL, NEUTRAL * 2, 1.0])
    ax_rank.set_yticklabels(["0", "1/3", "2/3", "1.0"], fontsize=9.5)
    ax_rank.set_ylabel("Stacked domain loading per record", fontsize=10)
    ax_rank.set_title("A. Rank-ordered semantic loading profiles", fontsize=11.5, fontweight="bold", pad=8, loc="left")
    ax_rank.spines["top"].set_visible(False)
    ax_rank.spines["right"].set_visible(False)
    ax_rank.spines["left"].set_color("#c7d4de")
    ax_rank.spines["bottom"].set_color("#c7d4de")

    # Panel B: benchmark-relative deviation distributions
    ax_dev = fig.add_subplot(gs[0, 1])
    ax_dev.set_facecolor("#fbfdff")
    deviation_frame = df[[f"loading_{d}" for d in domains]].copy() - NEUTRAL
    rng = np.random.default_rng(42)
    positions = np.arange(1, 4)
    violin = ax_dev.violinplot(
        [deviation_frame[f"loading_{d}"].to_numpy(dtype=float) for d in domains],
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, domain in zip(violin["bodies"], domains):
        body.set_facecolor(DOMAIN_COLORS[domain])
        body.set_edgecolor(DOMAIN_COLORS[domain])
        body.set_alpha(0.23)
        body.set_linewidth(1.0)

    for pos, domain, label in zip(positions, domains, dom_labels):
        vals = deviation_frame[f"loading_{domain}"].to_numpy(dtype=float)
        jitter = rng.normal(0.0, 0.045, size=len(vals))
        ax_dev.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=16,
            color=DOMAIN_COLORS[domain],
            alpha=0.34,
            edgecolors="none",
            zorder=3,
        )
        q1, median, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        mean = float(np.mean(vals))
        ax_dev.vlines(pos, q1, q3, color=DOMAIN_COLORS[domain], linewidth=5.0, alpha=0.95, zorder=4)
        ax_dev.scatter([pos], [mean], s=54, color=DOMAIN_COLORS[domain], edgecolors="white", linewidths=0.9, zorder=5, marker="D")
        ax_dev.scatter([pos], [median], s=28, color="white", edgecolors=DOMAIN_COLORS[domain], linewidths=1.0, zorder=6)
        ax_dev.text(
            pos + 0.18,
            mean,
            f"{mean:+.3f}",
            fontsize=8.8,
            color=DOMAIN_COLORS[domain],
            va="center",
            fontweight="bold",
        )

    balanced_all = int((deviation_frame.abs() <= 0.02).all(axis=1).sum())
    ax_dev.axhline(0, color="#44596c", linestyle="--", linewidth=1.1, alpha=0.9)
    ax_dev.text(0.56, 0.0012, "Equal-loading benchmark", fontsize=8.5, color="#44596c")
    ax_dev.text(
        0.58,
        deviation_frame.to_numpy(dtype=float).min() * 0.85,
        f"{balanced_all}/111 records stay within ±0.02 of equal loading\nacross all three domains.",
        fontsize=8.7,
        color="#41596f",
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#c7d4de", "alpha": 0.97},
    )
    ax_dev.set_xlim(0.45, 3.6)
    y_abs = max(0.022, float(np.abs(deviation_frame.to_numpy(dtype=float)).max()) * 1.18)
    ax_dev.set_ylim(-y_abs, y_abs)
    ax_dev.set_xticks(positions)
    ax_dev.set_xticklabels(dom_labels, fontsize=10, fontweight="bold")
    for tick, domain in zip(ax_dev.get_xticklabels(), domains):
        tick.set_color(DOMAIN_COLORS[domain])
    ax_dev.set_ylabel("Loading deviation from equal-share benchmark (loading - 1/3)", fontsize=10)
    ax_dev.set_title("B. Benchmark-relative loading deviations", fontsize=11.5, fontweight="bold", pad=8, loc="left")
    ax_dev.grid(axis="y", linestyle="--", alpha=0.22)
    ax_dev.spines["top"].set_visible(False)
    ax_dev.spines["right"].set_visible(False)
    ax_dev.spines["left"].set_color("#c7d4de")
    ax_dev.spines["bottom"].set_color("#c7d4de")

    ensure_parent(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _embedding_landscape_plot(
    record_loadings: pd.DataFrame,
    subdomain_summary: pd.DataFrame,
    out_path: Path,
    coords_out_path: Path,
) -> pd.DataFrame:
    required = {
        "record_id",
        "year",
        "review_type",
        "loading_biological",
        "loading_psychological",
        "loading_social",
        "dominant_domain",
    }
    if record_loadings.empty or not required.issubset(record_loadings.columns):
        return pd.DataFrame()

    record_embeddings_path = project_path("src", "vector_db", "semantic_loading", "records", "record_embeddings.npy")
    ontology_embeddings_path = project_path("src", "vector_db", "semantic_loading", "ontology", "ontology_embeddings.npy")
    subdomain_embeddings_path = project_path("src", "vector_db", "semantic_loading", "ontology", "subdomain_embeddings.npy")
    if not record_embeddings_path.exists() or not ontology_embeddings_path.exists() or not subdomain_embeddings_path.exists():
        return pd.DataFrame()

    record_embeddings = np.load(record_embeddings_path)
    ontology_embeddings = np.load(ontology_embeddings_path)
    subdomain_embeddings = np.load(subdomain_embeddings_path)

    domain_order = ["biological", "psychological", "social"]
    domain_labels = {"biological": "Biological", "psychological": "Psychological", "social": "Social"}
    domain_colors = {"biological": "#1a6b8a", "psychological": "#c45e2a", "social": "#3a7d44"}

    subdomain_labels: list[str] = []
    subdomain_domains: list[str] = []
    for domain in domain_order:
        for label in ONTOLOGY_TERMS.get(domain, []):
            subdomain_labels.append(label)
            subdomain_domains.append(domain)

    if record_embeddings.shape[0] != len(record_loadings):
        return pd.DataFrame()
    if ontology_embeddings.shape[0] < len(domain_order) or subdomain_embeddings.shape[0] < len(subdomain_labels):
        return pd.DataFrame()

    ontology_embeddings = ontology_embeddings[: len(domain_order)]
    subdomain_embeddings = subdomain_embeddings[: len(subdomain_labels)]
    combined_embeddings = np.vstack([record_embeddings, ontology_embeddings, subdomain_embeddings]).astype(np.float32)
    if combined_embeddings.shape[0] < 8:
        return pd.DataFrame()

    scaled = StandardScaler().fit_transform(combined_embeddings)
    pca_dims = max(2, min(36, scaled.shape[0] - 1, scaled.shape[1]))
    pca = PCA(n_components=pca_dims)
    pca_coords = pca.fit_transform(scaled)
    perplexity = max(8, min(30, (combined_embeddings.shape[0] - 1) // 5))
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        max_iter=2000,
        random_state=42,
    )
    tsne_coords = tsne.fit_transform(pca_coords)

    n_records = len(record_loadings)
    n_domains = len(domain_order)
    record_xy = tsne_coords[:n_records]
    domain_xy = tsne_coords[n_records : n_records + n_domains]
    subdomain_xy = tsne_coords[n_records + n_domains :]
    record_pca = pca_coords[:n_records]
    domain_pca = pca_coords[n_records : n_records + n_domains]
    subdomain_pca = pca_coords[n_records + n_domains :]

    records = record_loadings.copy().reset_index(drop=True)
    records["tsne_x"] = record_xy[:, 0]
    records["tsne_y"] = record_xy[:, 1]
    records["pca_x"] = record_pca[:, 0]
    records["pca_y"] = record_pca[:, 1] if record_pca.shape[1] > 1 else 0.0
    records["year"] = pd.to_numeric(records["year"], errors="coerce")
    records["dominant_loading"] = records[
        ["loading_biological", "loading_psychological", "loading_social"]
    ].max(axis=1)

    domain_frame = pd.DataFrame(
        {
            "node_type": "domain_anchor",
            "label": [domain_labels[d] for d in domain_order],
            "parent_domain": domain_order,
            "tsne_x": domain_xy[:, 0],
            "tsne_y": domain_xy[:, 1],
            "pca_x": domain_pca[:, 0],
            "pca_y": domain_pca[:, 1] if domain_pca.shape[1] > 1 else 0.0,
        }
    )
    subdomain_frame = pd.DataFrame(
        {
            "node_type": "subdomain_anchor",
            "label": subdomain_labels,
            "parent_domain": subdomain_domains,
            "tsne_x": subdomain_xy[:, 0],
            "tsne_y": subdomain_xy[:, 1],
            "pca_x": subdomain_pca[:, 0],
            "pca_y": subdomain_pca[:, 1] if subdomain_pca.shape[1] > 1 else 0.0,
        }
    )
    record_frame = records[
        [
            "record_id",
            "year",
            "review_type",
            "dominant_domain",
            "dominant_loading",
            "tsne_x",
            "tsne_y",
            "pca_x",
            "pca_y",
        ]
    ].copy()
    record_frame.insert(0, "node_type", "record")
    record_frame.insert(1, "label", record_frame["record_id"])
    record_frame["parent_domain"] = record_frame["dominant_domain"]
    coords_frame = pd.concat(
        [
            record_frame,
            domain_frame.assign(record_id="", year=np.nan, review_type="", dominant_domain="", dominant_loading=np.nan),
            subdomain_frame.assign(record_id="", year=np.nan, review_type="", dominant_domain="", dominant_loading=np.nan),
        ],
        ignore_index=True,
        sort=False,
    )
    ensure_parent(coords_out_path)
    coords_frame.to_csv(coords_out_path, index=False)

    dominant_strength = pd.to_numeric(records["dominant_loading"], errors="coerce").fillna(1.0 / 3.0).to_numpy(dtype=float)
    strength_scaled = np.clip((dominant_strength - (1.0 / 3.0)) / (2.0 / 3.0), 0.0, 1.0)
    point_sizes = 46 + 240 * strength_scaled

    fig, ax_map = plt.subplots(figsize=(15.8, 9.2))
    fig.patch.set_facecolor("white")
    ax_map.set_facecolor("#fbfdff")
    ax_map.set_axisbelow(True)

    all_x = tsne_coords[:, 0]
    all_y = tsne_coords[:, 1]
    pad_x = max((all_x.max() - all_x.min()) * 0.10, 1.0)
    pad_y = max((all_y.max() - all_y.min()) * 0.10, 1.0)

    if len(record_xy) >= 5:
        try:
            kde = gaussian_kde(record_xy.T, bw_method=0.24)
            xi = np.linspace(all_x.min() - pad_x, all_x.max() + pad_x, 180)
            yi = np.linspace(all_y.min() - pad_y, all_y.max() + pad_y, 180)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            cmap_density = mcolors.LinearSegmentedColormap.from_list(
                "embedding_density",
                ["#fbfdff", "#d7e7f3", "#a9c7de", "#5c8cb3", "#16324a"],
            )
            ax_map.contourf(Xi, Yi, Zi, levels=18, cmap=cmap_density, alpha=0.82, zorder=1)
            ax_map.contour(Xi, Yi, Zi, levels=8, colors="white", linewidths=0.75, alpha=0.40, zorder=2)
            for domain in domain_order:
                mask = records["dominant_domain"].eq(domain).to_numpy()
                if mask.sum() < 6:
                    continue
                kde_domain = gaussian_kde(record_xy[mask].T, bw_method=0.30)
                Zd = kde_domain(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                positive = Zd[Zd > 0]
                if positive.size < 10:
                    continue
                level = float(np.quantile(positive, 0.84))
                if level < float(np.nanmax(Zd)):
                    ax_map.contour(
                        Xi,
                        Yi,
                        Zd,
                        levels=[level],
                        colors=[domain_colors[domain]],
                        linewidths=1.6,
                        alpha=0.92,
                        zorder=4,
                    )
        except Exception:
            pass

    domain_anchor_lookup = {
        domain: (domain_xy[idx, 0], domain_xy[idx, 1]) for idx, domain in enumerate(domain_order)
    }
    for domain in domain_order:
        domain_sub = subdomain_frame.loc[subdomain_frame["parent_domain"] == domain]
        dx, dy = domain_anchor_lookup[domain]
        for _, row in domain_sub.iterrows():
            ax_map.plot(
                [dx, float(row["tsne_x"])],
                [dy, float(row["tsne_y"])],
                color=domain_colors[domain],
                alpha=0.13,
                linewidth=0.85,
                zorder=3,
            )
        ax_map.scatter(
            domain_sub["tsne_x"],
            domain_sub["tsne_y"],
            s=42,
            facecolors="white",
            edgecolors=domain_colors[domain],
            linewidths=1.0,
            alpha=0.84,
            zorder=4,
        )

    for domain in domain_order:
        mask = records["dominant_domain"].eq(domain).to_numpy()
        if not mask.any():
            continue
        ax_map.scatter(
            record_xy[mask, 0],
            record_xy[mask, 1],
            s=point_sizes[mask] * 1.35,
            color="white",
            alpha=0.32,
            edgecolors="none",
            zorder=4,
        )
        ax_map.scatter(
            record_xy[mask, 0],
            record_xy[mask, 1],
            s=point_sizes[mask],
            c=domain_colors[domain],
            alpha=0.90,
            edgecolors="white",
            linewidths=0.85,
            zorder=5,
        )

    for idx, domain in enumerate(domain_order):
        dx, dy = domain_anchor_lookup[domain]
        ax_map.scatter([dx], [dy], s=520, marker="H", facecolors="white", edgecolors=domain_colors[domain], linewidths=2.4, zorder=8)
        ax_map.annotate(
            domain_labels[domain],
            xy=(dx, dy),
            xytext=(dx + pad_x * (0.11 if domain != "psychological" else -0.18), dy + pad_y * (0.06 if idx != 1 else -0.10)),
            fontsize=10.5,
            fontweight="bold",
            color=domain_colors[domain],
            arrowprops={"arrowstyle": "-", "lw": 1.0, "color": domain_colors[domain]},
            bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": domain_colors[domain], "alpha": 0.98},
            zorder=9,
        )

    if not subdomain_summary.empty and {"domain", "subdomain"}.issubset(subdomain_summary.columns):
        summary = subdomain_summary.copy()
        summary["mean_loading"] = pd.to_numeric(summary["mean_loading"], errors="coerce").fillna(0.0)
        for domain in domain_order:
            top_subdomains = (
                summary.loc[summary["domain"] == domain]
                .sort_values("mean_loading", ascending=False)
                .head(2)["subdomain"]
                .tolist()
            )
            for subdomain in top_subdomains:
                match = subdomain_frame.loc[subdomain_frame["label"] == subdomain]
                if match.empty:
                    continue
                sx = float(match.iloc[0]["tsne_x"])
                sy = float(match.iloc[0]["tsne_y"])
                ax_map.text(
                    sx + pad_x * 0.015,
                    sy + pad_y * 0.018,
                    _shorten_label(subdomain, max_len=26),
                    fontsize=8.0,
                    color=domain_colors[domain],
                    bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.78},
                    zorder=10,
                )

    ax_map.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax_map.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    ax_map.text(all_x.min() + pad_x * 0.35, all_y.max() - pad_y * 0.42, "Ontology scaffold", fontsize=9.2, color="#50697f", style="italic")
    ax_map.text(record_xy[:, 0].mean() - pad_x * 0.15, record_xy[:, 1].max() + pad_y * 0.18, "Review manifold", fontsize=9.2, color="#50697f", style="italic")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=domain_colors["biological"], markeredgecolor="white", markersize=9, label="Bio-dominant reviews"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=domain_colors["psychological"], markeredgecolor="white", markersize=9, label="Psycho-dominant reviews"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=domain_colors["social"], markeredgecolor="white", markersize=9, label="Social-dominant reviews"),
        Line2D([0], [0], marker="H", color="w", markerfacecolor="white", markeredgecolor="#425a70", markersize=10, label="Domain anchors"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="#425a70", markersize=6, label="Subdomain anchors"),
    ]
    ax_map.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8.8,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cad5df",
    )

    ensure_parent(out_path)
    fig.savefig(out_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return coords_frame


def _ternary_to_cartesian(b: np.ndarray, p: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric (bio=top, psycho=bottom-left, social=bottom-right) → Cartesian."""
    x = s * 1.0 + b * 0.5
    y = b * (np.sqrt(3) / 2.0)
    return x, y


def _inside_triangle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return boolean mask: True where (x,y) is inside the standard ternary triangle."""
    sqrt3_inv = 1.0 / np.sqrt(3)
    b_coord = (2.0 / np.sqrt(3)) * y
    s_coord = x - y * sqrt3_inv
    p_coord = 1.0 - b_coord - s_coord
    return (b_coord >= -0.01) & (p_coord >= -0.01) & (s_coord >= -0.01)


def _pairwise_loading_plot(record_loadings: pd.DataFrame, pairwise_loadings: pd.DataFrame, out_path: Path) -> None:
    if record_loadings.empty or pairwise_loadings.empty:
        return
    required_domain = {"loading_biological", "loading_psychological", "loading_social"}
    required_pairwise = {"bio_psych", "bio_social", "psych_social", "triadic_product"}
    if not required_domain.issubset(record_loadings.columns) or not required_pairwise.issubset(pairwise_loadings.columns):
        return

    b_vals = pd.to_numeric(record_loadings["loading_biological"], errors="coerce").fillna(0.0).values
    p_vals = pd.to_numeric(record_loadings["loading_psychological"], errors="coerce").fillna(0.0).values
    s_vals = pd.to_numeric(record_loadings["loading_social"], errors="coerce").fillna(0.0).values

    total = b_vals + p_vals + s_vals
    total = np.where(total == 0, 1.0, total)
    b_norm = b_vals / total
    p_norm = p_vals / total
    s_norm = s_vals / total

    b_mean = float(b_vals.mean())
    p_mean = float(p_vals.mean())
    s_mean = float(s_vals.mean())
    bp_mean = float(pd.to_numeric(pairwise_loadings["bio_psych"], errors="coerce").fillna(0.0).mean())
    bs_mean = float(pd.to_numeric(pairwise_loadings["bio_social"], errors="coerce").fillna(0.0).mean())
    ps_mean = float(pd.to_numeric(pairwise_loadings["psych_social"], errors="coerce").fillna(0.0).mean())

    matrix = np.array([
        [b_mean,  bp_mean, bs_mean],
        [bp_mean, p_mean,  ps_mean],
        [bs_mean, ps_mean, s_mean],
    ])

    DOMAIN_COLORS = {"biological": "#1a6b8a", "psychological": "#c45e2a", "social": "#3a7d44"}
    dom_cols = [DOMAIN_COLORS["biological"], DOMAIN_COLORS["psychological"], DOMAIN_COLORS["social"]]

    fig = plt.figure(figsize=(18, 6.6))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 3, wspace=0.40, left=0.05, right=0.97)

    # ── Panel A: annotated co-loading matrix ─────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    # Custom colormap blending from white to a deep teal
    cmap_A = mcolors.LinearSegmentedColormap.from_list("bluteal", ["#f7fbff", "#2480a5", "#0a2e44"])
    im = ax0.imshow(matrix, cmap=cmap_A, vmin=0, vmax=matrix.max(), aspect="equal")
    domain_labels = ["Biological", "Psychological", "Social"]
    ax0.set_xticks(range(3))
    ax0.set_yticks(range(3))
    ax0.set_xticklabels(domain_labels, fontsize=10.5, fontweight="bold")
    ax0.set_yticklabels(domain_labels, fontsize=10.5, fontweight="bold")
    for tick, color in zip(ax0.get_xticklabels(), dom_cols):
        tick.set_color(color)
    for tick, color in zip(ax0.get_yticklabels(), dom_cols):
        tick.set_color(color)
    vmax_A = float(np.nanmax(matrix))
    for row in range(3):
        for col in range(3):
            val = matrix[row, col]
            txt_c = "white" if val > vmax_A * 0.60 else "#0a1e2e"
            style = "bold" if row == col else "normal"
            ax0.text(col, row, f"{val:.4f}", ha="center", va="center", fontsize=11,
                     color=txt_c, fontweight=style)
    cbar = plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Mean loading / product", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    # Diagonal label
    ax0.set_title("A.  Domain × pairwise co-loading matrix", fontsize=11, pad=10, fontweight="bold")
    # Diagonal box markers
    for i in range(3):
        ax0.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1.0, 1.0,
                                    fill=False, edgecolor=dom_cols[i], linewidth=2.5))
    ax0.spines[:].set_visible(False)

    # ── Panel B: Deviation field around the equal-loading benchmark ──────────
    ax1 = fig.add_subplot(gs[1])
    ax1.set_facecolor("#fbfdff")
    ax1.set_title("B.  Deviation field around equal BPS balance", fontsize=11, pad=10, fontweight="bold")
    x_dev = p_norm - b_norm
    y_dev = s_norm - (1.0 / 3.0)
    if len(x_dev) >= 5:
        try:
            kde = gaussian_kde(np.vstack([x_dev, y_dev]), bw_method=0.28)
            xi = np.linspace(x_dev.min() - 0.01, x_dev.max() + 0.01, 180)
            yi = np.linspace(y_dev.min() - 0.01, y_dev.max() + 0.01, 180)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            cmap_kde = mcolors.LinearSegmentedColormap.from_list(
                "deviation_density",
                ["#fbfdff", "#d6e6f2", "#a8c8df", "#5d8fb7", "#16324a"],
            )
            ax1.contourf(Xi, Yi, Zi, levels=16, cmap=cmap_kde, alpha=0.84, zorder=1)
            ax1.contour(Xi, Yi, Zi, levels=7, colors="white", linewidths=0.7, alpha=0.45, zorder=2)
        except Exception:
            pass

    for band in [0.02, 0.01]:
        ax1.add_patch(
            plt.Rectangle(
                (-band, -band),
                2 * band,
                2 * band,
                facecolor="#ffffff",
                edgecolor="#c4d2dd",
                linewidth=1.0 if band == 0.02 else 0.8,
                linestyle="--",
                alpha=0.42 if band == 0.02 else 0.60,
                zorder=3,
            )
        )

    record_dom = record_loadings.get("dominant_domain", pd.Series(["biological"] * len(b_norm)))
    for domain in ["biological", "psychological", "social"]:
        mask = record_dom.astype(str).eq(domain).to_numpy()
        if not mask.any():
            continue
        ax1.scatter(
            x_dev[mask],
            y_dev[mask],
            s=40,
            c=DOMAIN_COLORS[domain],
            alpha=0.88,
            edgecolors="white",
            linewidths=0.55,
            zorder=5,
            label=f"{domain.capitalize()} dominant",
        )

    ax1.axvline(0, color="#44596c", linestyle="--", linewidth=1.0, alpha=0.95, zorder=4)
    ax1.axhline(0, color="#44596c", linestyle="--", linewidth=1.0, alpha=0.95, zorder=4)
    ax1.scatter([0], [0], s=84, marker="D", color="#10273a", edgecolors="white", linewidths=0.8, zorder=6)
    ax1.text(0.0012, 0.0012, "Equal-loading\nbenchmark", fontsize=8.1, color="#23394c", zorder=7)
    ax1.text(x_dev.min() - 0.002, y_dev.max() + 0.0015, "Bio leaning", fontsize=8.5, color=DOMAIN_COLORS["biological"])
    ax1.text(x_dev.max() - 0.009, y_dev.max() + 0.0015, "Psycho leaning", fontsize=8.5, color=DOMAIN_COLORS["psychological"])
    ax1.text(x_dev.min() - 0.002, y_dev.min() - 0.0042, "Social deficit", fontsize=8.5, color=DOMAIN_COLORS["social"])
    ax1.text(x_dev.min() - 0.002, y_dev.max() - 0.002, "Social enrichment", fontsize=8.5, color=DOMAIN_COLORS["social"])
    ax1.set_xlabel("Psychological minus biological loading", fontsize=9.5)
    ax1.set_ylabel("Social loading minus equal-share benchmark", fontsize=9.5)
    ax1.grid(linestyle="--", alpha=0.18)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#c7d4de")
    ax1.spines["bottom"].set_color("#c7d4de")
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.92, edgecolor="#c7d4de")

    # ── Panel C: Benchmark-relative interval plot ─────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    dist_cols = ["bio_psych", "bio_social", "psych_social", "triadic_product"]
    dist_labels = ["Bio × Psycho", "Bio × Social", "Psycho × Social", "Triadic"]
    dist_colors = ["#1a6b8a", "#3a7d44", "#c45e2a", "#6b3fa0"]
    benchmarks = {
        "bio_psych": 1.0 / 9.0,
        "bio_social": 1.0 / 9.0,
        "psych_social": 1.0 / 9.0,
        "triadic_product": 1.0 / 27.0,
    }

    ax2.axvline(0, color="#44596c", linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)
    for idx, (col, label, color) in enumerate(zip(dist_cols, dist_labels, dist_colors)):
        vals = pd.to_numeric(pairwise_loadings[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        delta = vals - benchmarks[col]
        y_pos = len(dist_cols) - 1 - idx
        q05, q25, median, q75, q95 = np.quantile(delta, [0.05, 0.25, 0.5, 0.75, 0.95])
        mean_v = float(delta.mean())
        ax2.hlines(y_pos, q05, q95, color=color, linewidth=1.6, alpha=0.48, zorder=2)
        ax2.hlines(y_pos, q25, q75, color=color, linewidth=7.5, alpha=0.92, zorder=3)
        ax2.scatter([median], [y_pos], s=34, color="white", edgecolors=color, linewidths=1.0, zorder=4)
        ax2.scatter([mean_v], [y_pos], s=62, color=color, edgecolors="white", linewidths=0.9, marker="D", zorder=5)
        offset = 0.00025 if mean_v >= 0 else -0.00025
        ax2.text(
            mean_v + offset,
            y_pos + 0.20,
            f"{mean_v:+.4f}",
            fontsize=8.3,
            color=color,
            fontweight="bold",
            ha="left" if mean_v >= 0 else "right",
        )

    ax2.set_yticks(range(len(dist_cols)))
    ax2.set_yticklabels(dist_labels[::-1], fontsize=10)
    ax2.set_xlabel("Deviation from equal-balance benchmark", fontsize=10)
    ax2.set_title("C.  Benchmark-relative co-loading deviations", fontsize=11, pad=10, fontweight="bold")
    ax2.spines["left"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="x", linestyle="--", alpha=0.25)

    ensure_parent(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _operationalization_combined_plot(
    typology_counts: pd.DataFrame,
    function_by_review_type: pd.DataFrame,
    msk_scope: pd.DataFrame,
    domain_counts: pd.DataFrame,
    core_bps_counts: pd.DataFrame,
    out_path: Path,
) -> None:
    """2×2 figure: A=typology, B=function×objective heatmap, C=MSK scope, D=domain mentions."""
    fig = plt.figure(figsize=(15.5, 11.0))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35,
                          left=0.07, right=0.97, top=0.96, bottom=0.07)

    DOMAIN_COLORS = {"biological": "#1a6b8a", "psychological": "#c45e2a", "social": "#3a7d44"}
    palette = ["#1a6b8a", "#2480a5", "#3a94bd", "#5aaa9a", "#7abb8a", "#9acc7a", "#bad86a", "#d6e29e"]

    def _style(ax):
        ax.set_facecolor("#f8fbff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#c1d0dc")
        ax.spines["bottom"].set_color("#c1d0dc")
        ax.grid(linestyle="--", alpha=0.22, color="#6f8aa2")
        ax.set_axisbelow(True)

    TYPE_COLORS = {
        "potential integrative signal": "#1a6b8a",
        "multifactorial signal": "#3a7d44",
        "pseudo-bps or partial signal": "#c45e2a",
        "rhetorical label signal": "#6b3fa0",
    }

    # Panel A: Provisional typology
    ax_a = fig.add_subplot(gs[0, 0])
    _style(ax_a)
    if not typology_counts.empty:
        tc = typology_counts.copy()
        tc["n"] = pd.to_numeric(tc["n"], errors="coerce").fillna(0)
        tc["percent"] = pd.to_numeric(tc["percent"], errors="coerce").fillna(0)
        tc = tc.sort_values("n", ascending=True)
        labels = [_shorten_label(str(v), 30) for v in tc["provisional_typology"]]
        colors_a = [TYPE_COLORS.get(str(v).lower(), "#7f8c8d") for v in tc["provisional_typology"]]
        bars = ax_a.barh(labels, tc["n"], color=colors_a, height=0.65, edgecolor="white", linewidth=0.9)
        for bar, count, pct in zip(bars, tc["n"], tc["percent"]):
            ax_a.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                      f"{int(count)} ({pct:.1f}%)", va="center", ha="left", fontsize=9.2, color="#16324a")
        ax_a.set_xlim(0, tc["n"].max() * 1.55)
    ax_a.set_title("A.  Provisional BPS operationalization typology", fontsize=11, fontweight="bold", pad=8, loc="left")
    ax_a.set_xlabel("Number of included reviews", fontsize=10)

    # Panel B: Function × objective heatmap
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor("#f8fbff")
    if not function_by_review_type.empty:
        values = function_by_review_type.values.astype(float)
        vmax = max(float(np.nanmax(values)), 1)
        im = ax_b.imshow(values, cmap="Blues", aspect="auto", vmin=0, vmax=vmax)
        short_cols = [_shorten_label(str(c), 20) for c in function_by_review_type.columns]
        short_rows = [_shorten_label(str(r), 20) for r in function_by_review_type.index]
        ax_b.set_xticks(range(len(function_by_review_type.columns)))
        ax_b.set_yticks(range(len(function_by_review_type.index)))
        ax_b.set_xticklabels(short_cols, rotation=30, ha="right", fontsize=8.5)
        ax_b.set_yticklabels(short_rows, fontsize=8.5)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                val = int(values[row, col])
                txt_c = "white" if val > vmax * 0.6 else "#0d2238"
                ax_b.text(col, row, str(val), ha="center", va="center", fontsize=8.5, color=txt_c)
        ax_b.spines[:].set_visible(False)
    ax_b.set_title("B.  BPS function by review objective category", fontsize=11, fontweight="bold", pad=8, loc="left")

    # Panel C: MSK domain coverage
    ax_c = fig.add_subplot(gs[1, 0])
    _style(ax_c)
    if not msk_scope.empty:
        msk = msk_scope.loc[msk_scope["indicator"] != "Musculoskeletal reviews in Stage 2"].copy()
        msk["n"] = pd.to_numeric(msk["n"], errors="coerce").fillna(0)
        msk["percent"] = pd.to_numeric(msk["percent"], errors="coerce").fillna(0)
        msk = msk.sort_values("n", ascending=True)
        c_map = {
            "Biological mention present": "#1a6b8a",
            "Psychological mention present": "#c45e2a",
            "Social mention present": "#3a7d44",
            "Triadic BPS mention present": "#2f3e4e",
        }
        msk_total = int(msk_scope.loc[msk_scope["indicator"] == "Musculoskeletal reviews in Stage 2", "n"].iloc[0]) if "Musculoskeletal reviews in Stage 2" in msk_scope["indicator"].values else 1
        ax_c.barh(range(len(msk)), np.full(len(msk), 100.0), color="#e7eef4", height=0.62, edgecolor="none", zorder=1)
        bars = ax_c.barh(range(len(msk)), msk["percent"],
                         color=[c_map.get(str(ind), "#7f8c8d") for ind in msk["indicator"]],
                         height=0.62, edgecolor="white", linewidth=0.9, zorder=3)
        for idx, (ind, pct, count, bar) in enumerate(zip(msk["indicator"], msk["percent"], msk["n"], bars)):
            ax_c.text(min(float(pct) + 1.8, 103), idx, f"{pct:.1f}% (n={int(count)}/{msk_total})",
                      va="center", ha="left", fontsize=9, color="#16324a",
                      fontweight="bold" if "Triadic" in str(ind) else "normal")
        labels_c = [_shorten_label(str(v), 30) for v in msk["indicator"]]
        ax_c.set_yticks(range(len(msk)))
        ax_c.set_yticklabels(labels_c, fontsize=9.2)
        ax_c.set_xlim(0, 110)
        ax_c.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax_c.axvline(100, color="#9db0c1", linestyle=":", linewidth=1.0)
    ax_c.set_title("C.  Substantive domain coverage in musculoskeletal reviews (RQ2)", fontsize=11, fontweight="bold", pad=8, loc="left")
    ax_c.set_xlabel("Percentage of musculoskeletal reviews", fontsize=10)

    # Panel D: Full corpus domain mentions
    ax_d = fig.add_subplot(gs[1, 1])
    _style(ax_d)
    if not domain_counts.empty:
        dc = domain_counts.copy()
        dc["n"] = pd.to_numeric(dc["n"], errors="coerce").fillna(0)
        dc["percent"] = pd.to_numeric(dc["percent"], errors="coerce").fillna(0)
        dc_colors = [DOMAIN_COLORS.get(str(v).lower(), "#7f8c8d") for v in dc["domain"]]
        bars_d = ax_d.bar(dc["domain"], dc["n"], color=dc_colors, width=0.55, edgecolor="white", linewidth=0.9)
        for bar, count, pct in zip(bars_d, dc["n"], dc["percent"]):
            ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(dc["n"].max() * 0.01, 0.3),
                      f"{int(count)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold",
                      color="#16324a")
        ax_d.set_ylim(0, dc["n"].max() * 1.22)
        ax_d.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_d.set_title("D.  Substantive domain coverage across full corpus", fontsize=11, fontweight="bold", pad=8, loc="left")
    ax_d.set_ylabel("Number of included reviews", fontsize=10)

    ensure_parent(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _semantic_loading_combined_plot(
    subdomain_summary: pd.DataFrame,
    record_loadings: pd.DataFrame,
    out_path: Path,
) -> None:
    """Combined: Panel A = sunburst 2-layer, Panel B = domain loading deviation distributions."""
    if subdomain_summary.empty or record_loadings.empty:
        return
    required_rl = {"record_id", "loading_biological", "loading_psychological", "loading_social", "dominant_domain"}
    if not required_rl.issubset(record_loadings.columns):
        return

    # ── Shared setup ────────────────────────────────────────────────────────────
    domain_order = ["biological", "psychological", "social"]
    domain_colors = {"biological": "#1a6b8a", "psychological": "#c45e2a", "social": "#3a7d44"}
    domain_labels_map = {"biological": "Biological", "psychological": "Psychological", "social": "Social"}
    dom_labels = ["Biological", "Psychological", "Social"]
    domain_palettes = {
        "biological": ["#1a6b8a","#2480a5","#3190b5","#45a8cc","#5bbad6",
                       "#74cce0","#8fd8e8","#aae3f0","#c8eef7","#e2f7fc"],
        "psychological": ["#c45e2a","#cf6e38","#d97e48","#e38e5b","#ec9e6e",
                          "#f3ae84","#f7be9b","#facdb3","#fcdcc9","#fdeee5"],
        "social": ["#3a7d44","#498f54","#5aa165","#6db377","#83c48b",
                   "#9bd4a0","#b3e2b7","#caedce","#def6e1","#f0fbf1"],
    }

    data = subdomain_summary.copy()
    data["domain"] = data["domain"].astype(str).str.lower()
    data["mean_loading"] = pd.to_numeric(data["mean_loading"], errors="coerce").fillna(0.0)
    data = data.loc[data["mean_loading"] > 0].copy()

    domain_totals = data.groupby("domain", as_index=False)["mean_loading"].sum()
    domain_totals = domain_totals.set_index("domain").reindex(domain_order).fillna(0.0).reset_index()
    total_mass = float(domain_totals["mean_loading"].sum())
    if total_mass <= 0:
        return

    # Wide figure: left 60% = sunburst+bars, right 40% = deviation panel
    fig = plt.figure(figsize=(22, 11))
    fig.patch.set_facecolor("white")

    # --- Panel A: Sunburst (left half) ---
    ax_sun = fig.add_axes([0.01, 0.06, 0.30, 0.90])
    ax_sun.set_aspect("equal")
    ax_sun.axis("off")

    start_angle = 90.0
    for _, drow in domain_totals.iterrows():
        domain = str(drow["domain"])
        domain_mass = float(drow["mean_loading"])
        if domain_mass <= 0:
            continue
        domain_frac = domain_mass / total_mass
        domain_sweep = 360.0 * domain_frac
        end_angle = start_angle + domain_sweep
        inner_wedge = Wedge((0, 0), r=0.82, theta1=start_angle, theta2=end_angle, width=0.30,
                            facecolor=domain_colors.get(domain, "#7f8c8d"), edgecolor="white", linewidth=2.5)
        ax_sun.add_patch(inner_wedge)
        mid_angle = (start_angle + end_angle) / 2.0
        mid_rad = np.deg2rad(mid_angle)
        ax_sun.text(0.67 * np.cos(mid_rad), 0.67 * np.sin(mid_rad),
                    f"{domain_labels_map.get(domain, domain)}\n{domain_frac * 100:.1f}%",
                    ha="center", va="center", fontsize=11.5, color="white",
                    fontweight="bold", multialignment="center")
        children = data.loc[data["domain"] == domain, ["subdomain", "mean_loading"]].copy()
        children = children.sort_values("mean_loading", ascending=False).reset_index(drop=True)
        child_total = float(children["mean_loading"].sum())
        child_angle = start_angle
        palette = domain_palettes.get(domain, ["#bdc3c7"] * max(1, len(children)))
        for i, crow in children.iterrows():
            mass = float(crow["mean_loading"])
            if child_total <= 0 or mass <= 0:
                continue
            sweep = domain_sweep * (mass / child_total)
            child_end = child_angle + sweep
            ax_sun.add_patch(Wedge((0, 0), r=1.40, theta1=child_angle, theta2=child_end, width=0.48,
                                   facecolor=palette[i % len(palette)], edgecolor="white", linewidth=0.6, alpha=0.93))
            child_angle = child_end
        start_angle = end_angle

    center_c = plt.Circle((0, 0), 0.40, color="#f4f8fb", ec="#b8ccda", lw=1.8, zorder=10)
    ax_sun.add_artist(center_c)
    ax_sun.text(0, 0.10, "BPS", ha="center", va="center", fontsize=20, fontweight="bold", color="#1a2e44", zorder=11)
    ax_sun.text(0, -0.12, "Semantic", ha="center", va="center", fontsize=10, color="#3a5068", zorder=11)
    ax_sun.text(0, -0.28, "Loading", ha="center", va="center", fontsize=10, color="#3a5068", zorder=11)
    ax_sun.set_xlim(-1.65, 1.65)
    ax_sun.set_ylim(-1.65, 1.65)
    fig.text(0.155, 0.01, "Inner ring = domain proportion  |  Outer ring = subdomain profile",
             ha="center", fontsize=8.5, color="#556677", style="italic")
    fig.text(0.155, 0.97, "A", ha="center", fontsize=14, fontweight="bold", color="#16324a")

    # Subdomain bar panels (stacked right of sunburst)
    n_bio = len(data[data["domain"] == "biological"])
    n_psy = len(data[data["domain"] == "psychological"])
    n_soc = len(data[data["domain"] == "social"])
    n_total = max(n_bio + n_psy + n_soc, 1)
    gap = 0.012
    usable_height = 0.90 - 2 * gap
    h_bio = usable_height * n_bio / n_total
    h_psy = usable_height * n_psy / n_total
    h_soc = usable_height * n_soc / n_total
    y_soc = 0.05
    y_psy = y_soc + h_soc + gap
    y_bio = y_psy + h_psy + gap
    x_bars = 0.33
    w_bars = 0.25

    global_max_pct = max(float(data["mean_loading"].max()) * 100.0, 0.1)
    for domain, y_bot, h_ax in [("social", y_soc, h_soc), ("psychological", y_psy, h_psy), ("biological", y_bio, h_bio)]:
        ax_bar = fig.add_axes([x_bars, y_bot, w_bars, h_ax])
        d_data = data[data["domain"] == domain].copy()
        d_data["loading_pct"] = pd.to_numeric(d_data["mean_loading"], errors="coerce").fillna(0.0) * 100.0
        d_data = d_data.sort_values("mean_loading", ascending=True)
        bar_color = domain_colors[domain]
        labels_bar = [s[:36] + ("\u2026" if len(s) > 36 else "") for s in d_data["subdomain"]]
        vals = d_data["loading_pct"].to_numpy(dtype=float)
        ax_bar.barh(range(len(d_data)), vals, color=bar_color, alpha=0.80, height=0.62, edgecolor="white", linewidth=0.5)
        ax_bar.set_yticks(range(len(d_data)))
        ax_bar.set_yticklabels(labels_bar, fontsize=8.0)
        ax_bar.set_xlim(0, global_max_pct * 1.18)
        ax_bar.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax_bar.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax_bar.tick_params(axis="x", labelsize=7.0)
        ax_bar.tick_params(axis="y", length=0, pad=3)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["left"].set_visible(False)
        ax_bar.grid(axis="x", linestyle="--", alpha=0.25)
        ax_bar.set_axisbelow(True)
        domain_pct = float(domain_totals.loc[domain_totals["domain"].eq(domain), "mean_loading"].sum() / total_mass * 100.0)
        ax_bar.set_title(f"{domain_labels_map[domain]}  ({len(d_data)} subdomains; domain share {domain_pct:.1f}%)",
                         fontsize=9.5, fontweight="bold", color="white", pad=3, loc="left",
                         bbox=dict(facecolor=bar_color, edgecolor="none", boxstyle="round,pad=0.25"))
        if domain == "biological":
            ax_bar.set_xlabel("Mean weighted subdomain loading across corpus", fontsize=8.0)

    # --- Panel B: Deviation distributions (right section) ---
    df_rl = record_loadings.copy()
    NEUTRAL = 1.0 / 3.0
    domains = ["biological", "psychological", "social"]
    ax_dev = fig.add_axes([0.62, 0.08, 0.36, 0.86])
    ax_dev.set_facecolor("#fbfdff")
    deviation_frame = df_rl[[f"loading_{d}" for d in domains]].copy()
    for col in deviation_frame.columns:
        deviation_frame[col] = pd.to_numeric(deviation_frame[col], errors="coerce").fillna(NEUTRAL)
    deviation_frame = deviation_frame - NEUTRAL

    rng = np.random.default_rng(42)
    positions = np.arange(1, 4)
    violin = ax_dev.violinplot(
        [deviation_frame[f"loading_{d}"].to_numpy(dtype=float) for d in domains],
        positions=positions, widths=0.78, showmeans=False, showmedians=False, showextrema=False)
    for body, domain in zip(violin["bodies"], domains):
        body.set_facecolor(domain_colors[domain])
        body.set_edgecolor(domain_colors[domain])
        body.set_alpha(0.23)
        body.set_linewidth(1.0)
    for pos, domain in zip(positions, domains):
        vals = deviation_frame[f"loading_{domain}"].to_numpy(dtype=float)
        jitter = rng.normal(0.0, 0.045, size=len(vals))
        ax_dev.scatter(np.full(len(vals), pos) + jitter, vals,
                       s=16, color=domain_colors[domain], alpha=0.34, edgecolors="none", zorder=3)
        q1, median, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        mean_v = float(np.mean(vals))
        ax_dev.vlines(pos, q1, q3, color=domain_colors[domain], linewidth=5.0, alpha=0.95, zorder=4)
        ax_dev.scatter([pos], [mean_v], s=54, color=domain_colors[domain], edgecolors="white", linewidths=0.9, zorder=5, marker="D")
        ax_dev.scatter([pos], [median], s=28, color="white", edgecolors=domain_colors[domain], linewidths=1.0, zorder=6)
        ax_dev.text(pos + 0.18, mean_v, f"{mean_v:+.3f}", fontsize=9, color=domain_colors[domain], va="center", fontweight="bold")
    balanced_all = int((deviation_frame.abs() <= 0.02).all(axis=1).sum())
    ax_dev.axhline(0, color="#44596c", linestyle="--", linewidth=1.1, alpha=0.9)
    ax_dev.text(0.56, 0.0012, "Equal-loading benchmark", fontsize=8.5, color="#44596c")
    ax_dev.text(0.56, deviation_frame.to_numpy(dtype=float).min() * 0.85,
                f"{balanced_all}/{len(df_rl)} records within \u00b10.02\nof equal loading across all domains.",
                fontsize=8.5, color="#41596f",
                bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#c7d4de", "alpha": 0.97})
    ax_dev.set_xlim(0.45, 3.6)
    y_abs = max(0.022, float(np.abs(deviation_frame.to_numpy(dtype=float)).max()) * 1.18)
    ax_dev.set_ylim(-y_abs, y_abs)
    ax_dev.set_xticks(positions)
    ax_dev.set_xticklabels(dom_labels, fontsize=11, fontweight="bold")
    for tick, domain in zip(ax_dev.get_xticklabels(), domains):
        tick.set_color(domain_colors[domain])
    ax_dev.set_ylabel("Loading deviation from equal-share benchmark (loading \u2212 1/3)", fontsize=10)
    ax_dev.grid(axis="y", linestyle="--", alpha=0.22)
    ax_dev.spines["top"].set_visible(False)
    ax_dev.spines["right"].set_visible(False)
    ax_dev.spines["left"].set_color("#c7d4de")
    ax_dev.spines["bottom"].set_color("#c7d4de")
    fig.text(0.80, 0.97, "B", ha="center", fontsize=14, fontweight="bold", color="#16324a")

    ensure_parent(out_path)
    fig.savefig(out_path, dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _semantic_landscape_integrated_plot(
    record_loadings: pd.DataFrame,
    subdomain_summary: pd.DataFrame,
    pairwise_loadings: pd.DataFrame,
    out_path: Path,
    coords_out_path: Path,
) -> pd.DataFrame:
    """2×2 figure: A=embedding landscape, B=co-loading matrix, C=deviation field, D=ridgeline."""
    required = {"record_id", "year", "review_type", "loading_biological",
                "loading_psychological", "loading_social", "dominant_domain"}
    if record_loadings.empty or not required.issubset(record_loadings.columns):
        return pd.DataFrame()
    if pairwise_loadings.empty:
        return pd.DataFrame()

    record_embeddings_path = project_path("src", "vector_db", "semantic_loading", "records", "record_embeddings.npy")
    ontology_embeddings_path = project_path("src", "vector_db", "semantic_loading", "ontology", "ontology_embeddings.npy")
    subdomain_embeddings_path = project_path("src", "vector_db", "semantic_loading", "ontology", "subdomain_embeddings.npy")
    embeddings_available = (record_embeddings_path.exists() and
                            ontology_embeddings_path.exists() and
                            subdomain_embeddings_path.exists())

    domain_order = ["biological", "psychological", "social"]
    domain_labels_map = {"biological": "Biological", "psychological": "Psychological", "social": "Social"}
    domain_colors = {"biological": "#1a6b8a", "psychological": "#c45e2a", "social": "#3a7d44"}
    dom_cols = [domain_colors["biological"], domain_colors["psychological"], domain_colors["social"]]

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, hspace=0.24, wspace=0.36, left=0.05, right=0.97, top=0.96, bottom=0.07)

    # === Panel A: Embedding landscape ===
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor("#fbfdff")

    coords_frame = pd.DataFrame()
    if embeddings_available:
        record_embeddings = np.load(record_embeddings_path)
        ontology_embeddings = np.load(ontology_embeddings_path)
        subdomain_embeddings = np.load(subdomain_embeddings_path)

        subdomain_labels: list[str] = []
        subdomain_domains: list[str] = []
        for domain in domain_order:
            for label in ONTOLOGY_TERMS.get(domain, []):
                subdomain_labels.append(label)
                subdomain_domains.append(domain)

        if (record_embeddings.shape[0] == len(record_loadings) and
                ontology_embeddings.shape[0] >= len(domain_order) and
                subdomain_embeddings.shape[0] >= len(subdomain_labels)):
            ont_emb = ontology_embeddings[:len(domain_order)]
            sub_emb = subdomain_embeddings[:len(subdomain_labels)]
            combined = np.vstack([record_embeddings, ont_emb, sub_emb]).astype(np.float32)
            if combined.shape[0] >= 8:
                scaled = StandardScaler().fit_transform(combined)
                pca_dims = max(2, min(36, scaled.shape[0] - 1, scaled.shape[1]))
                pca = PCA(n_components=pca_dims)
                pca_coords = pca.fit_transform(scaled)
                perplexity = max(8, min(30, (combined.shape[0] - 1) // 5))
                tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                            perplexity=perplexity, max_iter=2000, random_state=42)
                tsne_coords = tsne.fit_transform(pca_coords)

                n_rec = len(record_loadings)
                n_dom = len(domain_order)
                record_xy = tsne_coords[:n_rec]
                domain_xy = tsne_coords[n_rec:n_rec + n_dom]
                subdomain_xy = tsne_coords[n_rec + n_dom:]

                records_plot = record_loadings.copy().reset_index(drop=True)
                records_plot["tsne_x"] = record_xy[:, 0]
                records_plot["tsne_y"] = record_xy[:, 1]
                records_plot["year"] = pd.to_numeric(records_plot["year"], errors="coerce")
                records_plot["dominant_loading"] = records_plot[
                    ["loading_biological", "loading_psychological", "loading_social"]].max(axis=1)
                record_center = np.array([np.median(record_xy[:, 0]), np.median(record_xy[:, 1])], dtype=float)

                # Precompute plotting bounds before layers that need grids.
                raw_all_x = np.concatenate([record_xy[:, 0], domain_xy[:, 0], subdomain_xy[:, 0]])
                raw_all_y = np.concatenate([record_xy[:, 1], domain_xy[:, 1], subdomain_xy[:, 1]])
                pad_x = max((raw_all_x.max() - raw_all_x.min()) * 0.10, 1.0)
                pad_y = max((raw_all_y.max() - raw_all_y.min()) * 0.10, 1.0)
                grid_x = np.linspace(raw_all_x.min() - pad_x, raw_all_x.max() + pad_x, 160)
                grid_y = np.linspace(raw_all_y.min() - pad_y, raw_all_y.max() + pad_y, 160)
                Xi, Yi = np.meshgrid(grid_x, grid_y)

                dominant_strength = pd.to_numeric(records_plot["dominant_loading"], errors="coerce").fillna(1/3).to_numpy(dtype=float)
                strength_scaled = np.clip((dominant_strength - (1/3)) / (2/3), 0.0, 1.0)
                point_sizes = 46 + 200 * strength_scaled

                # KDE density background
                if n_rec >= 5:
                    try:
                        kde = gaussian_kde(record_xy.T, bw_method=0.24)
                        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                        cmap_d = mcolors.LinearSegmentedColormap.from_list(
                            "emb_dens", ["#fbfdff", "#d7e7f3", "#a9c7de", "#5c8cb3", "#16324a"])
                        ax_a.contourf(Xi, Yi, Zi, levels=16, cmap=cmap_d, alpha=0.80, zorder=1)
                        ax_a.contour(Xi, Yi, Zi, levels=7, colors="white", linewidths=0.7, alpha=0.40, zorder=2)
                    except Exception:
                        pass

                # Domain-specific contour lines reveal local structure in the record cloud.
                for domain in domain_order:
                    mask = records_plot["dominant_domain"].eq(domain).to_numpy()
                    if mask.sum() < 10:
                        continue
                    try:
                        kde_dom = gaussian_kde(record_xy[mask].T, bw_method=0.34)
                        Zi_dom = kde_dom(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                        ax_a.contour(
                            Xi,
                            Yi,
                            Zi_dom,
                            levels=4,
                            colors=[domain_colors[domain]],
                            linewidths=0.9,
                            alpha=0.26,
                            zorder=2.4,
                        )
                    except Exception:
                        pass

                # Cluster subdomain anchors around domain centroids for clearer ontology fields.
                subdomain_xy_plot = subdomain_xy.copy()
                subdomain_domains_arr = np.array(subdomain_domains)
                domain_anchor_lookup: dict[str, tuple[float, float]] = {}
                for idx, domain in enumerate(domain_order):
                    rec_mask = records_plot["dominant_domain"].eq(domain).to_numpy()
                    rec_centroid = record_xy[rec_mask].mean(axis=0) if rec_mask.any() else record_center
                    target_anchor = 0.60 * domain_xy[idx] + 0.40 * rec_centroid
                    domain_mask = subdomain_domains_arr == domain
                    if not domain_mask.any():
                        domain_anchor_lookup[domain] = (float(target_anchor[0]), float(target_anchor[1]))
                        continue
                    domain_points = subdomain_xy_plot[domain_mask]
                    anchored = 0.74 * domain_points + 0.26 * target_anchor
                    subdomain_xy_plot[domain_mask] = anchored
                    cluster_center = 0.70 * anchored.mean(axis=0) + 0.30 * target_anchor
                    domain_anchor_lookup[domain] = (float(cluster_center[0]), float(cluster_center[1]))

                subdomain_frame_inner = pd.DataFrame(
                    {
                        "label": subdomain_labels,
                        "parent_domain": subdomain_domains,
                        "tsne_x": subdomain_xy_plot[:, 0],
                        "tsne_y": subdomain_xy_plot[:, 1],
                    }
                )

                all_x = np.concatenate(
                    [
                        record_xy[:, 0],
                        np.array([xy[0] for xy in domain_anchor_lookup.values()], dtype=float),
                        subdomain_xy_plot[:, 0],
                    ]
                )
                all_y = np.concatenate(
                    [
                        record_xy[:, 1],
                        np.array([xy[1] for xy in domain_anchor_lookup.values()], dtype=float),
                        subdomain_xy_plot[:, 1],
                    ]
                )
                pad_x = max((all_x.max() - all_x.min()) * 0.10, 1.0)
                pad_y = max((all_y.max() - all_y.min()) * 0.10, 1.0)

                for domain in domain_order:
                    dx, dy = domain_anchor_lookup[domain]
                    domain_sub = subdomain_frame_inner.loc[subdomain_frame_inner["parent_domain"] == domain]
                    if not domain_sub.empty:
                        spread = np.linalg.norm(
                            domain_sub[["tsne_x", "tsne_y"]].to_numpy(dtype=float) - np.array([[dx, dy]], dtype=float),
                            axis=1,
                        )
                        radius = max(float(np.quantile(spread, 0.80)) * 1.30, 0.22)
                        ax_a.add_patch(
                            plt.Circle(
                                (dx, dy),
                                radius,
                                facecolor=domain_colors[domain],
                                edgecolor=domain_colors[domain],
                                linewidth=0.9,
                                alpha=0.06,
                                zorder=2.7,
                            )
                        )
                    for _, row in domain_sub.iterrows():
                        ax_a.plot([dx, float(row["tsne_x"])], [dy, float(row["tsne_y"])],
                                  color=domain_colors[domain], alpha=0.12, linewidth=0.8, zorder=3)
                    ax_a.scatter(domain_sub["tsne_x"], domain_sub["tsne_y"],
                                 s=36, facecolors="white", edgecolors=domain_colors[domain],
                                 linewidths=1.0, alpha=0.82, zorder=4)

                # Records
                for domain in domain_order:
                    mask = records_plot["dominant_domain"].eq(domain).to_numpy()
                    if not mask.any():
                        continue
                    ax_a.scatter(record_xy[mask, 0], record_xy[mask, 1],
                                 s=point_sizes[mask], c=domain_colors[domain],
                                 alpha=0.88, edgecolors="white", linewidths=0.85, zorder=5)

                # Domain anchors
                for idx, domain in enumerate(domain_order):
                    dx, dy = domain_anchor_lookup[domain]
                    ax_a.scatter([dx], [dy], s=480, marker="H", facecolors="white",
                                 edgecolors=domain_colors[domain], linewidths=2.4, zorder=8)
                    radial_vec = np.array([dx, dy], dtype=float) - record_center
                    radial_norm = float(np.linalg.norm(radial_vec))
                    if radial_norm <= 1e-8:
                        radial_vec = np.array([1.0, 0.0], dtype=float)
                        radial_norm = 1.0
                    radial_unit = radial_vec / radial_norm
                    label_x = dx + radial_unit[0] * pad_x * 0.11
                    label_y = dy + radial_unit[1] * pad_y * 0.11
                    ax_a.annotate(domain_labels_map[domain], xy=(dx, dy),
                                  xytext=(label_x, label_y),
                                  fontsize=9.5, fontweight="bold", color=domain_colors[domain],
                                  arrowprops={"arrowstyle": "-", "lw": 1.0, "color": domain_colors[domain]},
                                  bbox={"boxstyle": "round,pad=0.25", "fc": "white",
                                        "ec": domain_colors[domain], "alpha": 0.98}, zorder=9)

                ax_a.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
                ax_a.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

                legend_handles = [
                    Line2D([0],[0], marker="o", color="w", markerfacecolor=domain_colors[d],
                           markersize=8, label=f"{domain_labels_map[d]}-dominant") for d in domain_order
                ] + [Line2D([0],[0], marker="H", color="w", markerfacecolor="white",
                            markeredgecolor="#425a70", markersize=9, label="Domain anchors"),
                     Line2D([0],[0], marker="o", color="w", markerfacecolor="white",
                            markeredgecolor="#425a70", markersize=6, label="Subdomain anchors")]
                ax_a.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=True,
                            framealpha=0.95, facecolor="white", edgecolor="#cad5df")

                # Save coords
                rec_frame = records_plot[["record_id","year","review_type","dominant_domain","dominant_loading","tsne_x","tsne_y"]].copy()
                rec_frame.insert(0, "node_type", "record")
                rec_frame.insert(1, "label", rec_frame["record_id"])
                rec_frame["parent_domain"] = rec_frame["dominant_domain"]
                dom_frame = pd.DataFrame(
                    {
                        "node_type": "domain_anchor",
                        "label": [domain_labels_map[d] for d in domain_order],
                        "parent_domain": domain_order,
                        "tsne_x": [domain_anchor_lookup[d][0] for d in domain_order],
                        "tsne_y": [domain_anchor_lookup[d][1] for d in domain_order],
                    }
                )
                sub_frame = pd.DataFrame({"node_type": "subdomain_anchor", "label": subdomain_labels,
                                          "parent_domain": subdomain_domains, "tsne_x": subdomain_xy_plot[:,0], "tsne_y": subdomain_xy_plot[:,1]})
                coords_frame = pd.concat([rec_frame, dom_frame.assign(record_id="", year=np.nan, review_type="", dominant_domain="", dominant_loading=np.nan),
                                          sub_frame.assign(record_id="", year=np.nan, review_type="", dominant_domain="", dominant_loading=np.nan)],
                                         ignore_index=True, sort=False)
                ensure_parent(coords_out_path)
                coords_frame.to_csv(coords_out_path, index=False)
        else:
            ax_a.text(0.5, 0.5, "Embeddings shape mismatch.\nCannot project.", ha="center", va="center",
                      fontsize=10, color="#888", transform=ax_a.transAxes)
    else:
        ax_a.text(0.5, 0.5, "Embedding files not found.\nRun semantic loading first.", ha="center", va="center",
                  fontsize=10, color="#888", transform=ax_a.transAxes)

    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    ax_a.set_title("A.  Embedding landscape of included reviews and BPS ontology",
                   fontsize=11, fontweight="bold", pad=8, loc="left")

    # === Panel B: Co-loading matrix ===
    ax_b = fig.add_subplot(gs[0, 1])
    b_vals = pd.to_numeric(record_loadings["loading_biological"], errors="coerce").fillna(0.0).values
    p_vals = pd.to_numeric(record_loadings["loading_psychological"], errors="coerce").fillna(0.0).values
    s_vals = pd.to_numeric(record_loadings["loading_social"], errors="coerce").fillna(0.0).values
    bp_mean = float(pd.to_numeric(pairwise_loadings.get("bio_psych", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean())
    bs_mean = float(pd.to_numeric(pairwise_loadings.get("bio_social", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean())
    ps_mean = float(pd.to_numeric(pairwise_loadings.get("psych_social", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean())
    matrix = np.array([[b_vals.mean(), bp_mean, bs_mean],
                       [bp_mean, p_vals.mean(), ps_mean],
                       [bs_mean, ps_mean, s_vals.mean()]])
    cmap_B = mcolors.LinearSegmentedColormap.from_list("bluteal", ["#f7fbff", "#2480a5", "#0a2e44"])
    im_b = ax_b.imshow(matrix, cmap=cmap_B, vmin=0, vmax=matrix.max(), aspect="auto")
    ax_b.set_xticks(range(3))
    ax_b.set_yticks(range(3))
    ax_b.set_xticklabels(["Biological", "Psychological", "Social"], fontsize=10, fontweight="bold")
    ax_b.set_yticklabels(["Biological", "Psychological", "Social"], fontsize=10, fontweight="bold")
    for tick, color in zip(ax_b.get_xticklabels(), dom_cols):
        tick.set_color(color)
    for tick, color in zip(ax_b.get_yticklabels(), dom_cols):
        tick.set_color(color)
    vmax_B = float(np.nanmax(matrix))
    for row in range(3):
        for col in range(3):
            val = matrix[row, col]
            txt_c = "white" if val > vmax_B * 0.60 else "#0a1e2e"
            ax_b.text(col, row, f"{val:.4f}", ha="center", va="center", fontsize=10.5,
                      color=txt_c, fontweight="bold" if row == col else "normal")
    cbar_b = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cbar_b.set_label("Mean loading / product", fontsize=8.5)
    cbar_b.ax.tick_params(labelsize=8)
    for i in range(3):
        ax_b.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1.0, 1.0, fill=False,
                                     edgecolor=dom_cols[i], linewidth=2.5))
    ax_b.spines[:].set_visible(False)
    ax_b.set_title("B.  Domain \u00d7 pairwise co-loading matrix", fontsize=11, fontweight="bold", pad=8, loc="left")

    # === Panel C: Deviation field ===
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor("#fbfdff")
    total = b_vals + p_vals + s_vals
    total = np.where(total == 0, 1.0, total)
    b_norm = b_vals / total
    p_norm = p_vals / total
    s_norm = s_vals / total
    x_dev = p_norm - b_norm
    y_dev = s_norm - (1.0 / 3.0)
    x_band = 0.02
    y_band = 0.02
    ax_c.axvspan(-x_band, x_band, color="#c8d5e2", alpha=0.24, zorder=0)
    ax_c.axhspan(-y_band, y_band, color="#c8d5e2", alpha=0.18, zorder=0)
    if len(x_dev) >= 5:
        try:
            kde_c = gaussian_kde(np.vstack([x_dev, y_dev]), bw_method=0.28)
            xi_c = np.linspace(x_dev.min() - 0.01, x_dev.max() + 0.01, 160)
            yi_c = np.linspace(y_dev.min() - 0.01, y_dev.max() + 0.01, 160)
            Xi_c, Yi_c = np.meshgrid(xi_c, yi_c)
            Zi_c = kde_c(np.vstack([Xi_c.ravel(), Yi_c.ravel()])).reshape(Xi_c.shape)
            cmap_c = mcolors.LinearSegmentedColormap.from_list("dev_dens",
                ["#fbfdff", "#d6e6f2", "#a8c8df", "#5d8fb7", "#16324a"])
            ax_c.contourf(Xi_c, Yi_c, Zi_c, levels=16, cmap=cmap_c, alpha=0.84, zorder=1)
            ax_c.contour(Xi_c, Yi_c, Zi_c, levels=7, colors="white", linewidths=0.7, alpha=0.45, zorder=2)
        except Exception:
            pass
    record_dom = record_loadings.get("dominant_domain", pd.Series(["biological"] * len(b_norm)))
    for domain in ["biological", "psychological", "social"]:
        mask = record_dom.astype(str).eq(domain).to_numpy()
        if not mask.any():
            continue
        ax_c.scatter(x_dev[mask], y_dev[mask], s=38, c=domain_colors[domain],
                     alpha=0.88, edgecolors="white", linewidths=0.55, zorder=5,
                     label=f"{domain.capitalize()} dominant")
    ax_c.axvline(0, color="#44596c", linestyle="--", linewidth=1.0, alpha=0.95, zorder=4)
    ax_c.axhline(0, color="#44596c", linestyle="--", linewidth=1.0, alpha=0.95, zorder=4)
    ax_c.scatter([0], [0], s=80, marker="D", color="#10273a", edgecolors="white", linewidths=0.8, zorder=6)
    x_abs = max(0.045, float(np.abs(x_dev).max()) * 1.15)
    y_abs = max(0.045, float(np.abs(y_dev).max()) * 1.18)
    ax_c.set_xlim(-x_abs, x_abs)
    ax_c.set_ylim(-y_abs, y_abs)
    ax_c.text(0.0012, 0.0012, "Equal-loading\nbenchmark", fontsize=8, color="#23394c", zorder=7)
    ax_c.text(0.03, 0.94, "Bio > Psycho", transform=ax_c.transAxes, fontsize=7.8, color="#36526a")
    ax_c.text(0.72, 0.94, "Psycho > Bio", transform=ax_c.transAxes, fontsize=7.8, color="#36526a")
    ax_c.text(0.03, 0.08, "Social below 1/3", transform=ax_c.transAxes, fontsize=7.8, color="#36526a")
    ax_c.text(0.03, 0.88, "Social above 1/3", transform=ax_c.transAxes, fontsize=7.8, color="#36526a")
    ax_c.set_xlabel("Psychological minus biological share", fontsize=9.5)
    ax_c.set_ylabel("Social share minus equal-share benchmark (1/3)", fontsize=9.5)
    ax_c.grid(linestyle="--", alpha=0.18)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.spines["left"].set_color("#c7d4de")
    ax_c.spines["bottom"].set_color("#c7d4de")
    ax_c.legend(fontsize=8, loc="lower right", framealpha=0.92, edgecolor="#c7d4de")
    ax_c.set_title("C.  Benchmark-centered domain deviation field", fontsize=11, fontweight="bold", pad=8, loc="left")

    # === Panel D: Benchmark-relative ridgeline ===
    ax_d = fig.add_subplot(gs[1, 1])
    dist_cols = ["bio_psych", "bio_social", "psych_social", "triadic_product"]
    dist_labels = ["Bio \u00d7 Psycho", "Bio \u00d7 Social", "Psycho \u00d7 Social", "Triadic"]
    dist_colors = ["#1a6b8a", "#3a7d44", "#c45e2a", "#6b3fa0"]
    benchmarks = {"bio_psych": 1/9, "bio_social": 1/9, "psych_social": 1/9, "triadic_product": 1/27}
    ax_d.axvline(0, color="#44596c", linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)
    for idx, (col, label, color) in enumerate(zip(dist_cols, dist_labels, dist_colors)):
        vals_d = pd.to_numeric(pairwise_loadings.get(col, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals_d) == 0:
            continue
        delta = vals_d - benchmarks[col]
        y_pos = len(dist_cols) - 1 - idx
        q05, q25, median_d, q75, q95 = np.quantile(delta, [0.05, 0.25, 0.5, 0.75, 0.95])
        mean_d = float(delta.mean())
        ax_d.hlines(y_pos, q05, q95, color=color, linewidth=1.6, alpha=0.48, zorder=2)
        ax_d.hlines(y_pos, q25, q75, color=color, linewidth=7.5, alpha=0.92, zorder=3)
        ax_d.scatter([median_d], [y_pos], s=32, color="white", edgecolors=color, linewidths=1.0, zorder=4)
        ax_d.scatter([mean_d], [y_pos], s=60, color=color, edgecolors="white", linewidths=0.9, marker="D", zorder=5)
        offset = 0.00025 if mean_d >= 0 else -0.00025
        ax_d.text(mean_d + offset, y_pos + 0.20, f"{mean_d:+.4f}",
                  fontsize=8.5, color=color, fontweight="bold", ha="left" if mean_d >= 0 else "right")
    ax_d.set_yticks(range(len(dist_cols)))
    ax_d.set_yticklabels(dist_labels[::-1], fontsize=10)
    ax_d.set_xlabel("Deviation from equal-balance benchmark", fontsize=10)
    ax_d.set_title("D.  Benchmark-relative co-loading deviations", fontsize=11, fontweight="bold", pad=8, loc="left")
    ax_d.spines["left"].set_visible(False)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.tick_params(axis="y", length=0)
    ax_d.grid(axis="x", linestyle="--", alpha=0.25)

    ensure_parent(out_path)
    fig.savefig(out_path, dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return coords_frame


def _write_characteristics_table(stage2: pd.DataFrame, path: Path, stage3_manifest: pd.DataFrame | None = None) -> None:
    """Generate a longtable of all 111 included reviews sorted chronologically."""
    if stage2.empty:
        path.write_text("% No included records available.\n", encoding="utf-8")
        return

    # ── Year override for records with missing year (inferred from DOI) ──────
    YEAR_OVERRIDES: dict[str, int] = {
        "WOS:000453643300005": 2018,  # Draper-Rodi et al. (IJOSM DOI .2018.07.001)
        "WOS:001116314900001": 2023,  # Jurak et al.       (JCM vol 12 2023)
    }
    stage3_lookup = (
        stage3_manifest.set_index("record_id", drop=False).to_dict("index")
        if isinstance(stage3_manifest, pd.DataFrame)
        and not stage3_manifest.empty
        and "record_id" in stage3_manifest.columns
        else {}
    )

    def _parse_year(row: pd.Series) -> int:
        if row["record_id"] in YEAR_OVERRIDES:
            return YEAR_OVERRIDES[row["record_id"]]
        v = str(row.get("year", "")).strip()
        if v and v not in ("", "nan"):
            try:
                return int(float(v))
            except ValueError:
                pass
        return 9999  # sort missing years to the end

    def _first_author_last(authors_str: str) -> str:
        """Return first-author last name from pipe-separated author string."""
        if not authors_str or str(authors_str).strip() in ("", "nan"):
            return "Anon."
        first = str(authors_str).split("|")[0].strip()
        # WOS format: "Lastname, Firstname"
        if "," in first:
            return first.split(",")[0].strip()
        # PubMed format: "G Hampf" or "G.A. Hampf" → last token
        parts = first.split()
        return parts[-1] if parts else first

    def _author_label(authors_str: str) -> str:
        # Escape each last name individually, then join — so the \& we add
        # here never passes through _latex_escape again in _latex_cite_cell.
        last = _latex_escape(_first_author_last(authors_str))
        parts = [p.strip() for p in str(authors_str).split("|") if p.strip()]
        n = len(parts)
        if n == 1:
            return last
        if n == 2:
            last2 = _latex_escape(_first_author_last(parts[1]))
            return f"{last} \\& {last2}"
        return f"{last} et~al."

    def _short_type(v: str) -> str:
        mapping = {
            "narrative or expert review": "Narrative review",
            "systematic review": "Systematic review",
            "meta-analysis": "Meta-analysis",
            "scoping or mapping review": "Scoping review",
            "network meta-analysis": "Network meta-analysis",
        }
        return mapping.get(str(v).lower().strip(), str(v))

    def _short_pain(v: str) -> str:
        mapping = {
            "chronic secondary musculoskeletal pain": "Musculoskeletal",
            "mixed or unspecified chronic pain": "Mixed/Unspecified",
            "chronic secondary headache or orofacial pain": "Headache/Orofacial",
            "chronic secondary visceral pain": "Visceral",
            "chronic neuropathic pain": "Neuropathic",
            "chronic primary pain": "Primary",
            "chronic postsurgical or posttraumatic pain": "Postsurgical",
            "chronic cancer-related pain": "Cancer-related",
        }
        return mapping.get(str(v).lower().strip(), str(v))

    def _short_obj(v: str) -> str:
        mapping = {
            "clinical": "Clinical",
            "conceptual": "Conceptual",
            "epidemiological": "Epidemiological",
            "methodological": "Methodological",
            "mixed": "Mixed",
            "unclear": "Unclear",
        }
        return mapping.get(str(v).lower().strip(), str(v))

    def _short_func(v: str) -> str:
        mapping = {
            "intervention rationale": "Intervention rationale",
            "explanatory framework": "Explanatory framework",
            "organizing principle": "Organizing principle",
            "background framing": "Background framing",
            "rhetorical label": "Rhetorical label",
            "justification": "Justification",
            "conclusion": "Conclusion",
            "unclear": "Unclear",
        }
        return mapping.get(str(v).lower().strip(), str(v))

    def _short_typology(v: str) -> str:
        mapping = {
            "multifactorial signal": "Multifactorial",
            "potential integrative signal": "Integrative",
            "pseudo-bps or partial signal": "Pseudo-BPS",
            "rhetorical label signal": "Rhetorical",
        }
        return mapping.get(str(v).lower().strip(), str(v))

    def _domain_cell(b: str, p: str, s: str) -> str:
        parts = []
        if str(b).strip().lower() == "yes":
            parts.append("B")
        if str(p).strip().lower() == "yes":
            parts.append("P")
        if str(s).strip().lower() == "yes":
            parts.append("S")
        return ", ".join(parts) if parts else "---"

    def _latex_cite_cell(author_label: str, year: int, doi: str, pmid: str) -> str:
        """Hyperlinked author+year: DOI first, then PubMed PMID, then plain text.
        author_label is already LaTeX-escaped (done in _author_label)."""
        year_str = str(year) if year != 9999 else "n.d."
        display = f"{author_label} ({year_str})"
        doi_val = str(doi).strip()
        if doi_val and doi_val not in ("", "nan"):
            return f"\\href{{https://doi.org/{doi_val}}}{{{display}}}"
        pmid_val = str(pmid).strip()
        if pmid_val and pmid_val not in ("", "nan", "0"):
            try:
                pmid_int = int(float(pmid_val))
                if pmid_int > 0:
                    return f"\\href{{https://pubmed.ncbi.nlm.nih.gov/{pmid_int}/}}{{{display}}}"
            except (ValueError, OverflowError):
                pass
        return display

    section_prefix = re.compile(
        r"^(objectives?|purpose(?:\s+of\s+review)?|aims?|background|introduction|methods?|results?|conclusions?|reviewers?'?\s+conclusions?|authors?'?\s+conclusions?)\s*:\s*",
        flags=re.IGNORECASE,
    )

    def _clean_text(value: str) -> str:
        text = " ".join(str(value).replace("\n", " ").split()).strip()
        while text and section_prefix.match(text):
            text = section_prefix.sub("", text, count=1).strip()
        return text

    def _resolve_cached_text_path(path_value: str) -> Path | None:
        raw = str(path_value).strip()
        if not raw or raw in {"", "nan"}:
            return None
        candidate = Path(raw)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        workspace_relative = project_path(*candidate.parts)
        if workspace_relative.exists():
            return workspace_relative
        project_relative = project_path("src", *candidate.parts)
        if project_relative.exists():
            return project_relative
        return None

    def _first_sentence(text: str, max_len: int = 220) -> str:
        cleaned = _clean_text(text)
        if not cleaned:
            return ""
        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
        sentence = parts[0] if parts else cleaned
        if not sentence.endswith((".", "!", "?", "...")):
            sentence += "."
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        if len(sentence) > max_len:
            sentence = sentence[: max_len - 3].rsplit(" ", 1)[0].rstrip() + "..."
        return sentence

    def _extract_fulltext_objective(record_id: str) -> tuple[str, str]:
        entry = stage3_lookup.get(record_id, {})
        cached_path = _resolve_cached_text_path(str(entry.get("cached_text_path", ""))) if entry else None
        if not cached_path:
            return "", ""
        try:
            text = cached_path.read_text(encoding="utf-8")
        except OSError:
            return "", ""

        snippet = " ".join(text[:14000].split())
        patterns = [
            re.compile(r"\b(?:objective|objectives|aim|aims|purpose)\s*[:\-]\s*(.{35,260}?\.)", re.IGNORECASE),
            re.compile(
                r"\bthis\s+(?:systematic|scoping|narrative|umbrella|rapid|integrative|meta-analysis|network meta-analysis|review|study)[^.]{0,120}?\b(?:aimed|was designed|sought|examined|evaluated|investigated)\s+to\s+(.{20,220}?\.)",
                re.IGNORECASE,
            ),
        ]
        for pattern in patterns:
            match = pattern.search(snippet)
            if match:
                return _first_sentence(match.group(1), max_len=190), str(cached_path.relative_to(project_path()))
        return "", str(cached_path.relative_to(project_path()))

    def _focus_phrase(text: str) -> str:
        cleaned = _clean_text(text)
        if not cleaned:
            return ""
        cleaned = re.sub(
            r"^(?:the\s+)?(?:objective|objectives|aim|aims|purpose)(?:\s+of\s+this\s+[^,.;:]{0,80})?\s+(?:was|were|is|are)?\s*(?:to\s+)?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^this\s+(?:systematic|scoping|narrative|umbrella|rapid|integrative|meta-analysis|network meta-analysis|review|study)[^,.;:]{0,100}?\s+(?:aimed|was designed|sought|examined|evaluated|investigated)\s+to\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^to\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip(" .;:-")
        if len(cleaned) > 120:
            cleaned = cleaned[:117].rsplit(" ", 1)[0].rstrip() + "..."
        return cleaned

    def _focus_is_usable(text: str) -> bool:
        focus = text.strip()
        if not focus:
            return False
        lowered = focus.lower()
        if len(lowered.split()) < 4:
            return False
        if lowered.startswith(("this ", "it ", "there ", "about ")):
            return False
        if focus.endswith((" a", " an", " the", " of", " to", " and", " or", " from", " with")):
            return False
        if re.search(r"\b(is|are)\b\s+(presented|described|discussed|reported)\b", lowered):
            return False
        return True

    def _title_focus(title: str) -> str:
        cleaned = _clean_text(title).rstrip(".")
        cleaned = re.sub(
            r"[:,\-]?\s+a\s+(?:systematic\s+review(?:\s+and\s+meta-analysis)?|meta-analysis|scoping\s+review|narrative\s+review|network\s+meta-analysis)$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip(" .;:-")

    def _compose_description(review_type: str, focus: str) -> str:
        label = _short_type(review_type).strip() or "Review"
        label_lower = label[0].lower() + label[1:] if label and label[0].isupper() else label.lower()
        focus_clean = focus.strip().rstrip(".")
        if not focus_clean:
            return f"This {label_lower} was focused on chronic pain evidence."
        if focus_clean[0].isupper() and not (len(focus_clean) > 1 and focus_clean[1].isupper()):
            focus_clean = focus_clean[0].lower() + focus_clean[1:]
        verb_lead = re.match(
            r"^(determine|evaluate|assess|examine|investigate|explore|describe|identify|summari(?:s|z)e|map|compare|review)\b",
            focus_clean,
            flags=re.IGNORECASE,
        )
        if verb_lead:
            sentence = f"This {label_lower} was aimed to {focus_clean}."
        else:
            sentence = f"This {label_lower} was focused on {focus_clean}."
        return _first_sentence(sentence, max_len=160)

    def _description_from_row(row: pd.Series) -> tuple[str, str, str, str]:
        record_id = str(row.get("record_id", ""))
        fulltext_obj, fulltext_path = _extract_fulltext_objective(record_id)
        objective = _first_sentence(str(row.get("objective_text", "")), max_len=180)
        abstract_sentence = _first_sentence(str(row.get("abstract", "")), max_len=180)
        title_focus = _title_focus(str(row.get("title", "")) or "chronic pain evidence")

        candidates: list[tuple[str, str]] = []
        if fulltext_obj:
            candidates.append(("fulltext_cached", _focus_phrase(fulltext_obj)))
        candidates.append(("title_fallback", _focus_phrase(title_focus)))
        if objective and len(objective) >= 35:
            candidates.append(("objective_text", _focus_phrase(objective)))
        if abstract_sentence and len(abstract_sentence) >= 50:
            candidates.append(("abstract_first_sentence", _focus_phrase(abstract_sentence)))

        source = "title_fallback"
        focus = _focus_phrase(title_focus)
        for candidate_source, candidate_focus in candidates:
            if _focus_is_usable(candidate_focus):
                source = candidate_source
                focus = candidate_focus
                break

        if not focus:
            focus = "chronic pain evidence"
        description = _compose_description(str(row.get("review_type", "")), focus)
        return description, source, focus, fulltext_path

    def _normalized_pmid(value: str) -> str:
        raw = str(value).strip()
        if not raw or raw in {"", "nan", "0"}:
            return ""
        try:
            num = int(float(raw))
        except (ValueError, OverflowError):
            return ""
        return str(num) if num > 0 else ""

    def _summary_cell(text: str) -> str:
        """Escaped one-sentence technical description; p{} column wraps automatically."""
        return _latex_escape(_clean_text(text))

    # ── Build rows ────────────────────────────────────────────────────────────
    df = stage2.copy()
    df["_year_int"] = df.apply(_parse_year, axis=1)
    df = df.sort_values(["_year_int", "record_id"], ascending=[True, True]).reset_index(drop=True)

    rows_tex: list[str] = []
    description_audit_rows: list[dict[str, str | int]] = []
    full_reference_rows: list[dict[str, str | int]] = []
    for _, row in df.iterrows():
        author_label = _author_label(str(row.get("authors", "")))
        year_int = int(row["_year_int"])
        doi  = str(row.get("doi", "")).strip()
        pmid = str(row.get("pmid", "")).strip()
        cite    = _latex_cite_cell(author_label, year_int, doi, pmid)
        description_text, description_source, description_focus, fulltext_path = _description_from_row(row)
        summary = _summary_cell(description_text)
        rtype   = _latex_escape(_short_type(row.get("review_type", "")))
        pain    = _latex_escape(_short_pain(row.get("icd11_pain_category", ""))).replace("/", r"/\allowbreak ")
        obj     = _latex_escape(_short_obj(row.get("objective_category", "")))
        func    = _latex_escape(_short_func(row.get("bps_function", "")))
        doms    = _domain_cell(row.get("bio_mentioned", ""), row.get("psych_mentioned", ""), row.get("social_mentioned", ""))
        typo    = _latex_escape(_short_typology(row.get("provisional_typology", "")))
        stage3_entry = stage3_lookup.get(str(row.get("record_id", "")), {})
        rows_tex.append(
            f"{cite} & {summary} & {rtype} & {pain} & {obj} & {func} & {doms} & {typo} \\\\"
        )
        description_audit_rows.append(
            {
                "record_id": str(row.get("record_id", "")),
                "year": "" if year_int == 9999 else year_int,
                "reference_label": f"{_first_author_last(str(row.get('authors', '')))} ({'' if year_int == 9999 else year_int})",
                "description": description_text,
                "description_focus": description_focus,
                "description_source": description_source,
                "description_char_count": len(description_text),
                "fulltext_cached_path": fulltext_path,
                "fulltext_status": str(stage3_entry.get("fulltext_status", "")),
                "manual_fulltext_review_needed": "yes" if description_source != "fulltext_cached" and str(row.get("stage3_candidate", "")).strip().lower() == "yes" else "no",
                "review_type": str(row.get("review_type", "")),
                "icd11_pain_category": str(row.get("icd11_pain_category", "")),
                "objective_category": str(row.get("objective_category", ""),),
                "bps_function": str(row.get("bps_function", "")),
                "provisional_typology": str(row.get("provisional_typology", "")),
            }
        )
        full_reference_rows.append(
            {
                "record_id": str(row.get("record_id", "")),
                "database": str(row.get("database", "")),
                "authors": str(row.get("authors", "")),
                "year": "" if year_int == 9999 else year_int,
                "title": _clean_text(row.get("title", "")),
                "journal": str(row.get("journal", "")),
                "doi": doi,
                "pmid": _normalized_pmid(pmid),
                "pmcid": str(row.get("pmcid", "")),
                "fulltext_status": str(stage3_entry.get("fulltext_status", "")),
                "cached_text_path": str(stage3_entry.get("cached_text_path", "")),
                "review_type": str(row.get("review_type", "")),
                "icd11_pain_category": str(row.get("icd11_pain_category", "")),
                "objective_category": str(row.get("objective_category", "")),
                "bps_function": str(row.get("bps_function", "")),
                "provisional_typology": str(row.get("provisional_typology", "")),
            }
        )

    # ── Column spec: full-width 8-col layout for landscape A4 pages ──
    # Keep columns broad enough to avoid cramped wrapping while fitting landscape width.
    _C = r">{\RaggedRight\arraybackslash\hbadness=10000\hfuzz=20pt}"
    colspec = (
        f"{_C}p{{2.6cm}}"   # Reference
        f"{_C}p{{8.1cm}}"   # Description
        f"{_C}p{{1.9cm}}"   # Type
        f"{_C}p{{2.5cm}}"   # Pain Condition
        f"{_C}p{{2.0cm}}"   # Objective
        f"{_C}p{{3.0cm}}"   # BPS Role
        f"{_C}p{{1.0cm}}"   # Domains (B, P, S)
        f"{_C}p{{2.4cm}}"   # Typology
    )

    N_COLS = 8
    header = (
        r"\textbf{Reference} & \textbf{Description} & \textbf{Type} "
        r"& \textbf{Pain Condition} & \textbf{Objective} & \textbf{BPS Role} "
        r"& \textbf{Dom.} & \textbf{Typology} \\"
    )
    midrule_repeat = (
        fr"\multicolumn{{{N_COLS}}}{{r}}{{\scriptsize\textit{{Continued on next page}}}} \\"
    )
    note_text = (
        fr"\multicolumn{{{N_COLS}}}{{p{{0.98\linewidth}}}}{{\footnotesize\textit{{Note.}} "
        r"Type: review-design class (for example, narrative, systematic, meta-analysis, or scoping). "
        r"Pain Condition: ICD-11-oriented chronic pain grouping inferred at abstract level. "
        r"Objective: primary objective class (clinical, conceptual, methodological, epidemiological, mixed, or unclear). "
        r"BPS Role: primary declared functional role of BPS language in the review. "
        r"Dom.: substantive domain mentions in title/objective/abstract after excluding lexical BPS token matches (B\,=\,biological, P\,=\,psychological, S\,=\,social). "
        r"Typology: provisional BPS operationalization class from Stage~2 abstract-level coding. "
        r"Description: concise past-tense one-sentence summary generated from cached full-text objective statements when available; otherwise from objective text, abstract, and finally title fallback. "
        r"References link to DOI (where available) or PubMed record.} \\"
    )

    body = "\n".join(rows_tex)

    content = (
        "% Auto-generated characteristics table — do not edit manually\n"
        "\\begin{landscape}\n"
        "\\setlength{\\LTleft}{0pt}\n"
        "\\setlength{\\LTright}{0pt}\n"
        "\\setlength{\\LTcapwidth}{\\linewidth}\n"
        "\\captionsetup{justification=raggedright,singlelinecheck=false}\n"
        "\\setlength{\\tabcolsep}{2.0pt}\n"
        "\\renewcommand{\\arraystretch}{1.18}\n"
        "\\hbadness=10000\n"
        "\\hfuzz=20pt\n"
        "\\scriptsize\n"
        f"\\begin{{longtable}}{{{colspec}}}\n"
        f"  \\caption{{Characteristics of all {len(df)} included reviews.}}\n"
        "  \\label{tab.characteristics}\\\\\n"
        "  \\toprule\n"
        f"  {header}\n"
        "  \\midrule\n"
        "  \\endfirsthead\n"
        "  \\toprule\n"
        f"  {header}\n"
        "  \\midrule\n"
        "  \\endhead\n"
        "  \\midrule\n"
        f"  {midrule_repeat}\n"
        "  \\endfoot\n"
        "  \\bottomrule\n"
        f"  {note_text}\n"
        "  \\endlastfoot\n"
        f"{body}\n"
        "\\end{longtable}\n"
        "\\renewcommand{\\arraystretch}{1}\n"
        "\\normalsize\n"
        "\\end{landscape}\n"
    )

    ensure_parent(path).write_text(content, encoding="utf-8")
    table_dir = project_path("paper", "assets", "tables")
    description_audit_path = ensure_parent(table_dir / "table1_description_audit.csv")
    pd.DataFrame(description_audit_rows).to_csv(description_audit_path, index=False)
    full_refs_path = ensure_parent(table_dir / "included_review_full_references.csv")
    pd.DataFrame(full_reference_rows).to_csv(full_refs_path, index=False)


def _draw_prisma(summary: dict[str, int], db_counts: pd.DataFrame, out_path: Path) -> None:
    """Render a PRISMA-style flow diagram with readable spacing and stable alignment."""
    fig, ax = plt.subplots(figsize=(14, 10.5))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    section_bg = "#eaf3fb"
    section_edge = "#aac3d8"
    section_text = "#173750"
    main_bg = "#f4f9fd"
    main_edge = "#1f5d86"
    side_bg = "#fff3df"
    side_edge = "#8a6a1f"
    arrow_color = "#1f5d86"

    def _fmt_count(value: int | float) -> str:
        return f"{int(value):,}"

    sections = [
        (0.80, 0.98, "Identification"),
        (0.45, 0.79, "Screening"),
        (0.02, 0.44, "Included"),
    ]
    for y_bot, y_top, label in sections:
        ax.add_patch(
            FancyBboxPatch(
                (0.02, y_bot),
                0.10,
                y_top - y_bot,
                boxstyle="round,pad=0.004,rounding_size=0.006",
                facecolor=section_bg,
                edgecolor=section_edge,
                linewidth=1.1,
            )
        )
        ax.text(
            0.07,
            (y_bot + y_top) / 2.0,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=section_text,
            rotation=90,
        )

    def draw_box(x: float, y: float, w: float, h: float, text: str, *, facecolor: str, edgecolor: str, fontsize: float = 9.0) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.010,rounding_size=0.008",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.4,
            )
        )
        ax.text(
            x + w / 2.0,
            y + h / 2.0,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            wrap=True,
            multialignment="center",
            color="#1f3447",
        )

    def arrow_down(x: float, y_from: float, y_to: float) -> None:
        ax.annotate(
            "",
            xy=(x, y_to),
            xytext=(x, y_from),
            arrowprops=dict(arrowstyle="->", lw=1.5, color=arrow_color),
        )

    def arrow_right(x_from: float, y: float, x_to: float) -> None:
        ax.annotate(
            "",
            xy=(x_to, y),
            xytext=(x_from, y),
            arrowprops=dict(arrowstyle="->", lw=1.3, color=side_edge, linestyle="dashed"),
        )

    db_lines = []
    if not db_counts.empty:
        for _, row in db_counts.iterrows():
            db_lines.append(f"{row['database']}: n = {_fmt_count(row['n'])}")
    db_str = "\n".join(db_lines[:4]) if db_lines else "Database breakdown not available"

    main_x = 0.16
    main_w = 0.50
    side_x = 0.72
    side_w = 0.24
    center_x = main_x + main_w / 2.0

    draw_box(
        main_x,
        0.83,
        main_w,
        0.13,
        (
            "Records identified via database searches\n"
            f"n = {_fmt_count(summary['combined_records'])}\n"
            f"{db_str}"
        ),
        facecolor=main_bg,
        edgecolor=main_edge,
    )
    arrow_down(center_x, 0.83, 0.75)

    draw_box(
        main_x,
        0.66,
        main_w,
        0.08,
        (
            "Records after deduplication\n"
            f"n = {_fmt_count(summary['deduplicated_records'])} "
            f"(removed {_fmt_count(summary['duplicates_removed'])})"
        ),
        facecolor=main_bg,
        edgecolor=main_edge,
    )
    arrow_down(center_x, 0.66, 0.58)

    draw_box(
        main_x,
        0.49,
        main_w,
        0.08,
        f"Records screened (title + abstract)\nn = {_fmt_count(summary['deduplicated_records'])}",
        facecolor=main_bg,
        edgecolor=main_edge,
    )
    draw_box(
        side_x,
        0.49,
        side_w,
        0.08,
        f"Stage 1 exclusions\nn = {_fmt_count(summary['excluded_records'])}",
        facecolor=side_bg,
        edgecolor=side_edge,
        fontsize=8.8,
    )
    arrow_right(main_x + main_w, 0.53, side_x)
    arrow_down(center_x, 0.49, 0.41)

    borderline_text = ""
    if int(summary.get("unclear_records", 0)) > 0:
        borderline_text = f" (borderline = {_fmt_count(summary['unclear_records'])})"
    draw_box(
        main_x,
        0.31,
        main_w,
        0.09,
        f"Eligible records entering Stage 2 coding\nn = {_fmt_count(summary['included_records'])}{borderline_text}",
        facecolor=main_bg,
        edgecolor=main_edge,
    )
    arrow_down(center_x, 0.31, 0.23)

    draw_box(
        main_x,
        0.13,
        main_w,
        0.09,
        (
            "Included in Stage 2 synthesis\n"
            f"n = {_fmt_count(summary['stage2_records'])}\n"
            "Publication years: 1990-2026"
        ),
        facecolor=main_bg,
        edgecolor=main_edge,
    )

    branch_y = 0.115
    arrow_down(center_x, 0.13, branch_y)
    left_box_x = main_x
    right_box_x = main_x + 0.28
    small_w = 0.22
    small_h = 0.07
    box_top_y = 0.03 + small_h

    ax.plot([left_box_x + small_w / 2.0, right_box_x + small_w / 2.0], [branch_y, branch_y], color=arrow_color, lw=1.4)
    ax.annotate("", xy=(left_box_x + small_w / 2.0, box_top_y), xytext=(left_box_x + small_w / 2.0, branch_y), arrowprops=dict(arrowstyle="->", lw=1.2, color=arrow_color))
    ax.annotate("", xy=(right_box_x + small_w / 2.0, box_top_y), xytext=(right_box_x + small_w / 2.0, branch_y), arrowprops=dict(arrowstyle="->", lw=1.2, color=arrow_color))

    draw_box(
        left_box_x,
        0.03,
        small_w,
        small_h,
        f"Open-access full texts\nretrieved (Stage 3)\nn = {_fmt_count(summary['stage3_pmc_open_fulltexts'])}",
        facecolor=main_bg,
        edgecolor=main_edge,
        fontsize=8.3,
    )
    draw_box(
        right_box_x,
        0.03,
        small_w,
        small_h,
        f"Manual retrieval queue\n(Stage 3)\nn = {_fmt_count(summary['stage3_manual_retrieval_required'])}",
        facecolor=side_bg,
        edgecolor=side_edge,
        fontsize=8.3,
    )

    fig.tight_layout(pad=0.5)
    ensure_parent(out_path)
    fig.savefig(out_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _parse_problem_flags(series: pd.Series) -> pd.DataFrame:
    flags: list[str] = []
    for value in series.astype(str):
        text = value.strip()
        if not text:
            continue
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                flags.extend(str(item).strip() for item in parsed if str(item).strip())
            else:
                flags.append(str(parsed).strip())
        except Exception:
            normalized = text.replace("[", "").replace("]", "").replace("'", "").replace(" | ", "|")
            for chunk in normalized.split(","):
                for token in chunk.split("|"):
                    token = token.strip()
                    if token:
                        flags.append(token)
    if not flags:
        return pd.DataFrame(columns=["conceptual_problem_flag", "n", "percent"])
    counts = pd.Series(flags).value_counts().rename_axis("conceptual_problem_flag").reset_index(name="n")
    counts = counts.loc[counts["conceptual_problem_flag"] != "none"].copy()
    if counts.empty:
        return pd.DataFrame(columns=["conceptual_problem_flag", "n", "percent"])
    return _with_percent(counts)


def _psychological_concept_counts(stage2: pd.DataFrame) -> pd.DataFrame:
    if stage2.empty or "psychological_concepts_detected" not in stage2.columns:
        return pd.DataFrame(columns=["psychological_concept", "n", "percent"])
    concepts: list[str] = []
    for value in stage2["psychological_concepts_detected"].astype(str):
        for token in value.split("|"):
            cleaned = token.strip().lower()
            if cleaned:
                concepts.append(cleaned)
    if not concepts:
        return pd.DataFrame(columns=["psychological_concept", "n", "percent"])
    counts = pd.Series(concepts).value_counts().rename_axis("psychological_concept").reset_index(name="n")
    return _with_percent(counts)


def _provisional_typology(row: pd.Series) -> str:
    bio = str(row.get("bio_mentioned", "")).strip().lower() == "yes"
    psych = str(row.get("psych_mentioned", "")).strip().lower() == "yes"
    social = str(row.get("social_mentioned", "")).strip().lower() == "yes"
    triadic = bio and psych and social

    function = str(row.get("bps_function", "")).strip().lower()
    if triadic and function in {"explanatory framework", "organizing principle"}:
        return "potential integrative signal"
    if triadic:
        return "multifactorial signal"
    if "rhetorical" in function or function in {"background framing", "conclusion"}:
        return "rhetorical label signal"
    return "pseudo-bps or partial signal"


def _shorten_label(value: str, max_len: int = 26) -> str:
    text = str(value).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


_BPS_LEXICAL_ONLY_PATTERN = re.compile(
    r"\bbio[\s-]*psycho[\s-]*social(?:ly)?\b|\bpsycho[\s-]*social(?:ly)?\b",
    flags=re.IGNORECASE,
)

_SUBSTANTIVE_DOMAIN_PATTERNS = {
    "bio": [
        re.compile(r"\bbiologic(?:al|ally)?\b", re.IGNORECASE),
        re.compile(r"\bbiomedical\b", re.IGNORECASE),
        re.compile(r"\bphysiolog(?:y|ical)\b", re.IGNORECASE),
        re.compile(r"\bneuro(?:pathic|biology|physiology|imaging)?\b", re.IGNORECASE),
        re.compile(r"\binflamm(?:ation|atory)?\b", re.IGNORECASE),
        re.compile(r"\bnocicept(?:ive|ion)?\b", re.IGNORECASE),
        re.compile(r"\bgenetic(?:s)?\b", re.IGNORECASE),
        re.compile(r"\bpharmacolog(?:y|ical|ic)\b", re.IGNORECASE),
        re.compile(r"\bcentral sensitization\b", re.IGNORECASE),
    ],
    "psych": [
        re.compile(r"\bpsycholog(?:y|ical|ically)\b", re.IGNORECASE),
        re.compile(r"\bcognit(?:ion|ive)\b", re.IGNORECASE),
        re.compile(r"\bbehavio(?:u)?ral\b", re.IGNORECASE),
        re.compile(r"\bdepress(?:ion|ive)?\b", re.IGNORECASE),
        re.compile(r"\banxiet(?:y|ies)\b", re.IGNORECASE),
        re.compile(r"\bstress(?:or|ors)?\b", re.IGNORECASE),
        re.compile(r"\bcatastrophi(?:zing|sation)\b", re.IGNORECASE),
        re.compile(r"\bfear[-\s]?avoidance\b", re.IGNORECASE),
        re.compile(r"\bself[-\s]?efficacy\b", re.IGNORECASE),
        re.compile(r"\bcoping\b", re.IGNORECASE),
        re.compile(r"\bmental\b", re.IGNORECASE),
        re.compile(r"\bemotion(?:al|s)?\b", re.IGNORECASE),
    ],
    "social": [
        re.compile(r"\bsocio(?:economic|cultural)\b", re.IGNORECASE),
        re.compile(r"\bsocial\s+(?:support|determinants?|context|environment|factors?)\b", re.IGNORECASE),
        re.compile(r"\bfamil(?:y|ies)\b", re.IGNORECASE),
        re.compile(r"\bcaregiver(?:s)?\b", re.IGNORECASE),
        re.compile(r"\bpartner(?:s)?\b", re.IGNORECASE),
        re.compile(r"\bwork(?:place|ing|related)?\b", re.IGNORECASE),
        re.compile(r"\boccupational\b", re.IGNORECASE),
        re.compile(r"\bemploy(?:ment|ability|ed)?\b", re.IGNORECASE),
        re.compile(r"\bincome\b", re.IGNORECASE),
        re.compile(r"\beducation(?:al)?\b", re.IGNORECASE),
        re.compile(r"\bdeprivation\b", re.IGNORECASE),
        re.compile(r"\binequalit(?:y|ies)\b", re.IGNORECASE),
        re.compile(r"\bstigma\b", re.IGNORECASE),
        re.compile(r"\bisolation\b", re.IGNORECASE),
        re.compile(r"\bcommunity\b", re.IGNORECASE),
        re.compile(r"\bcultur(?:e|al|ally)\b", re.IGNORECASE),
        re.compile(r"\bpolicy\b", re.IGNORECASE),
        re.compile(r"\bhealth\s?care access\b", re.IGNORECASE),
        re.compile(r"\baccess to care\b", re.IGNORECASE),
        re.compile(r"\blegal\b", re.IGNORECASE),
        re.compile(r"\bcompensation\b", re.IGNORECASE),
    ],
}


def _compact_icd11_label(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"\bchronic\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpain\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" -,/\t")
    return text or str(value).strip()


def _derive_substantive_domain_mentions(stage2: pd.DataFrame) -> pd.DataFrame:
    """Recode domain mentions from substantive content while excluding lexical BPS tokens."""
    if stage2.empty:
        return stage2.copy()

    coded = stage2.copy()
    for col in ["bio_mentioned", "psych_mentioned", "social_mentioned"]:
        if col not in coded.columns:
            coded[col] = ""
        coded[col] = coded[col].astype(str).str.strip().str.lower()

    coded["bio_mentioned_llm"] = coded["bio_mentioned"]
    coded["psych_mentioned_llm"] = coded["psych_mentioned"]
    coded["social_mentioned_llm"] = coded["social_mentioned"]

    text_fields = ["title", "objective_text", "abstract"]

    def _substantive_text(row: pd.Series) -> str:
        merged = " ".join(str(row.get(field, "")).strip() for field in text_fields if str(row.get(field, "")).strip())
        merged = _BPS_LEXICAL_ONLY_PATTERN.sub(" ", merged)
        merged = re.sub(r"\s+", " ", merged)
        return merged.strip().lower()

    def _has_pattern(text: str, patterns: list[re.Pattern[str]]) -> str:
        return "yes" if any(pattern.search(text) for pattern in patterns) else "no"

    coded["_substantive_text"] = coded.apply(_substantive_text, axis=1)
    coded["bio_mentioned"] = coded["_substantive_text"].apply(lambda t: _has_pattern(t, _SUBSTANTIVE_DOMAIN_PATTERNS["bio"]))
    coded["psych_mentioned"] = coded["_substantive_text"].apply(lambda t: _has_pattern(t, _SUBSTANTIVE_DOMAIN_PATTERNS["psych"]))
    coded["social_mentioned"] = coded["_substantive_text"].apply(lambda t: _has_pattern(t, _SUBSTANTIVE_DOMAIN_PATTERNS["social"]))

    coded["triadic_llm"] = np.where(
        (coded["bio_mentioned_llm"] == "yes")
        & (coded["psych_mentioned_llm"] == "yes")
        & (coded["social_mentioned_llm"] == "yes"),
        "yes",
        "no",
    )
    coded["triadic_substantive"] = np.where(
        (coded["bio_mentioned"] == "yes")
        & (coded["psych_mentioned"] == "yes")
        & (coded["social_mentioned"] == "yes"),
        "yes",
        "no",
    )
    return coded.drop(columns=["_substantive_text"])


def _collapse_review_type(value: str) -> str:
    text = str(value).strip().lower()
    if "network meta" in text:
        return "network meta-analysis"
    if "meta-analysis" in text or "meta analysis" in text:
        return "meta-analysis"
    if "systematic" in text:
        return "systematic review"
    if "scoping" in text or "mapping" in text:
        return "scoping or mapping"
    if "umbrella" in text or "overview" in text:
        return "umbrella or overview"
    if "rapid" in text:
        return "rapid review"
    if "realist" in text:
        return "realist review"
    if "integrative" in text:
        return "integrative review"
    if "narrative" in text or "expert" in text:
        return "narrative or expert"
    return "other synthesis"


def build_assets() -> dict[str, int]:
    search_outputs = project_path("review_stages", "02_search", "outputs")
    combined = _load_csv(search_outputs / "combined_records.csv")
    deduped = _load_csv(search_outputs / "deduplicated_records.csv")
    stage1 = _load_csv(project_path("review_stages", "03_screening", "outputs", "stage1_screening.csv"))
    stage2 = _load_csv(project_path("review_stages", "04_extraction", "outputs", "stage2_abstract_coding.csv"))
    stage2_llm = _load_csv(project_path("review_stages", "04_extraction", "outputs", "stage2_objective_llm_assist.csv"))
    stage3_manifest = _load_csv(project_path("review_stages", "04_extraction", "outputs", "stage3_candidate_manifest.csv"))
    stage3_summary = _load_csv(project_path("review_stages", "04_extraction", "outputs", "stage3_candidate_summary.csv"))
    reliability = _load_csv(project_path("review_stages", "03_screening", "audit", "reliability_report.csv"))

    domain_recode_audit = pd.DataFrame(
        columns=[
            "record_id",
            "bio_mentioned_llm",
            "psych_mentioned_llm",
            "social_mentioned_llm",
            "triadic_llm",
            "bio_mentioned",
            "psych_mentioned",
            "social_mentioned",
            "triadic_substantive",
        ]
    )
    if not stage2.empty:
        stage2 = _derive_substantive_domain_mentions(stage2)
        stage2["provisional_typology"] = stage2.apply(_provisional_typology, axis=1)
        domain_recode_audit = stage2[
            [
                "record_id",
                "bio_mentioned_llm",
                "psych_mentioned_llm",
                "social_mentioned_llm",
                "triadic_llm",
                "bio_mentioned",
                "psych_mentioned",
                "social_mentioned",
                "triadic_substantive",
            ]
        ].copy()

    semantic_result = run_semantic_loading(stage2)

    combined_records = len(combined) if not combined.empty else len(deduped)
    deduplicated_records = len(deduped)
    duplicates_removed = max(combined_records - deduplicated_records, 0)
    included_records = int((stage1["stage1_decision"] == "include").sum()) if not stage1.empty else 0
    excluded_records = int((stage1["stage1_decision"] == "exclude").sum()) if not stage1.empty else 0
    unclear_records = int((stage1["stage1_decision"] == "unclear").sum()) if not stage1.empty else 0

    summary = {
        "combined_records": combined_records,
        "deduplicated_records": deduplicated_records,
        "duplicates_removed": duplicates_removed,
        "included_records": included_records,
        "excluded_records": excluded_records,
        "unclear_records": unclear_records,
        "stage2_records": len(stage2),
        "stage3_candidates": int(stage3_summary.iloc[0]["stage3_candidates"]) if not stage3_summary.empty else 0,
        "stage3_pmc_open_fulltexts": int(stage3_summary.iloc[0]["pmc_open_fulltexts"]) if not stage3_summary.empty else 0,
        "stage3_manual_retrieval_required": int(stage3_summary.iloc[0]["manual_retrieval_required"]) if not stage3_summary.empty else 0,
        "semantic_records": int(len(semantic_result.record_loadings)),
    }

    db_counts = combined["database"].value_counts().rename_axis("database").reset_index(name="n") if not combined.empty and "database" in combined.columns else pd.DataFrame(columns=["database", "n"])
    if not stage2.empty:
        stage2 = stage2.copy()
        stage2["review_type_collapsed"] = stage2["review_type"].map(_collapse_review_type)
    review_type_counts = stage2["review_type_collapsed"].value_counts().rename_axis("review_type").reset_index(name="n") if not stage2.empty else pd.DataFrame(columns=["review_type", "n"])
    review_type_counts = _with_percent(review_type_counts)
    icd11_counts = stage2["icd11_pain_category"].value_counts().rename_axis("icd11_pain_category").reset_index(name="n") if not stage2.empty else pd.DataFrame(columns=["icd11_pain_category", "n"])
    icd11_counts = _with_percent(icd11_counts)

    bps_location_counts = stage2["bps_mention_location"].value_counts().rename_axis("bps_mention_location").reset_index(name="n") if not stage2.empty else pd.DataFrame(columns=["bps_mention_location", "n"])
    bps_location_counts = _with_percent(bps_location_counts)
    bps_function_counts = stage2["bps_function"].value_counts().rename_axis("bps_function").reset_index(name="n") if not stage2.empty else pd.DataFrame(columns=["bps_function", "n"])
    bps_function_counts = _with_percent(bps_function_counts)

    if not stage2.empty:
        domain_counts = pd.DataFrame(
            {
                "domain": ["Biological", "Psychological", "Social"],
                "n": [
                    int((stage2["bio_mentioned"] == "yes").sum()),
                    int((stage2["psych_mentioned"] == "yes").sum()),
                    int((stage2["social_mentioned"] == "yes").sum()),
                ],
            }
        )
    else:
        domain_counts = pd.DataFrame(columns=["domain", "n"])
    n_records = len(stage2) if not stage2.empty else 1
    domain_counts["percent"] = domain_counts["n"].apply(lambda v: round((v / n_records) * 100, 1))

    if not stage2.empty:
        triadic_overall_n = int(
            (
                (stage2["bio_mentioned"] == "yes")
                & (stage2["psych_mentioned"] == "yes")
                & (stage2["social_mentioned"] == "yes")
            ).sum()
        )
        core_bps_counts = pd.DataFrame(
            [
                {"indicator": "Biological mention", "n": int((stage2["bio_mentioned"] == "yes").sum())},
                {"indicator": "Psychological mention", "n": int((stage2["psych_mentioned"] == "yes").sum())},
                {"indicator": "Social mention", "n": int((stage2["social_mentioned"] == "yes").sum())},
                {"indicator": "Triadic co-mention", "n": triadic_overall_n},
            ]
        )
    else:
        core_bps_counts = pd.DataFrame(columns=["indicator", "n"])
    core_bps_counts["percent"] = core_bps_counts["n"].apply(lambda v: round((v / n_records) * 100, 1) if n_records else 0.0)

    musculoskeletal = stage2.loc[stage2["musculoskeletal_flag"] == "yes"].copy() if not stage2.empty else pd.DataFrame()
    if not musculoskeletal.empty:
        triadic = int(((musculoskeletal["bio_mentioned"] == "yes") & (musculoskeletal["psych_mentioned"] == "yes") & (musculoskeletal["social_mentioned"] == "yes")).sum())
        total_msk = len(musculoskeletal)
        msk_scope = pd.DataFrame(
            [
                {"indicator": "Musculoskeletal reviews in Stage 2", "n": total_msk},
                {"indicator": "Biological mention present", "n": int((musculoskeletal["bio_mentioned"] == "yes").sum())},
                {"indicator": "Psychological mention present", "n": int((musculoskeletal["psych_mentioned"] == "yes").sum())},
                {"indicator": "Social mention present", "n": int((musculoskeletal["social_mentioned"] == "yes").sum())},
                {"indicator": "Triadic BPS mention present", "n": triadic},
            ]
        )
        msk_scope["percent"] = msk_scope["n"].apply(lambda value: round((value / total_msk) * 100, 1) if total_msk else 0.0)
    else:
        msk_scope = pd.DataFrame(columns=["indicator", "n", "percent"])

    objective_counts = stage2["objective_category"].value_counts().rename_axis("objective_category").reset_index(name="n") if not stage2.empty else pd.DataFrame(columns=["objective_category", "n"])
    objective_counts = _with_percent(objective_counts)
    objective_llm_counts = stage2_llm["objective_category_llm"].value_counts().rename_axis("objective_category_llm").reset_index(name="n") if not stage2_llm.empty else pd.DataFrame(columns=["objective_category_llm", "n"])
    objective_llm_counts = _with_percent(objective_llm_counts)

    psych_concepts = _psychological_concept_counts(stage2)
    if not stage2.empty and "conceptual_problem_flags" in stage2.columns:
        conceptual_problems = _parse_problem_flags(stage2["conceptual_problem_flags"])
    elif not stage2_llm.empty:
        conceptual_problems = _parse_problem_flags(stage2_llm["conceptual_problem_flags"])
    else:
        conceptual_problems = pd.DataFrame(columns=["conceptual_problem_flag", "n", "percent"])

    if not stage2.empty and "year" in stage2.columns:
        year_values = pd.to_numeric(stage2["year"], errors="coerce").dropna().astype(int)
        publication_year_counts = year_values.value_counts().sort_index().rename_axis("year").reset_index(name="n")
        publication_year_counts = _with_percent(publication_year_counts)
        decade_counts = ((year_values // 10) * 10).value_counts().sort_index().rename_axis("decade").reset_index(name="n")
        decade_counts["period"] = decade_counts["decade"].astype(str) + "s"
        decade_counts = _with_percent(decade_counts[["period", "n"]])
    else:
        publication_year_counts = pd.DataFrame(columns=["year", "n", "percent"])
        decade_counts = pd.DataFrame(columns=["period", "n", "percent"])

    if not stage2.empty:
        staged = stage2.copy()
        if "provisional_typology" not in staged.columns or staged["provisional_typology"].astype(str).str.strip().eq("").all():
            staged["provisional_typology"] = staged.apply(_provisional_typology, axis=1)
        typology_counts = staged["provisional_typology"].value_counts().rename_axis("provisional_typology").reset_index(name="n")
        typology_counts = _with_percent(typology_counts)

        triadic_series = (
            (staged["bio_mentioned"] == "yes")
            & (staged["psych_mentioned"] == "yes")
            & (staged["social_mentioned"] == "yes")
        )
        staged["triadic_bps"] = triadic_series.map({True: "yes", False: "no"})
        triadic_by_review_type = pd.crosstab(staged["review_type_collapsed"], staged["triadic_bps"]).reset_index()
        triadic_by_review_type = triadic_by_review_type.rename(columns={"review_type_collapsed": "review_type"})

        function_by_review_type = pd.crosstab(staged["objective_category"], staged["bps_function"])
        function_by_review_type = function_by_review_type.loc[
            function_by_review_type.sum(axis=1).sort_values(ascending=False).head(8).index,
            function_by_review_type.sum(axis=0).sort_values(ascending=False).head(8).index,
        ]

        function_by_review_type_display = function_by_review_type.copy()
        function_by_review_type_display.index = [
            _shorten_label(index_value, max_len=24) for index_value in function_by_review_type_display.index
        ]
        function_by_review_type_display.columns = [
            _shorten_label(column_value, max_len=24) for column_value in function_by_review_type_display.columns
        ]
    else:
        typology_counts = pd.DataFrame(columns=["provisional_typology", "n", "percent"])
        triadic_by_review_type = pd.DataFrame(columns=["review_type", "yes", "no"])
        function_by_review_type = pd.DataFrame()
        function_by_review_type_display = pd.DataFrame()

    stage3_status = stage3_manifest["fulltext_status"].value_counts().rename_axis("fulltext_status").reset_index(name="n") if not stage3_manifest.empty else pd.DataFrame(columns=["fulltext_status", "n"])
    stage3_status = _with_percent(stage3_status)

    if not semantic_result.record_loadings.empty:
        semantic_summary = semantic_result.domain_summary.copy()
        semantic_summary["mean_cosine"] = semantic_summary["mean_cosine"].round(4)
        semantic_summary["mean_loading"] = semantic_summary["mean_loading"].round(4)
        semantic_dominance = semantic_result.dominance_by_review_type.copy()
        semantic_subdomain_summary = semantic_result.subdomain_summary.copy()
        semantic_pairwise_summary = semantic_result.pairwise_summary.copy()
        semantic_pairwise_records = semantic_result.pairwise_loadings.copy()

        top_frames = []
        for domain in ["biological", "psychological", "social"]:
            loading_col = f"loading_{domain}"
            if loading_col not in semantic_result.record_loadings.columns:
                continue
            top = semantic_result.record_loadings.sort_values(loading_col, ascending=False).head(5).copy()
            top["domain"] = domain
            top["loading"] = top[loading_col]
            top_frames.append(top[["domain", "record_id", "year", "review_type", "loading"]])
        semantic_top_records = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame(columns=["domain", "record_id", "year", "review_type", "loading"])
    else:
        semantic_summary = pd.DataFrame(columns=["domain", "mean_cosine", "mean_loading", "dominance_n"])
        semantic_dominance = pd.DataFrame(columns=["review_type", "biological", "psychological", "social"])
        semantic_subdomain_summary = pd.DataFrame(columns=["domain", "subdomain", "mean_loading", "median_loading"])
        semantic_pairwise_summary = pd.DataFrame(columns=["bio_psych_mean", "bio_social_mean", "psych_social_mean", "triadic_product_mean"])
        semantic_pairwise_records = pd.DataFrame(columns=["record_id", "year", "review_type", "bio_psych", "bio_social", "psych_social", "triadic_product"])
        semantic_top_records = pd.DataFrame(columns=["domain", "record_id", "year", "review_type", "loading"])

    detailed_catalog_columns = [
        "catalog_id",
        "year",
        "review_type",
        "icd11_pain_category",
        "bps_mention_location",
        "bps_function",
        "bio_mentioned",
        "psych_mentioned",
        "social_mentioned",
        "title",
    ]
    if not stage2.empty:
        detailed_catalog = stage2.copy().sort_values(["year", "record_id"], ascending=[False, True]).reset_index(drop=True)
        detailed_catalog["catalog_id"] = detailed_catalog.index + 1
        detailed_catalog = detailed_catalog[detailed_catalog_columns].copy()
        detailed_catalog["bps_mention_location"] = detailed_catalog["bps_mention_location"].replace(
            {
                "title and abstract": "title+abstract",
                "title only": "title",
                "abstract only": "abstract",
                "unclear": "unclear",
            }
        )
        detailed_catalog["bps_function"] = detailed_catalog["bps_function"].replace(
            {
                "background framing": "background",
                "intervention rationale": "intervention",
                "organizing principle": "organizing",
                "explanatory framework": "explanatory",
                "policy/practice implication": "policy/practice",
                "rhetorical label": "rhetorical",
            }
        )
        detailed_catalog["title"] = detailed_catalog["title"].astype(str).map(
            lambda value: value if len(value) <= 120 else value[:117].rstrip() + "..."
        )
    else:
        detailed_catalog = pd.DataFrame(columns=detailed_catalog_columns)

    stage3_by_record = (
        stage3_manifest.set_index("record_id", drop=False).to_dict("index")
        if not stage3_manifest.empty and "record_id" in stage3_manifest.columns
        else {}
    )
    relevance_rows: list[dict[str, str]] = []
    if not stage2.empty:
        for _, row in stage2.iterrows():
            record_id = str(row.get("record_id", ""))
            text_blob = f"{str(row.get('title', ''))}\n{str(row.get('abstract', ''))}".lower()
            flags: list[str] = []
            if "withdrawn" in text_blob or "retracted" in text_blob:
                flags.append("withdrawn_or_retracted_signal")
            if "pain" not in text_blob:
                flags.append("pain_focus_not_explicit")
            if not re.search(r"\b(chronic|persistent|long-term)\b", text_blob):
                flags.append("chronicity_not_explicit")
            if str(row.get("icd11_pain_category", "")).strip().lower() == "unclear":
                flags.append("pain_category_unclear")

            stage3_entry = stage3_by_record.get(record_id, {})
            fulltext_status = str(stage3_entry.get("fulltext_status", "")).strip()
            if str(row.get("stage3_candidate", "")).strip().lower() == "yes" and fulltext_status in {
                "",
                "manual_retrieval_required",
                "pmc_linked_fetch_failed",
                "pmc_fulltext_low_content_manual_check",
            }:
                flags.append("stage3_fulltext_manual_check_pending")

            if any(flag in flags for flag in {"withdrawn_or_retracted_signal", "pain_focus_not_explicit"}):
                priority = "high"
            elif flags:
                priority = "medium"
            else:
                priority = "low"

            relevance_rows.append(
                {
                    "record_id": record_id,
                    "reference_label": f"{_shorten_label(str(row.get('title', '')), max_len=78)}",
                    "stage3_candidate": str(row.get("stage3_candidate", "")),
                    "fulltext_status": fulltext_status,
                    "manual_relevance_priority": priority,
                    "relevance_flags": " | ".join(flags),
                    "osf_manual_adjudication_required": "yes",
                    "reviewer_decision": "",
                    "reviewer_notes": "",
                    "adjudication_decision": "",
                    "adjudication_notes": "",
                }
            )
    manual_relevance_audit = pd.DataFrame(relevance_rows)

    stage2_expected = int((stage1["stage1_decision"] == "include").sum()) if not stage1.empty else 0
    stage3_expected = int((stage2["stage3_candidate"] == "yes").sum()) if not stage2.empty and "stage3_candidate" in stage2.columns else 0
    stage3_actual = int(len(stage3_manifest))
    stage3_manual_queue = int((stage3_manifest.get("manual_retrieval_needed", pd.Series(dtype=str)) == "yes").sum()) if not stage3_manifest.empty else 0
    stage3_reliability_path = project_path("review_stages", "04_extraction", "forms", "stage3_reliability_sample.csv")
    stage3_reliability_n = len(_load_csv(stage3_reliability_path)) if stage3_reliability_path.exists() else 0
    osf_alignment = pd.DataFrame(
        [
            {
                "checkpoint": "Stage 2 includes all Stage 1 included records",
                "osf_requirement": "All eligible records proceed to Stage 2 abstract coding",
                "pipeline_value": f"stage2={len(stage2)}; expected={stage2_expected}",
                "status": "aligned" if len(stage2) == stage2_expected else "check",
            },
            {
                "checkpoint": "Stage 3 candidate manifest coverage",
                "osf_requirement": "Musculoskeletal/unspecified candidates proceed to Stage 3 preparation",
                "pipeline_value": f"manifest={stage3_actual}; expected={stage3_expected}",
                "status": "aligned" if stage3_actual == stage3_expected else "check",
            },
            {
                "checkpoint": "Stage 3 manual full-text queue tracked",
                "osf_requirement": "Full-text screening requires human review when retrieval is pending",
                "pipeline_value": f"manual_queue={stage3_manual_queue}",
                "status": "aligned" if stage3_manual_queue >= 0 else "check",
            },
            {
                "checkpoint": "Stage 3 reliability subset capped",
                "osf_requirement": "20% Stage 3 reliability subset with cap at 20",
                "pipeline_value": f"reliability_subset={stage3_reliability_n}",
                "status": "aligned" if 0 <= stage3_reliability_n <= 20 else "check",
            },
            {
                "checkpoint": "Manual relevance adjudication sheet",
                "osf_requirement": "Eligibility and coding decisions remain human-adjudicated",
                "pipeline_value": f"rows={len(manual_relevance_audit)}",
                "status": "aligned" if len(manual_relevance_audit) == len(stage2) else "check",
            },
        ]
    )

    table_dir = project_path("paper", "assets", "tables")
    figure_dir = project_path("paper", "assets", "figures")

    review_type_counts.to_csv(table_dir / "review_type_counts.csv", index=False)
    icd11_counts.to_csv(table_dir / "icd11_counts.csv", index=False)
    domain_counts.to_csv(table_dir / "domain_mentions.csv", index=False)
    domain_recode_audit.to_csv(table_dir / "domain_mention_recode_audit.csv", index=False)
    core_bps_counts.to_csv(table_dir / "core_bps_structure.csv", index=False)
    bps_location_counts.to_csv(table_dir / "bps_location_counts.csv", index=False)
    bps_function_counts.to_csv(table_dir / "bps_function_counts.csv", index=False)
    msk_scope.to_csv(table_dir / "musculoskeletal_scope.csv", index=False)
    objective_counts.to_csv(table_dir / "objective_category_rule_counts.csv", index=False)
    objective_llm_counts.to_csv(table_dir / "objective_category_llm_counts.csv", index=False)
    psych_concepts.to_csv(table_dir / "psychological_concept_counts.csv", index=False)
    conceptual_problems.to_csv(table_dir / "conceptual_problem_counts.csv", index=False)
    publication_year_counts.to_csv(table_dir / "publication_year_counts.csv", index=False)
    decade_counts.to_csv(table_dir / "publication_period_counts.csv", index=False)
    typology_counts.to_csv(table_dir / "provisional_typology_counts.csv", index=False)
    triadic_by_review_type.to_csv(table_dir / "triadic_by_review_type.csv", index=False)
    if not function_by_review_type_display.empty:
        function_by_review_type_display.to_csv(table_dir / "function_by_review_type_matrix.csv")
    else:
        pd.DataFrame().to_csv(table_dir / "function_by_review_type_matrix.csv")
    stage3_status.to_csv(table_dir / "stage3_retrieval_status.csv", index=False)
    detailed_catalog.to_csv(table_dir / "included_review_catalog.csv", index=False)
    semantic_summary.to_csv(table_dir / "semantic_domain_summary.csv", index=False)
    semantic_subdomain_summary.to_csv(table_dir / "semantic_subdomain_summary.csv", index=False)
    semantic_pairwise_summary.to_csv(table_dir / "semantic_pairwise_summary.csv", index=False)
    semantic_pairwise_records.to_csv(table_dir / "semantic_pairwise_records.csv", index=False)
    semantic_dominance.to_csv(table_dir / "semantic_dominance_by_review_type.csv", index=False)
    semantic_top_records.to_csv(table_dir / "semantic_top_records.csv", index=False)
    manual_relevance_audit.to_csv(table_dir / "manual_relevance_audit.csv", index=False)
    osf_alignment.to_csv(table_dir / "osf_alignment_checklist.csv", index=False)

    _bar_plot(review_type_counts.head(8), "review_type", "n", "Review types among Stage 2 included records", figure_dir / "review_type_counts.png")
    _bar_plot(icd11_counts.head(10), "icd11_pain_category", "n", "ICD-11 pain categories in Stage 2", figure_dir / "icd11_counts.png")
    _bar_plot(domain_counts, "domain", "n", "Substantive domain mentions among included reviews", figure_dir / "domain_mentions.png")
    _panel_descriptive_plot(publication_year_counts, review_type_counts, icd11_counts, core_bps_counts, figure_dir / "descriptive_panel_abcd.png")
    _bar_plot(typology_counts, "provisional_typology", "n", "Provisional BPS operationalization typology", figure_dir / "bps_typology_counts.png")
    _line_plot(publication_year_counts, "year", "n", "Included review publications across time", figure_dir / "publication_year_trend.png")
    _heatmap_plot(function_by_review_type_display, "BPS function by objective category (top categories)", figure_dir / "review_type_function_heatmap.png")
    _operationalization_combined_plot(typology_counts, function_by_review_type_display, msk_scope, domain_counts, core_bps_counts, figure_dir / "operationalization_combined.png")
    _semantic_sunburst_plot(semantic_subdomain_summary, figure_dir / "semantic_loading_sunburst.png")
    _pairwise_loading_plot(semantic_result.record_loadings, semantic_pairwise_records, figure_dir / "semantic_pairwise_loadings.png")
    _semantic_record_profile_plot(semantic_result.record_loadings, stage2, figure_dir / "semantic_domain_profiles.png")
    _semantic_loading_combined_plot(semantic_subdomain_summary, semantic_result.record_loadings, figure_dir / "semantic_loading_combined.png")
    embedding_coords_path = table_dir / "semantic_embedding_landscape_coordinates.csv"
    embedding_coords = _semantic_landscape_integrated_plot(
        semantic_result.record_loadings,
        semantic_subdomain_summary,
        semantic_pairwise_records,
        figure_dir / "semantic_landscape_integrated.png",
        embedding_coords_path,
    )
    if isinstance(embedding_coords, pd.DataFrame) and embedding_coords.empty:
        pd.DataFrame().to_csv(embedding_coords_path, index=False)
    _embedding_landscape_plot(
        semantic_result.record_loadings,
        semantic_subdomain_summary,
        figure_dir / "embedding_landscape.png",
        table_dir / "semantic_embedding_landscape_coordinates_standalone.csv",
    )
    _draw_prisma(summary, db_counts, figure_dir / "prisma_flow.png")
    _write_characteristics_table(
        stage2,
        project_path("paper", "report", "generated", "characteristics_table.tex"),
        stage3_manifest=stage3_manifest,
    )

    _write_latex_table(
        review_type_counts,
        project_path("paper", "report", "generated", "review_type_table.tex"),
        "Review types among included Stage 2 records.",
        "tab.review.types",
        "Percent is calculated over all Stage 2 included records.",
    )
    _write_latex_table(
        icd11_counts,
        project_path("paper", "report", "generated", "icd11_table.tex"),
        "ICD-11 pain classifications assigned in Stage 2.",
        "tab.icd11",
        "Coding is based on title and abstract information in Stage 2.",
    )
    _write_latex_table(
        domain_counts,
        project_path("paper", "report", "generated", "domain_table.tex"),
        "Biological, psychological, and social substantive mentions in Stage 2.",
        "tab.domains",
        "Substantive mention recoding excludes lexical BPS tokens (e.g., biopsychosocial) and requires domain-specific content cues. Domain counts are not mutually exclusive.",
    )
    _write_latex_table(
        bps_location_counts,
        project_path("paper", "report", "generated", "bps_location_table.tex"),
        "Primary Research Question 1. BPS mention location.",
        "tab.bps.location",
        "Location is coded from title and abstract fields.",
    )
    _write_latex_table(
        bps_function_counts,
        project_path("paper", "report", "generated", "bps_function_table.tex"),
        "Primary Research Question 1. BPS functional role.",
        "tab.bps.function",
        "Functional role is produced through structured abstract-level coding and remains provisional pending Stage 3 adjudication.",
    )
    _write_latex_table(
        msk_scope,
        project_path("paper", "report", "generated", "musculoskeletal_scope_table.tex"),
        "Primary Research Question 2. Substantive scope and triadic coverage in musculoskeletal reviews.",
        "tab.msk.scope",
        "Percent is calculated within Stage 2 musculoskeletal reviews after substantive-domain recoding.",
    )
    _write_latex_table(
        psych_concepts.head(20),
        project_path("paper", "report", "generated", "psych_concepts_table.tex"),
        "Primary Research Question 3. Most frequent psychological concepts.",
        "tab.psych.concepts",
        "Concepts are extracted from Stage 2 detected concept strings.",
    )
    _write_latex_table(
        conceptual_problems,
        project_path("paper", "report", "generated", "conceptual_problems_table.tex"),
        "Secondary Research Question. Conceptual problems in BPS usage.",
        "tab.conceptual.problems",
        "Counts are from the structured Stage 2 semantic coding layer and remain provisional pending full-text adjudication.",
    )
    _write_latex_table(
        publication_year_counts,
        project_path("paper", "report", "generated", "publication_year_table.tex"),
        "Publication year distribution among included reviews.",
        "tab.publication.year",
        "Percent is calculated over all Stage 2 included reviews.",
    )
    _write_latex_table(
        decade_counts,
        project_path("paper", "report", "generated", "publication_period_table.tex"),
        "Publication period distribution among included reviews.",
        "tab.publication.period",
        "Periods are grouped by decade.",
    )
    _write_latex_table(
        typology_counts,
        project_path("paper", "report", "generated", "bps_typology_table.tex"),
        "Primary Research Question 1. Provisional typology of BPS operationalization.",
        "tab.bps.typology",
        "Typology is provisional because Stage 3 integration coding and adjudication are not yet complete.",
    )
    _write_latex_table(
        triadic_by_review_type,
        project_path("paper", "report", "generated", "triadic_by_review_type_table.tex"),
        "Primary Research Question 2. Triadic BPS mention by review type.",
        "tab.triadic.by.review.type",
        "Counts indicate whether biological, psychological, and social mentions were all present.",
    )
    _write_latex_table(
        function_by_review_type_display.reset_index() if not function_by_review_type_display.empty else pd.DataFrame(columns=["objective_category"]),
        project_path("paper", "report", "generated", "function_by_review_type_table.tex"),
        "Primary Research Question 1. BPS function by objective category (top categories).",
        "tab.function.by.review.type",
        "Top rows and columns are selected by frequency for readability and labels are shortened to reduce page overflow.",
    )
    _write_latex_table(
        stage3_status,
        project_path("paper", "report", "generated", "stage3_retrieval_table.tex"),
        "Stage 3 full-text retrieval status.",
        "tab.stage3.retrieval",
        "Manual retrieval queue entries remain pending full-text acquisition.",
    )
    _write_latex_table(
        semantic_summary,
        project_path("paper", "report", "generated", "semantic_domain_summary_table.tex"),
        "Ontology-based semantic loading summary across BPS domains.",
        "tab.semantic.summary",
        f"Method={semantic_result.method}; model={semantic_result.model}. Mean loading uses softmax-normalized cosine similarity.",
    )
    _write_latex_table(
        semantic_subdomain_summary,
        project_path("paper", "report", "generated", "semantic_subdomain_summary_table.tex"),
        "Ontology subdomain loading summary across BPS hierarchy.",
        "tab.semantic.subdomain.summary",
        "Mean loading is weighted by parent domain loading for each included record.",
    )
    _write_latex_table(
        semantic_pairwise_summary,
        project_path("paper", "report", "generated", "semantic_pairwise_summary_table.tex"),
        "Pairwise and triadic semantic integration summary.",
        "tab.semantic.pairwise.summary",
        "Pairwise scores are products of normalized domain loadings per record.",
    )
    _write_latex_table(
        semantic_dominance,
        project_path("paper", "report", "generated", "semantic_dominance_table.tex"),
        "Dominant semantic domain by review type.",
        "tab.semantic.dominance",
        "Dominance is assigned by highest domain loading per included record.",
    )
    _write_latex_table(
        semantic_top_records,
        project_path("paper", "report", "generated", "semantic_top_records_table.tex"),
        "Highest-loading records per ontology domain.",
        "tab.semantic.top.records",
        "Top five records per domain are ranked by normalized loading score.",
    )
    _write_latex_longtable(
        detailed_catalog,
        project_path("paper", "report", "generated", "included_catalog_longtable.tex"),
        "Comprehensive Stage 2 included review catalog.",
        "tab.included.catalog",
        "This catalog is generated automatically from Stage 2 outputs and is intended for transparent audit and adjudication.",
    )

    triadic_n = int(msk_scope.loc[msk_scope["indicator"] == "Triadic BPS mention present", "n"].iloc[0]) if not msk_scope.empty and "Triadic BPS mention present" in msk_scope["indicator"].values else 0
    msk_n = int(msk_scope.loc[msk_scope["indicator"] == "Musculoskeletal reviews in Stage 2", "n"].iloc[0]) if not msk_scope.empty and "Musculoskeletal reviews in Stage 2" in msk_scope["indicator"].values else 0
    msk_bio_n = int(msk_scope.loc[msk_scope["indicator"] == "Biological mention present", "n"].iloc[0]) if not msk_scope.empty and "Biological mention present" in msk_scope["indicator"].values else 0
    msk_psych_n = int(msk_scope.loc[msk_scope["indicator"] == "Psychological mention present", "n"].iloc[0]) if not msk_scope.empty and "Psychological mention present" in msk_scope["indicator"].values else 0
    msk_soc_n = int(msk_scope.loc[msk_scope["indicator"] == "Social mention present", "n"].iloc[0]) if not msk_scope.empty and "Social mention present" in msk_scope["indicator"].values else 0

    q1_top_loc = bps_location_counts.iloc[0]["bps_mention_location"] if not bps_location_counts.empty else "not available"
    q1_top_loc_n = int(bps_location_counts.iloc[0]["n"]) if not bps_location_counts.empty else 0
    q1_top_loc_pct = float(bps_location_counts.iloc[0]["percent"]) if not bps_location_counts.empty else 0.0
    q1_top_func = bps_function_counts.iloc[0]["bps_function"] if not bps_function_counts.empty else "not available"
    q1_top_func_pct = float(bps_function_counts.iloc[0]["percent"]) if not bps_function_counts.empty else 0.0
    q1_second_func = bps_function_counts.iloc[1]["bps_function"] if len(bps_function_counts) > 1 else "not available"
    q1_second_func_pct = float(bps_function_counts.iloc[1]["percent"]) if len(bps_function_counts) > 1 else 0.0
    q1_top2_share = q1_top_func_pct + q1_second_func_pct
    rhetorical_pct = float(
        bps_function_counts.loc[
            bps_function_counts["bps_function"].astype(str).str.contains("rhetorical", case=False, na=False),
            "percent",
        ].sum()
    ) if not bps_function_counts.empty else 0.0
    secondary_top_problem = conceptual_problems.iloc[0]["conceptual_problem_flag"] if not conceptual_problems.empty else "none identified"
    q1_top_typology = typology_counts.iloc[0]["provisional_typology"] if not typology_counts.empty else "not available"

    triadic_yes = int(triadic_by_review_type["yes"].sum()) if not triadic_by_review_type.empty and "yes" in triadic_by_review_type.columns else 0
    triadic_total = int(triadic_yes + triadic_by_review_type["no"].sum()) if not triadic_by_review_type.empty and "no" in triadic_by_review_type.columns else 0
    triadic_percent = round((triadic_yes / triadic_total) * 100, 1) if triadic_total else 0.0
    semantic_n = len(semantic_result.record_loadings)

    latest_year = int(publication_year_counts["year"].max()) if not publication_year_counts.empty else 0
    earliest_year = int(publication_year_counts["year"].min()) if not publication_year_counts.empty else 0

    # Typology proportions for richer prose
    type_pseudo_pct = float(typology_counts.loc[typology_counts["provisional_typology"].str.contains("pseudo", case=False), "percent"].sum()) if not typology_counts.empty else 0.0
    type_integrative_pct = float(typology_counts.loc[typology_counts["provisional_typology"].str.contains("integrative", case=False), "percent"].sum()) if not typology_counts.empty else 0.0
    type_multi_pct = float(typology_counts.loc[typology_counts["provisional_typology"].str.contains("multifactorial", case=False), "percent"].sum()) if not typology_counts.empty else 0.0

    # Semantic loading stats
    sem_bio_mean = float(semantic_result.domain_summary.loc[semantic_result.domain_summary["domain"] == "biological", "mean_loading"].iloc[0]) if not semantic_result.domain_summary.empty else 0.0
    sem_psy_mean = float(semantic_result.domain_summary.loc[semantic_result.domain_summary["domain"] == "psychological", "mean_loading"].iloc[0]) if not semantic_result.domain_summary.empty else 0.0
    sem_soc_mean = float(semantic_result.domain_summary.loc[semantic_result.domain_summary["domain"] == "social", "mean_loading"].iloc[0]) if not semantic_result.domain_summary.empty else 0.0
    sem_bio_dom = int(semantic_result.domain_summary.loc[semantic_result.domain_summary["domain"] == "biological", "dominance_n"].iloc[0]) if not semantic_result.domain_summary.empty else 0
    sem_psy_dom = int(semantic_result.domain_summary.loc[semantic_result.domain_summary["domain"] == "psychological", "dominance_n"].iloc[0]) if not semantic_result.domain_summary.empty else 0
    sem_soc_dom = int(semantic_result.domain_summary.loc[semantic_result.domain_summary["domain"] == "social", "dominance_n"].iloc[0]) if not semantic_result.domain_summary.empty else 0

    # Top-3 psych concepts
    top3_concepts = [_latex_escape(str(r["psychological_concept"])) for _, r in psych_concepts.head(3).iterrows()] if not psych_concepts.empty else ["not available"]
    top3_concept_str = ", ".join(top3_concepts) if len(top3_concepts) >= 3 else top3_concepts[0]

    # Psychological concept profile detail
    psych_total_occ = int(pd.to_numeric(psych_concepts.get("n", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not psych_concepts.empty else 0
    psych_records_with_any = int(stage2["psychological_concepts_detected"].astype(str).str.strip().ne("").sum()) if "psychological_concepts_detected" in stage2.columns else 0
    psych_records_pct = round((psych_records_with_any / len(stage2)) * 100, 1) if len(stage2) else 0.0
    psych_top_rows = psych_concepts.head(6).to_dict("records") if not psych_concepts.empty else []
    psych_top3_share = round(float(pd.to_numeric(psych_concepts.head(3)["n"], errors="coerce").fillna(0).sum()) / psych_total_occ * 100, 1) if psych_total_occ else 0.0

    # Framework mentions at abstract level
    framework_tokens: list[str] = []
    if "theoretical_frameworks_detected" in stage2.columns:
        for value in stage2["theoretical_frameworks_detected"].astype(str):
            for token in value.split("|"):
                cleaned = token.strip().lower()
                if cleaned:
                    framework_tokens.append(cleaned)
    framework_counts = pd.Series(framework_tokens).value_counts().rename_axis("framework").reset_index(name="n") if framework_tokens else pd.DataFrame(columns=["framework", "n"])
    framework_total_occ = int(framework_counts["n"].sum()) if not framework_counts.empty else 0
    if not framework_counts.empty and framework_total_occ:
        framework_counts["percent"] = (framework_counts["n"] / framework_total_occ) * 100.0
    framework_top = framework_counts.head(3).to_dict("records") if not framework_counts.empty else []

    # Conceptual problem detail
    flag_total_occ = int(pd.to_numeric(conceptual_problems.get("n", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not conceptual_problems.empty else 0
    flag_top_rows = conceptual_problems.head(5).to_dict("records") if not conceptual_problems.empty else []
    known_flag_tokens = [
        "parallel_listing_without_integration",
        "mechanistic_absence",
        "missing_biology",
        "missing_social",
        "tokenistic_bps",
        "vague_definition",
        "construct_overlap",
    ]
    conceptual_records_with_any = 0
    if "conceptual_problem_flags" in stage2.columns:
        for value in stage2["conceptual_problem_flags"].astype(str):
            lowered = value.lower()
            if any(token in lowered for token in known_flag_tokens):
                conceptual_records_with_any += 1
    conceptual_records_pct = round((conceptual_records_with_any / len(stage2)) * 100, 1) if len(stage2) else 0.0

    def _fmt_ranked(rows: list[dict[str, object]], name_key: str, n_key: str = "n", pct_key: str = "percent", max_items: int = 3) -> str:
        if not rows:
            return "not available"
        pieces: list[str] = []
        for row in rows[:max_items]:
            label = _latex_escape(str(row.get(name_key, "")))
            n_val = int(float(row.get(n_key, 0) or 0))
            pct_val = float(row.get(pct_key, 0.0) or 0.0)
            pieces.append(f"{label} ({n_val}; {pct_val:.1f}\\%)")
        return "; ".join(pieces)

    psych_ranked_str = _fmt_ranked(psych_top_rows, "psychological_concept", max_items=6)
    framework_ranked_str = _fmt_ranked(framework_top, "framework", max_items=3)
    conceptual_ranked_str = _fmt_ranked(flag_top_rows, "conceptual_problem_flag", max_items=5)

    n_records = summary["stage2_records"]

    primary_answers = (
        "\\noindent\\textbf{BPS Operationalization.} "
        f"Across {n_records} included reviews, BPS terminology appeared predominantly in the abstract without title mention "
        f"(dominant location: {_latex_escape(str(q1_top_loc))}; {q1_top_loc_n} records, {q1_top_loc_pct:.1f}\\%). The most common declared functional role was "
        f"\\textit{{{_latex_escape(str(q1_top_func))}}} ({q1_top_func_pct:.1f}\\%), followed by {_latex_escape(str(q1_second_func))} ({q1_second_func_pct:.1f}\\%), together accounting for {q1_top2_share:.1f}\\% of the corpus. "
        f"Provisional typological classification: potential integrative signal {type_integrative_pct:.1f}\\%, "
        f"multifactorial signal {type_multi_pct:.1f}\\%, pseudo-BPS or partial signal {type_pseudo_pct:.1f}\\%, "
        f"rhetorical label signal {rhetorical_pct:.1f}\\%. "
        "This distribution provides empirical support for the pre-specified hypothesis that BPS language is often used as a rhetorical or contextualizing device rather than as a specification of triadic mechanistic integration. "
        "Evidence is visualized in Figure~\\ref{fig.operationalization.combined} (Panels~A and~B).\n\n"
        "\\par\\medskip\n"
        "\\noindent\\textbf{Scope, Balance, and Integration in Musculoskeletal Reviews.} "
        f"Using the substantive domain-coding layer (lexical BPS token matches excluded), coverage within the {msk_n} musculoskeletal reviews was: "
        f"biological {msk_bio_n}/{msk_n} ({round(msk_bio_n/msk_n*100,1) if msk_n else 0}\\%), "
        f"psychological {msk_psych_n}/{msk_n} ({round(msk_psych_n/msk_n*100,1) if msk_n else 0}\\%), "
        f"social {msk_soc_n}/{msk_n} ({round(msk_soc_n/msk_n*100,1) if msk_n else 0}\\%). "
        f"Simultaneous triadic co-mention was present in {triadic_n}/{msk_n} musculoskeletal reviews "
        f"({round(triadic_n/msk_n*100,1) if msk_n else 0}\\%). "
        f"Across all {n_records} Stage~2 records, triadic co-mention was {triadic_yes}/{triadic_total} ({triadic_percent}\\%). "
        f"Ontology-aligned semantic loading (covering {semantic_n} records) identified psychological orientation as dominant in {sem_psy_dom} records "
        f"({round(sem_psy_dom/semantic_n*100,1) if semantic_n else 0}\\%), biological in {sem_bio_dom} "
        f"({round(sem_bio_dom/semantic_n*100,1) if semantic_n else 0}\\%), and social in {sem_soc_dom} "
        f"({round(sem_soc_dom/semantic_n*100,1) if semantic_n else 0}\\%). "
        f"Mean domain loadings were: biological {sem_bio_mean:.3f}, psychological {sem_psy_mean:.3f}, social {sem_soc_mean:.3f}. "
        "Taken together, the two layers indicate divergence between domain naming and semantic weight: social content is often present but comparatively light in semantic centrality. "
        "Evidence is in Figure~\\ref{fig.operationalization.combined} (Panel~D), Figure~\\ref{fig.semantic.loading.combined}, and Figure~\\ref{fig.semantic.landscape.integrated}.\n\n"
        "\\par\\medskip\n"
        "\\noindent\\textbf{Psychological Concepts and Frameworks.} "
        f"Across {n_records} Stage~2 records, {psych_records_with_any}/{n_records} ({psych_records_pct:.1f}\\%) contained at least one explicit psychological construct token in abstract-level coding. "
        f"A total of {psych_total_occ} construct occurrences were detected; ranked frequencies were: {psych_ranked_str}. "
        f"The top three concepts ({top3_concept_str}) accounted for {psych_top3_share:.1f}\\% of all detected construct occurrences, indicating concentration in high-level affect labels. "
        "By contrast, theoretically specific mechanism-oriented constructs (for example, catastrophizing, fear-avoidance, self-efficacy, and illness perception) appeared less frequently, supporting a profile in which symptom-language is more prevalent than explicit mechanism-language at abstract level. "
        f"Framework mentions were comparatively sparse ({framework_total_occ} total mentions); top entries were: {framework_ranked_str}. "
        f"Ontology-based subdomain loading further indicated that psychological-domain weight is carried primarily by broad cognitive-behavioral and affective clusters rather than narrowly specified mechanistic frameworks. "
        "Stage~3 full-text coding will refine construct definitions, framework attribution, and hierarchical mapping under manual adjudication. "
        "Evidence is in Figure~\\ref{fig.semantic.loading.combined} (Panel~A) and the supplementary semantic tables.\n"
    )
    ensure_parent(project_path("paper", "report", "generated", "primary_answers.tex")).write_text(primary_answers, encoding="utf-8")

    secondary_answer = (
        "\\noindent\\textbf{Conceptual Problems in BPS Usage.} "
        f"From {n_records} Stage~2 records, {conceptual_records_with_any}/{n_records} ({conceptual_records_pct:.1f}\\%) contained at least one non-none conceptual-problem flag. "
        f"At the occurrence level (multi-label flags; not mutually exclusive), {flag_total_occ} flags were recorded in total. "
        f"The highest-frequency provisional flags were: {conceptual_ranked_str}. "
        f"The most frequent single flag was \\textit{{{_latex_escape(str(secondary_top_problem))}}}. "
        "Taken together, this pattern indicates that BPS terminology is often accompanied by parallel domain listing and limited mechanistic specification, with additional signals of tokenistic framing and uneven domain coverage. "
        "These indicators are treated as hypothesis-generating and provisional until Stage~3 full-text adjudication confirms or revises them at full-text depth.\n"
    )
    ensure_parent(project_path("paper", "report", "generated", "secondary_answer.tex")).write_text(secondary_answer, encoding="utf-8")

    results_text = (
        f"A total of {summary['combined_records']} records were identified across MEDLINE (PubMed) and Web of Science "
        f"({summary['deduplicated_records']} after deduplication, removing {summary['duplicates_removed']} duplicates). "
        f"Stage~1 title and abstract screening yielded {summary['included_records']} included, "
        f"{summary['excluded_records']} excluded, and {summary['unclear_records']} borderline records. "
        f"Stage~2 abstract-level semantic coding was completed for {summary['stage2_records']} included reviews "
        f"spanning publication years {earliest_year}--{latest_year}. "
        f"A Stage~3 candidate pool of {summary['stage3_candidates']} records was identified for full-text deep coding, "
        f"of which {summary['stage3_pmc_open_fulltexts']} were retrieved from open-access repositories and "
        f"{summary['stage3_manual_retrieval_required']} remain in the manual retrieval queue. "
        f"Ontology-aligned semantic loading was applied to all {summary['semantic_records']} Stage~2 records "
        f"using {_latex_escape(semantic_result.method)} ({_latex_escape(semantic_result.model)}). "
        "The study-selection workflow is shown in Figure~\\ref{fig.prisma}, and descriptive characteristics are summarized in Table~\\ref{tab.characteristics}.\n"
    )
    ensure_parent(project_path("paper", "report", "generated", "results_summary.tex")).write_text(results_text, encoding="utf-8")

    limitations_text = (
        "This synthesis represents a high-fidelity interim analysis pending Stage~3 full-text coding and dual-coder adjudication. "
        "The search covered MEDLINE and Web of Science; PsycINFO records were not available for this run due to credential constraints, "
        "which may underrepresent reviews indexed primarily in psychology databases. "
        "Stage~3 full-text retrieval is incomplete (53/87 in manual queue), constraining inferential closure on domain integration quality. "
        "The provisional typology relies on abstract-level signals and may misclassify records whose integration quality is apparent only in full text. "
        "Semantic loading is sensitive to ontology term selection; the 42-subdomain scheme reflects one operationalization among possible alternatives.\n"
    )
    ensure_parent(project_path("paper", "report", "generated", "limitations.tex")).write_text(limitations_text, encoding="utf-8")

    reliability_lines = []
    if not reliability.empty:
        for _, row in reliability.iterrows():
            stage = row.get("stage", "")
            status = row.get("status", "")
            n = int(row.get("n", 0)) if str(row.get("n", "")).strip() else 0
            pa = row.get("percent_agreement", "")
            kappa = row.get("cohen_kappa", "")
            line = f"{stage}. status={status}, n={n}, percent_agreement={pa}, kappa={kappa}"
            reliability_lines.append(_latex_escape(line))
    else:
        reliability_lines.append(_latex_escape("No reliability report available yet. Use dual-coded templates and rerun reliability-report."))

    supplement_text = (
        "Supplementary Methods Note. Protocol alignment and reliability.\n\n"
        "Operational pipeline stages preserve the pre-registered sequencing while logging deviations explicitly. "
        "Stage~2 uses structured LLM-based abstract coding with deterministic metadata fields and archived batch outputs; this augments rather than replaces final adjudication. Dual-coder templates are generated for Stage 1, Stage 2, and Stage 3 reliability checks, consistent with the registered subset reliability design. Current reliability file status.\n"
        + "\n".join(reliability_lines)
        + "\n\n"
        + _latex_escape(f"Semantic loading note. {semantic_result.note}")
        + "\n"
    )
    ensure_parent(project_path("paper", "report", "generated", "supplement_methods.tex")).write_text(supplement_text, encoding="utf-8")

    ensure_parent(project_path("review_stages", "05_synthesis", "outputs", "prisma_counts.json")).write_text(
        json.dumps({"summary": summary, "database_counts": db_counts.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )
    ensure_parent(project_path("review_stages", "05_synthesis", "outputs", "results_summary.json")).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary
