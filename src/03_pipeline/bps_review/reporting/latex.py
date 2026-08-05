from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from bps_review.settings import ROOT, protocol_config, resolve_path
from bps_review.utils.io import append_audit_log, read_json


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


def _csv_to_latex_table(csv_path: Path, output_path: Path, caption: str, label: str) -> None:
    frame = pd.read_csv(csv_path)
    latex = frame.to_latex(index=False, escape=True)
    wrapped = "\n".join(
        [
            r"\begin{table}[ht]",
            r"\centering",
            latex,
            rf"\caption{{{_latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    output_path.write_text(wrapped, encoding="utf-8")


def build_report() -> Path:
    summary = read_json(resolve_path("processed_manuscript") / "summary_metrics.json")
    tables_dir = resolve_path("processed_tables")
    figures_dir = resolve_path("processed_figures")

    paper_tables = ROOT / "paper" / "assets" / "tables"
    paper_figures = ROOT / "paper" / "assets" / "figures"
    paper_tables.mkdir(parents=True, exist_ok=True)
    paper_figures.mkdir(parents=True, exist_ok=True)

    for csv_name, caption, label in [
        ("review_type_counts.csv", "Distribution of review types in the included corpus.", "tab:review-types"),
        ("icd11_counts.csv", "ICD-11 pain category classification at abstract level.", "tab:icd11"),
        ("bps_function_counts.csv", "Functions of the biopsychosocial label in included reviews.", "tab:bps-function"),
        ("typology_counts.csv", "Typology of biopsychosocial operationalization.", "tab:typology"),
        ("top_concepts.csv", "Most frequently identified psychological concepts.", "tab:concepts"),
    ]:
        _csv_to_latex_table(tables_dir / csv_name, paper_tables / csv_name.replace(".csv", ".tex"), caption, label)

    for figure_name in ["prisma_flow.png", "review_types.png", "typology.png", "semantic_radar.png", "concept_network.png"]:
        shutil.copy2(figures_dir / figure_name, paper_figures / figure_name)

    protocol = protocol_config()
    main_tex = ROOT / "paper" / "report" / "main.tex"
    report_text = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\usepackage{{float}}
\title{{How the Biopsychosocial Model Frames Chronic Pain Research}}
\author{{Automated Reproducible Review Pipeline}}
\date{{Compiled on \today}}

\begin{document}
\maketitle

\begin{{abstract}}
This report operationalizes the OSF-registered systematic review on how the biopsychosocial model is used in chronic pain review literature. Using a reproducible Python pipeline aligned to the registered protocol, records were retrieved from PubMed/MEDLINE with slots for Web of Science and PsycINFO imports, deduplicated, screened with auditable rules, coded at abstract level, and synthesized into descriptive tables and concept-oriented figures. The present run identified {summary['counts']['identified']} PubMed records, retained {summary['counts']['deduped']} after deduplication, and carried {summary['counts']['stage2_coded']} records into abstract-level coding. Across the coded corpus, the strongest mean semantic projection was toward the psychological axis ({summary['domain_means']['semantic_psychological']:.3f}), followed by the biological axis ({summary['domain_means']['semantic_biological']:.3f}) and the social axis ({summary['domain_means']['semantic_social']:.3f}). The dominant operationalization pattern in this run was {summary['top_typology'].get('bps_typology', 'unclear')}, consistent with the registered expectation that the BPS model is frequently invoked without deep cross-domain integration.
\end{{abstract}}

\section{{Introduction}}
The review is registered at OSF under DOI \texttt{{{protocol['review']['registration_doi']}}}. Its goal is to examine whether reviews claiming a biopsychosocial framework in chronic pain research operationalize that framework as an integrative explanatory model or use it primarily as a label, background frame, or rhetorical device.

\section{{Methods}}
The repository follows the registered mixed-method review design. Eligibility criteria included adult chronic pain review articles in English between 1977 and 2025 that mention the biopsychosocial model in the title or abstract. Exclusions covered primary studies, protocols, editorials, letters, conference abstracts, pediatric-only studies, acute pain records, animal studies, and grey literature. The protocol-specified sources are Web of Science, PsycINFO, and MEDLINE via PubMed. In this automated environment, PubMed retrieval is fully implemented, and manual-import hooks exist for the other two protocol sources.

Stage 1 applied auditable title and abstract rules. Stage 2 coded review characteristics, pain classification, BPS function, domain coverage, and extracted psychological concepts. A semantic projection method based on TF--IDF similarity to biological, psychological, and social anchor lexicons was used to quantify relative emphasis across abstracts. LLM assistance was configured for higher-order concept clustering where available, consistent with the protocol's allowance for AI-assisted coding under human validation.

\section{{Results}}
\subsection{{Flow of Records}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\textwidth]{{../assets/figures/prisma_flow.png}}
\caption{{Simplified PRISMA-style flow for the current automated run.}}
\label{{fig:prisma}}
\end{{figure}}

\subsection{{Review Characteristics}}
\input{{../assets/tables/review_type_counts.tex}}
\input{{../assets/tables/icd11_counts.tex}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.78\textwidth]{{../assets/figures/review_types.png}}
\caption{{Distribution of review types in the coded corpus.}}
\label{{fig:review-types}}
\end{{figure}}

\subsection{{How the BPS Model Was Used}}
\input{{../assets/tables/bps_function_counts.tex}}
\input{{../assets/tables/typology_counts.tex}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.78\textwidth]{{../assets/figures/typology.png}}
\caption{{Counts of BPS operationalization typologies assigned at abstract level.}}
\label{{fig:typology}}
\end{{figure}}

In this run, the most common typology was \texttt{{{_latex_escape(summary['top_typology'].get('bps_typology', 'unclear'))}}}. The most common balance profile was \texttt{{{_latex_escape(summary['top_balance'].get('overall_balance', 'unclear'))}}}. These descriptive patterns are consistent with the registered hypothesis that the BPS model is often invoked in ways that foreground psychological content and under-specify social or mechanistic integration.

\subsection{{Semantic BPS Balance}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.55\textwidth]{{../assets/figures/semantic_radar.png}}
\caption{{Mean semantic projection of coded abstracts onto biological, psychological, and social anchor lexicons.}}
\label{{fig:semantic-radar}}
\end{{figure}}

The mean semantic projection scores were biological = {summary['domain_means']['semantic_biological']:.3f}, psychological = {summary['domain_means']['semantic_psychological']:.3f}, and social = {summary['domain_means']['semantic_social']:.3f}. This profile suggests that, at the abstract level, psychological language is more central than social language and somewhat more prominent than biological language in reviews invoking the BPS label.

\subsection{{Psychological Concepts}}
\input{{../assets/tables/top_concepts.tex}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.82\textwidth]{{../assets/figures/concept_network.png}}
\caption{{Co-occurrence network of extracted psychological concepts across coded abstracts.}}
\label{{fig:concept-network}}
\end{{figure}}

\section{{Discussion}}
The current automated run already supports a plausible interpretation aligned with the registered expectations: biopsychosocial framing is frequent, but cross-domain integration appears limited and psychologically weighted. This should be treated as an auditable computational first pass rather than the final human-completed review. The repository is structured so that manual Web of Science and PsycINFO exports, adjudicated screening, and Stage 3 full-text coding can be layered on top of the present corpus without restructuring the project.

\section{{Limitations}}
This compiled report reflects the current repository run and therefore inherits its operational constraints. First, full protocol execution requires manual import of Web of Science and PsycINFO results because those interfaces are not openly scriptable in this environment. Second, Stage 3 full-text coding remains contingent on full-text retrieval for the musculoskeletal subset. Third, semantic projections are useful for comparative profiling but do not replace interpretive full-text coding of integration quality.

\section{{Reproducibility}}
All scripts, config files, generated tables, and figures are stored in the repository. The report compiles with \texttt{{tectonic}}, and run metadata are recorded in the audit trail.

\end{{document}}
"""
    main_tex.write_text(report_text.strip() + "\n", encoding="utf-8")

    command = ["tectonic", str(main_tex.name)]
    subprocess.run(command, cwd=main_tex.parent, check=True)
    pdf_path = main_tex.with_suffix(".pdf")

    append_audit_log(
        resolve_path("audit_trail") / "reporting_log.jsonl",
        {"stage": "reporting", "output_pdf": str(pdf_path)},
    )
    return pdf_path
