from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from bps_review.utils.paths import project_path


def _agreement_report(frame: pd.DataFrame, a_col: str, b_col: str) -> dict[str, object]:
    if frame.empty:
        return {"status": "empty", "n": 0, "percent_agreement": None, "cohen_kappa": None}
    if a_col not in frame.columns or b_col not in frame.columns:
        return {"status": "not_ready", "n": 0, "percent_agreement": None, "cohen_kappa": None}

    subset = frame[[a_col, b_col]].fillna("").astype(str).apply(lambda col: col.str.strip())
    subset = subset.loc[(subset[a_col] != "") & (subset[b_col] != "")].copy()
    if subset.empty:
        return {"status": "not_ready", "n": 0, "percent_agreement": None, "cohen_kappa": None}

    n = len(subset)
    agreement = float((subset[a_col] == subset[b_col]).mean())
    kappa = float(cohen_kappa_score(subset[a_col], subset[b_col]))
    return {
        "status": "ok",
        "n": n,
        "percent_agreement": round(agreement, 4),
        "cohen_kappa": round(kappa, 4),
    }


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def stage_reliability_report() -> dict[str, object]:
    stage1 = _load(project_path("review_stages", "03_screening", "inputs", "stage1_double_screen_subset.csv"))
    stage2 = _load(project_path("review_stages", "04_extraction", "forms", "stage2_double_code_subset.csv"))
    stage3 = _load(project_path("review_stages", "04_extraction", "forms", "stage3_reliability_sample.csv"))

    report = {
        "stage1": _agreement_report(stage1, "reviewer_a_decision", "reviewer_b_decision"),
        "stage2": _agreement_report(stage2, "reviewer_a_objective_category", "reviewer_b_objective_category"),
        "stage3": _agreement_report(stage3, "reviewer_a_bps_typology", "reviewer_b_bps_typology"),
    }

    out_path = project_path("review_stages", "03_screening", "audit", "reliability_report.csv")
    rows = []
    for stage, values in report.items():
        rows.append({"stage": stage, **values})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return report
