from __future__ import annotations

import math

import pandas as pd

from bps_review.utils.io import write_csv
from bps_review.utils.paths import project_path


def _sample_size(total: int, proportion: float, cap: int) -> int:
    if total <= 0:
        return 0
    return min(cap, max(1, math.ceil(total * proportion)))


def prepare_screening_materials() -> dict[str, int]:
    search_path = project_path("review_stages", "02_search", "outputs", "deduplicated_records.csv")
    frame = pd.read_csv(search_path).fillna("")

    rayyan_columns = ["record_id", "title", "abstract", "year", "journal", "authors", "doi", "pmid", "database"]
    rayyan = frame[[column for column in rayyan_columns if column in frame.columns]].copy()
    write_csv(project_path("review_stages", "03_screening", "inputs", "rayyan_screening_queue.csv"), rayyan)
    write_csv(project_path("review_stages", "03_screening", "inputs", "stage1_screening_queue.csv"), rayyan)

    pilot_n = min(10, len(frame))
    pilot = frame.sample(n=pilot_n, random_state=42) if pilot_n else frame.head(0)
    write_csv(project_path("review_stages", "03_screening", "inputs", "pilot_screening_sample.csv"), pilot)

    reliability_n = _sample_size(len(frame), 0.20, 50)
    reliability = frame.sample(n=reliability_n, random_state=7) if reliability_n else frame.head(0)
    reliability = reliability.copy()
    reliability["reviewer_a_decision"] = ""
    reliability["reviewer_b_decision"] = ""
    reliability["reviewer_a_reason"] = ""
    reliability["reviewer_b_reason"] = ""
    reliability["adjudicated_decision"] = ""
    reliability["adjudication_notes"] = ""
    write_csv(project_path("review_stages", "03_screening", "inputs", "stage1_double_screen_subset.csv"), reliability)

    summary = {
        "screening_queue_records": len(rayyan),
        "rayyan_queue_records": len(rayyan),
        "pilot_screening_sample": len(pilot),
        "stage1_double_screen_subset": len(reliability),
    }
    write_csv(
        project_path("review_stages", "03_screening", "audit", "screening_preparation_summary.csv"),
        pd.DataFrame([summary]),
    )
    return summary
