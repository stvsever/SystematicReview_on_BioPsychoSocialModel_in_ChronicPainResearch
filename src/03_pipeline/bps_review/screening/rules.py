from __future__ import annotations

from datetime import date, datetime
import re

import pandas as pd

from bps_review.utils.io import load_yaml
from bps_review.utils.paths import project_path


BPS_TERMS = ("biopsychosocial", "bio-psycho-social", "bio psycho social")
PEDIATRIC_TERMS = ("child", "children", "adolescent", "adolescents", "pediatric", "paediatric", "youth")
ADULT_TERMS = ("adult", "adults", "older adult", "working-age")
ANIMAL_TERMS = ("rat", "rats", "mouse", "mice", "animal model", "rodent", "canine", "murine")
COMMENTARY_TERMS = ("editorial", "letter", "commentary", "perspective")
CHRONIC_PAIN_TERMS = (
    "chronic pain",
    "persistent pain",
    "musculoskeletal",
    "low back pain",
    "back pain",
    "neck pain",
    "fibromyalgia",
    "osteoarthritis",
    "temporomandibular",
    "complex regional pain",
    "neuropathic pain",
    "nociplastic pain",
    "pain chronification",
    "orofacial pain",
    "headache",
)


def _blob(row: pd.Series) -> str:
    return f'{row.get("title", "")}\n{row.get("abstract", "")}'.lower()


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _operational_window() -> tuple[date, date]:
    protocol = load_yaml(project_path("config", "protocol.yaml"))
    window = protocol["search"].get(
        "operational_date_window",
        protocol["search"].get("date_window", protocol["search"]["registered_date_window"]),
    )
    start_date = datetime.fromisoformat(window["start"]).date()
    end_date = datetime.fromisoformat(window["end"]).date()
    return start_date, end_date


def _parse_record_date(publication_date: str, year_value: str) -> date | None:
    text = (publication_date or "").strip()
    normalized = text.replace("/", "-")
    match_full = re.search(r"\b(19|20)\d{2}-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b", normalized)
    if match_full:
        year, month, day = match_full.group(0).split("-")
        try:
            return date(int(year), int(month), int(day))
        except ValueError:
            pass

    match_month = re.search(r"\b(19|20)\d{2}-(0?[1-9]|1[0-2])\b", normalized)
    if match_month:
        year, month = match_month.group(0).split("-")
        try:
            return date(int(year), int(month), 1)
        except ValueError:
            pass

    year_text = str(year_value or "").strip()
    if year_text:
        try:
            year = int(float(year_text))
            return date(year, 1, 1)
        except ValueError:
            return None
    match_year = re.search(r"(19|20)\d{2}", normalized)
    if match_year:
        return date(int(match_year.group(0)), 1, 1)
    return None


def _decision(row: pd.Series, start_date: date, end_date: date) -> tuple[str, str, str]:
    text = _blob(row)
    publication_types = str(row.get("publication_types", "")).lower()
    language = str(row.get("language", "")).lower()
    record_date = _parse_record_date(str(row.get("publication_date", "")), str(row.get("year", "")))

    if not _has_any(text, BPS_TERMS):
        return "exclude", "no biopsychosocial term in title/abstract", "high"
    if record_date and record_date > end_date:
        return "exclude", "outside operational search window", "high"
    if record_date and record_date < start_date:
        return "exclude", "outside operational search window", "high"
    if "protocol" in text or "protocol" in publication_types:
        return "exclude", "protocol", "high"
    if _has_any(publication_types, COMMENTARY_TERMS) or _has_any(text, COMMENTARY_TERMS):
        return "exclude", "commentary/editorial/letter", "medium"
    if _has_any(text, ANIMAL_TERMS) and "human" not in text:
        return "exclude", "animal/non-human focus", "medium"
    if _has_any(text, PEDIATRIC_TERMS) and not _has_any(text, ADULT_TERMS):
        return "exclude", "pediatric-only focus", "medium"
    if "acute pain" in text and "chronic pain" not in text and "persistent pain" not in text:
        return "exclude", "acute pain focus", "medium"
    if not _has_any(text, CHRONIC_PAIN_TERMS):
        return "exclude", "chronic pain focus unclear", "medium"
    if language and "eng" not in language and "english" not in language:
        return "exclude", "non-English record", "medium"
    if "review" not in publication_types and "review" not in text and "meta-analysis" not in text and "scoping" not in text:
        return "unclear", "review status unclear", "low"
    return "include", "", "medium"


def stage1_screen() -> pd.DataFrame:
    search_path = project_path("review_stages", "02_search", "outputs", "deduplicated_records.csv")
    frame = pd.read_csv(search_path).fillna("")
    start_date, end_date = _operational_window()
    decisions = frame.apply(lambda row: _decision(row, start_date, end_date), axis=1, result_type="expand")
    decisions.columns = ["stage1_decision", "stage1_reason", "stage1_confidence"]
    out = pd.concat([frame, decisions], axis=1)
    out["stage1_screened_by"] = "codex_machine_assist"
    out["stage1_screening_mode"] = "rule_based_provisional"
    output_path = project_path("review_stages", "03_screening", "outputs", "stage1_screening.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    summary = (
        out.groupby(["stage1_decision", "stage1_reason"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["stage1_decision", "n"], ascending=[True, False])
    )
    summary.to_csv(project_path("review_stages", "03_screening", "audit", "stage1_screening_summary.csv"), index=False)
    return out
