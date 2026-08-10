from __future__ import annotations

"""Generate the operational Stage 3 codebook from the schema itself.

The codebook that the review team, the OSF deposit, and the human coding form
all refer to is not maintained by hand. It is rendered from the same schema,
prompt, and configuration the pipeline runs on, so a field cannot exist in the
coder's instructions and be missing from the codebook, or carry different
allowed values in the two places.

Three levels of row are written:

* ``record``  one row per coded field, with its instruction and its allowed values;
* ``item``    one row per field inside a structured extraction item;
* ``derived`` one row per field the pipeline computes rather than asks for.
"""

import csv

from bps_review.fulltext.coding import schema as S
from bps_review.fulltext.coding.prompt import (
    CONTROLLED_LIST_VALUES,
    CONTROLLED_VALUES,
    FIELD_INSTRUCTIONS,
    ITEM_VALUE_LISTS,
)
from bps_review.fulltext.config import ITEM_CAPS, MAX_QUOTE_WORDS, OPEN_LIST_CAPS
from bps_review.utils.paths import project_path


CODEBOOK_COLUMNS = ["field", "stage", "level", "description", "allowed_values"]

# Fields the pipeline computes from the coded content and never asks for.
DERIVED_ROWS: list[tuple[str, str, str]] = [
    ("fulltext_eligibility",
     "Derived post-retrieval verdict. A recommendation for human adjudication, not a decision.",
     "include; uncertain; exclude"),
    ("fulltext_exclusion_reason",
     "The rule that produced an exclude or uncertain verdict, in words. Empty for include.",
     "free_text"),
    ("conceptual_yield",
     "Derived measure of how much conceptual material the paper actually yielded.",
     "high; moderate; low; minimal"),
    ("synthesis_priority",
     "Derived reading order for the later synthesis.",
     "core; supporting; background; not_relevant"),
    ("derived_typology",
     "The typology recomputed from coverage and integration by a fixed rule, for comparison "
     "with the coded bps_typology.",
     "true_integrative; multifactorial; pseudo_bps; rhetorical_bps; narrow_despite_label"),
    ("integration_index",
     "Derived index between 0 and 1 combining the three pairwise ladders and the triadic ladder.",
     "numeric"),
    ("n_subdomains_named",
     "How many distinct ontology subdomains the coding touched, across the three domains.",
     "numeric"),
    ("n_named_integration_edges",
     "How many integration claims name both the source and the target factor.",
     "numeric"),
    ("controlled_label_share",
     "Share of extracted items whose ontology anchor lands on the controlled spine. A "
     "measurement of the ontology against the literature, not of the coder.",
     "numeric"),
    ("presence flags",
     "One yes or no per conceptual element, read off the coded content rather than asked.",
     "yes; no"),
    ("adjudication_status",
     "Human adjudication state of this coded row. Eligibility decisions remain human.",
     "pending; agreed; adjudicated"),
]


def _shorten(text: str, limit: int = 240) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _record_allowed_values(name: str) -> str:
    if name in CONTROLLED_VALUES:
        return "; ".join(CONTROLLED_VALUES[name])
    if name in CONTROLLED_LIST_VALUES:
        return (f"multi-label list, max {OPEN_LIST_CAPS.get(name, 6)} entries from: "
                + "; ".join(CONTROLLED_LIST_VALUES[name]))
    if name in S.ITEM_MODELS:
        return f"extraction list, max {ITEM_CAPS[name]} items"
    if name in S.OPEN_LIST_FIELDS:
        return f"open list, max {OPEN_LIST_CAPS.get(name, 6)} entries"
    return "free_text"


def _item_allowed_values(list_name: str, field_name: str, annotation) -> str:
    values = ITEM_VALUE_LISTS.get(list_name, {}).get(field_name)
    if values:
        return "; ".join(values)
    if field_name.endswith("verbatim"):
        return f"verbatim quote, max {MAX_QUOTE_WORDS} words"
    if annotation == list[str]:
        return "list of short free-text labels"
    return "free_text"


def codebook_rows() -> list[dict[str, str]]:
    """Every codebook row, in the order the coder meets the fields."""
    rows: list[dict[str, str]] = [
        {"field": "record_id", "stage": "stage3", "level": "record",
         "description": "Internal unique record identifier.", "allowed_values": "free_text"},
        {"field": "full_text_available", "stage": "stage3", "level": "record",
         "description": "Whether the full text was available for coding.",
         "allowed_values": "yes; no; partial"},
    ]
    for name, instruction in FIELD_INSTRUCTIONS:
        rows.append({"field": name, "stage": "stage3", "level": "record",
                     "description": _shorten(instruction),
                     "allowed_values": _record_allowed_values(name)})
        model = S.ITEM_MODELS.get(name)
        if model is None:
            continue
        for sub_name, sub_field in model.model_fields.items():
            rows.append({
                "field": f"{name}.{sub_name}",
                "stage": "stage3",
                "level": "item",
                "description": f"Field inside each {name} item.",
                "allowed_values": _item_allowed_values(name, sub_name, sub_field.annotation),
            })
    for name, description, allowed in DERIVED_ROWS:
        rows.append({"field": name, "stage": "stage3", "level": "derived",
                     "description": description, "allowed_values": allowed})
    return rows


def codebook_path():
    return project_path("review_stages", "04_extraction", "codebooks", "stage3_codebook.csv")


def write_stage3_codebook() -> int:
    """Render the codebook CSV. Returns the number of rows written."""
    rows = codebook_rows()
    path = codebook_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CODEBOOK_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
