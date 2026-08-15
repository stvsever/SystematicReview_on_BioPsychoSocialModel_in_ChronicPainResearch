from __future__ import annotations

"""Reading a comparable vocabulary out of one extraction list.

Both overlap metrics, the lexical one in ``reliability.py`` and the semantic one
in ``semantic.py``, need the same thing first: the set of labels one coder wrote
in one paper, for one comparison space. That extraction is defined once here, so
the two metrics can never end up comparing slightly different label sets and
producing two numbers that look comparable and are not.

A space is declared in ``config.ExtractionSpace`` and answers one question. It
names the extraction list to read, whether the labels come from item keys or from
a sublist inside the items, an optional filter restricting which items count, and
the project vocabulary the labels are normalized against.

Normalization is deliberately shallow: case, whitespace, and punctuation are
flattened, and a label is mapped onto a project vocabulary only where that
vocabulary clearly carries it. A term the vocabularies do not know survives as
the coder wrote it, because a label the ontology cannot hold is a finding about
the ontology.
"""

import json
import re
from typing import Any

from bps_review.fulltext.coding.vocabulary import normalize_label
from bps_review.fulltext.config import (
    LIST_LABEL_KEY,
    LIST_LABEL_VOCAB,
    ExtractionSpace,
)


def normalize_text(value: object) -> str:
    """Case, whitespace, and punctuation flattened. Nothing else is touched."""
    cleaned = " ".join(str(value or "").strip().lower().split())
    return re.sub(r"[^a-z0-9 \-]+", "", cleaned)


def parse_items(value: Any) -> list[dict]:
    """The items of one extraction-list cell, however the cell is stored."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def _passes_filter(item: dict, space: ExtractionSpace) -> bool:
    if not space.filter_key:
        return True
    return str(item.get(space.filter_key, "") or "").strip() in space.filter_values


def _normalized(raw: object, vocabulary: str) -> str:
    mapped = normalize_label(raw, vocabulary) if vocabulary else raw
    return normalize_text(mapped)


def space_labels(value: Any, space: ExtractionSpace) -> set[str]:
    """The normalized label set one coder wrote for one paper, in one space."""
    labels: set[str] = set()
    for item in parse_items(value):
        if not _passes_filter(item, space):
            continue
        if space.sublist:
            entries = item.get(space.sublist, [])
            if isinstance(entries, str):
                entries = [entries]
            for entry in entries or []:
                label = _normalized(entry, space.vocabulary)
                if label:
                    labels.add(label)
            continue
        parts = []
        for index, key in enumerate(space.keys):
            raw = item.get(key, "")
            # The vocabulary maps the identifying label; the rest of a joined key
            # is a relation or a domain pair, which is already a closed value.
            part = _normalized(raw, space.vocabulary if index == 0 else "")
            if part:
                parts.append(part)
        if parts:
            labels.add(" | ".join(parts))
    return labels


def labels_of(value: Any, field: str) -> set[str]:
    """The identity labels of one extraction list, as the lexical metrics read them."""
    return space_labels(
        value,
        ExtractionSpace(
            name=field,
            label=field,
            question="",
            field=field,
            keys=LIST_LABEL_KEY.get(field, ()),
            vocabulary=LIST_LABEL_VOCAB.get(field, ""),
        ),
    )
