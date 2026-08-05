from __future__ import annotations

"""Fit one full text into the coding window without losing conceptual content.

Most open-access reviews fit the models' context windows whole, and when they do
they are sent whole. The reducer below only engages for the long tail (large
umbrella reviews, papers with heavy method and result sections), and then it
drops the least conceptual paragraphs first rather than truncating the tail of
the paper.

The rule is deliberately conservative and auditable:

* the title, the abstract, and every section heading are always kept;
* every paragraph is scored on how much biopsychosocial content it carries
  (integration cues weigh most, then domain cues, then concept, framework, and
  definition cues), with a bonus when several domains appear together, because a
  paragraph that names two domains in one place is where integration lives;
* the opening paragraphs of the introduction and the discussion are protected,
  because that is where the framing and the integrative claims concentrate;
* paragraphs are then kept in descending score order until the budget is spent
  and re-assembled in document order, with an explicit ``[... omitted ...]``
  marker wherever something was dropped, so the model can see that the text is
  partial and the coder can audit what was sent.
"""

import re

from bps_review.fulltext.config import CODING_TEXT_CHAR_BUDGET


BPS_CUES = (
    "biopsychosocial", "bio-psycho-social", "bio psycho social", "psychosocial",
    "multidimensional", "multifactorial",
)
BIO_CUES = (
    "nocicept", "sensitization", "inflammat", "neurophysiolog", "pathophysiolog",
    "biomechanic", "imaging", "genetic", "tissue", "muscle", "pharmacolog", "opioid",
    "neural", "brain", "spinal", "physiolog",
)
PSYCH_CUES = (
    "catastroph", "fear-avoidance", "fear avoidance", "kinesiophobia", "self-efficacy",
    "depress", "anxiety", "coping", "belief", "cognitive", "acceptance", "mindfulness",
    "emotion", "distress", "expectation", "behaviour", "behavior", "psycholog",
)
SOCIAL_CUES = (
    "social support", "socioeconomic", "work", "occupational", "employment", "family",
    "spouse", "cultur", "stigma", "healthcare system", "policy", "deprivation",
    "interpersonal", "social",
)
INTEGRATION_CUES = (
    "interact", "mediat", "moderat", "pathway", "mechanism", "bidirectional", "reciprocal",
    "vicious circle", "vicious cycle", "contribut to", "leads to", "influenc", "predict",
    "underl", "explains", "drives", "amplif", "interplay", "interrelat", "integrat",
)
DEFINITION_CUES = (
    "define", "defined as", "definition", "conceptuali", "operationali", "refers to",
    "construct", "framework", "model of", "terminology",
)
FRAMEWORK_CUES = (
    "engel", "fear-avoidance model", "cognitive behavioral", "cognitive behavioural",
    "gate control", "diathesis", "self-regulation", "common sense model", "operant",
    "theory", "theoretical",
)

CUE_WEIGHTS = (
    (INTEGRATION_CUES, 3.0),
    (DEFINITION_CUES, 2.0),
    (FRAMEWORK_CUES, 1.5),
    (BPS_CUES, 1.5),
)

DOMAIN_GROUPS = (BIO_CUES, PSYCH_CUES, SOCIAL_CUES)

# Sections whose opening paragraphs are protected from the reducer.
PROTECTED_SECTION_HEADS = ("introduction", "background", "discussion", "conclusion", "general discussion")

OMISSION_MARKER = "[... omitted: paragraph with no biopsychosocial conceptual content ...]"


def _count_cues(text: str, cues: tuple[str, ...]) -> int:
    return sum(1 for cue in cues if cue in text)


def paragraph_score(text: str) -> float:
    """How much biopsychosocial content a paragraph carries. Higher is kept first."""
    lowered = text.lower()
    domain_hits = [_count_cues(lowered, group) for group in DOMAIN_GROUPS]
    domains_present = sum(1 for hits in domain_hits if hits)

    score = 2.0 * sum(domain_hits)
    for cues, weight in CUE_WEIGHTS:
        score += weight * _count_cues(lowered, cues)
    # A paragraph where two or three domains meet is where integration lives.
    if domains_present >= 2:
        score *= 1.0 + 0.5 * (domains_present - 1)
    # A paragraph that names no domain at all carries much less, even when it is
    # full of generic conceptual language.
    if domains_present == 0:
        score *= 0.35
    # Very short fragments (stray headings, single sentences) carry little.
    if len(text) < 160:
        score *= 0.6
    return score


def _split_paragraphs(section_text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n{2,}|\n(?=[A-Z])", section_text) if part.strip()]
    return parts or ([section_text.strip()] if section_text.strip() else [])


def build_coding_text(paper: dict, budget: int = CODING_TEXT_CHAR_BUDGET) -> tuple[str, dict]:
    """Return (coding_text, stats) for one paper.

    ``paper`` needs ``title``, ``abstract``, and ``sections`` (a list of
    ``{"title": ..., "text": ...}``). ``stats`` records what was sent, so the
    notebook can report how many papers were reduced and by how much.
    """
    title = str(paper.get("title", "")).strip()
    abstract = str(paper.get("abstract", "")).strip()
    sections = paper.get("sections") or []

    header = f"TITLE: {title}\n\nABSTRACT:\n{abstract}\n"
    if paper.get("journal"):
        header = f"JOURNAL: {paper['journal']} ({paper.get('year', '')})\n" + header

    units: list[dict] = []
    for section_index, section in enumerate(sections):
        section_title = str(section.get("title", "")).strip() or "Section"
        head_key = section_title.lower()
        protected_section = any(head_key.startswith(name) for name in PROTECTED_SECTION_HEADS)
        for paragraph_index, paragraph in enumerate(_split_paragraphs(str(section.get("text", "")))):
            units.append(
                {
                    "order": len(units),
                    "section_index": section_index,
                    "section_title": section_title,
                    "text": paragraph,
                    "score": paragraph_score(paragraph),
                    "protected": protected_section and paragraph_index < 2,
                }
            )

    body_chars = sum(len(unit["text"]) for unit in units)
    available = max(0, budget - len(header))

    if body_chars <= available:
        kept = {unit["order"] for unit in units}
        reduced = False
    else:
        reduced = True
        kept = set()
        used = 0
        for unit in units:
            if unit["protected"]:
                kept.add(unit["order"])
                used += len(unit["text"])
        ranked = sorted(
            (unit for unit in units if unit["order"] not in kept),
            key=lambda unit: (-unit["score"], unit["order"]),
        )
        for unit in ranked:
            cost = len(unit["text"]) + 2
            if used + cost > available:
                continue
            kept.add(unit["order"])
            used += cost

    lines: list[str] = [header, "FULL TEXT:"]
    current_section = None
    previous_kept = True
    for unit in units:
        if unit["section_index"] != current_section:
            current_section = unit["section_index"]
            lines.append(f"\n## {unit['section_title']}")
            previous_kept = True
        if unit["order"] in kept:
            lines.append(unit["text"])
            previous_kept = True
        else:
            if previous_kept:
                lines.append(OMISSION_MARKER)
            previous_kept = False

    coding_text = "\n\n".join(line for line in lines if line)
    stats = {
        "n_paragraphs_total": len(units),
        "n_paragraphs_kept": len(kept),
        "body_chars_total": body_chars,
        "coding_text_chars": len(coding_text),
        "reduced": reduced,
        "kept_share": round(len(kept) / len(units), 4) if units else 1.0,
    }
    return coding_text, stats
