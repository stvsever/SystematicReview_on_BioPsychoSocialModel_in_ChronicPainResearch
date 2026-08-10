from __future__ import annotations

"""Repair, normalization, and the deterministic derivations of scheme 3.

Three things happen here, in this order.

1. **Repair.** A model response is merged onto the schema field by field: every
   controlled value is snapped to the closest legal value, every free-text field
   is trimmed, every structured item is rebuilt and dropped when it carries
   neither a label nor a quote. Repair fixes a malformed field. It never
   fabricates a coding: a response that is not a coding of this paper is rejected
   by ``assert_usable_payload`` and retried instead.

   Repair is deliberately conservative about labels. Only the two fields whose
   purpose is to point at the project ontology (``subdomain_label`` and
   ``concept_family``) are mapped onto a canonical label, and each of them sits
   next to a free-text field holding the review's own wording. Every other label
   is stored exactly as the review wrote it. The mapped form is added alongside,
   in the item table, so nothing the coder wrote is ever overwritten.

2. **Derivation.** Eligibility, integration depth, conceptual yield, ontology
   breadth, and synthesis priority are computed from the coded content by a fixed
   rule. They are never asked of the coder, so the filter is auditable, identical
   across providers, and recomputable from a cached run.

3. **Serialization.** One flat row per (paper, model) for the wide table, one row
   per extracted item for the item-level table, and the inverse operation, so the
   derived columns can be recomputed from disk without another API call.

The typology is the one judgement that is both coded and derived. The coder codes
``bps_typology`` as the dossier specifies, and the pipeline independently derives
``derived_typology`` from coverage and integration by rule. Comparing the two is a
direct test of whether the typology definition is operationally tight enough to be
applied the same way twice.
"""

import json

from bps_review.fulltext.coding import schema as S
from bps_review.fulltext.coding import vocabulary as V
from bps_review.fulltext.coding.prompt import (
    CONTROLLED_LIST_VALUES,
    CONTROLLED_VALUES,
    ITEM_VALUE_LISTS,
)
from bps_review.fulltext.config import (
    COVERAGE_DEPTH,
    ITEM_CAPS,
    ITEM_SUBLIST_CAP,
    MAX_NOTE_WORDS,
    MAX_QUOTE_WORDS,
    MAX_SUMMARY_WORDS,
    OPEN_LIST_CAPS,
    PAIRWISE_DEPTH,
    TRIADIC_DEPTH,
)


# --------------------------------------------------------------------------
# Normalization helpers.
# --------------------------------------------------------------------------
def clean_label(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_choice(value: object, allowed: list[str], fallback: str) -> str:
    """Snap a raw value onto the closest legal value of a controlled field."""
    text = clean_label(value)
    if not text:
        return fallback
    lookup = {option.lower(): option for option in allowed}
    lookup.update({option.lower().replace("_", " "): option for option in allowed})
    lookup.update({option.lower().replace("-", " "): option for option in allowed})
    if text in lookup:
        return lookup[text]
    spaced = text.replace("_", " ").replace("-", " ")
    if spaced in lookup:
        return lookup[spaced]
    # A model that answers "bio-psych" for "bio_psych", or "psycho-social" for
    # "psych_social", is answering correctly in a different dialect.
    aliases = {
        "bio psych": "bio_psych",
        "biopsychological": "bio_psych",
        "psycho social": "psych_social",
        "psychosocial": "psych_social",
        "bio social": "bio_social",
        "biosocial": "bio_social",
        "three domain": "triadic",
        "true integrative": "true_integrative",
        "pseudo bps": "pseudo_bps",
        "rhetorical bps": "rhetorical_bps",
        "narrow despite label": "narrow_despite_label",
        "psych dominant": "psych-dominant",
        "bio dominant": "bio-dominant",
        "social dominant": "social-dominant",
        "formally defined": "formally_defined",
        "operationalized only": "operationalized_only",
        "described informally": "described_informally",
        "named only": "named_only",
        "musculo skeletal": "musculoskeletal",
        "mixed": "mixed_or_other",
        "explicit bps term": "explicit_bps_term",
        "variant term only": "variant_term_only",
        "domain language only": "domain_language_only",
        "cited only": "cited_only",
        "bidirectional": "bidirectional or reciprocal",
        "reciprocal": "bidirectional or reciprocal",
        "not reported": "not reported",
        "musculoskeletal": "musculoskeletal",
    }
    snapped = aliases.get(spaced)
    if snapped and snapped in allowed:
        return snapped
    # A prefix match catches "mechanistic (via central sensitization)".
    for option in allowed:
        if spaced.startswith(option.lower().replace("_", " ")):
            return option
    return fallback


def _trim_words(text: object, max_words: int) -> str:
    words = " ".join(str(text or "").split())
    parts = words.split(" ")
    if len(parts) <= max_words:
        return words
    return " ".join(parts[:max_words])


def _string_list(values: object, cap: int, keep_case: bool = False) -> list[str]:
    """A de-duplicated list of short free-text labels.

    ``keep_case`` preserves the review's own capitalization, which matters for
    the open lists whose whole point is that they carry the paper's wording.

    Semicolons inside an entry are turned into commas. The wide table joins these
    lists with a semicolon, so an entry that contains one would come back as two
    entries when the row is read from disk, and these lists are exactly the
    fields where a free-text observation can run to a full sentence.
    """
    out: list[str] = []
    if isinstance(values, str):
        values = [part for part in values.split(";") if part.strip()]
    if not isinstance(values, (list, tuple)):
        return out
    for value in values:
        if isinstance(value, dict):
            value = value.get("label") or value.get("value") or ""
        label = " ".join(str(value or "").strip().split()) if keep_case else clean_label(value)
        label = label.replace(";", ",").strip(" .,:")
        if label and label not in out:
            out.append(label)
        if len(out) >= cap:
            break
    return out


def _controlled_string_list(values: object, allowed: list[str], cap: int) -> list[str]:
    """A list whose entries must each land on a closed vocabulary."""
    out: list[str] = []
    raw = values if isinstance(values, (list, tuple)) else [values]
    for value in raw:
        snapped = normalize_choice(value, allowed, "")
        if snapped and snapped not in out:
            out.append(snapped)
        if len(out) >= cap:
            break
    return out


def _repair_item(name: str, raw: dict) -> dict | None:
    model = S.ITEM_MODELS[name]
    value_lists = ITEM_VALUE_LISTS.get(name, {})
    label_vocabs = S.ITEM_LABEL_VOCAB.get(name, {})
    defaults = {field: info.default for field, info in model.model_fields.items()}
    item: dict[str, object] = {}
    for field_name, info in model.model_fields.items():
        raw_value = raw.get(field_name)
        if field_name in value_lists:
            item[field_name] = normalize_choice(raw_value, value_lists[field_name], str(defaults[field_name]))
        elif field_name in label_vocabs:
            # A spine pointer: mapped when it clearly matches, kept as written otherwise.
            item[field_name] = V.normalize_label(raw_value, label_vocabs[field_name])
        elif info.annotation == list[str]:
            item[field_name] = _string_list(raw_value, ITEM_SUBLIST_CAP, keep_case=True)
        elif field_name.endswith("verbatim"):
            item[field_name] = _trim_words(raw_value, MAX_QUOTE_WORDS + 10)
        else:
            item[field_name] = _trim_words(raw_value, MAX_NOTE_WORDS + 10)

    # The subdomains a domain-evidence item names belong to that item's domain.
    if name == "domain_evidence":
        kind = V.subdomain_kind_for_domain(str(item.get("domain", "")))
        if kind:
            item["subdomains_named"] = V.normalize_labels(item.get("subdomains_named"), kind,
                                                          cap=ITEM_SUBLIST_CAP)

    has_label = any(str(item.get(key, "")).strip() for key in S.ITEM_LABEL_KEY.get(name, ()))
    quote_key = S.ITEM_QUOTE_KEY.get(name, "")
    has_quote = bool(str(item.get(quote_key, "")).strip()) if quote_key else False
    if not has_label and not has_quote:
        return None
    return item


def repair_payload(record: dict, raw_payload: object) -> dict:
    """Merge a raw model response onto the schema, field by field."""
    raw = raw_payload if isinstance(raw_payload, dict) else {}
    if "records" in raw and isinstance(raw.get("records"), list) and raw["records"]:
        first = raw["records"][0]
        if isinstance(first, dict):
            raw = first

    defaults = {name: info.default for name, info in S.FullTextCodingRecord.model_fields.items()}
    out: dict[str, object] = {"record_id": record["record_id"]}

    for field_name, allowed in CONTROLLED_VALUES.items():
        out[field_name] = normalize_choice(raw.get(field_name), allowed, str(defaults[field_name]))

    for name in S.ITEM_MODELS:
        raw_items = raw.get(name)
        items: list[dict] = []
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    continue
                repaired = _repair_item(name, raw_item)
                if repaired is not None:
                    items.append(repaired)
                if len(items) >= ITEM_CAPS[name]:
                    break
        out[name] = items

    for name, kind in S.OPEN_LIST_FIELDS.items():
        cap = OPEN_LIST_CAPS.get(name, 6)
        if name in CONTROLLED_LIST_VALUES:
            out[name] = _controlled_string_list(raw.get(name), CONTROLLED_LIST_VALUES[name], cap)
        elif kind:
            # A vocabulary applies, so the entry is mapped when it clearly
            # matches and kept as written when it does not.
            out[name] = V.normalize_labels(raw.get(name), kind, cap=cap)
        else:
            out[name] = _string_list(raw.get(name), cap, keep_case=True)

    out["pain_condition_detail"] = _trim_words(raw.get("pain_condition_detail"), MAX_NOTE_WORDS)
    out["context_note"] = _trim_words(raw.get("context_note"), MAX_NOTE_WORDS)
    out["bps_operationalization_summary"] = _trim_words(
        raw.get("bps_operationalization_summary"), MAX_SUMMARY_WORDS)
    out["integration_mechanism_summary"] = _trim_words(
        raw.get("integration_mechanism_summary"), MAX_SUMMARY_WORDS)
    out["synthesis_note"] = _trim_words(raw.get("synthesis_note"), MAX_SUMMARY_WORDS)
    out["coding_rationale"] = _trim_words(raw.get("coding_rationale"), MAX_NOTE_WORDS + 20)
    return out


def assert_usable_payload(record: dict, raw_payload: object) -> None:
    """Raise unless the response is a real coding of this paper.

    The repair layer exists to fix individual malformed fields, not to fabricate a
    coding. A response that is not an object, that carries the wrong record_id, or
    that contains none of the schema's substantive keys is a failed call and must
    be retried rather than repaired into plausible-looking output.
    """
    if not isinstance(raw_payload, dict):
        raise ValueError(f"Model returned {type(raw_payload).__name__}, not a JSON object")

    payload = raw_payload
    if "records" in payload and isinstance(payload.get("records"), list) and payload["records"]:
        if isinstance(payload["records"][0], dict):
            payload = payload["records"][0]

    returned_id = str(payload.get("record_id", "")).strip()
    if returned_id and returned_id != record["record_id"]:
        raise ValueError(f"record_id mismatch: expected {record['record_id']}, received {returned_id}")

    substantive = set(CONTROLLED_VALUES) | set(S.ITEM_MODELS)
    present = [key for key in substantive if key in payload]
    if len(present) < 5:
        raise ValueError(
            f"Model returned only {len(present)} recognizable schema keys "
            f"(keys seen: {sorted(payload.keys())[:8]})"
        )


# --------------------------------------------------------------------------
# Deterministic derivations.
# --------------------------------------------------------------------------
PAIRWISE_FIELDS = ("integration_bio_psych", "integration_psych_social", "integration_bio_social")
COVERAGE_FIELDS = ("domain_coverage_bio", "domain_coverage_psych", "domain_coverage_social")

# The relations that carry taxonomic structure rather than mere association.
HIERARCHICAL_RELATIONS = ("is_a_subtype_of", "part_of_or_component_of",
                          "synonym_or_used_interchangeably")


def _count(record: S.FullTextCodingRecord, name: str) -> int:
    return len(getattr(record, name, []) or [])


def coverage_profile(record: S.FullTextCodingRecord) -> dict[str, int]:
    """The three coverage levels as depth scores, plus how many domains are real."""
    depths = {field: COVERAGE_DEPTH[getattr(record, field)] for field in COVERAGE_FIELDS}
    return {
        **depths,
        "coverage_total": sum(depths.values()),
        "domains_present": sum(1 for value in depths.values() if value >= 2),
        "domains_any": sum(1 for value in depths.values() if value >= 1),
    }


def integration_profile(record: S.FullTextCodingRecord) -> dict[str, int]:
    """The four integration levels as depth scores."""
    pairwise = {field: PAIRWISE_DEPTH[getattr(record, field)] for field in PAIRWISE_FIELDS}
    triadic = TRIADIC_DEPTH[record.integration_triadic]
    return {
        **pairwise,
        "pairwise_depth_total": sum(pairwise.values()),
        "pairwise_depth_max": max(pairwise.values()) if pairwise else 0,
        "triadic_depth": triadic,
        # One number per paper, comparable across the corpus: the pairwise mean
        # normalized to 0-1 plus the triadic rung, also normalized, averaged.
        "integration_index": round(
            0.5 * (sum(pairwise.values()) / (3 * 4)) + 0.5 * (triadic / 3), 4
        ),
    }


def ontology_profile(record: S.FullTextCodingRecord) -> dict[str, object]:
    """How much ontology this coding actually yielded, and how much of it is on-spine.

    Three numbers matter here. How many distinct subdomains the review touched,
    which is the breadth of its account. How many integration claims name both
    ends of the link, which is how many usable edges the paper contributes. And
    what share of the extracted labels landed on the project vocabularies, which
    measures the ontology against the literature rather than the other way round:
    a low share says the vocabularies need extending, not that the coder erred.
    """
    subdomains: dict[str, set[str]] = {"biological": set(), "psychological": set(), "social": set()}
    for item in record.biological_factors or []:
        if item.subdomain_label:
            subdomains["biological"].add(item.subdomain_label)
    for item in record.social_factors or []:
        if item.subdomain_label:
            subdomains["social"].add(item.subdomain_label)
    for item in record.psychological_concepts or []:
        if item.concept_family:
            subdomains["psychological"].add(item.concept_family)
    for item in record.domain_evidence or []:
        for label in item.subdomains_named or []:
            if item.domain in subdomains:
                subdomains[item.domain].add(label)

    on_spine, total_labels = 0, 0
    for name, (anchor_field, kind) in S.ITEM_ANCHOR.items():
        for item in getattr(record, name, []) or []:
            total_labels += 1
            anchor = str(getattr(item, anchor_field, "") or "").strip()
            if anchor and V.is_controlled(V.normalize_label(anchor, kind), kind):
                on_spine += 1

    named_edges = sum(
        1 for item in (record.integration_claims or [])
        if str(item.source_factor_label or "").strip() and str(item.target_factor_label or "").strip()
    )
    return {
        "n_subdomains_bio": len(subdomains["biological"]),
        "n_subdomains_psych": len(subdomains["psychological"]),
        "n_subdomains_social": len(subdomains["social"]),
        "n_subdomains_named": sum(len(values) for values in subdomains.values()),
        "n_named_integration_edges": named_edges,
        "n_emergent_labels": len(record.emergent_labels or []),
        "n_labels_checked": total_labels,
        "controlled_label_share": round(on_spine / total_labels, 4) if total_labels else 0.0,
    }


def bps_usage_profile(record: S.FullTextCodingRecord) -> dict[str, object]:
    """What the biopsychosocial label does in this paper, read off the coded items."""
    functions = list(record.bps_functions_present or [])
    for item in record.bps_usage_instances or []:
        if item.bps_function and item.bps_function not in functions:
            functions.append(item.bps_function)
    substantive = {"explanatory framework", "operational definition", "intervention rationale",
                   "organizing principle"}
    sections = sorted({item.section_located for item in (record.bps_usage_instances or [])
                       if item.section_located and item.section_located != "unclear"})
    return {
        "bps_function_set": "; ".join(sorted(function for function in functions if function)),
        "n_bps_functions": len([function for function in functions if function]),
        "bps_has_substantive_function": "yes" if substantive & set(functions) else "no",
        "bps_usage_sections": "; ".join(sections),
        "n_bps_usage_sections": len(sections),
    }


def derived_typology(record: S.FullTextCodingRecord) -> str:
    """The typology recomputed from coverage and integration by a fixed rule.

    This is not a replacement for the coded ``bps_typology``. It exists so the
    coded judgement can be checked against the rule the codebook describes, which
    is the sharpest available test of whether the typology is defined tightly
    enough to be applied consistently.
    """
    coverage = coverage_profile(record)
    integration = integration_profile(record)
    all_present = coverage["domains_present"] == 3
    any_present = coverage["domains_any"]

    if all_present and integration["triadic_depth"] >= 2:
        return "true_integrative"
    if all_present:
        return "multifactorial"
    if any_present <= 1:
        return "narrow_despite_label"
    if coverage["coverage_total"] <= 3:
        return "rhetorical_bps"
    return "pseudo_bps"


def eligibility(record: S.FullTextCodingRecord) -> tuple[str, str]:
    """The post-retrieval full-text filter: (verdict, reason).

    The verdict is a recommendation for a human adjudicator, not a final
    decision. It protects recall: anything that plausibly carries biopsychosocial
    content is kept, and everything doubtful becomes 'uncertain' rather than
    'exclude'.
    """
    coverage = coverage_profile(record)
    if coverage["domains_any"] == 0:
        return "exclude", "no biopsychosocial domain content in the full text"
    if coverage["domains_any"] == 1 and _count(record, "integration_claims") == 0:
        return "exclude", "single-domain review with no cross-domain claim"
    if record.source_type == "primary study":
        return "uncertain", "reads as a primary study rather than an evidence synthesis"
    if coverage["domains_present"] < 2:
        return "uncertain", "fewer than two domains substantively covered"
    if record.bps_label_used == "absent" and record.bps_typology == "unclear":
        return "uncertain", "no biopsychosocial vocabulary and no readable typology"
    if record.bps_typology == "unclear" and record.integration_triadic == "none":
        return "uncertain", "typology not readable and no triadic integration found"
    return "include", ""


def conceptual_yield(record: S.FullTextCodingRecord) -> str:
    """How much conceptual material this paper actually yielded.

    A measure of harvest, not of promise: it counts what was extracted, weighted
    toward the integration evidence, the named factors, and the defined concepts,
    because those carry the review's research questions and the ontology.
    """
    integration = integration_profile(record)
    ontology = ontology_profile(record)
    n_claims = _count(record, "integration_claims")
    n_concepts = _count(record, "psychological_concepts")
    n_defined = sum(
        1 for item in (record.psychological_concepts or [])
        if item.definitional_status in ("formally_defined", "operationalized_only")
    )
    n_frameworks = _count(record, "theoretical_frameworks")
    n_factors = _count(record, "biological_factors") + _count(record, "social_factors")
    n_relations = _count(record, "concept_relations")
    coverage = coverage_profile(record)

    if coverage["domains_present"] == 3 and integration["triadic_depth"] >= 2 and n_claims >= 3:
        return "high"
    if n_claims >= 4 and (n_defined >= 2 or n_frameworks >= 2):
        return "high"
    if ontology["n_subdomains_named"] >= 8 and n_claims >= 2:
        return "high"
    if n_claims >= 2 or n_relations >= 3 or (n_concepts >= 3 and coverage["domains_present"] >= 2):
        return "moderate"
    if n_claims >= 1 or n_concepts >= 1 or n_frameworks >= 1 or n_factors >= 2:
        return "low"
    return "minimal"


def synthesis_priority(record: S.FullTextCodingRecord, verdict: str, yield_level: str) -> str:
    """Reading order for the later synthesis."""
    if verdict == "exclude":
        return "not_relevant"
    if verdict == "include" and yield_level == "high":
        return "core"
    if yield_level in ("high", "moderate"):
        return "supporting"
    return "background"


def presence_flags(record: S.FullTextCodingRecord) -> dict[str, str]:
    """The binary presence of every conceptual element, as yes or no.

    Presence is read off the coded content, not asked of the coder. For the
    elements that have an extraction list, present means the coder actually
    returned at least one item, so a 'no' is the observable statement that this
    coder found nothing of that kind in this paper.

    These flags are what part of the cross-provider agreement is computed on.
    Whether two coders both found a theoretical framework in a paper is a
    question with one answer; whether they wrote the same label for it is a
    different question, answered by the set-overlap metrics.
    """
    domain_levels = {item.domain for item in (record.domain_evidence or []) if item.evidence_verbatim}
    n_defined = sum(
        1 for item in (record.psychological_concepts or [])
        if item.definitional_status in ("formally_defined", "operationalized_only")
    )
    named_edges = sum(
        1 for item in (record.integration_claims or [])
        if str(item.source_factor_label or "").strip() and str(item.target_factor_label or "").strip()
    )
    hierarchical = any(
        item.relation_type in HIERARCHICAL_RELATIONS for item in (record.concept_relations or [])
    )
    return {
        "present_bps_usage_evidence": "yes" if _count(record, "bps_usage_instances") else "no",
        "present_bps_definition": "yes" if _count(record, "bps_definitions") else "no",
        "present_integration_evidence": "yes" if _count(record, "integration_claims") else "no",
        "present_triadic_claim": "yes" if any(
            item.domains_linked == "triadic" for item in (record.integration_claims or [])
        ) else "no",
        "present_named_integration_edge": "yes" if named_edges else "no",
        "present_biological_factors": "yes" if _count(record, "biological_factors") else "no",
        "present_social_factors": "yes" if _count(record, "social_factors") else "no",
        "present_other_domain_factors": "yes" if _count(record, "other_domain_factors") else "no",
        "present_psychological_concepts": "yes" if _count(record, "psychological_concepts") else "no",
        "present_defined_concepts": "yes" if n_defined else "no",
        "present_concept_relations": "yes" if _count(record, "concept_relations") else "no",
        "present_hierarchical_relation": "yes" if hierarchical else "no",
        "present_theoretical_frameworks": "yes" if _count(record, "theoretical_frameworks") else "no",
        "present_instruments": "yes" if _count(record, "instruments") else "no",
        "present_conceptual_problems": "yes" if _count(record, "conceptual_problems") else "no",
        "present_domain_evidence_bio": "yes" if "biological" in domain_levels else "no",
        "present_domain_evidence_psych": "yes" if "psychological" in domain_levels else "no",
        "present_domain_evidence_social": "yes" if "social" in domain_levels else "no",
    }


def collect_quotes(record: S.FullTextCodingRecord) -> list[tuple[str, str]]:
    """Every verbatim quote in the record as (field, quote)."""
    quotes: list[tuple[str, str]] = []
    for name, quote_key in S.ITEM_QUOTE_KEY.items():
        for item in getattr(record, name, []) or []:
            text = str(getattr(item, quote_key, "") or "").strip()
            if text:
                quotes.append((name, text))
    return quotes


def derive(record: S.FullTextCodingRecord) -> dict[str, object]:
    """All derived fields for one coded record."""
    coverage = coverage_profile(record)
    integration = integration_profile(record)
    ontology = ontology_profile(record)
    usage = bps_usage_profile(record)
    verdict, reason = eligibility(record)
    yield_level = conceptual_yield(record)
    quotes = collect_quotes(record)
    n_defined = sum(
        1 for item in (record.psychological_concepts or [])
        if item.definitional_status in ("formally_defined", "operationalized_only")
    )
    computed_typology = derived_typology(record)
    counts = {f"n_{name}": _count(record, name) for name in S.ITEM_MODELS}
    return {
        **presence_flags(record),
        **usage,
        **ontology,
        "coverage_depth_bio": coverage["domain_coverage_bio"],
        "coverage_depth_psych": coverage["domain_coverage_psych"],
        "coverage_depth_social": coverage["domain_coverage_social"],
        "coverage_total": coverage["coverage_total"],
        "domains_present": coverage["domains_present"],
        "pairwise_depth_total": integration["pairwise_depth_total"],
        "pairwise_depth_max": integration["pairwise_depth_max"],
        "triadic_depth": integration["triadic_depth"],
        "integration_index": integration["integration_index"],
        **counts,
        "n_triadic_claims": sum(
            1 for item in (record.integration_claims or []) if item.domains_linked == "triadic"
        ),
        "n_defined_concepts": n_defined,
        "n_hierarchical_relations": sum(
            1 for item in (record.concept_relations or [])
            if item.relation_type in HIERARCHICAL_RELATIONS
        ),
        "n_evidence_quotes": len(quotes),
        "n_extracted_items": sum(counts.values()),
        "n_open_list_entries": sum(
            len(getattr(record, name, []) or []) for name in S.OPEN_LIST_FIELDS
        ),
        "derived_typology": computed_typology,
        "typology_matches_derived": "yes" if computed_typology == record.bps_typology else "no",
        "conceptual_yield": yield_level,
        "fulltext_eligibility": verdict,
        "fulltext_exclusion_reason": reason,
        "synthesis_priority": synthesis_priority(record, verdict, yield_level),
    }


# --------------------------------------------------------------------------
# Serialization.
# --------------------------------------------------------------------------
def serialize_row(record: S.FullTextCodingRecord, model_id: str,
                  coding_method: str = "llm_structured") -> dict[str, object]:
    """One flat row for the wide (paper x model) table."""
    payload = record.model_dump()
    row: dict[str, object] = {"record_id": record.record_id}

    for field_name in S.FullTextCodingRecord.model_fields:
        if field_name == "record_id":
            continue
        value = payload[field_name]
        if field_name in S.ITEM_MODELS:
            row[field_name] = json.dumps(value, ensure_ascii=False)
        elif field_name in S.OPEN_LIST_FIELDS:
            # Readable in a spreadsheet and parsable back into a list.
            row[field_name] = "; ".join(value)
        else:
            row[field_name] = value

    row.update(derive(record))
    row["coding_method"] = coding_method
    row["llm_model"] = model_id
    return row


def record_from_row(row: dict) -> S.FullTextCodingRecord:
    """Rebuild a coded record from one row of the persisted wide table.

    The wide table stores the coded content losslessly: structured items as JSON,
    open lists as semicolon-joined text, everything else as the value itself.
    Rebuilding the record makes the derived fields recomputable from disk, without
    another API call.
    """
    payload: dict[str, object] = {"record_id": str(row.get("record_id", ""))}
    for field_name in S.FullTextCodingRecord.model_fields:
        if field_name == "record_id":
            continue
        raw = row.get(field_name, "")
        if field_name in S.ITEM_MODELS:
            try:
                parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
            except json.JSONDecodeError:
                parsed = []
            payload[field_name] = parsed if isinstance(parsed, list) else []
        elif field_name in S.OPEN_LIST_FIELDS:
            if isinstance(raw, list):
                payload[field_name] = raw
            else:
                payload[field_name] = [part.strip() for part in str(raw or "").split(";") if part.strip()]
        else:
            text = str(raw or "")
            if text:
                payload[field_name] = text
    return S.FullTextCodingRecord.model_validate(payload)


def rederive_frame(long_df):
    """Recompute every derived column of a cached wide table from its content.

    Derived fields are never trusted from disk. Reloading a cached run therefore
    re-applies the current derivation rules, so a change to the eligibility logic
    or to the presence flags takes effect without re-coding the corpus.
    """
    if long_df.empty:
        return long_df
    frame = long_df.copy()
    derived = [derive(record_from_row(row)) for row in frame.to_dict(orient="records")]
    for column in derived[0]:
        frame[column] = [entry[column] for entry in derived]
    return frame


def item_rows(record: S.FullTextCodingRecord, model_id: str, model_label: str) -> list[dict[str, object]]:
    """One row per extracted item, for the item-level table and the quote check.

    Every row carries the item at both resolutions, and never trades one for the
    other. ``label_raw`` is exactly what the coder wrote, ``label_normalized`` is
    that same label mapped onto its own vocabulary when one applies, and
    ``anchor_label`` is where the item attaches to the project ontology, with
    ``anchor_controlled`` saying whether that attachment landed on the canonical
    list. An item with no anchor is not a defect: it is a term the ontology does
    not yet carry, and the synthesis needs to see it as such.
    """
    rows: list[dict[str, object]] = []
    for name in S.ITEM_MODELS:
        keys = S.ITEM_LABEL_KEY.get(name, ())
        quote_key = S.ITEM_QUOTE_KEY.get(name, "")
        identity_kind = S.ITEM_IDENTITY_VOCAB.get(name, "")
        anchor_field, anchor_kind = S.ITEM_ANCHOR.get(name, ("", ""))
        for index, item in enumerate(getattr(record, name, []) or []):
            payload = item.model_dump()
            parts = [str(payload.get(key, "") or "").strip() for key in keys]
            label = " | ".join(part for part in parts if part)
            head = parts[0] if parts else ""
            normalized = V.normalize_label(head, identity_kind) if identity_kind and head \
                else clean_label(head)
            anchor = str(payload.get(anchor_field, "") or "").strip() if anchor_field else ""
            anchor = V.normalize_label(anchor, anchor_kind) if anchor and anchor_kind else anchor
            rows.append(
                {
                    "record_id": record.record_id,
                    "model_label": model_label,
                    "model_id": model_id,
                    "extraction_field": name,
                    "item_index": index,
                    "label_raw": label,
                    "label_normalized": normalized,
                    "label_vocabulary": identity_kind,
                    "label_controlled": "yes" if identity_kind and V.is_controlled(normalized, identity_kind) else "no",
                    "anchor_label": anchor,
                    "anchor_vocabulary": anchor_kind,
                    "anchor_controlled": "yes" if anchor and anchor_kind and V.is_controlled(anchor, anchor_kind) else "no",
                    "quote": str(payload.get(quote_key, "") or ""),
                    "item_json": json.dumps(payload, ensure_ascii=False),
                }
            )
    return rows
