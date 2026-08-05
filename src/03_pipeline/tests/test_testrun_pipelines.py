"""Offline tests for the two cross-provider test-run pipelines.

Nothing here calls an API. The tests cover the parts that decide whether the
run is trustworthy: the agreement primitives, the deterministic repair and
derivation layer of the full-text scheme, the quote verification, and the
consistency between the prompt, the schema, and the reliability configuration.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from bps_review.extraction.llm_stage2 import FIELD_SPECIFICATION, Stage2StructuredRecord, _batch_prompt
from bps_review.fulltext.analysis.integrity import integration_evidence_discipline, verify_quote
from bps_review.fulltext.analysis.reliability import adjacent_agreement, compute_list_overlap
from bps_review.fulltext.coding.condense import build_coding_text, paragraph_score
from bps_review.fulltext.coding.derive import (
    assert_usable_payload,
    derive,
    normalize_choice,
    record_from_row,
    repair_payload,
    serialize_row,
)
from bps_review.fulltext.coding.prompt import CONTROLLED_VALUES, build_prompt, build_schema_spec
from bps_review.fulltext.coding.schema import FullTextCodingRecord
from bps_review.fulltext.config import (
    LIST_FIELDS,
    PRESENCE_FIELDS,
    RELIABILITY_FIELDS,
)
from bps_review.pilot.analysis.metrics import (
    cohen_kappa,
    fleiss_kappa,
    krippendorff_alpha,
    percent_agreement,
    unanimous_rate,
)
from bps_review.pilot.config import LIST_FIELDS as ABSTRACT_LIST_FIELDS
from bps_review.pilot.config import RELIABILITY_FIELDS as ABSTRACT_RELIABILITY_FIELDS


# --------------------------------------------------------------------------
# Agreement primitives
# --------------------------------------------------------------------------
def test_perfect_agreement_gives_kappa_one():
    columns = [["a", "b", "a", "b"]] * 3
    assert percent_agreement(columns[0], columns[1]) == 1.0
    assert fleiss_kappa(columns) == pytest.approx(1.0)
    assert krippendorff_alpha(columns) == pytest.approx(1.0)
    assert unanimous_rate(columns) == 1.0


def test_chance_level_agreement_gives_kappa_near_zero():
    first = ["a", "b"] * 10
    second = ["a", "a", "b", "b"] * 5
    assert cohen_kappa(first, second) == pytest.approx(0.0, abs=0.2)


def test_adjacent_agreement_counts_one_rung_apart_as_close():
    order = ["mechanistic", "directional", "descriptive", "mentioned", "none"]
    columns = [["mechanistic", "none"], ["directional", "none"]]
    # first item is one rung apart, second identical: both count as adjacent
    assert adjacent_agreement(columns, order) == 1.0
    far = [["mechanistic"], ["none"]]
    assert adjacent_agreement(far, order) == 0.0


# --------------------------------------------------------------------------
# The abstract-level prompt must name every field the schema validates
# --------------------------------------------------------------------------
def test_abstract_prompt_names_every_schema_field():
    """A prompt that omits a field name lets the repair layer fill it silently."""
    schema_fields = set(Stage2StructuredRecord.model_fields)
    assert schema_fields == set(FIELD_SPECIFICATION), (
        "the prompt's field specification and the validated schema must not drift apart"
    )


def test_abstract_prompt_carries_the_record_and_the_contract():
    prompt = _batch_prompt([{"record_id": "A001_1", "title": "t", "abstract": "a"}])
    assert "A001_1" in prompt
    assert "bio_mentioned" in prompt and "social_mentioned" in prompt
    assert "output_contract" in prompt


# --------------------------------------------------------------------------
# Full-text repair and derivation
# --------------------------------------------------------------------------
def _raw_payload(record_id: str = "F001_1") -> dict:
    return {
        "record_id": record_id,
        "review_track": "musculoskeletal",
        "source_type": "systematic review",
        "domain_coverage_bio": "elaborated",
        "domain_coverage_psych": "elaborated",
        "domain_coverage_social": "mentioned",
        "integration_bio_psych": "mechanistic",
        "integration_psych_social": "descriptive",
        "integration_bio_social": "none",
        "integration_triadic": "descriptive",
        "overall_balance": "balanced",
        "bps_typology": "true_integrative",
        "concept_definitions_present": "partial",
        "integration_claims": [
            {"domains_linked": "bio_psych", "integration_level": "mechanistic",
             "claim_verbatim": "central sensitization is amplified by catastrophizing",
             "mechanism_note": "via descending modulation", "section_located": "discussion",
             "evidence_basis": "theorized"},
            {"domains_linked": "triadic", "integration_level": "descriptive",
             "claim_verbatim": "biological psychological and social factors jointly shape outcome",
             "section_located": "introduction"},
        ],
        "domain_evidence": [
            {"domain": "biological", "coverage_level": "elaborated",
             "constructs_named": ["central sensitization"], "evidence_verbatim": "a passage about nociception",
             "section_located": "introduction"},
        ],
        "psychological_concepts": [
            {"concept_label": "pain catastrophizing", "definitional_status": "formally_defined",
             "definition_verbatim": "catastrophizing is defined as an exaggerated negative orientation"},
            {"concept_label": "self-efficacy", "definitional_status": "named_only"},
        ],
        "theoretical_frameworks": [
            {"framework_label": "fear-avoidance model", "role": "organizing framework",
             "framework_verbatim": "we use the fear-avoidance model"},
        ],
        "conceptual_problems": [
            {"problem_type": "missing_social", "problem_verbatim": "social factors were not examined"},
        ],
        "key_quotes": [],
        "coding_rationale": "clear three-domain account",
    }


def test_repair_snaps_dialects_onto_the_controlled_values():
    assert normalize_choice("Bio-Psych", ["bio_psych", "psych_social"], "psych_social") == "bio_psych"
    assert normalize_choice("TRUE INTEGRATIVE", ["true_integrative", "unclear"], "unclear") == "true_integrative"
    assert normalize_choice("something else", ["a", "b"], "b") == "b"
    assert normalize_choice(None, ["a", "b"], "b") == "b"


def test_repair_and_validate_roundtrip():
    record = {"record_id": "F001_1"}
    repaired = repair_payload(record, _raw_payload())
    coded = FullTextCodingRecord.model_validate(repaired)
    assert coded.integration_bio_psych == "mechanistic"
    assert len(coded.integration_claims) == 2
    assert len(coded.psychological_concepts) == 2


def test_repair_drops_items_without_a_label_or_a_quote():
    payload = _raw_payload()
    payload["theoretical_frameworks"].append({"role": "unclear"})
    coded = FullTextCodingRecord.model_validate(repair_payload({"record_id": "F001_1"}, payload))
    assert len(coded.theoretical_frameworks) == 1


def test_unusable_payloads_are_rejected_rather_than_repaired():
    with pytest.raises(ValueError):
        assert_usable_payload({"record_id": "F001_1"}, ["not", "an", "object"])
    with pytest.raises(ValueError):
        assert_usable_payload({"record_id": "F001_1"}, {**_raw_payload(), "record_id": "F999_9"})
    with pytest.raises(ValueError):
        assert_usable_payload({"record_id": "F001_1"}, {"record_id": "F001_1", "note": "hello"})
    assert_usable_payload({"record_id": "F001_1"}, _raw_payload())


def test_derivations_are_computed_from_the_content():
    coded = FullTextCodingRecord.model_validate(repair_payload({"record_id": "F001_1"}, _raw_payload()))
    derived = derive(coded)
    assert derived["present_integration_evidence"] == "yes"
    assert derived["present_triadic_claim"] == "yes"
    assert derived["present_defined_concepts"] == "yes"
    assert derived["domains_present"] == 3
    assert derived["fulltext_eligibility"] == "include"
    assert derived["derived_typology"] == "true_integrative"
    assert derived["typology_matches_derived"] == "yes"
    # two integration claims, one domain-evidence passage, one concept definition,
    # one framework passage, and one conceptual-problem passage
    assert derived["n_evidence_quotes"] == 6
    assert 0.0 <= derived["integration_index"] <= 1.0


def test_empty_coding_is_excluded_and_never_fabricated():
    empty = FullTextCodingRecord(record_id="F002_2")
    derived = derive(empty)
    assert derived["fulltext_eligibility"] == "exclude"
    assert derived["conceptual_yield"] == "minimal"
    assert derived["synthesis_priority"] == "not_relevant"
    assert derived["n_extracted_items"] == 0


def test_row_roundtrip_preserves_the_coded_content():
    coded = FullTextCodingRecord.model_validate(repair_payload({"record_id": "F001_1"}, _raw_payload()))
    row = serialize_row(coded, model_id="test/model")
    restored = record_from_row(row)
    assert restored.integration_triadic == coded.integration_triadic
    assert len(restored.psychological_concepts) == len(coded.psychological_concepts)
    assert derive(restored)["integration_index"] == derive(coded)["integration_index"]


# --------------------------------------------------------------------------
# Quote verification and evidence discipline
# --------------------------------------------------------------------------
def test_quote_verification_separates_exact_near_and_invented():
    source = "central sensitization is amplified by catastrophizing in chronic low back pain patients"
    assert verify_quote("central sensitization is amplified by catastrophizing", source)[0] == "exact"
    assert verify_quote("Central sensitization is amplified, by catastrophizing!", source)[0] == "exact"
    assert verify_quote("the authors never wrote anything remotely like this sentence", source)[0] == "unverified"
    assert verify_quote("too short", source)[0] == "too_short_to_check"


def test_evidence_discipline_flags_a_graded_link_without_a_quote():
    frame = pd.DataFrame([
        {
            "record_id": "F001_1",
            "model_label": "M",
            "integration_bio_psych": "mechanistic",
            "integration_psych_social": "none",
            "integration_bio_social": "none",
            "integration_triadic": "none",
            "integration_claims": json.dumps([]),
        }
    ])
    discipline = integration_evidence_discipline(frame)
    assert len(discipline) == 1
    assert discipline.iloc[0]["has_quoted_claim"] == "no"


# --------------------------------------------------------------------------
# Configuration consistency
# --------------------------------------------------------------------------
def test_every_reliability_field_exists_on_a_serialized_row():
    coded = FullTextCodingRecord.model_validate(repair_payload({"record_id": "F001_1"}, _raw_payload()))
    row = serialize_row(coded, model_id="test/model")
    missing = [field for field in RELIABILITY_FIELDS if field not in row]
    assert missing == [], f"reliability fields absent from the coded row: {missing}"
    assert all(field in row for field in LIST_FIELDS)
    assert all(row[field] in ("yes", "no") for field in PRESENCE_FIELDS)


def test_prompt_spec_covers_every_controlled_field():
    spec = build_schema_spec()
    assert set(CONTROLLED_VALUES).issubset(set(spec))
    prompt = build_prompt({"record_id": "F001_1"}, "TITLE: a paper")
    assert "F001_1" in prompt
    assert "integration_triadic" in prompt
    assert "mechanistic" in prompt


def test_abstract_reliability_and_list_fields_are_disjoint():
    assert not set(ABSTRACT_RELIABILITY_FIELDS) & set(ABSTRACT_LIST_FIELDS)


# --------------------------------------------------------------------------
# The condenser
# --------------------------------------------------------------------------
def test_condenser_sends_short_papers_whole():
    paper = {
        "title": "A biopsychosocial review",
        "abstract": "An abstract.",
        "sections": [{"title": "Introduction", "text": "One paragraph about pain.\n\nAnother paragraph."}],
    }
    text, stats = build_coding_text(paper, budget=100_000)
    assert stats["reduced"] is False
    assert stats["kept_share"] == 1.0
    assert "One paragraph about pain." in text


def test_condenser_prefers_paragraphs_where_domains_meet():
    integrative = ("Central sensitization interacts with catastrophizing and with workplace support, "
                   "so the mechanism spans all three domains and mediates disability outcomes.")
    filler = "The search was run in three databases and duplicates were removed by hand."
    assert paragraph_score(integrative) > paragraph_score(filler)


def test_condenser_marks_what_it_dropped():
    sections = [{"title": "Methods", "text": "\n\n".join([f"Filler paragraph number {i}. " * 30 for i in range(40)])}]
    paper = {"title": "T", "abstract": "A", "sections": sections}
    text, stats = build_coding_text(paper, budget=3000)
    assert stats["reduced"] is True
    assert "omitted" in text


def test_list_overlap_returns_one_row_per_open_list():
    payload = json.dumps([{"concept_label": "catastrophizing"}, {"concept_label": "self-efficacy"}])
    other = json.dumps([{"concept_label": "catastrophizing"}])
    frame = pd.DataFrame([
        {"record_id": "F1", "model_label": "DeepSeek-V4-Flash", "psychological_concepts": payload,
         "theoretical_frameworks": "[]", "conceptual_problems": "[]"},
        {"record_id": "F1", "model_label": "Nex-N2-Mini", "psychological_concepts": other,
         "theoretical_frameworks": "[]", "conceptual_problems": "[]"},
        {"record_id": "F1", "model_label": "Laguna-XS-2.1", "psychological_concepts": other,
         "theoretical_frameworks": "[]", "conceptual_problems": "[]"},
    ])
    overlap = compute_list_overlap(frame)
    assert len(overlap) == len(LIST_FIELDS)
    concepts = overlap[overlap["field"] == "psychological_concepts"].iloc[0]
    # two pairs share one of two labels (0.5), one pair is identical (1.0)
    assert concepts["mean_pairwise_jaccard"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
