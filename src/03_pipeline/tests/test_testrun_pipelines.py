"""Offline tests for the two cross-provider test-run pipelines.

Nothing here calls an API. The tests cover the parts that decide whether the
run is trustworthy: the agreement primitives, the deterministic repair and
derivation layer of the full-text scheme, the quote verification, and the
consistency between the prompt, the schema, and the reliability configuration.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bps_review.extraction.llm_stage2 import FIELD_SPECIFICATION, Stage2StructuredRecord, _batch_prompt
from bps_review.fulltext.analysis.integrity import integration_evidence_discipline, verify_quote
from bps_review.fulltext.analysis.integrity import extraction_yield
from bps_review.fulltext.analysis.reliability import (
    adjacent_agreement,
    build_reliability,
    compute_list_overlap,
)
from bps_review.fulltext.analysis.semantic import (
    SEMANTIC_SPACES,
    greedy_match,
    present_spaces,
    semantic_jaccard,
)
from bps_review.fulltext.analysis.spaces import space_labels
from bps_review.fulltext.coding.condense import build_coding_text, paragraph_score
from bps_review.fulltext.coding.derive import (
    assert_usable_payload,
    derive,
    item_rows,
    normalize_choice,
    record_from_row,
    repair_payload,
    serialize_row,
)
from bps_review.fulltext.coding.prompt import (
    CONTROLLED_VALUES,
    FIELD_INSTRUCTIONS,
    build_prompt,
    build_schema_spec,
)
from bps_review.fulltext.coding.schema import (
    FullTextCodingRecord,
    ITEM_LABEL_KEY,
    ITEM_MODELS,
    ITEM_QUOTE_KEY,
    OPEN_LIST_FIELDS,
)
from bps_review.fulltext.coding.vocabulary import is_controlled, normalize_label
from bps_review.fulltext.config import (
    EXTRACTION_SPACES,
    FULLTEXT_MODELS,
    ITEM_CAPS,
    LIST_FIELDS,
    LIST_LABEL_KEY,
    LIST_LABEL_KIND,
    PRESENCE_FIELDS,
    RELIABILITY_FIELDS,
    SPACE_BY_NAME,
)
from bps_review.fulltext.visualization.figures import EXTRACTION_YIELD_ORDER
from bps_review.graph.builder import (
    BPS_ENTITY_SUBGROUPS,
    FIELD_GROUPS,
    FieldView,
    graph_payload,
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
        "icd11_pain_category": "chronic secondary musculoskeletal pain",
        "population": "adult",
        "care_setting": "primary care",
        "primary_discipline": "physiotherapy or rehabilitation",
        "pain_condition_detail": "chronic low back pain",
        "pain_conditions": ["chronic low back pain", "CLBP"],
        "quality_assessment_reported": "yes",
        "quality_assessment_tools": ["AMSTAR-2"],
        "bps_label_used": "explicit_bps_term",
        "bps_primary_function": "explanatory framework",
        "bps_functions_present": ["explanatory framework", "intervention rationale"],
        "bps_definition_status": "formally_defined",
        "bps_model_variants": ["biopsychosocial model", "bio-psycho-social framework"],
        "bps_usage_instances": [
            {"usage_verbatim": "the biopsychosocial model explains persistent disability",
             "bps_function": "explanatory framework", "is_definitional": "no",
             "attributed_source": "Engel 1977", "section_located": "introduction"},
            {"usage_verbatim": "a biopsychosocial approach supports multidisciplinary care",
             "bps_function": "intervention rationale", "section_located": "discussion"},
        ],
        "bps_definitions": [
            {"definition_verbatim": "the model holds that biological psychological and social factors interact",
             "definition_type": "explicit_formal", "attributed_source": "Engel",
             "elements_named": ["biological", "psychological", "social"],
             "section_located": "introduction"},
        ],
        "domain_coverage_bio": "elaborated",
        "domain_coverage_psych": "elaborated",
        "domain_coverage_social": "mentioned",
        "coverage_lifestyle": "minimal",
        "coverage_spiritual_existential": "absent",
        "integration_bio_psych": "mechanistic",
        "integration_psych_social": "descriptive",
        "integration_bio_social": "none",
        "integration_triadic": "descriptive",
        "overall_balance": "balanced",
        "bps_typology": "true_integrative",
        "concept_definitions_present": "partial",
        "integration_claims": [
            {"domains_linked": "bio_psych", "integration_level": "mechanistic",
             "source_factor_label": "catastrophizing", "target_factor_label": "central sensitization",
             "direction": "unidirectional", "mediator_or_moderator": "descending modulation",
             "claim_verbatim": "central sensitization is amplified by catastrophizing",
             "mechanism_note": "via descending modulation", "section_located": "discussion",
             "evidence_basis": "theorized"},
            {"domains_linked": "triadic", "integration_level": "descriptive",
             "claim_verbatim": "biological psychological and social factors jointly shape outcome",
             "section_located": "introduction"},
        ],
        "domain_evidence": [
            {"domain": "biological", "coverage_level": "elaborated",
             "constructs_named": ["central sensitization"],
             "subdomains_named": ["central sensitisation"],
             "evidence_verbatim": "a passage about nociception",
             "section_located": "introduction"},
        ],
        "biological_factors": [
            {"factor_label": "central sensitisation", "subdomain_label": "central sensitization",
             "mechanism_level": "spinal or central nervous system",
             "factor_role": "mediator", "factor_verbatim": "central sensitisation was widely reported",
             "section_located": "results"},
            {"factor_label": "paraspinal muscle fatigue", "subdomain_label": "",
             "factor_role": "correlate", "factor_verbatim": "paraspinal muscle fatigue was noted"},
        ],
        "social_factors": [
            {"factor_label": "workplace support", "subdomain_label": "social support",
             "social_level": "workplace", "factor_role": "protective factor",
             "factor_verbatim": "supportive supervisors predicted return to work"},
        ],
        "other_domain_factors": [
            {"factor_label": "sleep hygiene", "domain": "lifestyle", "factor_role": "treatment target",
             "factor_verbatim": "sleep hygiene advice was part of the programme"},
        ],
        "psychological_concepts": [
            {"concept_label": "pain catastrophizing", "concept_family": "catastrophizing",
             "definitional_status": "formally_defined", "definition_source": "cited from other work",
             "measure_named": "PCS", "factor_role": "mediator",
             "definition_verbatim": "catastrophizing is defined as an exaggerated negative orientation"},
            {"concept_label": "self-efficacy", "definitional_status": "named_only"},
        ],
        "concept_relations": [
            {"source_concept": "kinesiophobia", "target_concept": "pain-related fear",
             "relation_type": "is_a_subtype_of", "explicitly_stated": "yes",
             "relation_verbatim": "kinesiophobia is a specific form of pain-related fear"},
            {"source_concept": "catastrophizing", "target_concept": "worry",
             "relation_type": "conflated_without_comment", "explicitly_stated": "no",
             "relation_verbatim": "the terms are used interchangeably throughout"},
        ],
        "theoretical_frameworks": [
            {"framework_label": "fear-avoidance model", "role": "organizing framework",
             "domains_covered": ["biological", "psychological"],
             "framework_verbatim": "we use the fear-avoidance model"},
        ],
        "instruments": [
            {"instrument_label": "Pain Catastrophizing Scale", "abbreviation": "PCS",
             "domain_measured": "psychological", "role": "predictor or covariate",
             "construct_measured_as_stated": "catastrophic thinking about pain",
             "instrument_verbatim": "catastrophizing was measured with the PCS"},
        ],
        "conceptual_problems": [
            {"problem_type": "missing_social", "problem_scope": "scope or coverage",
             "affected_labels": ["social support"], "named_by_authors": "no",
             "problem_verbatim": "social factors were not examined"},
        ],
        "key_quotes": [],
        "emergent_labels": ["paraspinal muscle fatigue", "flare-up literacy"],
        "conceptual_tensions": ["the model is invoked but never tested"],
        "additional_observations": ["the social domain appears only in the limitations"],
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
    assert len(coded.biological_factors) == 2
    assert len(coded.concept_relations) == 2
    assert coded.bps_primary_function == "explanatory framework"
    assert coded.bps_functions_present == ["explanatory framework", "intervention rationale"]


def test_repair_drops_items_without_a_label_or_a_quote():
    payload = _raw_payload()
    payload["theoretical_frameworks"].append({"role": "unclear"})
    coded = FullTextCodingRecord.model_validate(repair_payload({"record_id": "F001_1"}, payload))
    assert len(coded.theoretical_frameworks) == 1


def test_spine_pointers_are_mapped_while_free_text_labels_survive():
    """A mapped label must never cost us the review's own wording."""
    coded = FullTextCodingRecord.model_validate(repair_payload({"record_id": "F001_1"}, _raw_payload()))
    mapped = coded.biological_factors[0]
    assert mapped.subdomain_label == "Central Sensitization and Neuroplasticity"
    assert mapped.factor_label == "central sensitisation"        # exactly as written
    off_spine = coded.biological_factors[1]
    assert off_spine.subdomain_label == ""                       # nothing forced onto the spine
    assert off_spine.factor_label == "paraspinal muscle fatigue"
    # the framework label is never rewritten in place
    assert coded.theoretical_frameworks[0].framework_label == "fear-avoidance model"
    # and the free-text catch-all list is preserved verbatim
    assert "flare-up literacy" in coded.emergent_labels


def test_item_rows_carry_both_readings_of_a_label():
    coded = FullTextCodingRecord.model_validate(repair_payload({"record_id": "F001_1"}, _raw_payload()))
    rows = {(row["extraction_field"], row["item_index"]): row
            for row in item_rows(coded, model_id="test/model", model_label="M")}
    concept = rows[("psychological_concepts", 0)]
    assert concept["label_raw"] == "pain catastrophizing"
    assert concept["label_normalized"] == "pain catastrophizing"
    assert concept["label_controlled"] == "yes"
    assert concept["anchor_label"] == "catastrophizing and negative cognitive appraisal"
    unusual = rows[("biological_factors", 1)]
    assert unusual["label_normalized"] == "paraspinal muscle fatigue"
    assert unusual["anchor_label"] == ""
    assert unusual["anchor_controlled"] == "no"
    # an edge is identified by the pair it connects, not by a single label
    edge = rows[("concept_relations", 0)]
    assert edge["label_raw"] == "kinesiophobia | is_a_subtype_of | pain-related fear"


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
    assert derived["present_named_integration_edge"] == "yes"
    assert derived["present_hierarchical_relation"] == "yes"
    assert derived["present_biological_factors"] == "yes"
    assert derived["present_social_factors"] == "yes"
    assert derived["domains_present"] == 3
    assert derived["fulltext_eligibility"] == "include"
    assert derived["derived_typology"] == "true_integrative"
    assert derived["typology_matches_derived"] == "yes"
    # one quote per extracted item that carries one: 2 usage passages, 1 model
    # definition, 1 domain-evidence passage, 2 biological and 1 social factor,
    # 1 lifestyle factor, 1 concept definition, 2 relations, 2 integration
    # claims, 1 framework, 1 instrument, 1 conceptual problem
    assert derived["n_evidence_quotes"] == 16
    assert derived["n_named_integration_edges"] == 1
    assert 0.0 <= derived["integration_index"] <= 1.0


def test_ontology_derivations_measure_breadth_and_spine_coverage():
    coded = FullTextCodingRecord.model_validate(repair_payload({"record_id": "F001_1"}, _raw_payload()))
    derived = derive(coded)
    # one biological subdomain from the factor list and the same one from the
    # domain-evidence item, one psychological family, one social subdomain
    assert derived["n_subdomains_bio"] == 1
    assert derived["n_subdomains_psych"] == 1
    assert derived["n_subdomains_social"] == 1
    assert derived["n_emergent_labels"] == 2
    assert 0.0 < derived["controlled_label_share"] < 1.0
    assert derived["bps_has_substantive_function"] == "yes"
    assert "explanatory framework" in derived["bps_function_set"]
    assert derived["n_open_list_entries"] >= 8


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
    assert len(restored.concept_relations) == len(coded.concept_relations)
    assert restored.emergent_labels == coded.emergent_labels
    assert restored.bps_functions_present == coded.bps_functions_present


def test_open_list_entries_survive_the_row_roundtrip_intact():
    """The wide table joins open lists with a semicolon, so entries must not carry one."""
    payload = _raw_payload()
    payload["additional_observations"] = [
        "the social domain appears only in the limitations; the authors do not notice",
        "the model is credited to Engel but never quoted",
    ]
    coded = FullTextCodingRecord.model_validate(repair_payload({"record_id": "F001_1"}, payload))
    assert len(coded.additional_observations) == 2
    restored = record_from_row(serialize_row(coded, model_id="test/model"))
    assert restored.additional_observations == coded.additional_observations
    assert derive(restored)["integration_index"] == derive(coded)["integration_index"]
    assert derive(restored)["controlled_label_share"] == derive(coded)["controlled_label_share"]


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


def test_fulltext_prompt_names_every_schema_field():
    """A field the prompt never names is a field the repair layer fills silently."""
    named = {name for name, _ in FIELD_INSTRUCTIONS}
    coded = set(FullTextCodingRecord.model_fields) - {"record_id"}
    assert named == coded, (
        "the prompt's field instructions and the validated schema must not drift apart: "
        f"{sorted(coded - named)} unnamed, {sorted(named - coded)} unknown"
    )


def test_prompt_instructs_every_extraction_and_open_list():
    """A list the prompt never names is a list the model will never return."""
    spec = build_schema_spec()
    missing = [name for name in ITEM_MODELS if name not in spec]
    assert missing == [], f"extraction lists absent from the prompt: {missing}"
    missing_open = [name for name in OPEN_LIST_FIELDS if name not in spec]
    assert missing_open == [], f"open lists absent from the prompt: {missing_open}"
    assert all(spec[name].get("max_items") == ITEM_CAPS[name] for name in ITEM_MODELS)


def test_every_extraction_list_is_configured_end_to_end():
    """Caps, quote keys, and identifying labels must exist for every list."""
    assert set(ITEM_CAPS) == set(ITEM_MODELS)
    assert set(ITEM_QUOTE_KEY) == set(ITEM_MODELS)
    for name, model in ITEM_MODELS.items():
        assert ITEM_QUOTE_KEY[name] in model.model_fields
        assert LIST_LABEL_KEY.get(name, ()) or name not in LIST_FIELDS
        for key in LIST_LABEL_KEY.get(name, ()):
            assert key in model.model_fields


def test_preferred_labels_map_variants_but_keep_unknown_terms():
    assert normalize_label("central sensitisation", "bio_subdomain") == \
        "Central Sensitization and Neuroplasticity"
    assert normalize_label("catastrophising", "psych_concept") == "pain catastrophizing"
    assert normalize_label("Vlaeyen and Linton", "framework") == "fear-avoidance model"
    # a term the spine does not carry survives, cleaned but unchanged
    unusual = normalize_label("Flare-up literacy", "psych_concept")
    assert unusual == "flare-up literacy"
    assert not is_controlled(unusual, "psych_concept")


def test_abstract_reliability_and_list_fields_are_disjoint():
    assert not set(ABSTRACT_RELIABILITY_FIELDS) & set(ABSTRACT_LIST_FIELDS)


# --------------------------------------------------------------------------
# The expert-facing dossier must describe the scheme the pipeline runs
# --------------------------------------------------------------------------
def _dossier_content():
    """Load the coding-scheme dossier source, which lives outside the package."""
    root = Path(__file__).resolve().parents[3]
    path = root / "src" / "02_coding_schemes" / "_build" / "content.py"
    spec = importlib.util.spec_from_file_location("dossier_content", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scheme_3_dossier_documents_every_coded_field():
    """Experts evaluate the dossier, so the dossier has to be the scheme."""
    dossier = _dossier_content()
    documented = {f["name"] for sec in dossier.SCHEME_3["sections"]
                  if sec["kind"] == "fields" for f in sec["fields"]}
    coded = set(FullTextCodingRecord.model_fields) - {"record_id"}
    assert not (coded - documented), f"coded fields absent from the dossier: {sorted(coded - documented)}"


def test_scheme_3_dossier_documents_every_item_subfield():
    dossier = _dossier_content()
    by_name = {f["name"]: f for sec in dossier.SCHEME_3["sections"]
               if sec["kind"] == "fields" for f in sec["fields"]}
    for name, model in ITEM_MODELS.items():
        assert name in by_name, f"extraction list absent from the dossier: {name}"
        documented = {sf["name"] for sf in by_name[name]["subfields"]}
        assert documented == set(model.model_fields), (
            f"{name}: dossier subfields and schema fields differ "
            f"({sorted(set(model.model_fields) - documented)} missing, "
            f"{sorted(documented - set(model.model_fields))} extra)"
        )
        assert by_name[name]["cap"] == ITEM_CAPS[name]


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
    other = json.dumps([{"concept_label": "catastrophising"}])   # the same label, spelled differently
    empty = {field: "[]" for field in LIST_FIELDS}
    frame = pd.DataFrame([
        {"record_id": "F1", "model_label": "DeepSeek-V4-Flash", **empty,
         "psychological_concepts": payload},
        {"record_id": "F1", "model_label": "Nex-N2-Mini", **empty, "psychological_concepts": other},
        {"record_id": "F1", "model_label": "Laguna-XS-2.1", **empty, "psychological_concepts": other},
    ])
    overlap = compute_list_overlap(frame)
    assert len(overlap) == len(LIST_FIELDS)
    concepts = overlap[overlap["field"] == "psychological_concepts"].iloc[0]
    # two pairs share one of two labels (0.5), one pair is identical (1.0);
    # the two spellings count as one label because the vocabulary maps them
    assert concepts["mean_pairwise_jaccard"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)


def test_list_overlap_compares_relations_as_edges():
    edge = json.dumps([{"source_concept": "kinesiophobia", "relation_type": "is_a_subtype_of",
                        "target_concept": "pain-related fear"}])
    different_target = json.dumps([{"source_concept": "kinesiophobia",
                                    "relation_type": "is_a_subtype_of",
                                    "target_concept": "anxiety"}])
    empty = {field: "[]" for field in LIST_FIELDS}
    frame = pd.DataFrame([
        {"record_id": "F1", "model_label": "DeepSeek-V4-Flash", **empty, "concept_relations": edge},
        {"record_id": "F1", "model_label": "Nex-N2-Mini", **empty, "concept_relations": edge},
        {"record_id": "F1", "model_label": "Laguna-XS-2.1", **empty,
         "concept_relations": different_target},
    ])
    relations = compute_list_overlap(frame)
    row = relations[relations["field"] == "concept_relations"].iloc[0]
    # identical edge on one pair, a different endpoint on the other two
    assert row["mean_pairwise_jaccard"] == pytest.approx((1.0 + 0.0 + 0.0) / 3)


# --------------------------------------------------------------------------
# Semantic overlap of the open lists
# --------------------------------------------------------------------------
class _StubStore:
    """A vector store with hand-placed labels, so matching is testable offline."""

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = {label: np.asarray(values, dtype=np.float32) for label, values in vectors.items()}

    def vectors_for(self, texts):
        rows = [self._vectors[text] for text in texts if text in self._vectors]
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(rows)

    def known(self, texts):
        return [text for text in texts if text in self._vectors]


def test_greedy_match_pairs_each_label_at_most_once():
    similarity = np.array([[0.9, 0.8], [0.7, 0.95]])
    matched = greedy_match(similarity, 0.65)
    assert sorted((left, right) for left, right, _ in matched) == [(0, 0), (1, 1)]
    # the strongest pair is taken first, so a shared partner cannot be double counted
    contested = np.array([[0.9], [0.95]])
    assert len(greedy_match(contested, 0.65)) == 1


def test_semantic_jaccard_reduces_to_the_lexical_one_at_full_similarity():
    """At a threshold of 1.0 only identical vectors match, so the soft Jaccard is the hard one."""
    store = _StubStore({"a": [1.0, 0.0], "b": [0.0, 1.0], "a-prime": [0.94, 0.34]})
    first, second = ["a", "b"], ["a", "a-prime"]
    assert semantic_jaccard(first, second, store, 1.0) == pytest.approx(1 / 3)
    # a and a-prime are one concept at a permissive threshold, so both labels pair up
    assert semantic_jaccard(first, second, store, 0.9) == pytest.approx(1 / 3)
    assert semantic_jaccard(["a"], ["a-prime"], store, 0.9) == pytest.approx(1.0)


def test_every_extraction_list_is_compared(): 
    """A list left out of the overlap metrics is extraction nobody ever checks."""
    assert set(LIST_FIELDS) == set(ITEM_MODELS)
    assert set(LIST_LABEL_KIND) == set(ITEM_MODELS)


def test_list_identity_keys_do_not_drift_from_the_schema():
    """config spells the identity keys out, because the schema package imports it back."""
    assert LIST_LABEL_KEY == dict(ITEM_LABEL_KEY)


def test_every_extraction_list_has_an_identity_space():
    identity = {space.name for space in EXTRACTION_SPACES if space.layer == "identity"}
    assert identity == set(LIST_FIELDS)


def test_every_space_reads_a_real_field_of_a_real_item():
    """A space pointing at a field the scheme does not have would silently measure nothing."""
    for space in EXTRACTION_SPACES:
        model = ITEM_MODELS.get(space.field)
        assert model is not None, f"{space.name} reads an unknown extraction list {space.field}"
        fields = set(model.model_fields)
        for key in (*space.keys, space.sublist, space.filter_key):
            if key:
                assert key in fields, f"{space.name} reads {space.field}.{key}, which does not exist"


def test_spaces_are_uniquely_named():
    names = [space.name for space in EXTRACTION_SPACES]
    assert len(names) == len(set(names)) == len(SPACE_BY_NAME)


def test_a_space_reads_only_the_items_its_filter_allows():
    """A filtered space is a different question, not a smaller sample of the same one."""
    payload = json.dumps([
        {"concept_label": "catastrophizing", "definitional_status": "formally_defined"},
        {"concept_label": "self-efficacy", "definitional_status": "named_only"},
    ])
    # both labels are mapped onto their canonical vocabulary entry
    assert space_labels(payload, SPACE_BY_NAME["psychological_concepts"]) == {
        "pain catastrophizing", "pain self-efficacy"}
    assert space_labels(payload, SPACE_BY_NAME["defined_concepts"]) == {"pain catastrophizing"}


def test_a_vocabulary_space_reads_the_sublist_inside_the_items():
    """The constructs carrying a domain are invisible to any item-identity metric."""
    payload = json.dumps([
        {"domain": "biological", "constructs_named": ["central sensitization", "inflammation"]},
        {"domain": "social", "constructs_named": ["work support"]},
    ])
    assert space_labels(payload, SPACE_BY_NAME["domain_evidence"]) == {"biological", "social"}
    assert space_labels(payload, SPACE_BY_NAME["domain_evidence_constructs"]) == {
        "central sensitization", "inflammation", "work support"}
    assert space_labels(payload, SPACE_BY_NAME["domain_evidence_constructs_bio"]) == {
        "central sensitization", "inflammation"}


def test_spaces_a_run_cannot_answer_are_dropped_not_reported_empty():
    empty = {field: "[]" for field in LIST_FIELDS}
    frame = pd.DataFrame([
        {"record_id": "F1", "model_label": model, **empty,
         "psychological_concepts": json.dumps([{"concept_label": "catastrophizing"}])}
        for model in ("DeepSeek-V4-Flash", "Nex-N2-Mini", "Laguna-XS-2.1")
    ])
    available = {space.name for space in present_spaces(frame)}
    assert "psychological_concepts" in available
    assert "domain_evidence_constructs" not in available
    assert available < {space.name for space in SEMANTIC_SPACES}


# --------------------------------------------------------------------------
# Knowledge graph
# --------------------------------------------------------------------------
def _graph_frames():
    empty = {field: "[]" for field in ITEM_MODELS}
    concepts = json.dumps([
        {"concept_label": "catastrophizing", "concept_family": "pain-related cognition",
         "definitional_status": "formally_defined", "definition_verbatim": "a negative mental set"},
    ])
    row = serialize_row(FullTextCodingRecord.model_validate(_raw_payload("F001_1")), "vendor/model-x")
    long_df = pd.DataFrame([
        {**row, **empty, "psychological_concepts": concepts, "model_order": 1,
         "model_label": "Model-A", "provider": "Vendor A", "model_id": "vendor/model-x"},
        {**row, **empty, "psychological_concepts": "[]", "model_order": 2,
         "model_label": "Model-B", "provider": "Vendor B", "model_id": "vendor/model-y"},
    ])
    corpus_df = pd.DataFrame([{"record_id": "F001_1", "title": "A review of something"}])
    items_df = pd.DataFrame([
        {"record_id": "F001_1", "model_label": "Model-A", "extraction_field": "psychological_concepts",
         "item_index": 0, "label_raw": "catastrophizing", "label_normalized": "catastrophizing",
         "label_controlled": "yes", "anchor_label": "pain-related cognition",
         "quote": "a negative mental set", "item_json": "{}"},
    ])
    return corpus_df, long_df, items_df


def test_graph_payload_carries_every_coded_field_exactly_once():
    """A field that no group claims still has to reach the reviewer, under 'Other coded fields'."""
    corpus_df, long_df, items_df = _graph_frames()
    payload = graph_payload(corpus_df, long_df, items_df)
    field_nodes = [node for node in payload["nodes"] if node["type"] == "field"]
    coded_columns = [column for column in long_df.columns
                     if column not in {"record_id", "model_order", "model_label", "provider", "model_id"}]
    assert sorted(node["field"] for node in field_nodes) == sorted(coded_columns)
    assert len(field_nodes) == len({node["field"] for node in field_nodes})


def test_graph_payload_is_a_connected_tree_down_to_the_extracted_item():
    corpus_df, long_df, items_df = _graph_frames()
    payload = graph_payload(corpus_df, long_df, items_df)
    node_ids = {node["id"] for node in payload["nodes"]}
    assert all(edge["source"] in node_ids and edge["target"] in node_ids for edge in payload["edges"])
    # one root, and every other node reached by exactly one parent edge
    assert len(payload["edges"]) == len(payload["nodes"]) - 1
    concept_items = [node for node in payload["nodes"]
                     if node["type"] == "item" and node["field"] == "psychological_concepts"]
    assert [node["label"] for node in concept_items] == ["catastrophizing"]
    assert concept_items[0]["detail"]["Normalized label"] == "catastrophizing"


def _spec_columns(spec) -> set[str]:
    """Every column a group names, however deeply its headings nest."""
    if isinstance(spec, dict):
        return {column for child in spec.values() for column in _spec_columns(child)}
    return {view.column if isinstance(view, FieldView) else view for view in spec}


def _grouped_columns() -> set[str]:
    return {column for spec in FIELD_GROUPS.values() for column in _spec_columns(spec)}


def test_graph_groups_only_name_fields_the_scheme_produces():
    """Grouping must not drift away from the schema it lays out."""
    row = serialize_row(FullTextCodingRecord.model_validate(_raw_payload()), "vendor/model-x")
    produced = set(row) | {"model_order", "model_label", "provider", "model_id"}
    grouped = _grouped_columns()
    assert grouped <= produced, f"grouped but never produced: {sorted(grouped - produced)}"


def test_the_entity_layer_separates_the_triad_from_what_lies_beyond_it():
    """The three domains are siblings; what the registration adds is one level down.

    Lifestyle and the spiritual or existential are not a fourth and fifth domain.
    They are the account of what falls outside the three, so they sit under one
    heading of their own and the depth of the tree says so.
    """
    assert list(BPS_ENTITY_SUBGROUPS) == [
        "Biological factors",
        "Psychological factors",
        "Social factors",
        "Other factors",
    ]
    assert list(BPS_ENTITY_SUBGROUPS["Other factors"]) == [
        "Lifestyle factors",
        "Spiritual and existential factors",
    ]
    # Every entity owns at least one field, and every view reads a real column.
    row = serialize_row(FullTextCodingRecord.model_validate(_raw_payload()), "vendor/model-x")
    for entity, spec in BPS_ENTITY_SUBGROUPS.items():
        assert _spec_columns(spec), f"{entity} has no coding fields"
        for column in _spec_columns(spec):
            assert column in row, f"{entity} reads {column}, which the scheme never writes"


def test_a_list_holding_several_entities_is_split_into_one_node_each():
    """The domain evidence is one column carrying all three domains at once.

    Read as a single node it would put the biological evidence, the psychological
    evidence, and the social evidence in one undifferentiated list, which is the
    distinction this review exists to make.
    """
    corpus_df, long_df, items_df = _graph_frames()
    evidence = json.dumps([
        {"domain": "biological", "constructs_named": ["central sensitization"],
         "evidence_verbatim": "a bio passage"},
        {"domain": "social", "constructs_named": ["work support"],
         "evidence_verbatim": "a social passage"},
    ])
    long_df = long_df.assign(domain_evidence=evidence)
    payload = graph_payload(corpus_df, long_df, items_df)
    by_type = lambda kind: [node for node in payload["nodes"] if node["type"] == kind]  # noqa: E731

    entities = {node["label"] for node in by_type("subgroup")}
    assert {"Biological factors", "Social factors", "Psychological factors"} <= entities

    fields = {node["field"] for node in by_type("field")}
    assert "domain_evidence__biological" in fields and "domain_evidence__social" in fields
    # nothing was coded as psychological evidence, so that slice is dropped rather
    # than rendered as an empty node
    assert "domain_evidence__psychological" not in fields
    assert "domain_evidence" not in fields

    for slice_key, domain in (("domain_evidence__biological", "biological"),
                              ("domain_evidence__social", "social")):
        items = [node for node in by_type("item") if node["field"] == slice_key]
        assert items, f"{slice_key} carries no items"
        assert {node["detail"]["domain"] for node in items} == {domain}


def test_a_field_hangs_off_the_headings_that_name_it():
    """The path a node reports and the path its edges take have to be the same one."""
    corpus_df, long_df, items_df = _graph_frames()
    long_df = long_df.assign(
        other_domain_factors=json.dumps([
            {"factor_label": "sleep hygiene", "domain": "lifestyle"},
            {"factor_label": "meaning in life", "domain": "spiritual or existential"},
        ]),
    )
    payload = graph_payload(corpus_df, long_df, items_df)
    by_id = {node["id"]: node for node in payload["nodes"]}
    parent = {edge["target"]: edge["source"] for edge in payload["edges"]}

    nested = [node for node in payload["nodes"]
              if node["type"] == "field" and node.get("field_path") == [
                  "Other factors", "Lifestyle factors"]]
    assert nested, "the lifestyle slice never reached its heading"

    for node in payload["nodes"]:
        if node["type"] != "field" or not node.get("field_path"):
            continue
        walked = []
        cursor = by_id[parent[node["id"]]]
        while cursor["type"] == "subgroup":
            walked.append(cursor["label"])
            cursor = by_id[parent[cursor["id"]]]
        assert cursor["type"] == "group"
        assert list(reversed(walked)) == node["field_path"]


# --------------------------------------------------------------------------
# A complete current-scheme run, end to end
# --------------------------------------------------------------------------
def _full_scheme_frame(n_papers: int = 3) -> pd.DataFrame:
    """A coding table with every extraction list of the scheme filled.

    The run currently on disk was coded before the extraction layer existed, so
    it exercises only part of the analysis. This fixture stands in for the next
    run: it is built from the validated schema itself, which is exactly the shape
    the coders return once they answer the current prompt.
    """
    quotes = [{"claim_verbatim": "a quoted claim", "claim_type": "integrative",
               "section_located": "discussion", "why_it_matters": "it is central"}]
    # One evidence item per domain, because the evidence list is read as three
    # separate spaces and a fixture carrying only biology would not exercise them.
    evidence = [
        {"domain": domain, "coverage_level": "mentioned",
         "constructs_named": [construct], "subdomains_named": [construct],
         "evidence_verbatim": f"a passage about {construct}", "section_located": "results"}
        for domain, construct in (("biological", "central sensitization"),
                                  ("psychological", "pain catastrophizing"),
                                  ("social", "work support"))
    ]
    rows = []
    for paper in range(1, n_papers + 1):
        record_id = f"F{paper:03d}_1"
        for order, model in enumerate(FULLTEXT_MODELS, start=1):
            payload = _raw_payload(record_id)
            payload["key_quotes"] = quotes
            payload["domain_evidence"] = evidence
            # one problem the authors name themselves, one they only display
            payload["conceptual_problems"] = payload["conceptual_problems"] + [
                {"problem_type": "vague_definition", "problem_scope": "terminology",
                 "affected_labels": ["biopsychosocial"], "named_by_authors": "yes",
                 "problem_verbatim": "the authors call the term vague"}]
            # a little variation, so agreement is not degenerate
            payload["domain_coverage_social"] = ["elaborated", "mentioned", "absent"][order % 3]
            record = FullTextCodingRecord.model_validate(payload)
            row = serialize_row(record, model.openrouter_id)
            row.update({"model_order": order, "model_label": model.label,
                        "provider": model.provider, "model_id": model.openrouter_id})
            rows.append(row)
    return pd.DataFrame(rows)


def test_a_full_scheme_run_exercises_every_extraction_list():
    """Nothing the scheme extracts may fall outside the analysis on the next run."""
    long_df = _full_scheme_frame()
    for field in LIST_FIELDS:
        assert field in long_df.columns, f"{field} never reaches the coding table"
    overlap = compute_list_overlap(long_df)
    assert len(overlap) == len(LIST_FIELDS)
    assert set(overlap["field"]) == set(ITEM_MODELS)


def test_a_full_scheme_run_supports_every_comparison_space():
    """Every space the scheme declares must be answerable by a run that fills it."""
    long_df = _full_scheme_frame()
    available = {space.name for space in present_spaces(long_df)}
    declared = {space.name for space in EXTRACTION_SPACES}
    assert available == declared, f"unanswerable on a complete run: {sorted(declared - available)}"


def test_the_extraction_volume_panel_can_read_every_list():
    """The figure names lists by column; a rename would silently empty the panel."""
    yields = extraction_yield(_full_scheme_frame())
    for name in EXTRACTION_YIELD_ORDER:
        assert f"mean_{name}" in yields.columns, f"the yield table has no mean_{name}"
    assert set(EXTRACTION_YIELD_ORDER) == set(ITEM_MODELS)


def test_reliability_runs_over_a_full_scheme_table():
    """Every reliability field has to be present, or the whole stage raises."""
    results = build_reliability(_full_scheme_frame(), write=False)
    assert len(results["field_reliability"]) == len(RELIABILITY_FIELDS)
    assert results["summary"]["n_papers"] == 3
