"""Guard the published output tables of the full-text run.

They are a reformatting of a run, not an analysis, so every failure mode here is
a silent one: a list that loses its last item when it is split into columns, a
coding row that loses its paper when the bibliography is joined, or a structured
item whose JSON stops being unpacked into named columns. Each would still produce
a plausible-looking CSV.

These tables are also the run's store, so the last test is the important one: a
run written out and read back must be the same run, field for field.
"""

import json

import pandas as pd

from bps_review.fulltext.coding import schema as S
from bps_review.fulltext.publish import (
    CATEGORY_ORDER,
    CATEGORY_SINGULAR,
    ITEM_FIELD_ORDER,
    build_codings_long,
    build_codings_wide_per_paper,
    build_domain_factor_inventory,
    build_items_by_category,
    build_items_long,
    citation_columns,
    split_list_column,
)


BIBLIOGRAPHY = pd.DataFrame(
    [
        {
            "record_id": "F001_42080121",
            "citation": "Zhu et al. (2026), Frontiers in Aging Neuroscience",
            "title": "Myofascial pain in older adults",
            "authors": "Chang Zhu; Jie Xu",
            "publication_year": 2026,
            "journal": "Frontiers in Aging Neuroscience",
            "doi": "10.3389/fnagi.2026.1806386",
            "doi_url": "https://doi.org/10.3389/fnagi.2026.1806386",
        }
    ]
)


def _long_row(model_label: str, **overrides) -> dict:
    row = {
        "record_id": "F001_42080121",
        "model_order": 1,
        "model_label": model_label,
        "provider": "Test Provider",
        "model_id": "test/model",
        "pain_conditions": "myofascial pain syndrome; chronic low back pain",
        "quality_assessment_tools": "AMSTAR 2",
        "bps_model_variants": "",
        "bps_functions_present": "explanatory framework; intervention rationale",
        "emergent_labels": "geroscience framing",
        "conceptual_tensions": "one; two; three",
        "additional_observations": "",
        "fulltext_eligibility": "include",
        "bps_typology": "multifactorial",
        "derived_typology": "multifactorial",
        "overall_balance": "bio-dominant",
        "conceptual_yield": "high",
        "synthesis_priority": "core",
        "domain_coverage_bio": "elaborated",
        "domain_coverage_psych": "mentioned",
        "domain_coverage_social": "minimal",
        "integration_bio_psych": "mechanistic",
        "integration_psych_social": "mentioned",
        "integration_bio_social": "none",
        "integration_triadic": "partial",
        "integration_index": 0.4583,
        "n_extracted_items": 3,
        "n_evidence_quotes": 3,
        "biological_factors": json.dumps(
            [
                {
                    "factor_label": "central sensitization",
                    "subdomain_label": "central sensitization",
                    "mechanism_level": "spinal or central nervous system",
                    "factor_role": "determinant or risk factor",
                    "factor_verbatim": "central sensitization amplifies nociceptive input",
                    "section_located": "introduction",
                    "evidence_basis": "empirically_supported",
                }
            ]
        ),
        "coding_method": "llm_structured",
        "llm_model": "test/model",
    }
    row.update(overrides)
    return row


def _item(field: str, payload: dict, **overrides) -> dict:
    row = {
        "record_id": "F001_42080121",
        "model_label": "Model-A",
        "model_id": "test/model",
        "extraction_field": field,
        "item_index": 0,
        "label_raw": payload.get("factor_label", ""),
        "label_normalized": "",
        "label_vocabulary": "",
        "label_controlled": "no",
        "anchor_label": "",
        "anchor_vocabulary": "",
        "anchor_controlled": "no",
        "quote": payload.get(S.ITEM_QUOTE_KEY[field], ""),
        "item_json": json.dumps(payload),
    }
    row.update(overrides)
    return row


BIO_PAYLOAD = {
    "factor_label": "central sensitization",
    "subdomain_label": "central sensitization",
    "mechanism_level": "spinal or central nervous system",
    "factor_role": "determinant or risk factor",
    "factor_verbatim": "central sensitization amplifies nociceptive input",
    "section_located": "introduction",
    "evidence_basis": "empirically_supported",
}

ITEMS = pd.DataFrame(
    [
        _item(
            "biological_factors",
            BIO_PAYLOAD,
            label_raw="central sensitization",
            anchor_label="central sensitization",
            anchor_vocabulary="bio_subdomain",
            anchor_controlled="yes",
        )
    ]
)


def test_every_extraction_category_is_published_with_all_of_its_fields():
    """A category the schema defines but the export forgets would vanish silently."""
    assert {field for field, _ in CATEGORY_ORDER} == set(S.ITEM_MODELS)
    assert set(CATEGORY_SINGULAR) == set(S.ITEM_MODELS)
    for field, model in S.ITEM_MODELS.items():
        assert set(ITEM_FIELD_ORDER[field]) == set(model.model_fields), field


def test_split_list_column_keeps_every_item():
    frame = pd.DataFrame({"pain_conditions": ["a; b; c", "d", ""]})
    split = split_list_column(frame, "pain_conditions")
    assert "pain_conditions" not in split.columns
    assert list(split["pain_conditions_count"]) == [3, 1, 0]
    assert list(split["pain_conditions_1"]) == ["a", "d", ""]
    assert list(split["pain_conditions_3"]) == ["c", "", ""]


def test_split_list_column_does_not_write_empty_tail_columns():
    frame = pd.DataFrame({"conceptual_tensions": ["a; b", "c"]})
    split = split_list_column(frame, "conceptual_tensions")
    assert list(split.columns) == [
        "conceptual_tensions_count",
        "conceptual_tensions_1",
        "conceptual_tensions_2",
    ]


def test_codings_long_leads_with_the_citation_and_drops_json():
    long_df = pd.DataFrame([_long_row("Model-A"), _long_row("Model-B", model_order=2)])
    out = build_codings_long(long_df, citation_columns(BIBLIOGRAPHY))
    assert list(out.columns)[:4] == ["record_id", "citation", "title", "authors"]
    assert "biological_factors" not in out.columns
    assert len(out) == 2
    assert out["doi"].tolist() == ["10.3389/fnagi.2026.1806386"] * 2
    assert out["conceptual_tensions_3"].tolist() == ["three", "three"]


def test_codings_wide_per_paper_reports_agreement_on_both_headline_judgements():
    long_df = pd.DataFrame(
        [
            _long_row("Model-A"),
            _long_row("Model-B", model_order=2),
            _long_row(
                "Model-C",
                model_order=3,
                fulltext_eligibility="uncertain",
                bps_typology="pseudo_bps",
            ),
        ]
    )
    out = build_codings_wide_per_paper(long_df, citation_columns(BIBLIOGRAPHY))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_providers"] == 3
    assert row["n_providers_include"] == 2
    assert row["eligibility_agreement"] == "majority"
    assert row["modal_eligibility"] == "include"
    assert row["typology_agreement"] == "majority"
    assert row["modal_bps_typology"] == "multifactorial"
    assert row["model_a__bps_typology"] == "multifactorial"
    assert row["model_c__bps_typology"] == "pseudo_bps"


def test_three_different_answers_are_reported_as_no_majority():
    """A three-way split has no mode.

    Naming one of the tied values would invent a majority that does not exist,
    and picking it by set iteration order would make the table change between
    runs over the same coding.
    """
    long_df = pd.DataFrame(
        [
            _long_row("Model-A"),
            _long_row("Model-B", model_order=2, fulltext_eligibility="uncertain",
                      bps_typology="pseudo_bps"),
            _long_row("Model-C", model_order=3, fulltext_eligibility="exclude",
                      bps_typology="rhetorical_bps"),
        ]
    )
    out = build_codings_wide_per_paper(long_df, citation_columns(BIBLIOGRAPHY))
    row = out.iloc[0]
    assert row["eligibility_agreement"] == "no majority"
    assert row["modal_eligibility"] == ""
    assert row["typology_agreement"] == "no majority"
    assert row["modal_bps_typology"] == ""


def test_unanimous_is_reported_as_unanimous():
    long_df = pd.DataFrame(
        [_long_row("Model-A"), _long_row("Model-B", model_order=2), _long_row("Model-C", model_order=3)]
    )
    row = build_codings_wide_per_paper(long_df, citation_columns(BIBLIOGRAPHY)).iloc[0]
    assert row["eligibility_agreement"] == "unanimous"
    assert row["modal_eligibility"] == "include"
    assert row["typology_agreement"] == "unanimous"


def test_items_long_keeps_the_quote_in_its_own_column():
    out = build_items_long(ITEMS, citation_columns(BIBLIOGRAPHY))
    assert "item_json" not in out.columns
    assert out.iloc[0]["verbatim_quote"] == "central sensitization amplifies nociceptive input"
    assert out.iloc[0]["extraction_category"] == "biological_factors"
    assert out.iloc[0]["citation"] == "Zhu et al. (2026), Frontiers in Aging Neuroscience"


def test_items_by_category_unpacks_the_item_json_into_columns():
    tables = build_items_by_category(ITEMS, citation_columns(BIBLIOGRAPHY))
    factors = tables["01_biological_factors"]
    assert factors.iloc[0]["mechanism_level"] == "spinal or central nervous system"
    assert factors.iloc[0]["evidence_basis"] == "empirically_supported"
    assert factors.iloc[0]["item_label"] == "central sensitization"
    # The ontology anchor is written where one applies, and never as the label.
    assert factors.iloc[0]["ontology_anchor"] == "central sensitization"
    assert factors.iloc[0]["ontology_anchor_is_controlled"] == "yes"


def test_a_list_inside_an_item_stays_readable():
    payload = {
        "definition_verbatim": "a model of interacting biological, psychological and social factors",
        "definition_type": "explicit_formal",
        "attributed_source": "Engel 1977",
        "elements_named": ["biological", "psychological", "social"],
        "section_located": "introduction",
    }
    frame = pd.DataFrame([_item("bps_definitions", payload)])
    tables = build_items_by_category(frame, citation_columns(BIBLIOGRAPHY))
    definitions = tables["09_bps_definitions"]
    assert definitions.iloc[0]["elements_named"] == "biological | psychological | social"


def test_edge_categories_are_named_by_the_pair_they_connect():
    """A relation and an integration claim have no single label. Their name is the edge."""
    relation = {
        "source_concept": "catastrophizing",
        "target_concept": "fear avoidance",
        "relation_type": "antecedent_or_cause_of",
        "explicitly_stated": "yes",
        "relation_verbatim": "catastrophizing precedes fear avoidance",
        "section_located": "discussion",
    }
    frame = pd.DataFrame([_item("concept_relations", relation)])
    tables = build_items_by_category(frame, citation_columns(BIBLIOGRAPHY))
    relations = tables["07_concept_relations"]
    assert relations.iloc[0]["item_label"] == "catastrophizing | antecedent_or_cause_of | fear avoidance"


def test_categories_without_a_vocabulary_do_not_repeat_the_label_as_a_normalized_one():
    """Most categories have no controlled list for the item's own name.

    Showing a normalized label there would imply a mapping onto the project
    vocabulary that never happened.
    """
    tables = build_items_by_category(ITEMS, citation_columns(BIBLIOGRAPHY))
    factors = tables["01_biological_factors"]
    assert "label_normalized_for_matching" not in factors.columns

    combined = build_items_long(ITEMS, citation_columns(BIBLIOGRAPHY))
    assert combined.iloc[0]["label"] == "central sensitization"
    assert combined.iloc[0]["label_normalized_for_matching"] == ""
    assert combined.iloc[0]["label_is_controlled"] == ""
    # The ontology anchor is a different field, and it survives.
    assert combined.iloc[0]["ontology_anchor"] == "central sensitization"


def test_the_domain_inventory_carries_every_named_factor_of_every_domain():
    """The three domains plus the fourth, read together rather than four tables apart."""
    social = {
        "factor_label": "social support",
        "subdomain_label": "social support",
        "social_level": "interpersonal",
        "factor_role": "protective factor",
        "factor_verbatim": "social support buffered disability",
        "section_located": "results",
        "evidence_basis": "empirically_supported",
    }
    frame = pd.DataFrame(
        [
            ITEMS.iloc[0].to_dict(),
            _item("social_factors", social, label_raw="social support"),
        ]
    )
    tables = build_items_by_category(frame, citation_columns(BIBLIOGRAPHY))
    inventory = build_domain_factor_inventory(tables, citation_columns(BIBLIOGRAPHY))
    assert inventory["domain"].tolist() == ["biological", "social"]
    assert inventory["factor_label"].tolist() == ["central sensitization", "social support"]
    assert inventory["level_or_mechanism"].tolist() == [
        "spinal or central nervous system",
        "interpersonal",
    ]
    assert inventory.iloc[1]["verbatim_quote"] == "social support buffered disability"


def test_the_published_tables_are_a_complete_store_of_a_run(tmp_path):
    """A run written out and read back must be the same run.

    The published tables are the only copy of a run: the runner's own staged
    tables are discarded once they are written. That is only safe if this round
    trip is exact, so it is asserted field for field, including every item of
    every one of the thirteen extraction lists.
    """
    from bps_review.fulltext.coding.derive import item_rows, rederive_frame, serialize_row
    from bps_review.fulltext.coding.schema import FullTextCodingRecord
    from bps_review.fulltext.publish import build_output_tables, load_run_from_tables

    record = FullTextCodingRecord.model_validate(
        {
            "record_id": "F001_42080121",
            "review_track": "musculoskeletal",
            "source_type": "systematic review",
            "bps_label_used": "explicit_bps_term",
            "bps_primary_function": "explanatory framework",
            "bps_functions_present": ["explanatory framework", "intervention rationale"],
            "bps_definition_status": "formally_defined",
            "pain_conditions": ["chronic low back pain"],
            "quality_assessment_tools": ["AMSTAR 2"],
            "conceptual_tensions": ["one", "two"],
            "emergent_labels": ["geroscience framing"],
            "domain_coverage_bio": "elaborated",
            "domain_coverage_psych": "mentioned",
            "domain_coverage_social": "minimal",
            "integration_bio_psych": "mechanistic",
            "biological_factors": [BIO_PAYLOAD],
            "social_factors": [
                {
                    "factor_label": "workplace support",
                    "subdomain_label": "work environment",
                    "social_level": "workplace",
                    "factor_role": "protective factor",
                    "factor_verbatim": "supportive supervisors predicted return to work",
                    "section_located": "results",
                }
            ],
            "psychological_concepts": [
                {
                    "concept_label": "pain catastrophizing",
                    "concept_family": "catastrophizing",
                    "definitional_status": "formally_defined",
                    "definition_verbatim": "an exaggerated negative orientation toward pain",
                    "definition_source": "cited from other work",
                    "measure_named": "PCS",
                }
            ],
            "bps_definitions": [
                {
                    "definition_verbatim": "biological, psychological and social factors interact",
                    "definition_type": "explicit_formal",
                    "attributed_source": "Engel 1977",
                    "elements_named": ["biological", "psychological", "social"],
                }
            ],
            "integration_claims": [
                {
                    "domains_linked": "bio_psych",
                    "integration_level": "mechanistic",
                    "source_factor_label": "central sensitization",
                    "target_factor_label": "pain catastrophizing",
                    "claim_verbatim": "catastrophizing amplifies central sensitization",
                }
            ],
            "concept_relations": [
                {
                    "source_concept": "catastrophizing",
                    "target_concept": "fear avoidance",
                    "relation_type": "antecedent_or_cause_of",
                    "relation_verbatim": "catastrophizing precedes fear avoidance",
                }
            ],
            "coding_rationale": "a review that names its biology and its psychology",
        }
    )
    row = serialize_row(record, model_id="test/model")
    row.update(
        {
            "record_id": "F001_42080121",
            "model_order": 1,
            "model_label": "Model-A",
            "provider": "Test Provider",
            "model_id": "test/model",
        }
    )
    long_df = rederive_frame(pd.DataFrame([row]))
    items = pd.DataFrame(item_rows(record, "test/model", "Model-A"))

    corpus = tmp_path / "01_corpus"
    corpus.mkdir()
    pd.DataFrame(
        [
            {
                "record_id": "F001_42080121",
                "title": "Myofascial pain in older adults",
                "authors": "Chang Zhu | Jie Xu",
                "year": 2026,
                "journal": "Frontiers in Aging Neuroscience",
                "doi": "10.3389/fnagi.2026.1806386",
                "pmid": "42080121",
                "pmcid": "PMC13128561",
                "pmc_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13128561/",
            }
        ]
    ).to_csv(corpus / "articles.csv", index=False)

    codings = tmp_path / "02_model_codings"
    build_output_tables(codings, long_df, items, corpus_dir=corpus, verbose=False)
    back_long, back_items = load_run_from_tables(codings)

    for column in long_df.columns:
        assert column in back_long.columns, f"{column} was lost"
        assert str(back_long.iloc[0][column]) == str(long_df.iloc[0][column]), column
    # The open lists survive their trip through one column per item.
    assert back_long.iloc[0]["pain_conditions"] == "chronic low back pain"
    assert back_long.iloc[0]["bps_functions_present"] == (
        "explanatory framework; intervention rationale"
    )
    pd.testing.assert_frame_equal(
        items.astype(str).reset_index(drop=True),
        back_items[items.columns].astype(str).reset_index(drop=True),
    )
