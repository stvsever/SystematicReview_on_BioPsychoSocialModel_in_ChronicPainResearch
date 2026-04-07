# Stage 3 Codebook

Stage 3 is full-text deep coding for musculoskeletal pain reviews.

## Core Variables

- `record_id`
- `full_text_available`: yes, no, partial.
- `pain_condition_detail`
- `domain_coverage_bio`: elaborated, mentioned, minimal, absent.
- `domain_coverage_psych`: elaborated, mentioned, minimal, absent.
- `domain_coverage_social`: elaborated, mentioned, minimal, absent.
- `integration_bio_psych`: mechanistic, directional, descriptive, mentioned, none.
- `integration_psych_social`: mechanistic, directional, descriptive, mentioned, none.
- `integration_bio_social`: mechanistic, directional, descriptive, mentioned, none.
- `integration_triadic`: mechanistic, descriptive, partial, none.
- `integration_mechanism_summary`: concise free-text explanation of proposed cross-domain pathways.
- `overall_balance`: balanced, psych-dominant, bio-dominant, social-dominant, dyadic, unclear.
- `bps_typology`: true_integrative, multifactorial, pseudo_bps, rhetorical_bps, narrow_despite_label, unclear.
- `psychological_concepts`: normalized semicolon-separated list.
- `concept_definitions_present`: yes, partial, no.
- `theoretical_frameworks`: normalized semicolon-separated list.
- `integration_quotes_or_evidence`: supporting text snippets or section references.
- `conceptual_problems`: vague definitions, construct overlap, tokenistic BPS use, missing social analysis, missing biology, mechanistic absence, unclear boundaries, other.
- `coder_notes`

## Interpretation Rules

- `true_integrative`: explicit cross-domain causal or mechanistic interaction is central to the review's logic.
- `multifactorial`: multiple domains are covered meaningfully but mostly in parallel.
- `pseudo_bps`: BPS label used but one or more core domains are thin or absent.
- `rhetorical_bps`: BPS invoked mainly as framing or justification without analytic substance.
- `narrow_despite_label`: review claims BPS framing but substantive scope is essentially single-domain.
- Stage 3 should integrate quantitative tallies with qualitative evidence excerpts so that RQ2 and RQ3 can be answered at both distributional and interpretive levels.
