# Screening Rules

These rules translate the OSF registration into executable screening logic while preserving the principle that eligibility is human-validated.

## Include When

- The record is a review article.
- The title or abstract explicitly uses `biopsychosocial`, `bio-psycho-social`, or `bio psycho social`.
- The focus concerns chronic pain, persistent pain, or a chronic pain condition.
- The population is adult or mixed-age without pediatric-only restriction.
- The source is peer-reviewed and in English.

## Exclude When

- The record is a primary study, protocol, commentary, editorial, letter, conference abstract, book chapter, or grey-literature item.
- The focus is acute pain only.
- The population is exclusively pediatric.
- The record concerns animals only.
- Chronic pain is absent or peripheral.

## Borderline Handling

- `Maybe` is permitted for ambiguous review type, mixed acute/chronic populations, unclear pain duration, or unclear age group.
- Borderline cases must be logged to the audit trail with a free-text rationale.
- Stage 3 full-text screening resolves musculoskeletal ambiguities rather than Stage 1 over-excluding them.
