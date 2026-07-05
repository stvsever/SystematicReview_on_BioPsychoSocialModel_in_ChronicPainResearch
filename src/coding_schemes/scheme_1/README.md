# Scheme 1: Stage 1 Screening and Eligibility Decision Scheme

> **Status: DRAFT FOR EXPERT EVALUATION.** These coding schemes are a working draft circulated for expert evaluation. They have not been applied to a final review corpus. The current manuscript is a test run built with an earlier, coarser generation of these schemes; the refinements proposed here are awaiting expert sign-off before any re-run.

*Title and abstract eligibility for the BPS chronic pain corpus*

Rule-based provisional machine assist with mandatory human validation.

## What this scheme does

This scheme operationalizes title and abstract screening after search and deduplication. Its function is to decide whether a record enters the review corpus for downstream coding, using a human-validatable rule set centred on biopsychosocial language, chronic pain relevance, review design, and population eligibility.

Because everything downstream inherits this decision, the scheme is deliberately conservative: it protects recall at the boundary and pushes genuinely ambiguous records into a borderline register rather than excluding them early.

## At a glance

| Property | Value |
| --- | --- |
| Workflow position | Pre-extraction eligibility screen, run after search and deduplication and before Stage 2 abstract coding. |
| Operational mode | Deterministic rule set that emits a provisional decision, confidence, and reason. Every decision is validated by a human screener in Rayyan. |
| Unit of analysis | One bibliographic record (title, abstract, and publication metadata). |
| Provenance basis | Executable screening rules plus the OSF eligibility criteria. |
| Research questions | Gatekeeper for RQ1, RQ2, RQ3 (defines the analysable corpus) |

## Files in this folder

- [`scheme_1.html`](scheme_1.html) is the interactive evaluation surface. Open it in a browser, record a verdict and comments per section, then export your feedback as JSON.
- [`scheme_1.pdf`](scheme_1.pdf) is the formal dossier for sharing and printing.
- [`scheme_1.tex`](scheme_1.tex) is the LaTeX source (generated from `_build/content.py`).

## Proposed refinements awaiting expert sign-off

These are the enhancements that raise semantic resolution. They are proposals only and are not yet applied to the pipeline:

- **Proposed Decision Order.** Screeners currently apply the rules as an unordered checklist, which lets two screeners exclude the same record for different reasons and depresses reason-level agreement. We propose a fixed precedence so that the first triggered rule owns the recorded reason:

## Coded fields

### Decision Fields and Controlled Values

- `stage1_decision` (include, exclude, maybe, unclear): Eligibility verdict for the record.
- `stage1_reason`: Controlled reason attached to exclusions and unclear cases. See the exclusion catalogue below.
- `stage1_confidence` (high, medium, low): Screener or rule confidence in the decision.
- `stage1_screened_by`: Provenance of the provisional decision. Current outputs record codex_machine_assist before human validation.
- `stage1_screening_mode`: Screening mode. Current outputs record rule_based_provisional.

### Inclusion Logic

- `Review design`: The record is a review article or review-like evidence synthesis (systematic, meta-analysis, scoping, umbrella, narrative, realist, integrative, or expert review).
- `BPS lexical trigger`: The title or abstract explicitly contains a biopsychosocial term: biopsychosocial, bio-psycho-social, or bio psycho social.
- `Chronic pain focus`: The focus concerns chronic pain, persistent pain, or a named chronic pain condition.
- `Population`: The population is adult or mixed-age rather than pediatric-only.
- `Window and language`: The record is within the operational search window and in English.

### Exclusion Catalogue and Implemented Reason Labels

- `no biopsychosocial term in title/abstract`: No explicit biopsychosocial term is present in the title or abstract.
- `outside operational search window`: Publication date falls before or after the operational search dates configured in config/protocol.yaml.
- `protocol`: Protocol papers without results are excluded.
- `commentary/editorial/letter`: Commentary, editorial, and letter publication types are excluded.
- `animal/non-human focus`: Animal-only or non-human records are excluded unless a human focus is explicit.
- `pediatric-only focus`: Populations restricted to under 18 are excluded.
- `acute pain focus`: Acute-only pain records are excluded when chronicity is not also explicit.
- `chronic pain focus unclear`: Pain is present but chronic pain relevance is insufficiently clear.
- `non-English record`: Records not in English are excluded.
- `review status unclear`: The record cannot be confidently identified as a review or evidence synthesis from title, abstract, or publication metadata.

## Canonical source paths

- `src/protocol/decision_rules/screening_rules.md`
- `src/bps_review/screening/rules.py`
- `src/review_stages/03_screening/README.md`
- `src/protocol/osf/OSF_registration_HTBMFCPR.md`

## Regenerating this dossier

All three surfaces (PDF, HTML, README) are generated from one source of truth:

```bash
cd src/coding_schemes/_build
python3 build.py
```

Edit the scheme content in `_build/content.py`, not the generated files.
