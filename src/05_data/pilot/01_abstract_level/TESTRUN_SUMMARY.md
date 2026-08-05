# Abstract-level test run: cross-provider reliability

The Stage 2 abstract coding scheme was applied to 100 PubMed records by 3 large language models from 3 different providers, which gives 300 independent codings from 300 API calls. The models act as independent raters, so the run is a cross-provider inter-rater reliability check on the coding scheme and on the code that applies it.

## What was coded

- Corpus: 100 records from the operational PubMed query, most recent first, each with a usable abstract.
- Models: `deepseek/deepseek-v4-flash` (DeepSeek), `nex-agi/nex-n2-mini` (Nex AGI), `poolside/laguna-xs-2.1` (Poolside)
- Coding method counts: {'llm_structured': 300}
- Token usage: 1,375,595 tokens for $0.131

## Headline agreement

| Metric | Value |
| --- | --- |
| Mean Fleiss' kappa over all coded fields | 0.604 |
| Mean Krippendorff's alpha | 0.605 |
| Mean observed agreement | 77.9% |
| Mean unanimous rate | 69.2% |
| Abstracts where all models agree on Stage 3 candidacy | 91 of 100 |

## Per field

| Field | Fleiss' kappa | Krippendorff alpha | Observed agreement | Unanimous | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Quality assessment | 0.918 | 0.919 | 97.3% | 96.0% | Almost perfect |
| Social mention | 0.772 | 0.772 | 88.7% | 83.0% | Substantial |
| ICD-11 pain category | 0.759 | 0.759 | 84.7% | 78.0% | Substantial |
| Stage 3 candidate | 0.735 | 0.736 | 94.0% | 91.0% | Substantial |
| Review type | 0.708 | 0.709 | 81.7% | 74.0% | Substantial |
| Musculoskeletal flag | 0.568 | 0.569 | 72.3% | 63.0% | Moderate |
| Stage 3 priority | 0.568 | 0.569 | 72.3% | 63.0% | Moderate |
| Psychological mention | 0.536 | 0.537 | 85.3% | 78.0% | Moderate |
| Objective category | 0.512 | 0.514 | 72.3% | 60.0% | Moderate |
| Biological mention | 0.495 | 0.497 | 78.7% | 68.0% | Moderate |
| BPS function | 0.345 | 0.348 | 49.7% | 34.0% | Fair |
| Provisional typology | 0.327 | 0.329 | 57.7% | 43.0% | Fair |

Three of these fields are derived rather than asked: `stage3_candidate`, `stage3_priority`, `conceptual_problem_flags`. They are computed from the coded content by a fixed rule, so their agreement is the agreement of the judgements they read, not an independent judgement.

## Open extraction lists

Agreement on an open list is measured with set overlap, because two coders can both be right and still return different strings.

| List | Mean pairwise Jaccard | Distinct labels | Items with a label all models share |
| --- | --- | --- | --- |
| Psychological concepts | 0.555 | 263 | 68 |
| Theoretical frameworks | 0.470 | 87 | 41 |
| Conceptual problems | 0.441 | 7 | 50 |

## How the models behave

| Model | Provider | Stage 3 candidate | Musculoskeletal yes | Integrative signal | Mean domains | Mean concepts |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash | DeepSeek | 89.0% | 48.0% | 15.0% | 1.83 | 2.95 |
| Nex-N2-Mini | Nex AGI | 88.0% | 36.0% | 18.0% | 2.25 | 3.41 |
| Laguna-XS-2.1 | Poolside | 84.0% | 34.0% | 49.0% | 2.05 | 2.62 |

The closest pair is DeepSeek-V4-Flash and Nex-N2-Mini (82.0% observed agreement); the most distant pair is DeepSeek-V4-Flash and Laguna-XS-2.1 (75.3%).

## Consensus picture of the corpus

- Biological mention: 69 of 100 abstracts (69.0%).
- Psychological mention: 81 of 100 abstracts (81.0%).
- Social mention: 52 of 100 abstracts (52.0%).
- Provisional typology: pseudo-bps or partial signal 54, potential integrative signal 32, multifactorial signal 11, rhetorical label signal 3
- Stage 3 candidates by majority vote: 88 of 100.

## How to read this

These numbers describe agreement between three cheap models on a test corpus. They are not a finding about the biopsychosocial literature and they are not a validation of the coding scheme against a human standard. What they do show is where the scheme is specified tightly enough that independent coders converge, and where it is not, which is exactly the input the expert evaluation of the coding schemes needs.
