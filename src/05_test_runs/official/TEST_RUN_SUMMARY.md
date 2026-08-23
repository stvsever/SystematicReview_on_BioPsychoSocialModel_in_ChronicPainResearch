# Full-text test run: cross-provider reliability and evidence integrity

The Stage 3 full-text coding scheme was applied to 47 open-access review articles by 3 large language models from 3 different providers, which gives 141 independent codings and 9400 extracted items. The corpus is the open-access subset of the records the abstract-level run carried forward, so the two stages are one chain rather than two separate exercises.

## What was coded

- Corpus: 47 full texts retrieved from PubMed Central.
- Models: `deepseek/deepseek-v4-flash` (DeepSeek), `nex-agi/nex-n2-mini` (Nex AGI), `poolside/laguna-xs-2.1` (Poolside)
- Coding method counts: {'llm_structured': 141}
- Token usage: 5,133,914 tokens for $0.335

## Headline agreement

| Metric | Value |
| --- | --- |
| Mean Fleiss' kappa over all coded fields | 0.247 |
| Mean Krippendorff's alpha | 0.252 |
| Mean observed agreement | 69.2% |
| Categorical fields, mean kappa | 0.259 |
| Binary presence fields, mean kappa | 0.231 |
| Coverage ladder, within one rung | 92.5% |
| Integration ladder, within one rung | 56.6% |
| Papers where all models agree on eligibility | 38 of 47 |

## Per field

| Field | Fleiss' kappa | Krippendorff alpha | Observed | Within one rung | Interpretation |
| --- | --- | --- | --- | --- | --- |
| BPS usage evidence present | 1.000 | 1.000 | 100.0% | n/a | Almost perfect |
| Quality assessment reported | 0.910 | 0.910 | 95.7% | n/a | Almost perfect |
| Source type | 0.673 | 0.675 | 78.0% | n/a | Substantial |
| Spiritual or existential coverage | 0.600 | 0.602 | 87.9% | 96.5% | Moderate |
| Population | 0.562 | 0.565 | 77.3% | n/a | Moderate |
| Other-domain factors present | 0.554 | 0.558 | 85.8% | n/a | Moderate |
| ICD-11 pain category | 0.544 | 0.547 | 75.9% | n/a | Moderate |
| BPS definition present | 0.376 | 0.380 | 68.8% | n/a | Fair |
| Psychological coverage | 0.370 | 0.375 | 68.1% | 96.5% | Fair |
| Primary discipline | 0.350 | 0.355 | 67.4% | n/a | Fair |
| Care setting | 0.333 | 0.337 | 66.0% | n/a | Fair |
| Defined concepts present | 0.330 | 0.334 | 68.8% | n/a | Fair |
| Overall balance | 0.324 | 0.329 | 48.9% | n/a | Fair |
| Psychological evidence present | 0.319 | 0.324 | 97.2% | n/a | Fair |
| Psychological concepts present | 0.319 | 0.324 | 97.2% | n/a | Fair |
| BPS definition status | 0.316 | 0.321 | 62.4% | 86.5% | Fair |
| Biological coverage | 0.314 | 0.318 | 61.7% | 93.6% | Fair |
| Hierarchical relation present | 0.313 | 0.318 | 75.9% | n/a | Fair |
| Lifestyle coverage | 0.285 | 0.290 | 51.8% | 85.1% | Fair |
| BPS primary function | 0.280 | 0.285 | 45.4% | n/a | Fair |
| Social coverage | 0.252 | 0.257 | 51.1% | 90.8% | Fair |
| Instruments present | 0.248 | 0.254 | 73.0% | n/a | Fair |
| Concept relations present | 0.169 | 0.175 | 90.1% | n/a | Slight |
| BPS label used | 0.153 | 0.159 | 96.5% | n/a | Slight |
| Biological factors present | 0.130 | 0.136 | 92.9% | n/a | Slight |
| Biological evidence present | 0.130 | 0.136 | 92.9% | n/a | Slight |
| Social factors present | 0.128 | 0.134 | 84.4% | n/a | Slight |
| Social evidence present | 0.128 | 0.134 | 84.4% | n/a | Slight |
| Full-text eligibility | 0.108 | 0.114 | 85.8% | n/a | Slight |
| Review track | 0.107 | 0.114 | 54.6% | n/a | Slight |
| Concept definitions | 0.101 | 0.107 | 60.3% | 100.0% | Slight |
| Frameworks present | 0.098 | 0.104 | 91.5% | n/a | Slight |
| Synthesis priority | 0.033 | 0.040 | 62.4% | n/a | Slight |
| BPS typology | -0.007 | 0.000 | 34.0% | n/a | Poor |
| Conceptual problems present | -0.014 | -0.007 | 97.2% | n/a | Poor |
| Triadic claim present | -0.023 | -0.015 | 48.9% | n/a | Poor |
| Psych-social integration | -0.023 | -0.015 | 22.0% | 51.8% | Poor |
| Named integration edge present | -0.025 | -0.018 | 67.4% | n/a | Poor |
| Integration evidence present | -0.025 | -0.018 | 67.4% | n/a | Poor |
| Triadic integration | -0.088 | -0.080 | 29.1% | 73.8% | Poor |
| Bio-psych integration | -0.118 | -0.110 | 19.9% | 55.3% | Poor |
| Bio-social integration | -0.156 | -0.148 | 21.3% | 45.4% | Poor |

## Is the evidence real?

| Model | Quotes | Exact | Near | Verified | Mean words |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash | 2149 | 92.9% | 5.8% | 98.7% | 20.8 |
| Nex-N2-Mini | 4165 | 97.9% | 1.9% | 99.9% | 21.6 |
| Laguna-XS-2.1 | 1988 | 85.5% | 12.7% | 98.1% | 28.3 |

Every verbatim quote was matched back against the article it came from. 99.2% of 8302 checkable quotes were found in the source text, literally or with at most minor differences.

Of the 297 domain links graded above 'mentioned', 95.3% carry a quoted claim for exactly that pair. This is the check that decides whether the integration ladder can be trusted: a graded link with no passage behind it is a judgement the review cannot audit.

## How the models behave

| Model | Include | Core priority | True integrative | Any triadic | Mean integration index | Mean items |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash | 80.9% | 74.5% | 0.0% | 36.2% | 0.24 | 54.6 |
| Nex-N2-Mini | 100.0% | 100.0% | 27.7% | 97.9% | 0.67 | 97.5 |
| Laguna-XS-2.1 | 93.6% | 51.1% | 6.4% | 21.3% | 0.16 | 47.9 |

## Open extraction lists

| List | Mean pairwise Jaccard | Distinct labels | Papers with a shared label |
| --- | --- | --- | --- |
| BPS usage instances | 0.379 | 10 | 36 |
| BPS definitions | 0.185 | 9 | 3 |
| Domain evidence | 0.915 | 3 | 47 |
| Biological factors | 0.234 | 604 | 34 |
| Social factors | 0.228 | 437 | 31 |
| Psychological concepts | 0.399 | 477 | 44 |
| Lifestyle and existential factors | 0.167 | 279 | 14 |
| Concept relations | 0.026 | 1052 | 2 |
| Integration claims | 0.020 | 786 | 2 |
| Theoretical frameworks | 0.301 | 252 | 35 |
| Instruments | 0.267 | 280 | 20 |
| Conceptual problems | 0.292 | 13 | 29 |
| Key quotes | 0.460 | 8 | 42 |

## The same extraction, measured by meaning

The Jaccard above asks whether two providers wrote the same string, and it asks it only of the item identities. Both halves of that are too narrow. A scheme 3 item is not a label but a small record, and several of its fields are open vocabularies in their own right: which constructs a coder says carry the biological domain, which measure a construct is tied to, which components a definition of the model lists, which constructs a conceptual problem concerns. Each is a place where two coders can read a paper the same way and write different words.

Every one of those vocabularies is compared here. The scheme declares 33 comparison spaces and this run answers every one of them. Every label is embedded once with `openai/text-embedding-3-large`, and two labels count as the same concept at a cosine of 0.65, which turns the overlap into a soft Jaccard on the same 0 to 1 scale. Both columns are computed in the same pass over the same label sets, so reading one against the other compares two ways of measuring one thing rather than two instruments.

| Comparison space | Layer | Read from | Lexical | Semantic | Labels | Concepts |
| --- | --- | --- | --- | --- | --- | --- |
| BPS usage instances (controlled) | identity | `bps_usage_instances.bps_function` | 0.379 | 0.379 | 10 | 10 |
| BPS definitions | identity | `bps_definitions.attributed_source` | 0.185 | 0.185 | 9 | 9 |
| Domain evidence (controlled) | identity | `domain_evidence.domain` | 0.915 | 0.915 | 3 | 3 |
| Biological factors | identity | `biological_factors.factor_label` | 0.234 | 0.345 | 604 | 382 |
| Social factors | identity | `social_factors.factor_label` | 0.228 | 0.268 | 437 | 315 |
| Psychological concepts | identity | `psychological_concepts.concept_label` | 0.399 | 0.445 | 477 | 309 |
| Lifestyle and existential factors | identity | `other_domain_factors.factor_label` | 0.167 | 0.256 | 279 | 205 |
| Concept relations | identity | `concept_relations.source_concept + relation_type + target_concept` | 0.026 | 0.284 | 1052 | 315 |
| Integration claims | identity | `integration_claims.source_factor_label + domains_linked + target_factor_label` | 0.020 | 0.208 | 786 | 255 |
| Theoretical frameworks | identity | `theoretical_frameworks.framework_label` | 0.301 | 0.336 | 252 | 187 |
| Instruments | identity | `instruments.instrument_label` | 0.267 | 0.381 | 280 | 190 |
| Conceptual problems (controlled) | identity | `conceptual_problems.problem_type` | 0.292 | 0.292 | 13 | 13 |
| Key quotes (controlled) | identity | `key_quotes.claim_type` | 0.460 | 0.460 | 8 | 8 |
| Constructs carrying a domain | vocabulary | `domain_evidence.constructs_named` | 0.327 | 0.402 | 937 | 592 |
| Constructs carrying biology | vocabulary | `domain_evidence.constructs_named` | 0.288 | 0.374 | 397 | 258 |
| Constructs carrying psychology | vocabulary | `domain_evidence.constructs_named` | 0.454 | 0.529 | 228 | 149 |
| Constructs carrying the social | vocabulary | `domain_evidence.constructs_named` | 0.242 | 0.305 | 333 | 219 |
| Subdomains touched | vocabulary | `domain_evidence.subdomains_named` | 0.476 | 0.478 | 58 | 54 |
| Biological subdomain anchors | vocabulary | `biological_factors.subdomain_label` | 0.417 | 0.419 | 39 | 37 |
| Social subdomain anchors | vocabulary | `social_factors.subdomain_label` | 0.424 | 0.424 | 17 | 16 |
| Concept families | vocabulary | `psychological_concepts.concept_family` | 0.468 | 0.528 | 40 | 32 |
| Measures behind a construct | vocabulary | `psychological_concepts.measure_named` | 0.370 | 0.431 | 88 | 60 |
| What an instrument measures | vocabulary | `instruments.construct_measured_as_stated` | 0.183 | 0.324 | 316 | 199 |
| Domains a framework spans | vocabulary | `theoretical_frameworks.domains_covered` | 0.715 | 0.717 | 21 | 19 |
| Elements of a BPS definition | vocabulary | `bps_definitions.elements_named` | 0.207 | 0.261 | 154 | 101 |
| Who the model is credited to | vocabulary | `bps_usage_instances.attributed_source` | 0.186 | 0.186 | 22 | 22 |
| Constructs a problem concerns | vocabulary | `conceptual_problems.affected_labels` | 0.126 | 0.185 | 614 | 426 |
| Named mediators and moderators | vocabulary | `integration_claims.mediator_or_moderator` | 0.016 | 0.031 | 222 | 181 |
| Factors at the ends of a link | vocabulary | `integration_claims.source_factor_label + target_factor_label` | 0.029 | 0.164 | 770 | 366 |
| Concepts the review defines | filtered | `psychological_concepts.concept_label` | 0.230 | 0.239 | 63 | 56 |
| Frameworks doing real work | filtered | `theoretical_frameworks.framework_label` | 0.283 | 0.307 | 151 | 115 |
| Mechanistic integration claims | filtered | `integration_claims.source_factor_label + domains_linked + target_factor_label` | 0.021 | 0.184 | 544 | 208 |
| Problems the authors name themselves (controlled) | filtered | `conceptual_problems.problem_type` | 0.127 | 0.127 | 12 | 12 |

Mean over all spaces: 0.287 lexical against 0.345 semantic, over 6205 embedded labels. Over the 28 free-text spaces alone, where the semantic layer has something to merge, the mean is 0.329. The distance between the two columns is the part of the apparent disagreement that was only ever wording.

The controlled spaces are the control condition rather than a result. Where an item is identified by a value from a closed list, the two coders picked from the same menu, the semantic layer has nothing to merge, and the two columns are identical by construction. That they come out identical is the check that the method is not manufacturing agreement wherever it is applied.

Sensitivity to the threshold is in `03_reliability/semantic_overlap_summary.json`, so no reading here depends on where exactly the line is drawn.

## How much of the extraction lands on the project ontology

| Extraction list | Items | Anchored | On the controlled spine | Distinct labels written |
| --- | --- | --- | --- | --- |
| Psychological concepts | 1564 | 99.3% | 88.7% | 578 |
| Biological factors | 1180 | 99.1% | 91.4% | 760 |
| Social factors | 888 | 99.0% | 98.0% | 582 |
| Instruments | 529 | 100.0% | 31.2% | 321 |
| Theoretical frameworks | 513 | 100.0% | 38.4% | 282 |

The controlled share measures the ontology against the literature, not the coder against the ontology. A label the vocabularies do not carry is kept as the review wrote it and listed in `15_off_spine_labels.csv`, which is the working list for extending the vocabularies after expert evaluation.

## Consensus picture of the corpus

- Biological coverage: elaborated 29, mentioned 17, absent 1
- Psychological coverage: elaborated 31, mentioned 15, absent 1
- Social coverage: mentioned 30, elaborated 12, absent 3, minimal 2
- Bio-psych integration: directional 21, mechanistic 15, none 10, descriptive 1
- Psych-social integration: directional 20, none 12, mechanistic 11, descriptive 2, mentioned 2
- Bio-social integration: none 25, directional 10, mechanistic 7, descriptive 3, mentioned 2
- Triadic integration: none 28, descriptive 9, partial 6, mechanistic 4
- BPS typology: multifactorial 24, pseudo_bps 17, true_integrative 5, narrow_despite_label 1
- Coded typology matches the rule-derived typology in 53.2% of codings.

## Where the outputs are

- `01_corpus/`: the paper list with full citations and DOIs, the retrieval log, and the corpus manifest; the article text itself stays local
- `02_model_codings/`: start here. Every coding of the run, long and wide: the codings, the extracted items per category, and everything per provider
- `03_reliability/`: agreement, consensus, overlap, ontology coverage, and quote verification tables
- `04_figures/`: the static review figures
- `05_knowledge_graph/index.html`: the self-contained interactive knowledge graph, from the coding scheme down to the quoted sentence behind one extracted item

The article full texts are licensed material and are intentionally excluded from Git, together with the API call trail behind the run. Everything the coding derived from the text is committed: the coded rows, the extracted items, the verbatim evidence quotes, and every aggregate table.

## How to read this

These numbers describe agreement between three cheap models on a small open-access corpus. They are not a finding about the biopsychosocial literature and they are not a validation of the coding scheme against a human standard. What they do show is which parts of the scheme are specified tightly enough that independent coders converge, and which parts are not, which is the input the expert evaluation needs.
