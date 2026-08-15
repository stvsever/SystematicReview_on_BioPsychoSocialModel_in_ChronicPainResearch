# Full-text test run: cross-provider reliability and evidence integrity

The Stage 3 full-text coding scheme was applied to 47 open-access review articles by 3 large language models from 3 different providers, which gives 141 independent codings and 4080 extracted items. The corpus is the open-access subset of the records the abstract-level run carried forward, so the two stages are one chain rather than two separate exercises.

## What was coded

- Corpus: 47 full texts retrieved from PubMed Central.
- Models: `deepseek/deepseek-v4-flash` (DeepSeek), `nex-agi/nex-n2-mini` (Nex AGI), `poolside/laguna-xs-2.1` (Poolside)
- Coding method counts: {'llm_structured': 141}
- Token usage: 3,155,248 tokens for $0.264

## Headline agreement

| Metric | Value |
| --- | --- |
| Mean Fleiss' kappa over all coded fields | 0.178 |
| Mean Krippendorff's alpha | 0.184 |
| Mean observed agreement | 65.9% |
| Categorical fields, mean kappa | 0.217 |
| Binary presence fields, mean kappa | 0.119 |
| Coverage ladder, within one rung | 92.2% |
| Integration ladder, within one rung | 67.6% |
| Papers where all models agree on eligibility | 38 of 47 |

## Per field

| Field | Fleiss' kappa | Krippendorff alpha | Observed | Within one rung | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Source type | 0.833 | 0.834 | 89.4% | n/a | Almost perfect |
| Psychological coverage | 0.551 | 0.554 | 78.0% | 97.9% | Moderate |
| Defined concepts present | 0.388 | 0.392 | 70.2% | n/a | Fair |
| Overall balance | 0.372 | 0.377 | 51.1% | n/a | Fair |
| Frameworks present | 0.337 | 0.342 | 92.9% | n/a | Fair |
| Concept definitions | 0.321 | 0.326 | 67.4% | 100.0% | Fair |
| Biological coverage | 0.312 | 0.317 | 63.8% | 95.0% | Fair |
| Social coverage | 0.273 | 0.278 | 48.2% | 83.7% | Fair |
| Triadic claim present | 0.225 | 0.231 | 64.5% | n/a | Fair |
| Social evidence present | 0.175 | 0.181 | 74.5% | n/a | Slight |
| Review track | 0.116 | 0.122 | 52.5% | n/a | Slight |
| Triadic integration | 0.071 | 0.078 | 34.0% | 81.6% | Slight |
| Full-text eligibility | 0.063 | 0.070 | 85.1% | n/a | Slight |
| Psych-social integration | 0.062 | 0.069 | 26.2% | 64.5% | Slight |
| Bio-psych integration | 0.039 | 0.046 | 31.9% | 71.6% | Slight |
| Synthesis priority | 0.027 | 0.033 | 53.2% | n/a | Slight |
| Integration evidence present | 0.014 | 0.021 | 85.8% | n/a | Slight |
| BPS typology | 0.000 | 0.008 | 31.9% | n/a | Slight |
| Bio-social integration | -0.006 | 0.001 | 29.1% | 52.5% | Poor |
| Conceptual problems present | -0.007 | 0.000 | 98.6% | n/a | Poor |
| Psychological concepts present | -0.014 | -0.007 | 97.2% | n/a | Poor |
| Psychological evidence present | -0.014 | -0.007 | 97.2% | n/a | Poor |
| Biological evidence present | -0.037 | -0.029 | 92.9% | n/a | Poor |

## Is the evidence real?

| Model | Quotes | Exact | Near | Verified | Mean words |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash | 703 | 90.8% | 8.4% | 99.1% | 23.5 |
| Nex-N2-Mini | 1735 | 97.5% | 2.4% | 99.9% | 28.0 |
| Laguna-XS-2.1 | 868 | 78.7% | 19.9% | 98.6% | 33.3 |

Every verbatim quote was matched back against the article it came from. 99.4% of 3306 checkable quotes were found in the source text, literally or with at most minor differences.

Of the 356 domain links graded above 'mentioned', 95.2% carry a quoted claim for exactly that pair. This is the check that decides whether the integration ladder can be trusted: a graded link with no passage behind it is a judgement the review cannot audit.

## How the models behave

| Model | Include | Core priority | True integrative | Any triadic | Mean integration index | Mean items |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash | 83.0% | 44.7% | 4.3% | 51.1% | 0.27 | 20.9 |
| Nex-N2-Mini | 100.0% | 100.0% | 51.1% | 97.9% | 0.68 | 41.4 |
| Laguna-XS-2.1 | 91.5% | 55.3% | 6.4% | 53.2% | 0.35 | 24.5 |

## Open extraction lists

Every extraction list the run carries is compared, by the identity of its items.
Recomputed from the stored codings by the current code: the project vocabularies
have been extended since this run was coded, so more spellings now normalize onto
one label than when the run was first analysed.

| List | Mean pairwise Jaccard | Distinct labels | Papers with a shared label |
| --- | --- | --- | --- |
| Domain evidence (controlled) | 0.878 | 3 | 47 |
| Psychological concepts | 0.414 | 424 | 45 |
| Integration claims | 0.585 | 4 | 37 |
| Theoretical frameworks | 0.341 | 245 | 34 |
| Conceptual problems (controlled) | 0.296 | 9 | 27 |
| Key quotes (controlled) | 0.433 | 7 | 41 |

## The same extraction, measured by meaning

The Jaccard above asks whether two providers wrote the same string, and it asks it only of the item identities. Both halves of that are too narrow. A scheme 3 item is not a label but a small record, and several of its fields are open vocabularies in their own right: which constructs a coder says carry the biological domain, which measure a construct is tied to, which components a definition of the model lists, which constructs a conceptual problem concerns. Each is a place where two coders can read a paper the same way and write different words.

Every one of those vocabularies is compared here. The scheme declares 33 comparison spaces and this run supports 13 of them (6 item identities, 4 vocabularies read from inside the items, 3 restricted to the items that carry weight). The rest belong to extraction lists this run does not carry, because it was coded on the previous generation of scheme 3. Every label is embedded once with `openai/text-embedding-3-large`, and two labels count as the same concept at a cosine of 0.65, which turns the overlap into a soft Jaccard on the same 0 to 1 scale. Both columns are computed in one pass over the same label sets, so reading one against the other compares two ways of measuring one thing rather than two instruments.

| Comparison space | Layer | Read from | Lexical | Semantic | Labels | Concepts |
| --- | --- | --- | --- | --- | --- | --- |
| Domain evidence (controlled) | identity | `domain_evidence.domain` | 0.878 | 0.878 | 3 | 3 |
| Psychological concepts | identity | `psychological_concepts.concept_label` | 0.414 | 0.463 | 424 | 281 |
| Integration claims | identity | `integration_claims.source_factor_label + domains_linked + target_factor_label` | 0.585 | 0.585 | 4 | 3 |
| Theoretical frameworks | identity | `theoretical_frameworks.framework_label` | 0.341 | 0.436 | 245 | 170 |
| Conceptual problems (controlled) | identity | `conceptual_problems.problem_type` | 0.296 | 0.296 | 9 | 9 |
| Key quotes (controlled) | identity | `key_quotes.claim_type` | 0.433 | 0.433 | 7 | 7 |
| Constructs carrying a domain | vocabulary | `domain_evidence.constructs_named` | 0.307 | 0.383 | 1139 | 727 |
| Constructs carrying biology | vocabulary | `domain_evidence.constructs_named` | 0.255 | 0.357 | 507 | 332 |
| Constructs carrying psychology | vocabulary | `domain_evidence.constructs_named` | 0.406 | 0.467 | 324 | 218 |
| Constructs carrying the social | vocabulary | `domain_evidence.constructs_named` | 0.237 | 0.285 | 333 | 222 |
| Concepts the review defines | filtered | `psychological_concepts.concept_label` | 0.272 | 0.289 | 144 | 122 |
| Frameworks doing real work | filtered | `theoretical_frameworks.framework_label` | 0.280 | 0.349 | 165 | 124 |
| Mechanistic integration claims | filtered | `integration_claims.source_factor_label + domains_linked + target_factor_label` | 0.430 | 0.446 | 4 | 3 |

Mean over all spaces: 0.395 lexical against 0.436 semantic, over 1574 embedded labels. Over the 10 free-text spaces alone, where the semantic layer has something to merge, the mean is 0.406. The distance between the two columns is the part of the apparent disagreement that was only ever wording.

The largest single vocabulary in the run was invisible to every metric until now. `domain_evidence.constructs_named` carries 2,522 named constructs, more than the psychological-concept list itself, and it is where the coders say what actually carries each domain. Split by domain, the three providers converge least on the social (0.285) and most on the psychological (0.467), which is the same asymmetry the coverage ladder shows, now measured on the content rather than on the grade.

The controlled spaces are the control condition rather than a result. Where an item is identified by a value from a closed list, the two coders picked from the same menu, the semantic layer has nothing to merge, and the two columns are identical by construction. That they come out identical is the check that the method is not manufacturing agreement wherever it is applied.

Sensitivity to the threshold is in `03_reliability/semantic_overlap_summary.json`: the mean moves by 0.02 across cosine 0.60 to 0.75, so no reading here depends on where exactly the line is drawn.

## Review surfaces

- `01_corpus/`: the corpus inventory, identifiers, and retrieval log. The article texts stay local and are never pushed.
- `02_model_codings/`: every article by provider coding, the item-level table, the raw audit trail, and the usage manifest.
- `03_reliability/`: agreement, consensus, lexical and semantic overlap, and quote verification.
- `04_figures/`: the five static review figures.
- `05_knowledge_graph/index.html`: the self-contained interactive knowledge graph over this run, from the coding scheme down to the quoted sentence behind one extracted item. Open it in a desktop browser; it needs no server.

## Consensus picture of the corpus

- Biological coverage: elaborated 30, mentioned 17
- Psychological coverage: elaborated 30, mentioned 17
- Social coverage: mentioned 22, elaborated 15, absent 9, minimal 1
- Bio-psych integration: mechanistic 21, directional 19, none 5, descriptive 2
- Psych-social integration: mechanistic 13, none 13, directional 12, descriptive 8, mentioned 1
- Bio-social integration: none 28, mechanistic 9, directional 7, mentioned 2, descriptive 1
- Triadic integration: none 19, descriptive 12, partial 10, mechanistic 6
- BPS typology: pseudo_bps 18, true_integrative 17, multifactorial 12
- Coded typology matches the rule-derived typology in 60.3% of codings.

## How to read this

These numbers describe agreement between three cheap models on a small open-access corpus. They are not a finding about the biopsychosocial literature and they are not a validation of the coding scheme against a human standard. What they do show is which parts of the scheme are specified tightly enough that independent coders converge, and which parts are not, which is the input the expert evaluation needs.
