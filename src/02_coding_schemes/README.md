# Coding Scheme Dossiers

> **Status: DRAFT FOR EXPERT EVALUATION.** These coding schemes are a working draft circulated for expert evaluation. They have not been applied to a final review corpus. The current manuscript is a test run that exercised an earlier, coarser generation of these schemes. The workflow itself has since been validated end to end in two cross-provider test runs, in which three large language models from three different providers applied the abstract-level and the full-text scheme independently and their agreement was quantified per coded field. The full run on the review corpus is deliberately held until this evaluation is complete.

This directory contains one communication-ready dossier for each distinct coding scheme in the systematic review workflow. Every scheme is provided in three synchronized surfaces:

- an **interactive HTML** evaluation surface with per-section feedback boxes and JSON export or import,
- a compiled **PDF** for sharing and printing,
- an explanatory **README**.

Open [`index.html`](index.html) for the aggregated dashboard: a pipeline map, links to every scheme, and a console that merges exported feedback files into one consolidated view.

## Why these schemes are circulated now

The current manuscript is a **test run** (it exercised an earlier, coarser generation of these schemes in the Python workflow with an LLM, gemini-2.5-flash). This release raises their semantic quality and resolution (operational anchors for every value, positive and negative indicators, explicit boundary rules, worked examples from the real corpus, a comprehensive psychological concept taxonomy, and clearly labelled proposed refinements). Nothing here has been applied to a final corpus yet. The schemes are being circulated for expert evaluation first; the pipeline will be re-run only after sign-off.

## Scope: two reviews, one shared instrument

The team plans two parallel reviews: one on musculoskeletal chronic pain and one on neuropathic chronic pain. These coding schemes are a single uniform instrument for both. The pain-condition family (musculoskeletal or neuropathic) is the varying input that selects which records enter each review; the coding logic, value vocabularies, and anchors are shared so the two reviews stay directly comparable. Uniformity is kept wherever it is defensible. It is relaxed in exactly two places, where forcing it would distort the science: the routing flags that assign a record to a review (a musculoskeletal flag and a parallel neuropathic flag), and the biological subdomain ontology, which carries a shared core plus a musculoskeletal extension and a neuropathic extension because the biological mechanisms of the two pain families genuinely differ. The psychological and social layers, the integration ladder, the typology, and the concept taxonomy stay identical across both tracks.

## Inventory

### Scheme 1: Stage 1 Screening and Eligibility Decision Scheme

- **Stage:** Stage 1
- **Purpose:** This scheme operationalizes title and abstract screening after search and deduplication. Its function is to decide whether a record enters the review corpus for downstream coding, using a human-validatable rule set centred on biopsychosocial language, chronic pain relevance, review design, and population eligibility.
- **Files:** [`scheme_1/scheme_1.html`](scheme_1/scheme_1.html), [`scheme_1/scheme_1.pdf`](scheme_1/scheme_1.pdf), [`scheme_1/README.md`](scheme_1/README.md)

### Scheme 2: Stage 2 Abstract-Level Structured Coding Scheme

- **Stage:** Stage 2
- **Purpose:** This scheme standardizes abstract-level extraction for all eligible chronic pain reviews. It is the main corpus-wide coding layer used to describe review characteristics, classify the function of biopsychosocial language, detect biological, psychological, and social content, flag conceptual problems, and generate a provisional biopsychosocial typology for downstream synthesis.
- **Files:** [`scheme_2/scheme_2.html`](scheme_2/scheme_2.html), [`scheme_2/scheme_2.pdf`](scheme_2/scheme_2.pdf), [`scheme_2/README.md`](scheme_2/README.md)

### Scheme 3: Stage 3 Full-Text Deep Coding Scheme

- **Stage:** Stage 3
- **Purpose:** This scheme is the full-text deep coding framework for Stage 3 candidate reviews. It is applied as one uniform instrument to both planned reviews: the musculoskeletal chronic pain review and the neuropathic chronic pain review. The pain-condition family is the varying input that decides which records each review reads; the coding fields, value vocabularies, and anchors are identical across both tracks so the two reviews stay directly comparable.
- **Files:** [`scheme_3/scheme_3.html`](scheme_3/scheme_3.html), [`scheme_3/scheme_3.pdf`](scheme_3/scheme_3.pdf), [`scheme_3/README.md`](scheme_3/README.md)

### Scheme 4: Stage 3 Retrieval and Manual Relevance Triage Scheme

- **Stage:** Stage 3
- **Purpose:** This scheme governs the transition from Stage 2 abstract coding to Stage 3 full-text work. It standardizes which candidate reviews need manual retrieval, which records require manual relevance adjudication, and how retrieval status, risk signals, and reviewer decisions are recorded before deep coding begins.
- **Files:** [`scheme_4/scheme_4.html`](scheme_4/scheme_4.html), [`scheme_4/scheme_4.pdf`](scheme_4/scheme_4.pdf), [`scheme_4/README.md`](scheme_4/README.md)

### Scheme 5: Psychological Concept Clustering and Framework Mapping Scheme

- **Stage:** Cross-stage
- **Purpose:** This scheme standardizes higher-order concept mapping after concept detection. It groups extracted psychological concepts from chronic pain review records into interpretable families and links them to likely theoretical frameworks. The goal is cross-record comparability when the raw concepts are heterogeneous, overlapping, or variably named.
- **Files:** [`scheme_5/scheme_5.html`](scheme_5/scheme_5.html), [`scheme_5/scheme_5.pdf`](scheme_5/scheme_5.pdf), [`scheme_5/README.md`](scheme_5/README.md)

### Scheme 6: BPS Ontology and Semantic Loading Benchmark Scheme

- **Stage:** Synthesis
- **Purpose:** This scheme supplies the ontology scaffold used to quantify semantic emphasis across biological, psychological, and social axes. It is not a manual adjudication form, but it is an operational text-classification framework: it standardizes the domain and subdomain prompts against which review records are embedded and compared.
- **Files:** [`scheme_6/scheme_6.html`](scheme_6/scheme_6.html), [`scheme_6/scheme_6.pdf`](scheme_6/scheme_6.pdf), [`scheme_6/README.md`](scheme_6/README.md)

## Evaluation workflow

1. Open [`index.html`](index.html) and read the status note.
2. Open each scheme, read the anchored definitions and the sections marked **Proposed**, and record a verdict and comments per section.
3. Export one JSON file per scheme (the button is in the top bar).
4. Load every exported file into the console on `index.html` to see a consolidated view and export a single bundle for the team.

## Regeneration

All surfaces are generated from `_build/content.py` by `_build/build.py`. Do not hand-edit the generated `.tex`, `.html`, or `README.md` files; edit the content model and rebuild:

```bash
cd src/coding_schemes/_build
python3 build.py
```

## Notes

- The dossiers prioritize the operational implementation used by the pipeline. Where protocol prose, codebooks, and generated outputs diverge, the dossiers state that explicitly.
- The underlying source files and outputs remain in their original project paths so reviewers can inspect the raw materials directly.
- No em dashes are used in any generated file.
