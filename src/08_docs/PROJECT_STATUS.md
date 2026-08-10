# Project Status

Last updated: 2026-08-10

## Hierarchical goals

1. Preserve the OSF-registered review logic in machine-readable form.
2. Build a reproducible search to reporting pipeline that can be rerun and audited.
3. Keep automated assistance clearly separated from adjudicated review decisions.
4. Validate the coding schemes with experts before the pipeline is re-run on the full corpus.
5. Produce manuscript-ready assets and a compiled PDF from the current repository state.

## Where the project stands

The manuscript in `paper/report/main.pdf` is a **test run on an earlier, coarser generation of
the coding schemes**. The schemes have since been revised for higher semantic resolution and are
now circulated for expert evaluation. The pipeline is deliberately held until that evaluation is
complete.

Two cross-provider test runs were executed to validate the workflow itself before that re-run.

### Coding schemes (expert evaluation package)

- Six dossiers regenerated from one source of truth (`src/02_coding_schemes/_build/content.py`).
- Each dossier now carries the purpose of the scheme and the scheme itself, and nothing else.
  Worked examples, reliability architecture, documented divergences, test-run corpus counts, and
  the proposed-refinement sections were removed.
- Every coded field carries its own expert-feedback box, every section carries a section-level
  box, and the overall assessment is the last block on the page.
- HTML, PDF, and README are in sync for all six schemes; the em-dash check passes.

### Scheme 3 raised to extraction resolution (2026-08-10)

The full-text scheme was rebuilt so that it names what a review carries rather than only grading
how much of it there is. The graded ladders are unchanged, which keeps the earlier readings
comparable; everything below is additional.

- Thirteen structured extraction lists (was six), seven open free-text lists, 82 fields inside
  the list items, and a ceiling of 116 extracted items per coding.
- New in this revision: per-passage coding of what the biopsychosocial label does and how the
  model is defined (RQ1); a named-factor inventory per domain with ontology subdomain, role, and
  level (RQ2); relations between psychological constructs, hierarchical and semantic, as edges
  (RQ3, registered and previously uncoded); instruments with what they are said to measure;
  conceptual problems with their scope and the constructs they concern (SQ1); lifestyle and
  spiritual or existential coverage, which the registration names alongside the triad.
- Integration claims now name the source factor and the target factor, so the corpus yields a
  map of what this literature says connects to what rather than a count of domain pairs.
- A vocabulary module holds the preferred labels (the Scheme 6 subdomain ontology, the Scheme 5
  concept families, frameworks, instruments, pain conditions, attributed sources). Mapping is
  conservative, a mapped label never replaces the review's own wording, and off-spine terms are
  kept verbatim and reported as the working list for extending the ontology.
- The operational codebook CSV and the human coding form are now generated from the schema, so
  the dossier, the codebook, the form, and the running code cannot drift apart. Two tests assert
  that the dossier documents every coded field and every item subfield.
- The pipeline integrates the change without a re-run: derived columns are recomputed on load,
  and the notebook reports the new ontology-coverage tables when it is next executed.

### Abstract-level test run (scheme 2)

- 100 records drawn from the operational PubMed query, most recent first, each with a usable
  abstract; 60 carry a PubMed Central id.
- 3 cheap models from 3 providers x 100 records = 300 codings from 300 API calls, no failures,
  1.38 million tokens, 0.13 US dollars.
- Mean Fleiss' kappa 0.60 across 12 coded fields. Substantial agreement on the descriptive
  fields (ICD-11 pain family 0.76, review type 0.71); fair agreement on the two fields carrying
  the review's own argument (BPS function 0.35, provisional typology 0.33).
- 88 of 100 records carried forward by majority vote; 91 of 100 unanimous on that call.
- Outputs under `src/05_data/pilot/01_abstract_level/`.

### Full-text test run (scheme 3)

Executed on the previous generation of scheme 3, before the extraction layer above was added.
The counts below describe that earlier instrument.

- The open-access subset of the abstract-level candidate set: 47 full texts retrieved and parsed
  from PubMed Central out of 88 candidates (53 had a PMC id).
- The same 3 models x 47 papers = 141 codings, with graded coverage and integration ladders and
  a verbatim quote behind every graded judgement.
- Every extracted quote is matched back against its source article, and every domain link graded
  above `mentioned` is checked for a quoted claim supporting exactly that pair.
- Outputs under `src/05_data/pilot/02_fulltext_level/`.

### A defect the test run exposed and fixed

The Stage 2 prompt listed value vocabularies but never named the fields to return. Models
answered with the subset whose option lists they could see and omitted `bio_mentioned`,
`psych_mentioned`, `social_mentioned`, `musculoskeletal_flag`, and
`quality_assessment_reported`. The deterministic repair layer filled those five from the lexical
rule-based coder, so the output looked like a complete structured coding while being keyword
matching for the review's core RQ2 variables. The prompt now carries an explicit field
specification and output contract, and a test asserts that the specification and the validated
schema cannot drift apart again.

## Repository state

- `src/` is grouped into numbered sections (`01_protocol` through `09_artifacts`); the mapping
  from a section name to its directory lives only in `bps_review/utils/paths.py`.
- Protocol metadata in `src/03_pipeline/config/protocol.yaml`, search strings in
  `search_queries.yaml`.
- PubMed direct API access confirmed. Web of Science Starter and EDS remain
  `missing_credentials`; those sources are imported manually.
- Combined normalized corpus 3874 records; deduplicated 3372; provisional Stage 1 includes 109;
  Stage 2 abstract-coded 109; Stage 3 candidate set 87.
- Tables and figures generated under `paper/assets/`; manuscript compiled to
  `paper/report/main.pdf`.
- Test suite passing (33 tests, including offline coverage of the agreement primitives, the
  repair and derivation layer, quote verification, prompt-schema consistency, and the
  agreement between the expert dossier and the coded schema).

## Open constraints

- Full automated access to Web of Science and PsycINFO is not assumed in this environment.
- Stage 3 full-text coding depends on open-access availability or manually supplied full texts.
  In the test run, 47 of 88 candidates yielded a retrievable open-access full text.
- Any protocol deviations must be written to
  `src/06_review_stages/01_protocol/outputs/deviations.md`.
- The test-run numbers describe three cheap models and must not be read as findings about the
  literature.

## Next steps

1. Collect expert feedback on the six coding schemes through the HTML surfaces.
2. Apply the accepted revisions to `src/02_coding_schemes/_build/content.py` and to the coding
   implementation.
3. Swap `TESTRUN_MODELS` for state-of-the-art models and re-run both stages on the full corpus.
4. Regenerate the manuscript from the new outputs.
