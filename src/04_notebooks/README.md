# Notebooks

Two executed end-to-end notebooks, one per coding level, plus a reserved slot for the synthesis
level. Both are stored with their outputs, so they can be read without running anything.

| Notebook | What it runs | Scale | Cost |
| --- | --- | --- | --- |
| [`01_abstractlevel_testrun.ipynb`](01_abstractlevel_testrun.ipynb) | Stage 2 abstract coding scheme, three providers | 100 abstracts x 3 models = 300 codings | 0.13 USD |
| [`02_fulltextlevel_testrun.ipynb`](02_fulltextlevel_testrun.ipynb) | Stage 3 full-text deep coding scheme, same three providers | 47 full texts x 3 models = 141 codings, 9,400 extracted items | 0.34 USD |
| [`03_synthesislevel_testrun.ipynb`](03_synthesislevel_testrun.ipynb) | Reserved until the coding schemes are signed off | | |

They are one chain: notebook 01 exports the consensus candidate set, notebook 02 retrieves the
open-access full texts of that set and codes them. Both write their outputs to
[`src/05_test_runs/`](../05_test_runs/); the full-text run is
[`src/05_test_runs/tests/02_pilot_fulltext/`](../05_test_runs/tests/02_pilot_fulltext/).

## Running them

Both notebooks load cached results by default and re-render every table and figure from them, so
a top-to-bottom run costs nothing and takes under a minute. To call the API again, set
`FORCE_RERUN = True` in the coding section, or use the CLI:

```bash
python -m bps_review run-abstract-testrun --force-corpus --force-coding
python -m bps_review build-fulltext-corpus
python -m bps_review run-fulltext-testrun --force-coding
```

An `OPENROUTER_API_KEY` in `.env` is required for a re-run; `NCBI_EMAIL` and `NCBI_API_KEY` are
recommended for the PubMed and PubMed Central retrieval.

## Swapping the models

The three test-run models are cheap stand-ins. They are defined once, in
`src/03_pipeline/bps_review/pilot/config.py`, and reused by both stages:

```python
TESTRUN_MODELS = [
    TestRunModel(1, "DeepSeek-V4-Flash", "01_deepseek_v4_flash", "deepseek/deepseek-v4-flash", "DeepSeek"),
    TestRunModel(2, "Nex-N2-Mini", "02_nex_n2_mini", "nex-agi/nex-n2-mini", "Nex AGI"),
    TestRunModel(3, "Laguna-XS-2.1", "03_laguna_xs_2_1", "poolside/laguna-xs-2.1", "Poolside"),
]
```

Replacing that list with state-of-the-art models is the whole model swap. A model whose endpoint
needs different reasoning or completion settings gets an entry in `MODEL_RUNTIME` in
`bps_review/fulltext/config.py`.
