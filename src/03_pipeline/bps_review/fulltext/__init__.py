"""Cross-provider full-text test run for the Stage 3 coding scheme.

Applies the Stage 3 full-text deep coding scheme to the open-access subset of the
records the abstract-level run carried forward, once per model, with the same
three cheap models from three providers. It grades the depth of each domain and
every pairwise and triadic integration, carries a verbatim quote for each of
those judgements, verifies every quote against the source article, and quantifies
categorical agreement, binary presence agreement, and extraction overlap.

    from bps_review.fulltext import run_fulltext_testrun_pipeline
    out = run_fulltext_testrun_pipeline()                   # reuse cached corpus and codings
    out = run_fulltext_testrun_pipeline(force_coding=True)  # re-code every paper via the API
"""

from bps_review.fulltext.config import FULLTEXT_MODELS, MODEL_LABELS
from bps_review.fulltext.pipeline import ensure_corpus, run_fulltext_testrun_pipeline

__all__ = ["run_fulltext_testrun_pipeline", "ensure_corpus", "FULLTEXT_MODELS", "MODEL_LABELS"]
