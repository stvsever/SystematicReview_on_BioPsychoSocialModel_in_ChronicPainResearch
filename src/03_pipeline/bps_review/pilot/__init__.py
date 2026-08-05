"""Cross-provider abstract-level test run for the Stage 2 coding scheme.

Applies the Stage 2 abstract coding scheme to a fresh PubMed sample once per
model, with three cheap models from three providers, and quantifies how much the
providers agree. The models act as independent raters, so the run is an
inter-rater reliability check on the scheme and on the code that applies it.

    from bps_review.pilot import run_abstract_testrun
    out = run_abstract_testrun()                    # reuse cached sample and codings
    out = run_abstract_testrun(force_coding=True)   # re-code every abstract via the API
"""

from bps_review.pilot.config import MODEL_LABELS, TESTRUN_MODELS, TestRunModel
from bps_review.pilot.pipeline import run_abstract_testrun

__all__ = ["run_abstract_testrun", "TESTRUN_MODELS", "MODEL_LABELS", "TestRunModel"]
