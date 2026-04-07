PYTHON ?= python

.PHONY: install search search-wos search-psycinfo api-check dedupe prep-screen screen reliability code code-llm prep-stage3 assets report review test run-all

install:
	$(PYTHON) -m pip install -e .

search:
	$(PYTHON) -m bps_review search-pubmed

search-wos:
	$(PYTHON) -m bps_review search-wos

search-psycinfo:
	$(PYTHON) -m bps_review search-psycinfo

api-check:
	$(PYTHON) -m bps_review check-api-access

dedupe:
	$(PYTHON) -m bps_review dedupe

prep-screen:
	$(PYTHON) -m bps_review prepare-screening

screen:
	$(PYTHON) -m bps_review screen-stage1

reliability:
	$(PYTHON) -m bps_review reliability-report

code:
	$(PYTHON) -m bps_review extract-stage2

code-llm:
	$(PYTHON) -m bps_review assist-stage2-llm

prep-stage3:
	$(PYTHON) -m bps_review prepare-stage3

assets:
	$(PYTHON) -m bps_review build-assets

report:
	tectonic --reruns 2 --outdir paper/report paper/report/main.tex

test:
	pytest -q

run-all:
	$(PYTHON) -m bps_review run-all

review: search dedupe prep-screen screen code prep-stage3 reliability assets report
