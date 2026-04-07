from __future__ import annotations

import argparse
import json
from typing import Any, Callable

import pandas as pd

from bps_review.extraction.llm_stage2 import assist_stage2_objectives
from bps_review.extraction.stage2 import extract_stage2
from bps_review.extraction.stage3_prep import prepare_stage3_candidates
from bps_review.reporting.build_assets import build_assets
from bps_review.reporting.semantic_loading import run_semantic_loading
from bps_review.search.access import check_external_api_access
from bps_review.search.dedupe import deduplicate_search_corpus
from bps_review.search.eds import search_eds_psycinfo
from bps_review.search.pubmed import search_pubmed
from bps_review.screening.reliability import stage_reliability_report
from bps_review.search.wos import search_wos_starter
from bps_review.screening.prep import prepare_screening_materials
from bps_review.screening.rules import stage1_screen
from bps_review.utils.paths import project_path


def _safe_step(step_name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        result = func(*args, **kwargs)
        if hasattr(result, "__len__") and not isinstance(result, (str, bytes, dict)):
            return {"step": step_name, "status": "ok", "rows": len(result)}
        if isinstance(result, dict):
            payload = {"step": step_name, "status": "ok"}
            payload.update(result)
            return payload
        return {"step": step_name, "status": "ok"}
    except EnvironmentError as exc:
        return {"step": step_name, "status": "missing_credentials", "detail": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive execution wrapper
        return {"step": step_name, "status": "failed", "detail": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="BPS chronic pain systematic review pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pubmed_parser = subparsers.add_parser("search-pubmed", help="Run the operational PubMed search and normalize records")
    pubmed_parser.add_argument("--query-key", default="pubmed_operational_primary")
    wos_parser = subparsers.add_parser("search-wos", help="Run Web of Science Starter API search when API access is available")
    wos_parser.add_argument("--query-key", default="wos_starter_operational")
    psycinfo_parser = subparsers.add_parser("search-psycinfo", help="Run PsycINFO search through EDS API when access is available")
    psycinfo_parser.add_argument("--query-key", default="psycinfo_eds_operational")

    subparsers.add_parser("check-api-access", help="Check PubMed, Web of Science Starter, and EDS API connectivity")
    subparsers.add_parser("dedupe", help="Combine normalized search files and remove duplicates")
    subparsers.add_parser("prepare-screening", help="Create Rayyan and reliability-preparation files")
    subparsers.add_parser("screen-stage1", help="Run provisional title/abstract screening rules")
    subparsers.add_parser("reliability-report", help="Summarize available inter-rater agreement metrics from double-coded subsets")
    subparsers.add_parser("extract-stage2", help="Run abstract-level Stage 2 coding with structured LLM extraction and rule-based fallback")
    subparsers.add_parser("assist-stage2-llm", help="Run structured LLM Stage 2 coding and write the standalone LLM audit outputs")
    subparsers.add_parser("prepare-stage3", help="Prepare Stage 3 candidate manifest and fetch PMC full texts where available")
    subparsers.add_parser("semantic-loading", help="Run ontology-based semantic loading analysis and vector export")
    subparsers.add_parser("build-assets", help="Generate tables, figures, and manuscript fragments")
    subparsers.add_parser(
        "run-all",
        help="Run all available retrieval and synthesis stages with graceful fallback when optional APIs are unavailable",
    )

    args = parser.parse_args()

    if args.command == "search-pubmed":
        frame = search_pubmed(query_key=args.query_key)
        print(json.dumps({"records": len(frame), "query_key": args.query_key}, indent=2))
    elif args.command == "search-wos":
        frame = search_wos_starter(query_key=args.query_key)
        print(json.dumps({"records": len(frame), "query_key": args.query_key}, indent=2))
    elif args.command == "search-psycinfo":
        frame = search_eds_psycinfo(query_key=args.query_key)
        print(json.dumps({"records": len(frame), "query_key": args.query_key}, indent=2))
    elif args.command == "check-api-access":
        print(json.dumps(check_external_api_access(), indent=2))
    elif args.command == "dedupe":
        frame = deduplicate_search_corpus()
        print(json.dumps({"deduplicated_records": len(frame)}, indent=2))
    elif args.command == "prepare-screening":
        summary = prepare_screening_materials()
        print(json.dumps(summary, indent=2))
    elif args.command == "screen-stage1":
        frame = stage1_screen()
        print(json.dumps({"screened_records": len(frame)}, indent=2))
    elif args.command == "reliability-report":
        print(json.dumps(stage_reliability_report(), indent=2))
    elif args.command == "extract-stage2":
        frame = extract_stage2()
        print(json.dumps({"stage2_records": len(frame)}, indent=2))
    elif args.command == "assist-stage2-llm":
        frame = assist_stage2_objectives()
        print(json.dumps({"stage2_llm_rows": len(frame)}, indent=2))
    elif args.command == "prepare-stage3":
        summary = prepare_stage3_candidates()
        print(json.dumps(summary, indent=2))
    elif args.command == "semantic-loading":
        stage2_path = project_path("review_stages", "04_extraction", "outputs", "stage2_abstract_coding.csv")
        stage2 = pd.read_csv(stage2_path).fillna("") if stage2_path.exists() else pd.DataFrame()
        result = run_semantic_loading(stage2)
        print(
            json.dumps(
                {
                    "status": result.status,
                    "method": result.method,
                    "model": result.model,
                    "records": len(result.record_loadings),
                    "note": result.note,
                },
                indent=2,
            )
        )
    elif args.command == "build-assets":
        summary = build_assets()
        print(json.dumps(summary, indent=2))
    elif args.command == "run-all":
        api_status = check_external_api_access()
        run_report: dict[str, Any] = {
            "api_status": api_status,
            "retrieval": [],
            "pipeline": [],
            "semantic_coding": {},
        }

        run_report["retrieval"].append(_safe_step("pubmed_operational_primary", search_pubmed, "pubmed_operational_primary"))
        run_report["retrieval"].append(_safe_step("pubmed_audit_sensitivity", search_pubmed, "pubmed_audit_sensitivity"))
        run_report["retrieval"].append(_safe_step("wos_starter_operational", search_wos_starter, "wos_starter_operational"))
        run_report["retrieval"].append(_safe_step("psycinfo_eds_operational", search_eds_psycinfo, "psycinfo_eds_operational"))

        deduped = deduplicate_search_corpus()
        prep_summary = prepare_screening_materials()
        stage1 = stage1_screen()
        stage2 = extract_stage2()
        run_report["semantic_coding"] = {
            "step": "extract_stage2_semantic_coding",
            "status": "ok",
            "rows": len(stage2),
            "coding_methods": sorted(stage2["coding_method"].astype(str).unique().tolist()) if "coding_method" in stage2.columns else [],
            "models": sorted(stage2["llm_model"].astype(str).replace("", pd.NA).dropna().unique().tolist()) if "llm_model" in stage2.columns else [],
        }
        stage3_summary = prepare_stage3_candidates()
        reliability = stage_reliability_report()
        asset_summary = build_assets()

        run_report["pipeline"] = [
            {"step": "dedupe", "status": "ok", "rows": len(deduped)},
            {"step": "prepare_screening", "status": "ok", **prep_summary},
            {"step": "screen_stage1", "status": "ok", "rows": len(stage1)},
            {"step": "extract_stage2", "status": "ok", "rows": len(stage2)},
            {"step": "prepare_stage3", "status": "ok", **stage3_summary},
            {"step": "reliability_report", "status": "ok", **reliability},
            {"step": "build_assets", "status": "ok", **asset_summary},
        ]
        print(json.dumps(run_report, indent=2))
