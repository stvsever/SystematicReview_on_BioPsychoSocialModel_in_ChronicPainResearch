from __future__ import annotations

"""Run the Stage 3 full-text coding scheme once per model, for every paper.

One paper is one request: full texts are far too long to batch, and a per-paper
call keeps a failure local to a single cell of the (paper x model) grid.

The runner is built for a fast, cheap, resilient run:

* papers are coded concurrently inside a model, and all models run in parallel,
  so the whole grid finishes in one pass;
* every attempt is wrapped in a hard wall-clock timeout, because a provider that
  trickles bytes will otherwise never trip the socket read timeout;
* a response that is not a real coding of this paper is rejected rather than
  repaired, and retried with backoff;
* a paper that never codes is written as an explicit ``coding_failed`` row rather
  than as a fabricated one, so gaps stay visible in the analysis;
* token usage is recorded per call, so the real cost of the run is reportable.

Outputs, all under ``src/05_data/pilot/02_fulltext_level/02_model_codings``:
``by_model/<model>.csv``, ``audit/<model>.jsonl``, ``all_model_codings_long.csv``,
``all_extracted_items_long.csv``, ``run_manifest.json`` and ``run.log``.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bps_review.fulltext.coding.condense import build_coding_text
from bps_review.fulltext.coding.derive import (
    assert_usable_payload,
    item_rows,
    rederive_frame,
    repair_payload,
    serialize_row,
)
from bps_review.fulltext.coding.prompt import SYSTEM_PROMPT, build_prompt
from bps_review.fulltext.coding.schema import FullTextCodingRecord
from bps_review.fulltext.config import (
    FULLTEXT_MODELS,
    HARD_TIMEOUT_SECONDS,
    MAX_MODEL_WORKERS,
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    MAX_WORKERS,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BACKOFF_SECONDS,
    codings_dir,
    items_csv,
    long_codings_csv,
    model_runtime,
)
from bps_review.fulltext.corpus.pmc import load_corpus_records
from bps_review.llm.openrouter import chat_completion_json_with_usage
from bps_review.pilot.config import TestRunModel
from bps_review.utils.io import append_jsonl, ensure_parent, write_csv, write_json


IDENTITY_COLUMNS = ["record_id", "model_order", "model_label", "provider", "model_id"]

# Dedicated pool used only to enforce a hard wall-clock timeout per attempt.
_TIMEOUT_POOL = ThreadPoolExecutor(max_workers=64, thread_name_prefix="bps-ft-timeout")


def code_one_paper(record: dict, model_id: str) -> tuple[FullTextCodingRecord, dict]:
    """Code a single paper with a single model. Raises on an unusable response."""
    coding_text, text_stats = build_coding_text(record)
    prompt = build_prompt(record, coding_text)
    runtime = model_runtime(model_id)
    payload, usage = chat_completion_json_with_usage(
        prompt,
        model=model_id,
        temperature=0.0,
        system_prompt=SYSTEM_PROMPT,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_tokens=runtime.get("max_output_tokens", MAX_OUTPUT_TOKENS),
        reasoning=runtime.get("reasoning"),
    )
    assert_usable_payload(record, payload)
    repaired = repair_payload(record, payload)
    coded = FullTextCodingRecord.model_validate(repaired)
    meta = {
        "text_stats": text_stats,
        "usage": usage,
        "coding_text_chars": len(coding_text),
    }
    return coded, meta


def _attempt(record: dict, model_id: str) -> tuple[FullTextCodingRecord | None, dict]:
    """Code one paper with retries and a hard timeout. Never raises."""
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        started = time.time()
        try:
            future = _TIMEOUT_POOL.submit(code_one_paper, record, model_id)
            coded, meta = future.result(timeout=HARD_TIMEOUT_SECONDS)
            audit = {
                "record_id": record["record_id"],
                "model_id": model_id,
                "status": "ok",
                "attempts": attempt,
                "seconds": round(time.time() - started, 2),
                "coding_text_chars": meta["coding_text_chars"],
                "text_reduced": meta["text_stats"]["reduced"],
                "kept_share": meta["text_stats"]["kept_share"],
                "usage": meta["usage"],
            }
            return coded, audit
        except FutureTimeout:
            last_error = f"hard timeout after {HARD_TIMEOUT_SECONDS:.0f}s"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    return None, {
        "record_id": record["record_id"],
        "model_id": model_id,
        "status": "failed",
        "attempts": MAX_RETRIES,
        "detail": last_error[:400],
    }


def _failed_row(record: dict) -> dict:
    """An explicit failure row. Never a fabricated coding."""
    empty = FullTextCodingRecord(record_id=record["record_id"])
    row = serialize_row(empty, model_id="", coding_method="coding_failed")
    row["coding_rationale"] = "The model returned no usable coding for this paper after all retries."
    return row


def code_corpus_with_model(
    records: list[dict],
    model: TestRunModel,
    max_workers: int = MAX_WORKERS,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Code every paper once with one model. Returns (wide, items, audits)."""
    coded_by_id: dict[str, FullTextCodingRecord] = {}
    audits: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_attempt, record, model.openrouter_id): record["record_id"]
            for record in records
        }
        for future in as_completed(futures):
            coded, audit = future.result()
            audits.append(audit)
            if coded is not None:
                coded_by_id[audit["record_id"]] = coded

    rows: list[dict] = []
    item_records: list[dict] = []
    for record in records:
        coded = coded_by_id.get(record["record_id"])
        if coded is None:
            row = _failed_row(record)
        else:
            row = serialize_row(coded, model_id=model.openrouter_id)
            item_records.extend(item_rows(coded, model.openrouter_id, model.label))
        row["record_id"] = record["record_id"]
        row["model_order"] = model.order
        row["model_label"] = model.label
        row["provider"] = model.provider
        row["model_id"] = model.openrouter_id
        rows.append(row)

    frame = pd.DataFrame(rows)
    lead = [column for column in IDENTITY_COLUMNS if column in frame.columns]
    rest = [column for column in frame.columns if column not in lead]
    return frame[lead + rest].copy(), pd.DataFrame(item_records), audits


def _log(log_path: Path | None, message: str, verbose: bool) -> None:
    stamped = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
    if verbose:
        print(stamped, flush=True)
    if log_path is not None:
        ensure_parent(log_path)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")


def _usage_totals(audits: list[dict]) -> dict[str, float]:
    """Token and dollar totals, taken from what the provider actually billed."""
    prompt_tokens = sum(int((audit.get("usage") or {}).get("prompt_tokens", 0) or 0) for audit in audits)
    completion_tokens = sum(int((audit.get("usage") or {}).get("completion_tokens", 0) or 0) for audit in audits)
    reasoning_tokens = sum(
        int(((audit.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
        for audit in audits
    )
    cost = sum(float((audit.get("usage") or {}).get("cost", 0) or 0) for audit in audits)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": round(cost, 4),
    }


def run_fulltext_testrun(
    models: list[TestRunModel] | None = None,
    records: list[dict] | None = None,
    parallel_models: bool = True,
    per_model_workers: int = MAX_WORKERS,
    verbose: bool = True,
    log_path: Path | None = None,
) -> pd.DataFrame:
    """Code the whole corpus with every model and persist all outputs."""
    models = models or FULLTEXT_MODELS
    records = records if records is not None else load_corpus_records()

    out_dir = codings_dir()
    by_model_dir = out_dir / "by_model"
    audit_dir = out_dir / "audit"
    if log_path is None:
        log_path = out_dir / "run.log"
    ensure_parent(log_path)
    log_path.write_text("", encoding="utf-8")

    _log(log_path, f"Coding {len(records)} full texts x {len(models)} models "
                   f"= {len(records) * len(models)} codings "
                   f"(parallel_models={parallel_models}, workers={per_model_workers})", verbose)

    usage_by_model: dict[str, dict] = {}

    def _one(model: TestRunModel):
        started = time.time()
        _log(log_path, f"start  {model.label}", verbose)
        frame, items, audits = code_corpus_with_model(records, model, max_workers=per_model_workers)
        write_csv(by_model_dir / f"{model.slug}.csv", frame)
        if not items.empty:
            write_csv(by_model_dir / f"{model.slug}_items.csv", items)
        audit_path = audit_dir / f"{model.slug}.jsonl"
        ensure_parent(audit_path)
        audit_path.write_text("", encoding="utf-8")
        for audit in sorted(audits, key=lambda item: item["record_id"]):
            append_jsonl(audit_path, audit)
        usage_by_model[model.label] = _usage_totals(audits)
        ok = int((frame["coding_method"] == "llm_structured").sum())
        n_items = 0 if items.empty else len(items)
        _log(log_path, f"done   {model.label}: {ok}/{len(frame)} coded, {n_items} extracted items, "
                       f"{time.time() - started:.1f}s", verbose)
        return model, frame, items

    if parallel_models:
        with ThreadPoolExecutor(max_workers=min(MAX_MODEL_WORKERS, len(models))) as executor:
            results = list(executor.map(_one, models))
    else:
        results = [_one(model) for model in models]

    frames = {model.label: frame for model, frame, _ in results}
    item_frames = [items for _, _, items in results if not items.empty]

    combined = pd.concat([frames[model.label] for model in models], ignore_index=True)
    write_csv(long_codings_csv(), combined)
    all_items = pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame()
    write_csv(items_csv(), all_items)

    total_usage = {
        key: round(sum(usage[key] for usage in usage_by_model.values()), 4)
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "cost_usd")
    }
    manifest = {
        "n_papers": len(records),
        "n_models": len(models),
        "n_codings": int(len(combined)),
        "n_extracted_items": int(len(all_items)),
        "models": [
            {"order": m.order, "label": m.label, "openrouter_id": m.openrouter_id, "provider": m.provider}
            for m in models
        ],
        "coding_method_counts": combined["coding_method"].value_counts().to_dict(),
        "max_workers": per_model_workers,
        "model_runtime": {m.label: model_runtime(m.openrouter_id) for m in models},
        "token_usage_by_model": usage_by_model,
        "token_usage_total": total_usage,
    }
    write_json(out_dir / "run_manifest.json", manifest)
    _log(log_path, f"ALL DONE: {len(combined)} codings, {len(all_items)} extracted items, "
                   f"{total_usage['total_tokens']} tokens, ${total_usage['cost_usd']:.3f} "
                   f"| methods={manifest['coding_method_counts']}", verbose)
    return combined


def load_or_run(force: bool = False) -> pd.DataFrame:
    """Load the cached long coding table if present, otherwise run the test run.

    The derived columns are recomputed from the coded content on load, so a
    cached run always reports the eligibility, yield, priority, and presence
    flags of the current rules rather than the ones in force when it was written.
    """
    path = long_codings_csv()
    if path.exists() and not force:
        return rederive_frame(pd.read_csv(path).fillna(""))
    return run_fulltext_testrun()


def load_items() -> pd.DataFrame:
    """Load the cached item-level extraction table."""
    path = items_csv()
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")
