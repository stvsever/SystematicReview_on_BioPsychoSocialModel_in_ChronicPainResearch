from __future__ import annotations

"""Run the Stage 2 abstract coding scheme once per model, for every record.

Every model codes every abstract through exactly the same structured-JSON path
the main pipeline uses (``bps_review.extraction.llm_stage2._parse_batch``), so
the coding logic under test is identical across providers and identical to the
one the review will run. The runner adds:

* one record per request, coded concurrently through a ThreadPoolExecutor, with
  all models running in parallel, so the whole grid finishes in one pass;
* a hard wall-clock timeout per attempt, because a provider that trickles bytes
  never trips the socket read timeout;
* retries with backoff; a record that still never codes falls back to the
  deterministic rule-based coder and is flagged as such, so the field grid stays
  complete and every fallback stays visible in the analysis;
* per-call token usage, so the real cost of the run is reportable rather than
  estimated.

Outputs, all under ``src/05_data/pilot/01_abstract_level/02_model_codings``:
``by_model/<model>.csv``, ``audit/<model>.jsonl``, ``all_model_codings_long.csv``,
``run_manifest.json`` and ``run.log``.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bps_review.extraction.llm_stage2 import _parse_batch
from bps_review.pilot.coding.data import load_testrun_records
from bps_review.pilot.config import (
    BATCH_SIZE,
    HARD_TIMEOUT_SECONDS,
    MAX_MODEL_WORKERS,
    MAX_RETRIES,
    MAX_WORKERS,
    RETRY_BACKOFF_SECONDS,
    TESTRUN_MODELS,
    TestRunModel,
    codings_dir,
    long_codings_csv,
)
from bps_review.utils.io import append_jsonl, ensure_parent, write_csv, write_json
from bps_review.utils.paths import PROJECT_ROOT


IDENTITY_COLUMNS = ["record_id", "model_order", "model_label", "provider", "model_id"]

# Dedicated pool used only to enforce a hard wall-clock timeout around each
# request attempt. A request that trickles forever is abandoned here (its worker
# thread is left to die on its own) and retried.
_TIMEOUT_POOL = ThreadPoolExecutor(max_workers=64, thread_name_prefix="bps-timeout")


def _chunks(items: list, size: int) -> list[list]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _attempt_batch(batch_index: int, batch: list[dict], model_id: str) -> tuple[list[dict], dict]:
    """Code one request payload with retries and a hard timeout. Never raises."""
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        started = time.time()
        try:
            future = _TIMEOUT_POOL.submit(_parse_batch, batch_index, batch, model_id)
            rows, audit_payload = future.result(timeout=HARD_TIMEOUT_SECONDS)
            audit = {
                "batch_index": batch_index,
                "model_id": model_id,
                "record_ids": [record["record_id"] for record in batch],
                "status": "ok",
                "attempts": attempt,
                "seconds": round(time.time() - started, 2),
                "usage": audit_payload.get("usage", {}),
            }
            return rows, audit
        except FutureTimeout:
            last_error = f"hard timeout after {HARD_TIMEOUT_SECONDS:.0f}s"
        except Exception as exc:  # transient API or parse failure
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    audit = {
        "batch_index": batch_index,
        "model_id": model_id,
        "record_ids": [record["record_id"] for record in batch],
        "status": "failed",
        "attempts": MAX_RETRIES,
        "detail": last_error[:400],
    }
    return [], audit


def _rule_based_row(record: dict) -> dict:
    """The deterministic fallback coding, used only when the API never answers."""
    from bps_review.extraction.llm_stage2 import _batch_fallback_rows

    row = _batch_fallback_rows([record], "no usable response after all retries")[0]
    row["coding_method"] = "rule_based_fallback"
    row["llm_model"] = ""
    return row


def code_records_with_model(
    records: list[dict],
    model: TestRunModel,
    max_workers: int = MAX_WORKERS,
    batch_size: int = BATCH_SIZE,
) -> tuple[pd.DataFrame, list[dict]]:
    """Code every record once with a single model, resilient to failures."""
    batches = _chunks(records, batch_size)
    coded_by_id: dict[str, dict] = {}
    audits: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_attempt_batch, index, batch, model.openrouter_id): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            rows, audit = future.result()
            audits.append(audit)
            for row in rows:
                coded_by_id[row["record_id"]] = row

    # Anything that never coded: deterministic fallback so the grid stays full.
    fallbacks = 0
    for record in records:
        if record["record_id"] not in coded_by_id:
            coded_by_id[record["record_id"]] = _rule_based_row(record)
            fallbacks += 1
    if fallbacks:
        audits.append({"model_id": model.openrouter_id, "status": "rule_based_fallback", "count": fallbacks})

    ordered_rows = []
    for record in records:
        row = dict(coded_by_id[record["record_id"]])
        row["record_id"] = record["record_id"]
        row["model_order"] = model.order
        row["model_label"] = model.label
        row["provider"] = model.provider
        row["model_id"] = model.openrouter_id
        ordered_rows.append(row)

    frame = pd.DataFrame(ordered_rows)
    lead = [column for column in IDENTITY_COLUMNS if column in frame.columns]
    rest = [column for column in frame.columns if column not in lead]
    return frame[lead + rest].copy(), audits


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
    cost = sum(float((audit.get("usage") or {}).get("cost", 0) or 0) for audit in audits)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": round(cost, 4),
    }


def run_testrun(
    models: list[TestRunModel] | None = None,
    records: list[dict] | None = None,
    parallel_models: bool = True,
    per_model_workers: int = MAX_WORKERS,
    verbose: bool = True,
    log_path: Path | None = None,
) -> pd.DataFrame:
    """Run every model over every abstract and persist the coded outputs.

    Each model's coded CSV and audit trail are written as soon as that model
    finishes, so partial progress is never lost. Returns the combined long
    dataframe (one row per abstract per model).
    """
    models = models or TESTRUN_MODELS
    records = records if records is not None else load_testrun_records()

    out_dir = codings_dir()
    by_model_dir = out_dir / "by_model"
    audit_dir = out_dir / "audit"
    if log_path is None:
        log_path = out_dir / "run.log"
    ensure_parent(log_path)
    log_path.write_text("", encoding="utf-8")

    n_calls = len(_chunks(records, BATCH_SIZE)) * len(models)
    _log(log_path, f"Coding {len(records)} abstracts x {len(models)} models "
                   f"= {len(records) * len(models)} codings in {n_calls} API calls "
                   f"(parallel_models={parallel_models}, workers={per_model_workers})", verbose)

    usage_by_model: dict[str, dict] = {}

    def _one(model: TestRunModel) -> tuple[TestRunModel, pd.DataFrame]:
        started = time.time()
        _log(log_path, f"start  {model.label}", verbose)
        frame, audits = code_records_with_model(records, model, max_workers=per_model_workers)
        write_csv(by_model_dir / f"{model.slug}.csv", frame)
        audit_path = audit_dir / f"{model.slug}.jsonl"
        ensure_parent(audit_path)
        audit_path.write_text("", encoding="utf-8")
        for audit in sorted(audits, key=lambda item: item.get("batch_index", -1)):
            append_jsonl(audit_path, audit)
        usage_by_model[model.label] = _usage_totals(audits)
        structured = int((frame["coding_method"] == "llm_structured").sum())
        _log(log_path, f"done   {model.label}: {structured}/{len(frame)} llm_structured "
                       f"in {time.time() - started:.1f}s", verbose)
        return model, frame

    if parallel_models:
        with ThreadPoolExecutor(max_workers=min(MAX_MODEL_WORKERS, len(models))) as executor:
            results = list(executor.map(_one, models))
    else:
        results = [_one(model) for model in models]

    frames = {model.label: frame for model, frame in results}
    combined = pd.concat([frames[model.label] for model in models], ignore_index=True)
    write_csv(long_codings_csv(), combined)

    total_usage = {
        key: round(sum(usage[key] for usage in usage_by_model.values()), 4)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd")
    }
    manifest = {
        "n_abstracts": len(records),
        "n_models": len(models),
        "n_codings": int(len(combined)),
        "n_api_calls": n_calls,
        "models": [
            {"order": m.order, "label": m.label, "openrouter_id": m.openrouter_id, "provider": m.provider}
            for m in models
        ],
        "coding_method_counts": combined["coding_method"].value_counts().to_dict(),
        "batch_size": BATCH_SIZE,
        "max_workers": per_model_workers,
        "token_usage_by_model": usage_by_model,
        "token_usage_total": total_usage,
    }
    write_json(out_dir / "run_manifest.json", manifest)

    try:
        destination = long_codings_csv().relative_to(PROJECT_ROOT)
    except ValueError:
        destination = long_codings_csv().name
    _log(log_path, f"ALL DONE: {len(combined)} codings -> {destination} "
                   f"| {total_usage['total_tokens']} tokens | ${total_usage['cost_usd']:.3f} "
                   f"| methods={manifest['coding_method_counts']}", verbose)
    return combined


def load_or_run(force: bool = False) -> pd.DataFrame:
    """Load the cached long coding table if present, otherwise run the test run."""
    path = long_codings_csv()
    if path.exists() and not force:
        return pd.read_csv(path).fillna("")
    return run_testrun()
