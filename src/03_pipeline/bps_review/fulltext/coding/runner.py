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

import json
import random
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
    MAX_TRANSIENT_RETRIES,
    MAX_WORKERS,
    REPAIR_WORKERS,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BACKOFF_SECONDS,
    TRANSIENT_BACKOFF_CAP_SECONDS,
    TRANSIENT_BACKOFF_SECONDS,
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


# Transient upstream conditions, as opposed to a malformed response. A congested
# provider answers 429 or 5xx, and the only useful response is to wait longer
# than the few seconds a malformed-response retry needs. One run lost 24 of 141
# codings to a run of 503s that a linear backoff gave up on inside 15 seconds.
#
# The last three are the same condition wearing different clothes: a provider
# that accepts the request and then fails to generate returns a 200 carrying
# ``finish_reason=error``, no choices at all, or an empty completion. None of
# those is a bad answer from the model, so none of them should be retried as if
# waiting would not help.
_TRANSIENT_MARKERS = (
    "429", "500", "502", "503", "504", "overloaded", "rate limit", "timeout",
    "finish_reason=error", "returned no choices", "returned no content",
)


def _is_transient(error: str) -> bool:
    lowered = error.lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


def _backoff_seconds(attempt: int, transient: bool) -> float:
    """How long to wait before the next attempt.

    A malformed response is retried promptly, because waiting does not make the
    model answer differently. A congested provider is retried on an exponential
    schedule with jitter, because waiting is the only thing that helps and
    because a thundering herd of workers retrying in lockstep is what keeps a
    provider congested.
    """
    if not transient:
        return RETRY_BACKOFF_SECONDS * attempt
    capped = min(TRANSIENT_BACKOFF_SECONDS * (2 ** (attempt - 1)), TRANSIENT_BACKOFF_CAP_SECONDS)
    return capped * random.uniform(0.6, 1.4)


def _attempt(record: dict, model_id: str) -> tuple[FullTextCodingRecord | None, dict]:
    """Code one paper with retries and a hard timeout. Never raises."""
    last_error = ""
    attempts_allowed = MAX_RETRIES
    attempt = 0
    while attempt < attempts_allowed:
        attempt += 1
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
        transient = _is_transient(last_error)
        # A provider outage is worth more patience than a bad answer, so a
        # transient failure buys extra attempts rather than burning the budget
        # that a malformed response needs.
        if transient:
            attempts_allowed = min(MAX_TRANSIENT_RETRIES, max(attempts_allowed, attempt + 1))
        if attempt < attempts_allowed:
            time.sleep(_backoff_seconds(attempt, transient))
    return None, {
        "record_id": record["record_id"],
        "model_id": model_id,
        "status": "failed",
        "attempts": attempt,
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


def _read_audits(path: Path) -> dict[str, dict]:
    """The audit entries already on disk for one model, keyed by record."""
    if not path.exists():
        return {}
    entries: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        entries[entry["record_id"]] = entry
    return entries


def _sum_usage(left: dict, right: dict) -> dict:
    """Cumulative token usage across a run and the repair passes after it."""
    keys = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "cost_usd")
    merged = {key: (left.get(key, 0) or 0) + (right.get(key, 0) or 0) for key in keys}
    merged["cost_usd"] = round(merged["cost_usd"], 4)
    return merged


def repair_failed_codings(
    models: list[TestRunModel] | None = None,
    records: list[dict] | None = None,
    per_model_workers: int = REPAIR_WORKERS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Re-code only the cells that failed, and splice the results into the run.

    A failed coding is almost always an upstream outage rather than a paper the
    model cannot read, so the whole grid does not need re-coding to fill one in.
    This re-codes exactly the ``(paper, model)`` cells written as ``coding_failed``,
    replaces those rows, their extracted items, and their audit entries in place,
    and adds what the repair cost to the manifest so the reported cost stays the
    true cost of the table on disk.

    It runs at lower concurrency than the main pass on purpose: a repair follows
    a provider that was already struggling, and hammering it again is what
    produced the gap in the first place.
    """
    models = models or FULLTEXT_MODELS
    records = records if records is not None else load_corpus_records()
    record_by_id = {record["record_id"]: record for record in records}

    out_dir = codings_dir()
    by_model_dir = out_dir / "by_model"
    audit_dir = out_dir / "audit"
    log_path = out_dir / "run.log"

    manifest_path = out_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    usage_by_model = dict(manifest.get("token_usage_by_model", {}))

    repaired_total = 0
    still_failed_total = 0

    for model in models:
        frame_path = by_model_dir / f"{model.slug}.csv"
        if not frame_path.exists():
            continue
        frame = pd.read_csv(frame_path).fillna("")
        failed_ids = [
            str(record_id)
            for record_id, method in zip(frame["record_id"], frame["coding_method"])
            if method == "coding_failed" and str(record_id) in record_by_id
        ]
        if not failed_ids:
            continue

        _log(log_path, f"repair {model.label}: re-coding {len(failed_ids)} failed codings "
                       f"(workers={per_model_workers})", verbose)
        targets = [record_by_id[record_id] for record_id in failed_ids]

        coded_by_id: dict[str, FullTextCodingRecord] = {}
        audits: list[dict] = []
        with ThreadPoolExecutor(max_workers=per_model_workers) as executor:
            futures = [executor.submit(_attempt, record, model.openrouter_id) for record in targets]
            for future in as_completed(futures):
                coded, audit = future.result()
                audits.append(audit)
                if coded is not None:
                    coded_by_id[audit["record_id"]] = coded

        if not coded_by_id:
            still_failed_total += len(failed_ids)
            _log(log_path, f"repair {model.label}: 0 of {len(failed_ids)} recovered", verbose)
            usage_by_model[model.label] = _sum_usage(usage_by_model.get(model.label, {}), _usage_totals(audits))
            continue

        # Splice the recovered rows into the model's wide frame, keeping the
        # column set and the row order the frame already has.
        repaired_rows = {}
        new_items: list[dict] = []
        for record_id, coded in coded_by_id.items():
            row = serialize_row(coded, model_id=model.openrouter_id)
            row["record_id"] = record_id
            row["model_order"] = model.order
            row["model_label"] = model.label
            row["provider"] = model.provider
            row["model_id"] = model.openrouter_id
            repaired_rows[record_id] = row
            new_items.extend(item_rows(coded, model.openrouter_id, model.label))

        # Rebuilt rather than assigned in place: a repaired row is written as
        # strings, and writing those into columns pandas typed as numeric when it
        # read the CSV is a dtype error waiting to happen.
        order = list(frame["record_id"])
        kept = frame[~frame["record_id"].isin(repaired_rows)]
        replacement = pd.DataFrame(
            [repaired_rows[record_id] for record_id in order if record_id in repaired_rows]
        )
        frame = pd.concat([kept, replacement], ignore_index=True)
        frame["record_id"] = frame["record_id"].astype(str)
        frame = (frame.set_index("record_id").reindex(order).reset_index())
        lead = [column for column in IDENTITY_COLUMNS if column in frame.columns]
        rest = [column for column in frame.columns if column not in lead]
        frame = frame[lead + rest]
        write_csv(frame_path, frame)

        items_path = by_model_dir / f"{model.slug}_items.csv"
        existing_items = pd.read_csv(items_path).fillna("") if items_path.exists() else pd.DataFrame()
        items_frame = pd.DataFrame(new_items)
        if not existing_items.empty:
            # A repaired paper had no items before, but drop any anyway so a
            # second repair of the same cell cannot double-count it.
            existing_items = existing_items[~existing_items["record_id"].isin(repaired_rows)]
            items_frame = pd.concat([existing_items, items_frame], ignore_index=True)
        if not items_frame.empty:
            items_frame = items_frame.sort_values("record_id", kind="stable").reset_index(drop=True)
            write_csv(items_path, items_frame)

        audit_path = audit_dir / f"{model.slug}.jsonl"
        entries = _read_audits(audit_path)
        for audit in audits:
            entries[audit["record_id"]] = audit
        ensure_parent(audit_path)
        audit_path.write_text("", encoding="utf-8")
        for record_id in sorted(entries):
            append_jsonl(audit_path, entries[record_id])

        usage_by_model[model.label] = _sum_usage(usage_by_model.get(model.label, {}), _usage_totals(audits))
        repaired_total += len(repaired_rows)
        still_failed_total += len(failed_ids) - len(repaired_rows)
        _log(log_path, f"repair {model.label}: {len(repaired_rows)} of {len(failed_ids)} recovered, "
                       f"{len(new_items)} extracted items added", verbose)

    # Rebuild the combined tables from the per-model files, whether or not this
    # pass recovered anything, so the long table always matches its parts.
    frames = []
    item_frames = []
    for model in models:
        frame_path = by_model_dir / f"{model.slug}.csv"
        if frame_path.exists():
            frames.append(pd.read_csv(frame_path).fillna(""))
        items_path = by_model_dir / f"{model.slug}_items.csv"
        if items_path.exists():
            item_frames.append(pd.read_csv(items_path).fillna(""))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_csv(long_codings_csv(), combined)
    all_items = pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame()
    write_csv(items_csv(), all_items)

    manifest["n_codings"] = int(len(combined))
    manifest["n_extracted_items"] = int(len(all_items))
    manifest["coding_method_counts"] = combined["coding_method"].value_counts().to_dict()
    manifest["token_usage_by_model"] = usage_by_model
    manifest["token_usage_total"] = {
        key: round(sum(usage.get(key, 0) or 0 for usage in usage_by_model.values()), 4)
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "cost_usd")
    }
    manifest["repair_passes"] = int(manifest.get("repair_passes", 0)) + 1
    write_json(manifest_path, manifest)

    _log(log_path, f"REPAIR DONE: {repaired_total} recovered, {still_failed_total} still failed, "
                   f"{len(combined)} codings, {len(all_items)} extracted items "
                   f"| methods={manifest['coding_method_counts']}", verbose)
    return rederive_frame(combined)


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
