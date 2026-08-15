from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

from bps_review.utils.env import load_environment


API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_CHAT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
# Semantic label matching needs the larger embedding model: the labels are short
# noun phrases whose difference is often a single qualifier, and the small model
# does not separate those reliably.
SEMANTIC_EMBEDDING_MODEL = "openai/text-embedding-3-large"


def _headers() -> dict[str, str]:
    load_environment()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def resolve_default_model() -> str:
    explicit = os.environ.get("OPENROUTER_MODEL", "").strip()
    if explicit:
        return explicit
    try:
        response = requests.get(f"{API_BASE}/models", headers=_headers(), timeout=60)
        response.raise_for_status()
        items = response.json().get("data", [])
        gemini_flash = [item["id"] for item in items if "gemini" in item["id"].lower() and "flash" in item["id"].lower()]
        if gemini_flash:
            return sorted(gemini_flash)[-1]
        mini = [item["id"] for item in items if "mini" in item["id"].lower()]
        if mini:
            return sorted(mini)[0]
    except requests.RequestException:
        pass
    return DEFAULT_CHAT_MODEL


def chat_completion(prompt: str, model: str | None = None, temperature: float = 0.0) -> str:
    load_environment()
    chosen_model = model or os.environ.get("OPENROUTER_MODEL") or resolve_default_model()
    payload = {
        "model": chosen_model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "You are assisting a protocol-aligned systematic review pipeline. Return concise, structured outputs."},
            {"role": "user", "content": prompt},
        ],
    }
    response = requests.post(f"{API_BASE}/chat/completions", headers=_headers(), data=json.dumps(payload), timeout=120)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    content = (choices[0].get("message") or {}).get("content") if choices else None
    if not content:
        error = data.get("error") or {}
        raise RuntimeError(
            f"Model {chosen_model} returned no content "
            f"(error={error.get('message') or error or 'none'})"
        )
    return content


def _extract_json_blob(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    raise ValueError("No valid JSON object found in model output.")


# Route around a degraded upstream provider. OpenRouter serves most models from
# several providers; sorting by throughput and allowing fallbacks keeps a single
# slow or stalled provider from hanging the request.
PROVIDER_ROUTING = {"sort": "throughput", "allow_fallbacks": True}


def chat_completion_json(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
    system_prompt: str | None = None,
    timeout: int = 180,
    max_tokens: int | None = None,
) -> Any:
    """Request one JSON object from a chat model."""
    payload, _ = chat_completion_json_with_usage(
        prompt,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    return payload


def chat_completion_json_with_usage(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
    system_prompt: str | None = None,
    timeout: int = 180,
    max_tokens: int | None = None,
    reasoning: dict | None = None,
) -> tuple[Any, dict]:
    """Request one JSON object and also return the provider's token usage.

    ``max_tokens`` caps the completion. It matters for the full-text coding
    stage, where a long structured extraction must still fit inside the
    completion window of the model used, and the usage block is what makes the
    real cost of a run reportable rather than estimated.

    ``reasoning`` is passed through to OpenRouter. On a reasoning model the
    thinking tokens are drawn from the same completion budget as the answer, so a
    long structured task needs either the reasoning turned off or a budget large
    enough for both. Which of the two applies is a property of the model, so it
    is decided by the caller.
    """
    load_environment()
    chosen_model = model or os.environ.get("OPENROUTER_MODEL") or resolve_default_model()
    payload = {
        "model": chosen_model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "provider": PROVIDER_ROUTING,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
                or "You are assisting a protocol-aligned systematic review pipeline. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    if reasoning is not None:
        payload["reasoning"] = reasoning
    response = requests.post(f"{API_BASE}/chat/completions", headers=_headers(), data=json.dumps(payload), timeout=timeout)
    if response.status_code >= 400:
        # Raised rather than retried here without the JSON mode, the token cap,
        # and the reasoning settings this call was made with. That silent
        # downgrade produced answers that looked like codings and were not, and
        # it hid the status code from the caller's retry policy, which needs it
        # to tell a congested provider from a bad request.
        raise RuntimeError(
            f"OpenRouter returned HTTP {response.status_code} for {chosen_model}: "
            f"{response.text[:300]}"
        )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        # An upstream provider error arrives as a 200 with no choices and an
        # error block. Naming it is what lets the caller retry it as the
        # transient condition it usually is.
        error = data.get("error") or {}
        raise RuntimeError(
            f"Model {chosen_model} returned no choices "
            f"(error={error.get('message') or error or 'none'})"
        )
    choice = choices[0] or {}
    content = (choice.get("message") or {}).get("content")
    usage = data.get("usage") or {}
    if not content:
        # An empty answer with finish_reason "length" means the completion budget
        # was spent before the answer began, which happens when a reasoning model
        # thinks for the whole allowance. Say so, so the caller can retry with a
        # larger budget instead of seeing an opaque type error.
        finish = choice.get("finish_reason")
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        raise ValueError(
            f"Model {chosen_model} returned empty content (finish_reason={finish}, "
            f"reasoning_tokens={reasoning_tokens}, completion_tokens={usage.get('completion_tokens')})"
        )
    return _extract_json_blob(content), usage


def embed_batch_with_usage(
    texts: list[str], model: str | None = None, timeout: int = 120
) -> tuple[list[list[float]], dict]:
    """Embed one batch and return the vectors in input order plus provider usage."""
    load_environment()
    chosen_model = model or os.environ.get("OPENROUTER_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    payload = {"model": chosen_model, "input": texts}
    response = requests.post(
        f"{API_BASE}/embeddings", headers=_headers(), data=json.dumps(payload), timeout=timeout
    )
    response.raise_for_status()
    body = response.json()
    data = sorted(body.get("data", []), key=lambda item: item.get("index", 0))
    vectors = [item.get("embedding", []) for item in data]
    if len(vectors) != len(texts):
        raise RuntimeError("Embedding response length does not match input length.")
    return vectors, body.get("usage") or {}


def embed_texts_concurrent(
    texts: list[str],
    model: str | None = None,
    batch_size: int = 96,
    max_workers: int = 8,
    timeout: int = 120,
) -> tuple[list[list[float]], dict]:
    """Embed many short texts in parallel batches, preserving input order.

    The label vocabulary of one run is a few hundred short strings, so the wall
    time is dominated by round trips rather than by tokens. Batches go out on a
    thread pool and the usage blocks are summed, so the embedding step reports
    its own cost the same way the coding step does.
    """
    if not texts:
        return [], {}
    batches = [(start, texts[start:start + batch_size]) for start in range(0, len(texts), batch_size)]
    results: dict[int, list[list[float]]] = {}
    usage_total = {"prompt_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "requests": 0}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as pool:
        futures = {
            pool.submit(embed_batch_with_usage, chunk, model, timeout): start
            for start, chunk in batches
        }
        for future in as_completed(futures):
            start = futures[future]
            vectors, usage = future.result()
            results[start] = vectors
            usage_total["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            usage_total["total_tokens"] += int(usage.get("total_tokens") or 0)
            usage_total["cost_usd"] += float(usage.get("cost") or 0.0)
            usage_total["requests"] += 1
    ordered: list[list[float]] = []
    for start, _ in batches:
        ordered.extend(results[start])
    return ordered, usage_total


def embed_texts(texts: list[str], model: str | None = None, batch_size: int = 32) -> list[list[float]]:
    """Generate embeddings for a list of input texts via OpenRouter.

    The function preserves input order and raises if the API call fails.
    """
    if not texts:
        return []

    load_environment()
    chosen_model = model or os.environ.get("OPENROUTER_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL

    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        payload = {
            "model": chosen_model,
            "input": chunk,
        }
        response = requests.post(f"{API_BASE}/embeddings", headers=_headers(), data=json.dumps(payload), timeout=120)
        response.raise_for_status()
        data = response.json().get("data", [])
        data = sorted(data, key=lambda item: item.get("index", 0))
        chunk_embeddings = [item.get("embedding", []) for item in data]
        if len(chunk_embeddings) != len(chunk):
            raise RuntimeError("Embedding response length does not match input length.")
        all_embeddings.extend(chunk_embeddings)

    return all_embeddings
