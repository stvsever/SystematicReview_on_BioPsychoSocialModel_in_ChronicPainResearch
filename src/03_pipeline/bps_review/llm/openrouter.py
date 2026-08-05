from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from bps_review.utils.env import load_environment


API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_CHAT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


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
    return data["choices"][0]["message"]["content"]


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
        fallback_text = chat_completion(prompt, model=chosen_model, temperature=temperature)
        return _extract_json_blob(fallback_text), {}
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]
    content = choice["message"].get("content")
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
