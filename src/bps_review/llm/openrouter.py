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


def chat_completion_json(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
    system_prompt: str | None = None,
) -> Any:
    load_environment()
    chosen_model = model or os.environ.get("OPENROUTER_MODEL") or resolve_default_model()
    payload = {
        "model": chosen_model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": system_prompt
                or "You are assisting a protocol-aligned systematic review pipeline. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    response = requests.post(f"{API_BASE}/chat/completions", headers=_headers(), data=json.dumps(payload), timeout=180)
    if response.status_code >= 400:
        fallback_text = chat_completion(prompt, model=chosen_model, temperature=temperature)
        return _extract_json_blob(fallback_text)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return _extract_json_blob(content)


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
