from __future__ import annotations

import re
from functools import lru_cache

import pycountry


SPECIAL_COUNTRY_PATTERNS = {
    "usa": "United States",
    "u.s.a.": "United States",
    "us": "United States",
    "u.s.": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "the netherlands": "Netherlands",
}


@lru_cache(maxsize=1)
def _country_names() -> list[str]:
    names = []
    for country in pycountry.countries:
        names.append(country.name)
        if hasattr(country, "official_name"):
            names.append(country.official_name)
        if hasattr(country, "common_name"):
            names.append(country.common_name)
    names.extend(SPECIAL_COUNTRY_PATTERNS.values())
    return sorted(set(names), key=len, reverse=True)


def infer_country_from_text(text: str) -> str:
    lowered = f" {text.lower()} "
    for token, mapped in SPECIAL_COUNTRY_PATTERNS.items():
        if f" {token} " in lowered:
            return mapped
    for country_name in _country_names():
        pattern = rf"\b{re.escape(country_name.lower())}\b"
        if re.search(pattern, lowered):
            return country_name
    return ""
