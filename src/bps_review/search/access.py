from __future__ import annotations

import requests

from bps_review.search.pubmed import _request as pubmed_request
from bps_review.utils.env import get_env


def check_pubmed_access() -> dict[str, str]:
    response = pubmed_request("einfo.fcgi", {"db": "pubmed", "retmode": "json"})
    response.raise_for_status()
    return {"service": "pubmed", "status": "ok"}


def check_wos_starter_access() -> dict[str, str]:
    api_key = get_env("CLARIVATE_API_KEY")
    if not api_key:
        return {
            "service": "wos_starter",
            "status": "missing_credentials",
            "required": "CLARIVATE_API_KEY",
        }
    response = requests.get(
        "https://api.clarivate.com/apis/wos-starter/v1/documents",
        headers={"X-ApiKey": api_key},
        params={"q": 'TS=("pain")', "limit": 1, "page": 1},
        timeout=60,
    )
    if response.ok:
        return {"service": "wos_starter", "status": "ok"}
    return {"service": "wos_starter", "status": f"http_{response.status_code}", "detail": response.text[:300]}


def check_eds_access() -> dict[str, str]:
    user = get_env("EDS_API_USER")
    password = get_env("EDS_API_PASSWORD")
    interface_id = get_env("EDS_API_INTERFACE_ID") or get_env("EDS_API_INTERFACE")
    if not user or not password or not interface_id:
        return {
            "service": "eds_api",
            "status": "missing_credentials",
            "required": "EDS_API_USER, EDS_API_PASSWORD, EDS_API_INTERFACE_ID (or EDS_API_INTERFACE), EDS_API_PROFILE, EDS_API_ORG",
        }
    response = requests.post(
        "https://eds-api.ebscohost.com/authservice/rest/uidauth",
        json={"UserId": user, "Password": password, "InterfaceId": interface_id},
        timeout=60,
    )
    if response.ok:
        return {"service": "eds_api", "status": "ok"}
    return {"service": "eds_api", "status": f"http_{response.status_code}", "detail": response.text[:300]}


def check_external_api_access() -> list[dict[str, str]]:
    return [check_pubmed_access(), check_wos_starter_access(), check_eds_access()]
