#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from azure.identity import AzureCliCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("attach_search_uami")
DEFAULT_API_VERSION = "2025-05-01"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attach a user-assigned managed identity to an existing Azure AI Search service using the Search Management Update API.")
    p.add_argument("--subscription-id", default=os.getenv("AZURE_SUBSCRIPTION_ID", ""))
    p.add_argument("--search-resource-group", default=os.getenv("AZURE_SEARCH_RESOURCE_GROUP", ""))
    p.add_argument("--search-service-name", default=os.getenv("AZURE_SEARCH_SERVICE_NAME", ""))
    p.add_argument("--uami-resource-id", default=os.getenv("FIQ_SEARCH_UAMI_RESOURCE_ID", ""))
    p.add_argument("--api-version", default=DEFAULT_API_VERSION)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def setup_logging(level_name: str, log_path: Path) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S", handlers=handlers, force=True)


def mgmt_url(subscription_id: str, resource_group: str, search_service_name: str, api_version: str) -> str:
    return (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}"
        f"/providers/Microsoft.Search/searchServices/{search_service_name}"
        f"?api-version={api_version}"
    )


def auth_headers(credential: AzureCliCredential, scope: str = "https://management.azure.com/.default") -> dict[str, str]:
    token = credential.get_token(scope)
    return {"Authorization": f"Bearer {token.token}", "Content-Type": "application/json"}


def compute_identity_update(existing: dict[str, Any], uami_resource_id: str) -> dict[str, Any]:
    identity = existing.get("identity") or {}
    existing_type = identity.get("type") or "None"
    existing_uamis = identity.get("userAssignedIdentities") or {}
    new_uamis = dict(existing_uamis)
    new_uamis[uami_resource_id] = {}

    if existing_type == "SystemAssigned":
        new_type = "SystemAssigned, UserAssigned"
    elif existing_type in ("UserAssigned", "SystemAssigned, UserAssigned"):
        new_type = existing_type
    elif existing_type == "None":
        new_type = "UserAssigned"
    else:
        raise ValueError(f"Unsupported existing identity type: {existing_type}")

    return {
        "existing_identity_type": existing_type,
        "new_identity_type": new_type,
        "user_assigned_identity_count_before": len(existing_uamis),
        "user_assigned_identity_count_after": len(new_uamis),
        "identity": {
            "type": new_type,
            "userAssignedIdentities": new_uamis,
        },
    }


def write_artifacts(run_id: str, payload: dict[str, Any]) -> tuple[Path, Path, Path]:
    out_dir = Path(__file__).resolve().parents[2] / "exploration" / "deep_dive" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"attach_search_uami_{run_id}.json"
    md_path = out_dir / f"attach_search_uami_{run_id}.md"
    log_path = out_dir / f"attach_search_uami_{run_id}.log"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# Attach Search UAMI ({run_id})",
        "",
        "## Inputs",
        "",
    ]
    for key, value in payload["metadata"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Result",
            "",
            f"- success: {'yes' if payload['result']['success'] else 'no'}",
            f"- existing_identity_type: {payload['result'].get('existing_identity_type', '')}",
            f"- new_identity_type: {payload['result'].get('new_identity_type', '')}",
            f"- uami_count_before: {payload['result'].get('user_assigned_identity_count_before', '')}",
            f"- uami_count_after: {payload['result'].get('user_assigned_identity_count_after', '')}",
            f"- response_status: {payload['result'].get('status_code', '')}",
        ]
    )
    if not payload["result"]["success"]:
        lines.extend(
            [
                "",
                "## Error",
                "",
                f"- type: {payload['result'].get('error_type', '')}",
                f"- message: {payload['result'].get('error_message', '')}",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path, log_path


def main() -> None:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parents[2] / "exploration" / "deep_dive" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_log_path = out_dir / f"attach_search_uami_{run_id}.log"
    setup_logging(args.log_level, temp_log_path)
    _ = load_config()

    required = {
        "subscription_id": args.subscription_id,
        "search_resource_group": args.search_resource_group,
        "search_service_name": args.search_service_name,
        "uami_resource_id": args.uami_resource_id,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Missing required inputs: {', '.join(missing)}")

    credential = AzureCliCredential()
    headers = auth_headers(credential)
    url = mgmt_url(args.subscription_id, args.search_resource_group, args.search_service_name, args.api_version)

    logger.info("GET search service rg=%s name=%s", args.search_resource_group, args.search_service_name)
    get_response = requests.get(url, headers=headers, timeout=120)
    get_response.raise_for_status()
    existing = get_response.json()

    identity_update = compute_identity_update(existing, args.uami_resource_id)
    patch_body = {"identity": identity_update["identity"]}

    logger.info(
        "PATCH search service existing_identity_type=%s new_identity_type=%s uami=%s",
        identity_update["existing_identity_type"],
        identity_update["new_identity_type"],
        args.uami_resource_id,
    )

    result: dict[str, Any] = {
        "success": False,
        "existing_identity_type": identity_update["existing_identity_type"],
        "new_identity_type": identity_update["new_identity_type"],
        "user_assigned_identity_count_before": identity_update["user_assigned_identity_count_before"],
        "user_assigned_identity_count_after": identity_update["user_assigned_identity_count_after"],
    }
    try:
        patch_response = requests.patch(url, headers=headers, json=patch_body, timeout=120)
        patch_response.raise_for_status()
        body = patch_response.json()
        result.update(
            {
                "success": True,
                "status_code": patch_response.status_code,
                "response_identity_type": ((body.get("identity") or {}).get("type", "")),
                "response_user_assigned_identities": sorted(((body.get("identity") or {}).get("userAssignedIdentities") or {}).keys()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        response = getattr(exc, "response", None)
        result.update(
            {
                "success": False,
                "status_code": getattr(response, "status_code", None),
                "error_type": type(exc).__name__,
                "error_message": response.text if response is not None else str(exc),
            }
        )

    payload = {
        "metadata": {
            "run_id": run_id,
            "subscription_id": args.subscription_id,
            "search_resource_group": args.search_resource_group,
            "search_service_name": args.search_service_name,
            "uami_resource_id": args.uami_resource_id,
            "api_version": args.api_version,
            "management_url": url,
        },
        "result": result,
    }
    md_path, json_path, _ = write_artifacts(run_id, payload)
    logger.info("Artifacts markdown=%s json=%s log=%s", md_path, json_path, temp_log_path)
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
