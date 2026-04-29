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
from azure.identity import AzureCliCredential, get_bearer_token_provider

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("ai_search_project_connection_setup")
DEFAULT_CACHE_PATH = REPO_ROOT / "exploration" / "deep_dive" / "output" / "ai_search_project_connection_cache.json"


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Create or update the direct Azure AI Search project connection for AzureAISearchTool.")
    p.add_argument("--connection-name", default=os.getenv("AZURE_AI_SEARCH_PROJECT_CONNECTION_NAME", "ai-search-direct"))
    p.add_argument("--search-endpoint", default=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT", ""))
    p.add_argument("--auth-type", choices=["AAD", "ApiKey"], default=os.getenv("AZURE_AI_SEARCH_CONNECTION_AUTH_TYPE", "AAD"))
    p.add_argument("--search-admin-key", default=os.getenv("AZURE_SEARCH_ADMIN_KEY", ""))
    p.add_argument("--project-resource-id", default=os.getenv("PROJECT_RESOURCE_ID", ""))
    p.add_argument("--api-version", default="2025-06-01")
    p.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH))
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--print-body", action="store_true", help="Print the final request body for inspection.")
    _ = cfg
    return p.parse_args()


def _setup_logging(level_name: str, log_path: Path) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S", handlers=handlers, force=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _derive_project_resource_id(cfg: Any) -> str:
    if getattr(cfg, "subscription_id", "") and getattr(cfg, "resource_group", "") and getattr(cfg, "resource_name", "") and getattr(cfg, "project_name", ""):
        return (
            f"/subscriptions/{cfg.subscription_id}"
            f"/resourceGroups/{cfg.resource_group}"
            f"/providers/Microsoft.CognitiveServices/accounts/{cfg.resource_name}"
            f"/projects/{cfg.project_name}"
        )
    return ""


def _write_outputs(run_id: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = REPO_ROOT / "exploration" / "deep_dive" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"ai_search_project_connection_setup_{run_id}.json"
    md_path = out_dir / f"ai_search_project_connection_setup_{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# AI Search Project Connection Setup ({run_id})",
        "",
        "## Inputs",
        "",
    ]
    for k, v in payload["metadata"].items():
        if k == "credentials_supplied":
            v = "yes" if v else "no"
        lines.append(f"- {k}: {v}")
    lines.extend(
        [
            "",
            "## Result",
            "",
            f"- status_code: {payload['result']['status_code']}",
            f"- connection_name: {payload['result']['connection_name']}",
            f"- connection_id: {payload['result']['connection_id']}",
            f"- target: {payload['result']['target']}",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main() -> None:
    args = parse_args()
    cfg = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "exploration" / "deep_dive" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"ai_search_project_connection_setup_{run_id}.log"
    _setup_logging(args.log_level, log_path)

    project_resource_id = args.project_resource_id or _derive_project_resource_id(cfg)
    if not project_resource_id:
        raise SystemExit("Could not resolve project resource id. Set PROJECT_RESOURCE_ID or provide subscription/resource group/foundry resource/project vars in .env.")

    search_endpoint = args.search_endpoint.rstrip("/")
    if not search_endpoint:
        raise SystemExit("Missing search endpoint. Set AZURE_SEARCH_SERVICE_ENDPOINT or pass --search-endpoint.")

    auth_type = args.auth_type
    if auth_type == "ApiKey" and not args.search_admin_key:
        raise SystemExit("ApiKey auth selected but no search admin key provided. Set AZURE_SEARCH_ADMIN_KEY or pass --search-admin-key.")

    connection_url = (
        f"https://management.azure.com{project_resource_id}"
        f"/connections/{args.connection_name}?api-version={args.api_version}"
    )

    properties: dict[str, Any] = {
        "category": "CognitiveSearch",
        "target": search_endpoint,
        "authType": auth_type,
    }
    if auth_type == "ApiKey":
        properties["credentials"] = {"key": args.search_admin_key}

    body = {
        "properties": properties,
    }

    if args.print_body:
        print(json.dumps(body, indent=2))

    cred = AzureCliCredential(process_timeout=60)
    token_provider = get_bearer_token_provider(cred, "https://management.azure.com/.default")
    headers = {
        "Authorization": f"Bearer {token_provider()}",
        "Content-Type": "application/json",
    }
    logger.info("PUT connection_name=%s auth_type=%s target=%s", args.connection_name, auth_type, search_endpoint)
    response = requests.put(connection_url, json=body, headers=headers, timeout=120)
    response.raise_for_status()
    result_json = response.json()

    connection_id = (
        result_json.get("id")
        or f"{project_resource_id}/connections/{args.connection_name}"
    )
    payload = {
        "metadata": {
            "run_id": run_id,
            "project_resource_id": project_resource_id,
            "connection_name": args.connection_name,
            "search_endpoint": search_endpoint,
            "auth_type": auth_type,
            "credentials_supplied": auth_type == "ApiKey",
            "api_version": args.api_version,
        },
        "result": {
            "status_code": response.status_code,
            "connection_name": args.connection_name,
            "connection_id": connection_id,
            "target": search_endpoint,
            "response_json": result_json,
        },
    }
    md_path, json_path = _write_outputs(run_id, payload)

    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "created_at_utc": _utc_now(),
                "project_resource_id": project_resource_id,
                "connection_name": args.connection_name,
                "connection_id": connection_id,
                "search_endpoint": search_endpoint,
                "auth_type": auth_type,
                "api_version": args.api_version,
                "response_json": result_json,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Artifacts markdown=%s json=%s log=%s cache=%s", md_path, json_path, log_path, cache_path)


if __name__ == "__main__":
    main()
