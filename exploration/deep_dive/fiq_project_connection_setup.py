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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("fiq_project_connection_setup")
DEFAULT_KNOWLEDGE_CACHE_PATH = Path(__file__).resolve().parent / "output" / "fiq_knowledge_cache.json"
DEFAULT_CONNECTION_CACHE_PATH = Path(__file__).resolve().parent / "output" / "fiq_project_connection_cache.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create or update the Foundry project connection for the Foundry IQ MCP endpoint.")
    p.add_argument("--knowledge-cache-path", default=str(DEFAULT_KNOWLEDGE_CACHE_PATH))
    p.add_argument("--connection-cache-path", default=str(DEFAULT_CONNECTION_CACHE_PATH))
    p.add_argument("--project-resource-id", default=os.getenv("PROJECT_RESOURCE_ID", ""))
    p.add_argument("--project-connection-name", default=os.getenv("PROJECT_CONNECTION_NAME", "fiq-knowledge-base"))
    p.add_argument("--api-version", default="2025-10-01-preview")
    p.add_argument("--mcp-endpoint", default="")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _setup_logging(level_name: str, log_path: Path) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S", handlers=handlers, force=True)


def _write_outputs(run_id: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"fiq_project_connection_setup_{run_id}.json"
    md_path = out_dir / f"fiq_project_connection_setup_{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# FIQ Project Connection Setup ({run_id})",
        "",
        "## Inputs",
        "",
    ]
    for k, v in payload["metadata"].items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Result", "", f"- status_code: {payload['result']['status_code']}", f"- target: {payload['result']['target']}", f"- success: {payload['result']['success']}"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main() -> None:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"fiq_project_connection_setup_{run_id}.log"
    _setup_logging(args.log_level, log_path)
    _ = load_config()

    project_resource_id = args.project_resource_id
    if not project_resource_id:
        raise SystemExit("Missing PROJECT_RESOURCE_ID or --project-resource-id.")

    mcp_endpoint = args.mcp_endpoint
    if not mcp_endpoint:
        knowledge_cache = Path(args.knowledge_cache_path)
        if not knowledge_cache.exists():
            raise SystemExit(f"Knowledge cache not found: {knowledge_cache}")
        mcp_endpoint = str(json.loads(knowledge_cache.read_text(encoding='utf-8')).get("mcp_endpoint") or "")
    if not mcp_endpoint:
        raise SystemExit("Could not resolve MCP endpoint.")

    connection_url = f"https://management.azure.com{project_resource_id}/connections/{args.project_connection_name}?api-version={args.api_version}"
    connection_body = {
        "name": args.project_connection_name,
        "type": "Microsoft.MachineLearningServices/workspaces/connections",
        "properties": {
            "authType": "ProjectManagedIdentity",
            "category": "RemoteTool",
            "target": mcp_endpoint,
            "isSharedToAll": True,
            "audience": "https://search.azure.com/",
            "metadata": {"ApiType": "Azure"},
        },
    }
    cred = AzureCliCredential(process_timeout=60)
    token_provider = get_bearer_token_provider(cred, "https://management.azure.com/.default")
    headers = {"Authorization": f"Bearer {token_provider()}", "Content-Type": "application/json"}
    logger.info("PUT project connection name=%s", args.project_connection_name)
    response = requests.put(connection_url, json=connection_body, headers=headers, timeout=120)
    response.raise_for_status()

    payload = {
        "metadata": {
            "run_id": run_id,
            "project_resource_id": project_resource_id,
            "project_connection_name": args.project_connection_name,
            "mcp_endpoint": mcp_endpoint,
            "api_version": args.api_version,
        },
        "result": {
            "success": True,
            "status_code": response.status_code,
            "target": mcp_endpoint,
            "connection_url": connection_url,
            "response_json": response.json(),
        },
    }
    md_path, json_path = _write_outputs(run_id, payload)

    cache_payload = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project_resource_id": project_resource_id,
        "project_connection_name": args.project_connection_name,
        "mcp_endpoint": mcp_endpoint,
        "api_version": args.api_version,
        "response_json": response.json(),
    }
    cache_path = Path(args.connection_cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
    logger.info("Artifacts markdown=%s json=%s log=%s cache=%s", md_path, json_path, log_path, cache_path)


if __name__ == "__main__":
    main()
