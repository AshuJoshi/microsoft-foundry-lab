#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("fiq_knowledge_setup")
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "output" / "fiq_knowledge_cache.json"


@dataclass
class KnowledgeSetupRecord:
    kind: str
    name: str
    success: bool
    status_code: int | None
    url: str
    error_type: str | None = None
    error_message: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create or update Azure AI Search knowledge sources and a Foundry IQ knowledge base.")
    p.add_argument("--api-version", default="2025-11-01-preview")
    p.add_argument("--knowledge-base-name", default=os.getenv("FIQ_KNOWLEDGE_BASE_NAME", "zava-agentic-kb"))
    p.add_argument("--hr-index-name", default="hrdocs")
    p.add_argument("--health-index-name", default="healthdocs")
    p.add_argument("--hr-source-name", default="hrdocs-knowledge-source")
    p.add_argument("--health-source-name", default="healthdocs-knowledge-source")
    p.add_argument("--auth-mode", choices=["api-key", "managed-identity"], default=os.getenv("FIQ_AOAI_AUTH_MODE", "managed-identity"))
    p.add_argument("--uami-resource-id", default=os.getenv("FIQ_SEARCH_UAMI_RESOURCE_ID", ""))
    p.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _setup_logging(level_name: str, log_path: Path) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S", handlers=handlers, force=True)


def _write_outputs(run_id: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"fiq_knowledge_setup_{run_id}.json"
    md_path = out_dir / f"fiq_knowledge_setup_{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# FIQ Knowledge Setup ({run_id})",
        "",
        "## Inputs",
        "",
    ]
    for k, v in payload["metadata"].items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Results", "", "| Kind | Name | Success | Status | URL | Error |", "|---|---|---|---:|---|---|"])
    for rec in payload["records"]:
        err = "" if rec["success"] else f"{rec['error_type']}: {rec['error_message']}"
        err = err.replace("|", "\\|")
        lines.append(f"| {rec['kind']} | {rec['name']} | {'yes' if rec['success'] else 'no'} | {rec['status_code'] or '-'} | {rec['url']} | {err} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def _search_headers(search_api_key: str) -> dict[str, str]:
    return {"Content-Type": "application/json", "api-key": search_api_key}


def _put_json(url: str, body: dict[str, Any], headers: dict[str, str]) -> requests.Response:
    resp = requests.put(url, json=body, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp


def _build_model_parameters(
    auth_mode: str,
    azure_openai_endpoint: str,
    chat_deployment: str,
    chat_model_name: str,
    azure_openai_key: str,
    uami_resource_id: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "resourceUri": azure_openai_endpoint,
        "deploymentId": chat_deployment,
        "modelName": chat_model_name,
    }
    if auth_mode == "api-key":
        params["apiKey"] = azure_openai_key
    else:
        params["authIdentity"] = {
            "@odata.type": "#Microsoft.Azure.Search.DataUserAssignedIdentity",
            "userAssignedIdentity": uami_resource_id,
        }
    return params


def main() -> None:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"fiq_knowledge_setup_{run_id}.log"
    _setup_logging(args.log_level, log_path)
    _ = load_config()

    search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT", "").rstrip("/")
    search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY", "")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY", "")
    chat_deployment = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT", "")
    chat_model_name = os.getenv("AZURE_OPENAI_CHATGPT_MODEL_NAME", "")
    required = [search_endpoint, search_api_key, azure_openai_endpoint, chat_deployment, chat_model_name]
    if args.auth_mode == "api-key":
        required.append(azure_openai_key)
    else:
        required.append(args.uami_resource_id)
    if not all(required):
        auth_requirements = "AZURE_OPENAI_KEY" if args.auth_mode == "api-key" else "FIQ_SEARCH_UAMI_RESOURCE_ID/--uami-resource-id"
        raise SystemExit(
            "Missing required env vars: AZURE_SEARCH_SERVICE_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, "
            f"AZURE_OPENAI_ENDPOINT, {auth_requirements}, AZURE_OPENAI_CHATGPT_DEPLOYMENT, AZURE_OPENAI_CHATGPT_MODEL_NAME"
        )

    headers = _search_headers(search_api_key)
    records: list[KnowledgeSetupRecord] = []

    source_specs = [
        (args.hr_source_name, args.hr_index_name, "Zava HR documents"),
        (args.health_source_name, args.health_index_name, "Zava health documents"),
    ]
    for source_name, index_name, description in source_specs:
        body = {
            "name": source_name,
            "kind": "searchIndex",
            "description": description,
            "searchIndexParameters": {
                "searchIndexName": index_name,
                "sourceDataFields": [{"name": "blob_path"}, {"name": "snippet"}],
                "searchFields": [{"name": "snippet"}],
            },
        }
        url = f"{search_endpoint}/knowledgesources/{source_name}?api-version={args.api_version}"
        logger.info("PUT knowledge source name=%s index=%s", source_name, index_name)
        try:
            resp = _put_json(url, body, headers)
            records.append(KnowledgeSetupRecord("knowledge_source", source_name, True, resp.status_code, url))
        except Exception as exc:  # noqa: BLE001
            response = getattr(exc, "response", None)
            message = response.text if response is not None and response.text else str(exc)
            records.append(KnowledgeSetupRecord("knowledge_source", source_name, False, getattr(response, "status_code", None), url, type(exc).__name__, message))

    kb_body = {
        "name": args.knowledge_base_name,
        "description": "Knowledge base for Foundry Agent grounding with extractive data",
        "outputMode": "extractiveData",
        "knowledgeSources": [{"name": args.health_source_name}, {"name": args.hr_source_name}],
        "models": [
            {
                "kind": "azureOpenAI",
                "azureOpenAIParameters": _build_model_parameters(
                    auth_mode=args.auth_mode,
                    azure_openai_endpoint=azure_openai_endpoint,
                    chat_deployment=chat_deployment,
                    chat_model_name=chat_model_name,
                    azure_openai_key=azure_openai_key,
                    uami_resource_id=args.uami_resource_id,
                ),
            }
        ],
        "retrievalReasoningEffort": {"kind": "minimal"},
    }
    kb_url = f"{search_endpoint}/knowledgebases/{args.knowledge_base_name}?api-version={args.api_version}"
    logger.info("PUT knowledge base name=%s", args.knowledge_base_name)
    try:
        kb_resp = _put_json(kb_url, kb_body, headers)
        records.append(KnowledgeSetupRecord("knowledge_base", args.knowledge_base_name, True, kb_resp.status_code, kb_url))
    except Exception as exc:  # noqa: BLE001
        response = getattr(exc, "response", None)
        message = response.text if response is not None and response.text else str(exc)
        records.append(KnowledgeSetupRecord("knowledge_base", args.knowledge_base_name, False, getattr(response, "status_code", None), kb_url, type(exc).__name__, message))

    mcp_endpoint = f"{search_endpoint}/knowledgebases/{args.knowledge_base_name}/mcp?api-version={args.api_version}"
    payload = {
        "metadata": {
            "run_id": run_id,
            "search_endpoint": search_endpoint,
            "api_version": args.api_version,
            "knowledge_base_name": args.knowledge_base_name,
            "mcp_endpoint": mcp_endpoint,
            "auth_mode": args.auth_mode,
            "uami_resource_id": args.uami_resource_id if args.auth_mode == "managed-identity" else "",
            "hr_index_name": args.hr_index_name,
            "health_index_name": args.health_index_name,
        },
        "records": [asdict(r) for r in records],
    }
    md_path, json_path = _write_outputs(run_id, payload)

    cache_payload = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "search_endpoint": search_endpoint,
        "api_version": args.api_version,
        "knowledge_base_name": args.knowledge_base_name,
        "mcp_endpoint": mcp_endpoint,
        "auth_mode": args.auth_mode,
        "uami_resource_id": args.uami_resource_id if args.auth_mode == "managed-identity" else "",
        "knowledge_sources": [args.hr_source_name, args.health_source_name],
        "indexes": [args.hr_index_name, args.health_index_name],
        "records": [asdict(r) for r in records],
    }
    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
    logger.info("Artifacts markdown=%s json=%s log=%s cache=%s", md_path, json_path, log_path, cache_path)
    failed = [r for r in records if not r.success]
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
