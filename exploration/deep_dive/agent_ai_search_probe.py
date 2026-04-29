#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import AISearchIndexResource, AzureAISearchTool, AzureAISearchToolResource, PromptAgentDefinition
from azure.identity import DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("agent_ai_search_probe")
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "output" / "ai_search_indexes_cache.json"


@dataclass
class PromptCase:
    name: str
    prompt: str
    target_index: str | None
    expected_markers: list[str] = field(default_factory=list)


@dataclass
class ProbeRecord:
    index_name: str
    case_name: str
    run_index: int
    success: bool
    status_code: int | None
    latency_ms: int
    model: str
    agent_name: str
    conversation_id: str | None
    output_preview: str | None
    output_text: str | None
    output_item_types: list[str]
    citation_count: int
    citation_urls: list[str]
    citation_annotations: list[dict[str, Any]]
    mentioned_dates: list[str]
    expected_markers: list[str]
    matched_markers: list[str]
    expectation_met: bool | None
    error_type: str | None
    error_message: str | None
    response_headers: dict[str, str] | None


class HeaderRecorder:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.last_status_code: int | None = None
        self.last_headers: dict[str, str] = {}

    def on_request(self, request: httpx.Request) -> None:
        _ = request

    def on_response(self, response: httpx.Response) -> None:
        self.last_status_code = response.status_code
        self.last_headers = {k.lower(): v for k, v in response.headers.items()}


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Agent-based probe for AzureAISearchTool using existing Azure AI Search project connections.")
    p.add_argument("--model", default=cfg.default_model_deployment_name)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--cases", default="all", help="Comma-separated cases from: vacation_senior,recognition,mental_health_copay,office_visit_primary,unknown")
    p.add_argument("--index-names", default="hrdocs,healthdocs", help="Comma-separated Azure AI Search index names to probe.")
    p.add_argument("--project-connection-name", default=os.getenv("AZURE_AI_SEARCH_PROJECT_CONNECTION_NAME", ""))
    p.add_argument("--project-connection-id", default="")
    p.add_argument("--search-endpoint", default=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT", ""), help="Used to resolve a matching project connection when name/id is omitted.")
    p.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH), help="Optional cache from ai_search_index_setup.py to use as index-name default.")
    p.add_argument("--query-type", default="semantic")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--tool-choice", choices=["auto", "required"], default="required")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _setup_logging(level_name: str, log_path: Path) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S", handlers=handlers, force=True)


def _attach_hooks(openai_client: Any, rec: HeaderRecorder) -> None:
    inner = getattr(openai_client, "_client", None)
    if inner is not None and hasattr(inner, "event_hooks"):
        inner.event_hooks = {"request": [rec.on_request], "response": [rec.on_response]}


def _extract_citation_urls_from_item(item: Any) -> list[str]:
    urls: list[str] = []
    if getattr(item, "type", None) != "message":
        return urls
    for content in getattr(item, "content", []) or []:
        if getattr(content, "type", None) != "output_text":
            continue
        for ann in getattr(content, "annotations", []) or []:
            if getattr(ann, "type", None) == "url_citation" and getattr(ann, "url", None):
                urls.append(str(ann.url))
    return urls


def _extract_citation_annotations_from_item(item: Any) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    if getattr(item, "type", None) != "message":
        return annotations
    for content in getattr(item, "content", []) or []:
        if getattr(content, "type", None) != "output_text":
            continue
        text_value = getattr(content, "text", None)
        for ann in getattr(content, "annotations", []) or []:
            if getattr(ann, "type", None) != "url_citation":
                continue
            annotations.append(
                {
                    "type": str(getattr(ann, "type", "") or ""),
                    "url": str(getattr(ann, "url", "") or ""),
                    "title": str(getattr(ann, "title", "") or ""),
                    "start_index": getattr(ann, "start_index", None),
                    "end_index": getattr(ann, "end_index", None),
                    "text_excerpt": text_value,
                }
            )
    uniq: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for ann in annotations:
        key = (ann.get("url"), ann.get("title"), ann.get("start_index"), ann.get("end_index"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(ann)
    return uniq


def _extract_dates(text: str) -> list[str]:
    patterns = [r"\b\d{4}-\d{2}-\d{2}\b", r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b"]
    found: list[str] = []
    for pat in patterns:
        found.extend(re.findall(pat, text, flags=re.IGNORECASE))
    out: list[str] = []
    seen: set[str] = set()
    for value in found:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _marker_matches(marker: str, output_text: str) -> bool:
    if marker.lower() in output_text.lower():
        return True
    if marker.startswith("$") and marker[1:] in output_text:
        return True
    if marker.lower() == "i don't know" and "don't know" in output_text.lower():
        return True
    return False


def _build_cases() -> list[PromptCase]:
    return [
        PromptCase("vacation_senior", "How much vacation time does the Senior vacation tier receive? Answer with just the duration.", "hrdocs", ["4 weeks"]),
        PromptCase("recognition", "Name one employee recognition program mentioned in the HR documents.", "hrdocs", ["Innovator of the Month"]),
        PromptCase("mental_health_copay", "What is the typical in-network copayment for mental health and substance abuse services under Northwind Health Plus? Answer with just the amount.", "healthdocs", ["$30"]),
        PromptCase("office_visit_primary", "What is the office visit copay for primary care physicians under Northwind Health Plus? Answer with just the amount.", "healthdocs", ["$35"]),
        PromptCase("unknown", "What is the moon reimbursement policy? If the indexed content does not contain the answer, say exactly 'I don't know'.", None, ["I don't know"]),
    ]


def _to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "as_dict"):
        try:
            return obj.as_dict()
        except Exception:
            pass
    if isinstance(obj, dict):
        return obj
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def _resolve_connection(project_client: AIProjectClient, *, connection_id: str, connection_name: str, search_endpoint: str) -> tuple[str, str | None]:
    if connection_id:
        return connection_id, connection_name or None
    connections = list(project_client.connections.list())
    if connection_name:
        for conn in connections:
            raw = _to_dict(conn)
            if (raw.get("name") or getattr(conn, "name", None)) == connection_name:
                return str(raw.get("id") or ""), str(connection_name)
        raise SystemExit(f"Project connection not found by name: {connection_name}")
    if search_endpoint:
        matches: list[tuple[str, str | None]] = []
        for conn in connections:
            raw = _to_dict(conn)
            target = str(raw.get("target") or getattr(conn, "target", "") or "")
            if target.rstrip("/").lower() == search_endpoint.rstrip("/").lower():
                matches.append((str(raw.get("id") or ""), str(raw.get("name") or getattr(conn, "name", None) or "")))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise SystemExit(f"Multiple project connections match search endpoint {search_endpoint}. Pass --project-connection-name explicitly.")
    raise SystemExit("Could not resolve Azure AI Search project connection. Pass --project-connection-name or --project-connection-id.")


def _write_outputs(run_id: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"agent_ai_search_probe_{run_id}.json"
    md_path = out_dir / f"agent_ai_search_probe_{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines: list[str] = []
    lines.append(f"# Agent AI Search Probe ({run_id})")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for k, v in payload["metadata"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Index | Case | Run | Success | Status | Latency (ms) | Expectation | Detail |")
    lines.append("|---|---|---:|---|---:|---:|---|---|")
    for rec in payload["records"]:
        detail = rec["output_preview"] if rec["success"] else f"{rec['error_type']}: {rec['error_message']}"
        detail = (detail or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {rec['index_name']} | {rec['case_name']} | {rec['run_index']} | {'yes' if rec['success'] else 'no'} | {rec['status_code'] or '-'} | {rec['latency_ms']} | {rec['expectation_met']} | {detail[:160]} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main() -> None:
    args = parse_args()
    cfg = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"agent_ai_search_probe_{run_id}.log"
    _setup_logging(args.log_level, log_path)

    requested_cases = {c.strip() for c in args.cases.split(",") if c.strip()} if args.cases.strip().lower() != "all" else None
    all_cases = _build_cases()
    selected_cases = [c for c in all_cases if requested_cases is None or c.name in requested_cases]
    if not selected_cases:
        raise SystemExit("No valid cases selected.")

    index_names = [name.strip() for name in args.index_names.split(",") if name.strip()]
    cache_path = Path(args.cache_path)
    if cache_path.exists() and not index_names:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        index_names = [str(name) for name in cache.get("indexes", [])]
    if not index_names:
        raise SystemExit("No Azure AI Search index names provided.")

    logger.info("run_id=%s model=%s indexes=%s", run_id, args.model, index_names)
    records: list[ProbeRecord] = []

    with DefaultAzureCredential() as credential, AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client, project_client.get_openai_client() as openai_client:
        recorder = HeaderRecorder()
        _attach_hooks(openai_client, recorder)
        connection_id, resolved_connection_name = _resolve_connection(
            project_client,
            connection_id=args.project_connection_id,
            connection_name=args.project_connection_name,
            search_endpoint=args.search_endpoint,
        )
        logger.info("project_connection_id=%s project_connection_name=%s", connection_id, resolved_connection_name)

        for index_name in index_names:
            agent_name = f"{cfg.agent_name_prefix}-ais-{index_name}-{datetime.now(timezone.utc).strftime('%H%M%S')}".lower().replace("_", "-")
            agent = project_client.agents.create_version(
                agent_name=agent_name,
                definition=PromptAgentDefinition(
                    model=args.model,
                    instructions=(
                        "You are a grounded retrieval assistant. Use the attached Azure AI Search tool to answer from the indexed content. "
                        "If the indexed content does not contain the answer, say 'I don't know'."
                    ),
                    tools=[
                        AzureAISearchTool(
                            azure_ai_search=AzureAISearchToolResource(
                                indexes=[
                                    AISearchIndexResource(
                                        project_connection_id=connection_id,
                                        index_name=index_name,
                                        query_type=args.query_type,
                                        top_k=args.top_k,
                                    )
                                ]
                            )
                        )
                    ],
                ),
                description="Exploration agent for AzureAISearchTool.",
            )
            try:
                applicable_cases = [case for case in selected_cases if case.target_index in {None, index_name}]
                for case in applicable_cases:
                    for run_index in range(1, args.runs + 1):
                        conversation = openai_client.conversations.create()
                        logger.info("START index=%s case=%s run=%s conversation_id=%s", index_name, case.name, run_index, conversation.id)
                        t0 = time.perf_counter()
                        try:
                            recorder.clear()
                            resp = openai_client.responses.create(
                                conversation=conversation.id,
                                input=case.prompt,
                                tool_choice=args.tool_choice,
                                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                            )
                            latency_ms = int((time.perf_counter() - t0) * 1000)
                            output_text = resp.output_text or ""
                            output_item_types: list[str] = []
                            citation_urls: list[str] = []
                            citation_annotations: list[dict[str, Any]] = []
                            for item in getattr(resp, "output", []) or []:
                                output_item_types.append(str(getattr(item, "type", "") or ""))
                                citation_urls.extend(_extract_citation_urls_from_item(item))
                                citation_annotations.extend(_extract_citation_annotations_from_item(item))
                            matched = [m for m in case.expected_markers if _marker_matches(m, output_text)]
                            normalized_annotations = []
                            seen: set[tuple[Any, ...]] = set()
                            for ann in citation_annotations:
                                key = (ann.get("url"), ann.get("title"), ann.get("start_index"), ann.get("end_index"))
                                if key in seen:
                                    continue
                                seen.add(key)
                                normalized_annotations.append(ann)
                            rec = ProbeRecord(
                                index_name=index_name,
                                case_name=case.name,
                                run_index=run_index,
                                success=True,
                                status_code=recorder.last_status_code,
                                latency_ms=latency_ms,
                                model=getattr(resp, "model", None) or args.model,
                                agent_name=agent.name,
                                conversation_id=conversation.id,
                                output_preview=output_text[:500] if output_text else None,
                                output_text=output_text or None,
                                output_item_types=output_item_types,
                                citation_count=len(normalized_annotations),
                                citation_urls=sorted(set(citation_urls)),
                                citation_annotations=normalized_annotations,
                                mentioned_dates=_extract_dates(output_text),
                                expected_markers=case.expected_markers,
                                matched_markers=matched,
                                expectation_met=len(matched) == len(case.expected_markers),
                                error_type=None,
                                error_message=None,
                                response_headers=recorder.last_headers,
                            )
                            records.append(rec)
                            logger.info("DONE  index=%s case=%s run=%s expectation=%s served_model=%s", index_name, case.name, run_index, rec.expectation_met, rec.model)
                        except Exception as exc:  # noqa: BLE001
                            latency_ms = int((time.perf_counter() - t0) * 1000)
                            resp = getattr(exc, "response", None)
                            headers = recorder.last_headers
                            status = recorder.last_status_code
                            if resp is not None:
                                status = getattr(resp, "status_code", status)
                                try:
                                    headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
                                except Exception:
                                    pass
                            records.append(
                                ProbeRecord(
                                    index_name=index_name,
                                    case_name=case.name,
                                    run_index=run_index,
                                    success=False,
                                    status_code=status,
                                    latency_ms=latency_ms,
                                    model=args.model,
                                    agent_name=agent.name,
                                    conversation_id=conversation.id,
                                    output_preview=None,
                                    output_text=None,
                                    output_item_types=[],
                                    citation_count=0,
                                    citation_urls=[],
                                    citation_annotations=[],
                                    mentioned_dates=[],
                                    expected_markers=case.expected_markers,
                                    matched_markers=[],
                                    expectation_met=False,
                                    error_type=type(exc).__name__,
                                    error_message=str(exc),
                                    response_headers=headers,
                                )
                            )
                            logger.warning("FAIL  index=%s case=%s run=%s error=%s", index_name, case.name, run_index, exc)
                        finally:
                            try:
                                openai_client.conversations.delete(conversation_id=conversation.id)
                            except Exception:
                                logger.warning("cleanup failed for conversation_id=%s", conversation.id)
            finally:
                try:
                    project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
                    logger.info("deleted agent_name=%s agent_version=%s", agent.name, agent.version)
                except Exception:
                    logger.warning("cleanup failed for agent_name=%s agent_version=%s", agent.name, agent.version)

    payload = {
        "metadata": {
            "run_id": run_id,
            "model": args.model,
            "indexes": index_names,
            "project_connection_id": connection_id,
            "project_connection_name": resolved_connection_name,
            "query_type": args.query_type,
            "top_k": args.top_k,
            "tool_choice": args.tool_choice,
            "project_endpoint": cfg.project_endpoint,
        },
        "records": [asdict(r) for r in records],
    }
    md_path, json_path = _write_outputs(run_id, payload)
    logger.info("Artifacts markdown=%s json=%s log=%s", md_path, json_path, log_path)


if __name__ == "__main__":
    main()
