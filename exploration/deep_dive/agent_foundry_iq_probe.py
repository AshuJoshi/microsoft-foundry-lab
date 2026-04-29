#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MCPTool, PromptAgentDefinition
from azure.identity import DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("agent_foundry_iq_probe")
DEFAULT_KNOWLEDGE_CACHE_PATH = Path(__file__).resolve().parent / "output" / "fiq_knowledge_cache.json"
DEFAULT_CONNECTION_CACHE_PATH = Path(__file__).resolve().parent / "output" / "fiq_project_connection_cache.json"


@dataclass
class PromptCase:
    name: str
    prompt: str
    expected_markers: list[str] = field(default_factory=list)


@dataclass
class ProbeRecord:
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
    p = argparse.ArgumentParser(description="Agent-based probe for Foundry IQ MCP-backed knowledge bases.")
    p.add_argument("--model", default=cfg.default_model_deployment_name)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--cases", default="all", help="Comma-separated cases from: hr_policy,health_fact,cross_source_compare,unknown")
    p.add_argument("--knowledge-cache-path", default=str(DEFAULT_KNOWLEDGE_CACHE_PATH))
    p.add_argument("--connection-cache-path", default=str(DEFAULT_CONNECTION_CACHE_PATH))
    p.add_argument("--mcp-endpoint", default="")
    p.add_argument("--project-connection-name", default="")
    p.add_argument("--tool-choice", choices=["auto", "required"], default="required")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _setup_logging(level_name: str, log_path: Path) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S", handlers=handlers, force=True)


def _build_cases() -> list[PromptCase]:
    return [
        PromptCase("hr_policy", "How much vacation time does the Senior vacation tier receive? Answer with just the duration.", ["4 weeks"]),
        PromptCase("health_fact", "What is the typical in-network copayment for mental health and substance abuse services under Northwind Health Plus? Answer with just the amount.", ["$30"]),
        PromptCase("cross_source_compare", "What is one employee recognition program from the HR documents, and what is the primary care physician office visit copay under Northwind Health Plus? Answer in one sentence.", ["Innovator of the Month", "$35"]),
        PromptCase("unknown", "What is the moon reimbursement policy? If the knowledge base does not contain the answer, say exactly 'I don't know'.", ["I don't know"]),
    ]


def _attach_hooks(openai_client: Any, recorder: HeaderRecorder) -> None:
    inner = getattr(openai_client, "_client", None)
    if inner is not None and hasattr(inner, "event_hooks"):
        inner.event_hooks = {"request": [recorder.on_request], "response": [recorder.on_response]}


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
    out: list[dict[str, Any]] = []
    if getattr(item, "type", None) != "message":
        return out
    for content in getattr(item, "content", []) or []:
        if getattr(content, "type", None) != "output_text":
            continue
        text_value = getattr(content, "text", None)
        for ann in getattr(content, "annotations", []) or []:
            out.append(
                {
                    "type": str(getattr(ann, "type", "") or ""),
                    "url": str(getattr(ann, "url", "") or ""),
                    "title": str(getattr(ann, "title", "") or ""),
                    "filename": str(getattr(ann, "filename", "") or ""),
                    "file_id": str(getattr(ann, "file_id", "") or ""),
                    "container_id": str(getattr(ann, "container_id", "") or ""),
                    "start_index": getattr(ann, "start_index", None),
                    "end_index": getattr(ann, "end_index", None),
                    "text_excerpt": text_value,
                }
            )
    uniq: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for ann in out:
        key = (ann.get("type"), ann.get("url"), ann.get("filename"), ann.get("file_id"), ann.get("start_index"), ann.get("end_index"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(ann)
    return uniq


def _extract_dates(text: str) -> list[str]:
    pats = [r"\b\d{4}-\d{2}-\d{2}\b", r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b"]
    found: list[str] = []
    for pat in pats:
        found.extend(re.findall(pat, text, flags=re.IGNORECASE))
    out: list[str] = []
    for value in found:
        if value not in out:
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


def _write_outputs(run_id: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"agent_foundry_iq_probe_{run_id}.json"
    md_path = out_dir / f"agent_foundry_iq_probe_{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [f"# Agent Foundry IQ Probe ({run_id})", "", "## Inputs", ""]
    for k, v in payload["metadata"].items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Results", "", "| Case | Run | Success | Status | Latency (ms) | Expectation | Detail |", "|---|---:|---|---:|---:|---|---|"])
    for rec in payload["records"]:
        detail = rec["output_preview"] if rec["success"] else f"{rec['error_type']}: {rec['error_message']}"
        detail = (detail or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {rec['case_name']} | {rec['run_index']} | {'yes' if rec['success'] else 'no'} | {rec['status_code'] or '-'} | {rec['latency_ms']} | {rec['expectation_met']} | {detail[:160]} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main() -> None:
    args = parse_args()
    cfg = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"agent_foundry_iq_probe_{run_id}.log"
    _setup_logging(args.log_level, log_path)

    requested = {c.strip() for c in args.cases.split(",") if c.strip()} if args.cases.strip().lower() != "all" else None
    all_cases = _build_cases()
    selected_cases = [c for c in all_cases if requested is None or c.name in requested]
    if not selected_cases:
        raise SystemExit("No valid cases selected.")

    mcp_endpoint = args.mcp_endpoint
    knowledge_cache = Path(args.knowledge_cache_path)
    if not mcp_endpoint and knowledge_cache.exists():
        mcp_endpoint = str(json.loads(knowledge_cache.read_text(encoding="utf-8")).get("mcp_endpoint") or "")
    if not mcp_endpoint:
        raise SystemExit("Could not resolve MCP endpoint. Pass --mcp-endpoint or create fiq_knowledge_setup cache first.")

    project_connection_name = args.project_connection_name
    connection_cache = Path(args.connection_cache_path)
    if not project_connection_name and connection_cache.exists():
        project_connection_name = str(json.loads(connection_cache.read_text(encoding="utf-8")).get("project_connection_name") or "")
    if not project_connection_name:
        raise SystemExit("Could not resolve project connection name. Pass --project-connection-name or create fiq_project_connection_setup cache first.")

    records: list[ProbeRecord] = []
    with DefaultAzureCredential() as credential, AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client, project_client.get_openai_client() as openai_client:
        recorder = HeaderRecorder()
        _attach_hooks(openai_client, recorder)
        agent_name = f"{cfg.agent_name_prefix}-fiq-{datetime.now(timezone.utc).strftime('%H%M%S')}".lower().replace("_", "-")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=args.model,
                instructions=(
                    "You are a helpful assistant that must use the knowledge base to answer all questions. "
                    "You must never answer from your own knowledge. "
                    "If the knowledge base does not contain the answer, respond with 'I don't know'."
                ),
                tools=[
                    MCPTool(
                        server_label="knowledge-base",
                        server_url=mcp_endpoint,
                        require_approval="never",
                        allowed_tools=["knowledge_base_retrieve"],
                        project_connection_id=project_connection_name,
                    )
                ],
            ),
            description="Exploration agent for Foundry IQ MCP-backed knowledge bases.",
        )
        try:
            for case in selected_cases:
                for run_index in range(1, args.runs + 1):
                    conversation = openai_client.conversations.create()
                    logger.info("START case=%s run=%s conversation_id=%s", case.name, run_index, conversation.id)
                    t0 = time.perf_counter()
                    try:
                        recorder.clear()
                        resp = openai_client.responses.create(
                            conversation=conversation.id,
                            tool_choice=args.tool_choice,
                            input=case.prompt,
                            extra_body={"agent_reference": {"name": agent.name, "version": agent.version, "type": "agent_reference"}},
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
                        records.append(
                            ProbeRecord(
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
                                citation_count=len(citation_annotations),
                                citation_urls=sorted(set(citation_urls)),
                                citation_annotations=citation_annotations,
                                mentioned_dates=_extract_dates(output_text),
                                expected_markers=case.expected_markers,
                                matched_markers=matched,
                                expectation_met=len(matched) == len(case.expected_markers),
                                error_type=None,
                                error_message=None,
                                response_headers=recorder.last_headers,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        latency_ms = int((time.perf_counter() - t0) * 1000)
                        records.append(
                            ProbeRecord(
                                case_name=case.name,
                                run_index=run_index,
                                success=False,
                                status_code=recorder.last_status_code,
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
                                response_headers=recorder.last_headers,
                            )
                        )
                        logger.warning("FAIL  case=%s run=%s error=%s", case.name, run_index, exc)
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
            "mcp_endpoint": mcp_endpoint,
            "project_connection_name": project_connection_name,
            "tool_choice": args.tool_choice,
            "project_endpoint": cfg.project_endpoint,
        },
        "records": [asdict(r) for r in records],
    }
    md_path, json_path = _write_outputs(run_id, payload)
    logger.info("Artifacts markdown=%s json=%s log=%s", md_path, json_path, log_path)


if __name__ == "__main__":
    main()
