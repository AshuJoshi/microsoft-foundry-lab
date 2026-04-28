#!/usr/bin/env python3
"""V2 of search_agent_probe.py — adds a follow-up turn within the same conversation.

By default, every case now runs a search follow-up in the same conversation after
the initial search turn.  Use ``--no-followup`` to disable the follow-up and get
the same single-turn behavior as V1.

The follow-up prompt asks the model to refine or extend its previous search answer,
which exercises the model's ability to issue a *second* web-search tool call within
an ongoing conversation — the exact scenario where some models (e.g. Claude) have
been reported to fail.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, WebSearchApproximateLocation, WebSearchTool
from azure.identity import DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("search_agent_probe_v2")

SCRIPT_PREFIX = "search_agent_probe_v2"


@dataclass
class PromptCase:
    name: str
    prompt: str
    followup_prompt: str


@dataclass
class ProbeRecord:
    case_name: str
    turn: str  # "initial" or "followup"
    run_index: int
    success: bool
    status_code: int | None
    latency_ms: int
    model: str
    agent_name: str
    conversation_id: str | None
    prompt: str
    output_preview: str | None
    output_text: str | None
    citation_count: int
    citation_urls: list[str]
    citation_annotations: list[dict[str, Any]]
    mentioned_dates: list[str]
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
    p = argparse.ArgumentParser(
        description="Agent-based probe for Foundry WebSearchTool with same-conversation follow-up.",
    )
    p.add_argument("--model", default=cfg.default_model_deployment_name, help="Deployment/model name to test.")
    p.add_argument("--topic", default="Microsoft Foundry announcements", help="Topic substituted into prompt cases.")
    p.add_argument("--days-window", type=int, default=30, help="Date window for time-bounded prompts.")
    p.add_argument("--runs", type=int, default=1, help="Runs per case.")
    p.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    p.add_argument(
        "--cases",
        default="all",
        help="Comma-separated case set from: baseline,recent_window,strict_dates. Default: all.",
    )
    p.add_argument("--tool-choice", choices=["auto", "required"], default="required")
    p.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--country", default="US")
    p.add_argument("--region", default="WA")
    p.add_argument("--city", default="Seattle")
    p.add_argument(
        "--followup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send a search follow-up in the same conversation after each initial turn (default: on). "
        "Use --no-followup to disable.",
    )
    return p.parse_args()


def _setup_logging(level_name: str, log_path: Path) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )


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
                    "type": str(getattr(ann, "type", "")),
                    "url": str(getattr(ann, "url", "") or ""),
                    "title": str(getattr(ann, "title", "") or ""),
                    "start_index": getattr(ann, "start_index", None),
                    "end_index": getattr(ann, "end_index", None),
                    "text_excerpt": text_value,
                }
            )
    return annotations


def _extract_dates(text: str) -> list[str]:
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b",
    ]
    found: list[str] = []
    for pat in patterns:
        found.extend(re.findall(pat, text, flags=re.IGNORECASE))
    seen: set[str] = set()
    uniq: list[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def _build_cases(topic: str, days_window: int) -> list[PromptCase]:
    since_date = (datetime.now(timezone.utc) - timedelta(days=days_window)).date().isoformat()
    return [
        PromptCase(
            "baseline",
            f"Find two recent updates about {topic}. Cite sources with links.",
            f"Now search for one more update about {topic} that you did not mention above. Cite the source with a link.",
        ),
        PromptCase(
            "recent_window",
            (
                f"Find updates about {topic}. Use only sources from the last {days_window} days "
                f"(on or after {since_date}). Include at least two source links and mention the source dates."
            ),
            (
                f"Search again and find one additional update about {topic} from the last {days_window} days "
                f"(on or after {since_date}) that was not in your previous answer. Cite the source with a link and date."
            ),
        ),
        PromptCase(
            "strict_dates",
            (
                f"Research {topic}. Return exactly 3 bullets. Each bullet must include a date and a source link. "
                f"If a source date cannot be verified, say that explicitly."
            ),
            (
                f"Search for 2 more updates about {topic} that were not in your previous answer. "
                f"Return exactly 2 bullets, each with a date and a source link."
            ),
        ),
    ]


def _run_turn(
    *,
    openai_client: Any,
    recorder: HeaderRecorder,
    agent_name: str,
    conversation_id: str,
    prompt: str,
    case_name: str,
    turn: str,
    stream: bool,
    tool_choice: str,
    model: str,
    run_index: int,
) -> ProbeRecord:
    recorder.clear()
    t0 = time.perf_counter()
    output_text = ""
    citation_urls: list[str] = []
    citation_annotations: list[dict[str, Any]] = []
    try:
        if stream:
            s = openai_client.responses.create(
                stream=True,
                conversation=conversation_id,
                input=prompt,
                tool_choice=tool_choice,
                extra_body={"agent_reference": {"name": agent_name, "type": "agent_reference"}},
            )
            for event in s:
                if event.type == "response.output_text.delta":
                    output_text += event.delta or ""
                elif event.type == "response.output_item.done":
                    citation_urls.extend(_extract_citation_urls_from_item(event.item))
                    citation_annotations.extend(_extract_citation_annotations_from_item(event.item))
                elif event.type == "response.completed" and not output_text:
                    output_text = event.response.output_text or ""
        else:
            resp = openai_client.responses.create(
                conversation=conversation_id,
                input=prompt,
                tool_choice=tool_choice,
                extra_body={"agent_reference": {"name": agent_name, "type": "agent_reference"}},
            )
            output_text = resp.output_text or ""
            for item in getattr(resp, "output", []) or []:
                citation_urls.extend(_extract_citation_urls_from_item(item))
                citation_annotations.extend(_extract_citation_annotations_from_item(item))

        latency_ms = int((time.perf_counter() - t0) * 1000)
        urls = sorted(set(citation_urls))
        uniq_annotations: list[dict[str, Any]] = []
        seen_ann: set[tuple[Any, ...]] = set()
        for ann in citation_annotations:
            key = (
                ann.get("url"),
                ann.get("title"),
                ann.get("start_index"),
                ann.get("end_index"),
            )
            if key in seen_ann:
                continue
            seen_ann.add(key)
            uniq_annotations.append(ann)
        return ProbeRecord(
            case_name=case_name,
            turn=turn,
            run_index=run_index,
            success=True,
            status_code=recorder.last_status_code,
            latency_ms=latency_ms,
            model=getattr(locals().get("resp", None), "model", model),
            agent_name=agent_name,
            conversation_id=conversation_id,
            prompt=prompt,
            output_preview=output_text[:500] if output_text else None,
            output_text=output_text or None,
            citation_count=len(urls),
            citation_urls=urls,
            citation_annotations=uniq_annotations,
            mentioned_dates=_extract_dates(output_text),
            error_type=None,
            error_message=None,
            response_headers=recorder.last_headers,
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - t0) * 1000)
        resp = getattr(exc, "response", None)
        status = recorder.last_status_code
        headers: dict[str, str] = {}
        if resp is not None:
            status = getattr(resp, "status_code", status)
            try:
                headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
            except Exception:  # noqa: BLE001
                headers = {}
        return ProbeRecord(
            case_name=case_name,
            turn=turn,
            run_index=run_index,
            success=False,
            status_code=status,
            latency_ms=latency_ms,
            model=model,
            agent_name=agent_name,
            conversation_id=conversation_id,
            prompt=prompt,
            output_preview=None,
            output_text=None,
            citation_count=0,
            citation_urls=[],
            citation_annotations=[],
            mentioned_dates=[],
            error_type=type(exc).__name__,
            error_message=str(exc),
            response_headers=headers or recorder.last_headers,
        )


def _write_outputs(run_id: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{SCRIPT_PREFIX}_{run_id}.json"
    md_path = out_dir / f"{SCRIPT_PREFIX}_{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# Search Agent Probe V2 ({run_id})")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for k, v in payload["metadata"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Case | Turn | Run | Success | Status | Latency (ms) | Citations | x-request-id | Detail |")
    lines.append("|---|---|---:|---|---:|---:|---:|---|---|")
    for rec in payload["records"]:
        detail = rec["output_preview"] if rec["success"] else f"{rec['error_type']}: {rec['error_message']}"
        detail = (detail or "").replace("|", "\\|").replace("\n", " ")
        headers = rec.get("response_headers") or {}
        lines.append(
            f"| {rec['case_name']} | {rec['turn']} | {rec['run_index']} | {'yes' if rec['success'] else 'no'} | "
            f"{rec['status_code'] or '-'} | {rec['latency_ms']} | {rec['citation_count']} | "
            f"{headers.get('x-request-id', '-')} | {detail[:160]} |"
        )
    lines.append("")
    lines.append("## Failure Headers")
    lines.append("")
    lines.append("| Case | Turn | Run | Status | APIM Request ID | x-request-id | x-ms-region |")
    lines.append("|---|---|---:|---:|---|---|---|")
    for rec in payload["records"]:
        if rec["success"]:
            continue
        headers = rec.get("response_headers") or {}
        lines.append(
            f"| {rec['case_name']} | {rec['turn']} | {rec['run_index']} | {rec['status_code'] or '-'} | "
            f"{headers.get('apim-request-id', '-')} | {headers.get('x-request-id', '-')} | "
            f"{headers.get('x-ms-region', '-')} |"
        )
    lines.append("")
    lines.append("## Prompts")
    lines.append("")
    for rec in payload["records"]:
        lines.append(f"### {rec['case_name']} / {rec['turn']} / run {rec['run_index']}")
        lines.append("")
        lines.append(f"> {rec['prompt']}")
        lines.append("")
    lines.append("## Citations")
    lines.append("")
    for rec in payload["records"]:
        lines.append(f"### {rec['case_name']} / {rec['turn']} / run {rec['run_index']}")
        if not rec.get("citation_annotations"):
            lines.append("")
            lines.append("No citation annotations captured.")
            lines.append("")
            continue
        lines.append("")
        lines.append("| Title | URL |")
        lines.append("|---|---|")
        for ann in rec.get("citation_annotations", []):
            title = str(ann.get("title") or "-").replace("|", "\\|").replace("\n", " ")
            url = str(ann.get("url") or "-").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {title} | {url} |")
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main() -> None:
    args = parse_args()
    cfg = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "output"
    log_path = out_dir / f"{SCRIPT_PREFIX}_{run_id}.log"
    _setup_logging(args.log_level, log_path)

    all_cases = _build_cases(args.topic, args.days_window)
    if args.cases.strip().lower() == "all":
        selected_cases = all_cases
    else:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        selected_cases = [case for case in all_cases if case.name in wanted]
        if not selected_cases:
            raise SystemExit("No valid cases selected. Choose from: baseline, recent_window, strict_dates")

    turns_per_case = 2 if args.followup else 1
    total_calls = len(selected_cases) * args.runs * turns_per_case
    progress = 0
    location = WebSearchApproximateLocation(country=args.country, city=args.city, region=args.region)

    logger.info("run_id=%s", run_id)
    logger.info("model=%s", args.model)
    logger.info("runs=%s", args.runs)
    logger.info("followup=%s", args.followup)
    logger.info("cases=%s", [c.name for c in selected_cases])
    logger.info("tool_choice=%s stream=%s", args.tool_choice, args.stream)

    records: list[ProbeRecord] = []
    metadata = {
        "model": args.model,
        "topic": args.topic,
        "days_window": args.days_window,
        "runs_per_case": args.runs,
        "followup": args.followup,
        "tool_choice": args.tool_choice,
        "stream": args.stream,
        "project_endpoint": cfg.project_endpoint,
        "location": f"{args.city}, {args.region}, {args.country}",
    }

    with DefaultAzureCredential() as credential, AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client, project_client.get_openai_client() as openai_client:
        recorder = HeaderRecorder()
        _attach_hooks(openai_client, recorder)

        run_stamp = datetime.now(timezone.utc).strftime("%H%M%S")
        agent_name = f"{cfg.agent_name_prefix}-search-v2-{args.model.lower().replace('.', '-').replace('_', '-')}-{run_stamp}"
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=args.model,
                instructions="You are a web research assistant. Use web search when needed and cite sources.",
                tools=[WebSearchTool(user_location=location)],
            ),
            description="Exploration agent for Foundry WebSearchTool (v2 with follow-up).",
        )

        try:
            logger.info("agent_name=%s agent_version=%s", agent.name, agent.version)
            for case in selected_cases:
                for run_index in range(1, args.runs + 1):
                    conversation = openai_client.conversations.create()
                    logger.info(
                        "CONV  case=%s run=%s conversation_id=%s",
                        case.name,
                        run_index,
                        conversation.id,
                    )
                    try:
                        # --- initial turn ---
                        progress += 1
                        logger.info(
                            "START progress=%s/%s case=%s turn=initial run=%s",
                            progress, total_calls, case.name, run_index,
                        )
                        rec = _run_turn(
                            openai_client=openai_client,
                            recorder=recorder,
                            agent_name=agent.name,
                            conversation_id=conversation.id,
                            prompt=case.prompt,
                            case_name=case.name,
                            turn="initial",
                            stream=args.stream,
                            tool_choice=args.tool_choice,
                            model=args.model,
                            run_index=run_index,
                        )
                        records.append(rec)
                        if rec.success:
                            logger.info(
                                "DONE  progress=%s/%s case=%s turn=initial run=%s status=%s latency_ms=%s citations=%s served_model=%s",
                                progress, total_calls, case.name, run_index,
                                rec.status_code, rec.latency_ms, rec.citation_count, rec.model,
                            )
                        else:
                            logger.warning(
                                "FAIL  progress=%s/%s case=%s turn=initial run=%s status=%s latency_ms=%s error=%s",
                                progress, total_calls, case.name, run_index,
                                rec.status_code, rec.latency_ms, rec.error_message,
                            )

                        # --- follow-up turn (same conversation) ---
                        if args.followup:
                            progress += 1
                            logger.info(
                                "START progress=%s/%s case=%s turn=followup run=%s conversation_id=%s",
                                progress, total_calls, case.name, run_index, conversation.id,
                            )
                            followup_rec = _run_turn(
                                openai_client=openai_client,
                                recorder=recorder,
                                agent_name=agent.name,
                                conversation_id=conversation.id,
                                prompt=case.followup_prompt,
                                case_name=case.name,
                                turn="followup",
                                stream=args.stream,
                                tool_choice=args.tool_choice,
                                model=args.model,
                                run_index=run_index,
                            )
                            records.append(followup_rec)
                            if followup_rec.success:
                                logger.info(
                                    "DONE  progress=%s/%s case=%s turn=followup run=%s status=%s latency_ms=%s citations=%s served_model=%s",
                                    progress, total_calls, case.name, run_index,
                                    followup_rec.status_code, followup_rec.latency_ms,
                                    followup_rec.citation_count, followup_rec.model,
                                )
                            else:
                                logger.warning(
                                    "FAIL  progress=%s/%s case=%s turn=followup run=%s status=%s latency_ms=%s error=%s",
                                    progress, total_calls, case.name, run_index,
                                    followup_rec.status_code, followup_rec.latency_ms,
                                    followup_rec.error_message,
                                )
                    finally:
                        try:
                            openai_client.conversations.delete(conversation_id=conversation.id)
                        except Exception:  # noqa: BLE001
                            logger.warning("cleanup failed for conversation_id=%s", conversation.id)
        finally:
            try:
                project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
                logger.info("deleted agent_name=%s agent_version=%s", agent.name, agent.version)
            except Exception:  # noqa: BLE001
                logger.warning("cleanup failed for agent_name=%s agent_version=%s", agent.name, agent.version)

    payload = {"metadata": metadata, "records": [asdict(r) for r in records]}
    md_path, json_path = _write_outputs(run_id, payload)
    logger.info("Artifacts markdown=%s json=%s log=%s", md_path, json_path, log_path)


if __name__ == "__main__":
    main()
