#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
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

logger = logging.getLogger("agent_web_search_stepwise_probe")


@dataclass
class UsageSnapshot:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class StepRecord:
    timestamp_utc: str
    command: str
    status_code: int | None
    success: bool
    latency_ms: int
    prompt_chars: int
    output_chars: int
    usage: UsageSnapshot
    conversation_id: str | None
    output_preview: str | None = None
    output_text: str | None = None
    output_item_types: list[str] = field(default_factory=list)
    citation_count: int = 0
    citation_urls: list[str] = field(default_factory=list)
    citation_annotations: list[dict[str, Any]] = field(default_factory=list)
    mentioned_dates: list[str] = field(default_factory=list)
    case_name: str | None = None
    apim_request_id: str | None = None
    x_request_id: str | None = None
    x_ms_region: str | None = None
    served_model: str | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class ProbeState:
    run_id: str
    created_at_utc: str
    project_endpoint: str
    mode: str
    model: str
    topic: str
    days_window: int
    country: str
    region: str
    city: str
    agent_name: str | None = None
    agent_version: str | None = None
    conversation_id: str | None = None
    last_response_id: str | None = None
    records: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PromptCase:
    name: str
    prompt: str


class HeaderRecorder:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.status_code: int | None = None
        self.headers: dict[str, str] = {}

    def on_request(self, request: httpx.Request) -> None:
        _ = request

    def on_response(self, response: httpx.Response) -> None:
        self.status_code = response.status_code
        self.headers = {k.lower(): v for k, v in response.headers.items()}


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(
        description=(
            "Stepwise agent-based web-search probe that preserves remote agent and conversation state "
            "until explicit cleanup. Designed for project runtime now and published runtime later."
        )
    )
    p.add_argument(
        "--state",
        default="",
        help="Path to state JSON file. Defaults to output/agent_web_search_stepwise_<run_id>.state.json",
    )
    p.add_argument("--mode", choices=["project"], default="project", help="Runtime mode. Only 'project' is implemented currently.")
    p.add_argument("--model", default=cfg.default_model_deployment_name, help="Deployment/model name.")
    p.add_argument("--topic", default="Microsoft Foundry announcements", help="Topic substituted into named prompt cases.")
    p.add_argument("--days-window", type=int, default=30, help="Date window for time-bounded named prompt cases.")
    p.add_argument("--country", default="US")
    p.add_argument("--region", default="WA")
    p.add_argument("--city", default="Seattle")
    p.add_argument("--log-level", default="INFO")

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("create-agent", help="Create a remote Prompt agent configured with WebSearchTool.")
    sub.add_parser("create-conversation", help="Create a remote conversation and persist it in state.")

    p_case = sub.add_parser("ask-case", help="Run one deterministic named web-search case.")
    p_case.add_argument("--case", required=True, help="Case name from: baseline,recent_window,strict_dates")
    p_case.add_argument("--max-output-tokens", type=int, default=260)
    p_case.add_argument("--tool-choice", choices=["auto", "required"], default="required")

    p_ask = sub.add_parser("ask", help="Send one arbitrary web-search prompt to the remote agent.")
    p_ask.add_argument("--message", required=True)
    p_ask.add_argument("--max-output-tokens", type=int, default=260)
    p_ask.add_argument("--tool-choice", choices=["auto", "required"], default="required")

    p_show = sub.add_parser("show-state", help="Print current local state.")
    p_show.add_argument("--summary", action="store_true", help="Print a concise run summary instead of the full state.")

    p_cleanup = sub.add_parser("cleanup", help="Delete remote conversation and/or agent.")
    p_cleanup.add_argument("--delete-conversation", action="store_true")
    p_cleanup.add_argument("--delete-agent", action="store_true")
    return p.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_state_path(run_id: str) -> Path:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"agent_web_search_stepwise_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "state.json"


def _state_path(args: argparse.Namespace) -> Path:
    if args.state:
        return Path(args.state)
    return _default_state_path(_run_id())


def _log_path_for_state(state_path: Path) -> Path:
    return state_path.with_name("state.log")


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


def _build_cases(topic: str, days_window: int) -> dict[str, PromptCase]:
    since_date = (datetime.now(timezone.utc) - timedelta(days=days_window)).date().isoformat()
    cases = [
        PromptCase("baseline", f"Find two recent updates about {topic}. Cite sources with links."),
        PromptCase(
            "recent_window",
            (
                f"Find updates about {topic}. Use only sources from the last {days_window} days "
                f"(on or after {since_date}). Include at least two source links and mention the source dates."
            ),
        ),
        PromptCase(
            "strict_dates",
            (
                f"Research {topic}. Return exactly 3 bullets. Each bullet must include a date and a source link. "
                f"If a source date cannot be verified, say that explicitly."
            ),
        ),
    ]
    return {case.name: case for case in cases}


def _attach_hooks(openai_client: Any, recorder: HeaderRecorder) -> None:
    inner = getattr(openai_client, "_client", None)
    if inner is not None and hasattr(inner, "event_hooks"):
        inner.event_hooks = {"request": [recorder.on_request], "response": [recorder.on_response]}


def _call_with_capture(recorder: HeaderRecorder, fn: Any) -> tuple[Any | None, dict[str, str], int, Exception | None]:
    recorder.reset()
    t0 = time.perf_counter()
    try:
        out = fn()
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return out, recorder.headers.copy(), latency_ms, None
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - t0) * 1000)
        headers = recorder.headers.copy()
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
            except Exception:  # noqa: BLE001
                pass
        return None, headers, latency_ms, exc


def _usage_snapshot(response: Any) -> UsageSnapshot:
    usage = getattr(response, "usage", None)
    if usage is None:
        return UsageSnapshot()
    return UsageSnapshot(
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def _output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    return text if isinstance(text, str) else ""


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


def _normalize_annotations(citation_annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    return uniq_annotations


def _load_state(path: Path) -> ProbeState:
    return ProbeState(**json.loads(path.read_text(encoding="utf-8")))


def _save_state(path: Path, state: ProbeState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def _record_success(state: ProbeState, command: str, prompt: str, response: Any, headers: dict[str, str], latency_ms: int, case_name: str | None = None) -> None:
    output_text = _output_text(response)
    usage = _usage_snapshot(response)
    output_item_types: list[str] = []
    citation_urls: list[str] = []
    citation_annotations: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        output_item_types.append(str(getattr(item, "type", "")))
        citation_urls.extend(_extract_citation_urls_from_item(item))
        citation_annotations.extend(_extract_citation_annotations_from_item(item))
    normalized_annotations = _normalize_annotations(citation_annotations)
    state.records.append(
        asdict(
            StepRecord(
                timestamp_utc=_utc_now(),
                command=command,
                status_code=headers.get(":status") and int(headers[":status"]) or None,
                success=True,
                latency_ms=latency_ms,
                prompt_chars=len(prompt),
                output_chars=len(output_text),
                usage=usage,
                conversation_id=state.conversation_id,
                output_preview=output_text[:500] if output_text else None,
                output_text=output_text or None,
                output_item_types=output_item_types,
                citation_count=len(normalized_annotations),
                citation_urls=sorted(set(citation_urls)),
                citation_annotations=normalized_annotations,
                mentioned_dates=_extract_dates(output_text),
                case_name=case_name,
                apim_request_id=headers.get("apim-request-id"),
                x_request_id=headers.get("x-request-id"),
                x_ms_region=headers.get("x-ms-region"),
                served_model=getattr(response, "model", None) or state.model,
            )
        )
    )


def _record_failure(state: ProbeState, command: str, prompt: str, headers: dict[str, str], latency_ms: int, err: Exception, case_name: str | None = None) -> None:
    state.records.append(
        asdict(
            StepRecord(
                timestamp_utc=_utc_now(),
                command=command,
                status_code=headers.get(":status") and int(headers[":status"]) or None,
                success=False,
                latency_ms=latency_ms,
                prompt_chars=len(prompt),
                output_chars=0,
                usage=UsageSnapshot(),
                conversation_id=state.conversation_id,
                case_name=case_name,
                apim_request_id=headers.get("apim-request-id"),
                x_request_id=headers.get("x-request-id"),
                x_ms_region=headers.get("x-ms-region"),
                error_type=type(err).__name__,
                error_message=str(err),
            )
        )
    )


def _record_lifecycle(
    state: ProbeState,
    *,
    command: str,
    success: bool,
    latency_ms: int = 0,
    detail: str | None = None,
    headers: dict[str, str] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    headers = headers or {}
    state.records.append(
        asdict(
            StepRecord(
                timestamp_utc=_utc_now(),
                command=command,
                status_code=headers.get(":status") and int(headers[":status"]) or None,
                success=success,
                latency_ms=latency_ms,
                prompt_chars=0,
                output_chars=len(detail or ""),
                usage=UsageSnapshot(),
                conversation_id=state.conversation_id,
                output_preview=detail,
                output_text=detail,
                apim_request_id=headers.get("apim-request-id"),
                x_request_id=headers.get("x-request-id"),
                x_ms_region=headers.get("x-ms-region"),
                served_model=state.model,
                error_type=error_type,
                error_message=error_message,
            )
        )
    )


def _summarize_state(state: ProbeState) -> dict[str, Any]:
    command_counts: dict[str, int] = {}
    success_count = 0
    failure_count = 0
    last_record = state.records[-1] if state.records else None
    for rec in state.records:
        command = str(rec.get("command") or "")
        command_counts[command] = command_counts.get(command, 0) + 1
        if rec.get("success"):
            success_count += 1
        else:
            failure_count += 1
    return {
        "run_id": state.run_id,
        "state_path_shape": "run-folder",
        "model": state.model,
        "topic": state.topic,
        "location": f"{state.city}, {state.region}, {state.country}",
        "agent_name": state.agent_name,
        "agent_version": state.agent_version,
        "conversation_id": state.conversation_id,
        "record_count": len(state.records),
        "success_count": success_count,
        "failure_count": failure_count,
        "command_counts": command_counts,
        "last_command": None if last_record is None else last_record.get("command"),
        "last_case": None if last_record is None else last_record.get("case_name"),
        "last_status": None if last_record is None else last_record.get("status_code"),
        "last_citations": None if last_record is None else last_record.get("citation_count"),
        "last_served_model": None if last_record is None else last_record.get("served_model"),
        "last_preview": None if last_record is None else last_record.get("output_preview"),
    }


def main() -> None:
    args = parse_args()
    state_path = _state_path(args)
    log_path = _log_path_for_state(state_path)
    _setup_logging(args.log_level, log_path)
    cfg = load_config()

    if state_path.exists():
        state = _load_state(state_path)
    else:
        state = ProbeState(
            run_id=_run_id(),
            created_at_utc=_utc_now(),
            project_endpoint=cfg.project_endpoint,
            mode=args.mode,
            model=args.model,
            topic=args.topic,
            days_window=args.days_window,
            country=args.country,
            region=args.region,
            city=args.city,
        )
        _save_state(state_path, state)

    case_map = _build_cases(state.topic, state.days_window)
    logger.info(
        "state=%s command=%s mode=%s model=%s location=%s, %s, %s topic=%s",
        state_path,
        args.command,
        state.mode,
        state.model,
        state.city,
        state.region,
        state.country,
        state.topic,
    )

    recorder = HeaderRecorder()
    with DefaultAzureCredential() as credential, AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client:
        with project_client.get_openai_client() as openai_client:
            _attach_hooks(openai_client, recorder)

            if args.command == "create-agent":
                location = WebSearchApproximateLocation(country=state.country, region=state.region, city=state.city)
                agent_name = f"{cfg.agent_name_prefix}-web-step-{_run_id().lower()}"
                response, headers, latency_ms, err = _call_with_capture(
                    recorder,
                    lambda: project_client.agents.create_version(
                        agent_name=agent_name,
                        definition=PromptAgentDefinition(
                            model=state.model,
                            instructions="You are a web research assistant. Use web search when needed, ground answers in current sources, and cite links.",
                            tools=[WebSearchTool(user_location=location)],
                        ),
                        description="Stepwise exploration agent for Foundry WebSearchTool.",
                    ),
                )
                if err is not None:
                    _record_lifecycle(
                        state,
                        command="create-agent",
                        success=False,
                        latency_ms=latency_ms,
                        detail=f"agent_name={agent_name}",
                        headers=headers,
                        error_type=type(err).__name__,
                        error_message=str(err),
                    )
                    _save_state(state_path, state)
                    raise SystemExit(1)
                agent = response
                state.agent_name = agent.name
                state.agent_version = str(agent.version)
                _record_lifecycle(
                    state,
                    command="create-agent",
                    success=True,
                    latency_ms=latency_ms,
                    detail=f"agent_name={state.agent_name} agent_version={state.agent_version}",
                    headers=headers,
                )
                _save_state(state_path, state)
                logger.info("created agent_name=%s agent_version=%s", state.agent_name, state.agent_version)
                return

            if args.command == "create-conversation":
                if not state.agent_name:
                    raise SystemExit("State file has no agent yet. Run create-agent first.")
                response, headers, latency_ms, err = _call_with_capture(recorder, lambda: openai_client.conversations.create())
                if err is not None:
                    _record_lifecycle(
                        state,
                        command="create-conversation",
                        success=False,
                        latency_ms=latency_ms,
                        headers=headers,
                        error_type=type(err).__name__,
                        error_message=str(err),
                    )
                    _save_state(state_path, state)
                    raise SystemExit(1)
                conversation = response
                state.conversation_id = conversation.id
                _record_lifecycle(
                    state,
                    command="create-conversation",
                    success=True,
                    latency_ms=latency_ms,
                    detail=f"conversation_id={state.conversation_id}",
                    headers=headers,
                )
                _save_state(state_path, state)
                logger.info("created conversation_id=%s", state.conversation_id)
                return

            if args.command == "show-state":
                payload = _summarize_state(state) if args.summary else asdict(state)
                print(json.dumps(payload, indent=2))
                return

            if args.command in {"ask-case", "ask"}:
                if not state.agent_name:
                    raise SystemExit("State file has no agent yet. Run create-agent first.")
                if not state.conversation_id:
                    raise SystemExit("State file has no conversation yet. Run create-conversation first.")

                if args.command == "ask-case":
                    case_name = args.case.strip()
                    if case_name not in case_map:
                        raise SystemExit(f"Unknown case: {case_name}. Valid cases: {', '.join(sorted(case_map))}")
                    prompt = case_map[case_name].prompt
                else:
                    case_name = None
                    prompt = args.message

                logger.info("START command=%s case=%s conversation_id=%s", args.command, case_name or "-", state.conversation_id)
                response, headers, latency_ms, err = _call_with_capture(
                    recorder,
                    lambda: openai_client.responses.create(
                        conversation=state.conversation_id,
                        input=prompt,
                        max_output_tokens=args.max_output_tokens,
                        tool_choice=args.tool_choice,
                        extra_body={"agent_reference": {"name": state.agent_name, "type": "agent_reference"}},
                    ),
                )
                if err is not None:
                    _record_failure(state, args.command, prompt, headers, latency_ms, err, case_name=case_name)
                    _save_state(state_path, state)
                    logger.warning("FAIL command=%s case=%s latency_ms=%s error=%s", args.command, case_name or "-", latency_ms, err)
                    raise SystemExit(1)

                _record_success(state, args.command, prompt, response, headers, latency_ms, case_name=case_name)
                state.last_response_id = getattr(response, "id", None)
                _save_state(state_path, state)
                last = state.records[-1]
                logger.info(
                    "DONE  command=%s case=%s total_tokens=%s citations=%s served_model=%s",
                    args.command,
                    case_name or "-",
                    ((last.get("usage") or {}).get("total_tokens")),
                    last.get("citation_count"),
                    last.get("served_model"),
                )
                return

            if args.command == "cleanup":
                if args.delete_conversation and state.conversation_id:
                    deleted_id = state.conversation_id
                    _, headers, latency_ms, err = _call_with_capture(recorder, lambda: openai_client.conversations.delete(deleted_id))
                    if err is None:
                        logger.info("deleted conversation_id=%s", deleted_id)
                        _record_lifecycle(
                            state,
                            command="cleanup-conversation",
                            success=True,
                            latency_ms=latency_ms,
                            detail=f"conversation_id={deleted_id}",
                            headers=headers,
                        )
                    else:
                        _record_lifecycle(
                            state,
                            command="cleanup-conversation",
                            success=False,
                            latency_ms=latency_ms,
                            detail=f"conversation_id={deleted_id}",
                            headers=headers,
                            error_type=type(err).__name__,
                            error_message=str(err),
                        )
                    state.conversation_id = None
                if args.delete_agent and state.agent_name and state.agent_version:
                    deleted_name = state.agent_name
                    deleted_version = state.agent_version
                    _, headers, latency_ms, err = _call_with_capture(
                        recorder,
                        lambda: project_client.agents.delete_version(agent_name=deleted_name, agent_version=deleted_version),
                    )
                    if err is None:
                        logger.info("deleted agent_name=%s agent_version=%s", deleted_name, deleted_version)
                        _record_lifecycle(
                            state,
                            command="cleanup-agent",
                            success=True,
                            latency_ms=latency_ms,
                            detail=f"agent_name={deleted_name} agent_version={deleted_version}",
                            headers=headers,
                        )
                    else:
                        _record_lifecycle(
                            state,
                            command="cleanup-agent",
                            success=False,
                            latency_ms=latency_ms,
                            detail=f"agent_name={deleted_name} agent_version={deleted_version}",
                            headers=headers,
                            error_type=type(err).__name__,
                            error_message=str(err),
                        )
                    state.agent_name = None
                    state.agent_version = None
                _save_state(state_path, state)
                return


if __name__ == "__main__":
    main()
