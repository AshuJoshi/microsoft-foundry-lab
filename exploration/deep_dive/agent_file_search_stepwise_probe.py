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
from azure.ai.projects.models import FileSearchTool, PromptAgentDefinition
from azure.identity import DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("agent_file_search_stepwise_probe")


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
    file_search_results: list[dict[str, Any]] = field(default_factory=list)
    citation_count: int = 0
    citation_urls: list[str] = field(default_factory=list)
    citation_files: list[str] = field(default_factory=list)
    citation_annotations: list[dict[str, Any]] = field(default_factory=list)
    case_name: str | None = None
    expected_markers: list[str] = field(default_factory=list)
    matched_markers: list[str] = field(default_factory=list)
    expectation_met: bool | None = None
    apim_request_id: str | None = None
    x_request_id: str | None = None
    x_ms_region: str | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class ProbeState:
    run_id: str
    created_at_utc: str
    project_endpoint: str
    mode: str
    model: str
    vector_store_id: str
    sample_corpus: str | None = None
    cache_path: str | None = None
    agent_name: str | None = None
    agent_version: str | None = None
    conversation_id: str | None = None
    last_response_id: str | None = None
    records: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PromptCase:
    name: str
    prompt: str
    expected_markers: list[str] = field(default_factory=list)


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
            "Stepwise agent-based file-search probe that preserves remote agent and conversation state "
            "until explicit cleanup. Designed for project runtime now and published runtime later."
        )
    )
    p.add_argument("--state", default="", help="Path to state JSON file. Defaults to output/agent_file_search_stepwise_<run_id>.state.json")
    p.add_argument("--mode", choices=["project"], default="project", help="Runtime mode. Only 'project' is implemented currently.")
    p.add_argument("--model", default=cfg.default_model_deployment_name, help="Deployment/model name.")
    p.add_argument(
        "--cache-path",
        default="exploration/deep_dive/output/file_search_vector_store.json",
        help="Cache file produced by vector_store_index.py",
    )
    p.add_argument("--log-level", default="INFO")

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("create-agent", help="Create a remote Prompt agent configured with FileSearchTool.")
    sub.add_parser("create-conversation", help="Create a remote conversation and persist it in state.")

    p_case = sub.add_parser("ask-case", help="Run one deterministic named file-search case.")
    p_case.add_argument("--case", required=True, help="Case name from: ids,vendor,total_due,highest_total,quote_ink,summary")
    p_case.add_argument("--max-output-tokens", type=int, default=220)
    p_case.add_argument("--tool-choice", choices=["auto", "required"], default="required")

    p_ask = sub.add_parser("ask", help="Send one arbitrary file-search prompt to the remote agent.")
    p_ask.add_argument("--message", required=True)
    p_ask.add_argument("--max-output-tokens", type=int, default=220)
    p_ask.add_argument("--tool-choice", choices=["auto", "required"], default="required")

    sub.add_parser("show-state", help="Print current local state.")

    p_cleanup = sub.add_parser("cleanup", help="Delete remote conversation and/or agent.")
    p_cleanup.add_argument("--delete-conversation", action="store_true")
    p_cleanup.add_argument("--delete-agent", action="store_true")
    return p.parse_args()


def _build_cases() -> dict[str, PromptCase]:
    cases = [
        PromptCase("ids", "List all invoice IDs and PO numbers found in the indexed invoice files.", ["INV-1001", "INV-1005", "PO-7781", "PO-7931"]),
        PromptCase("vendor", "What is the vendor for invoice INV-1004? Answer with the vendor name only.", ["Northwind IT Services"]),
        PromptCase("total_due", "What is the total due for invoice INV-1002? Answer with the amount only.", ["$565.00"]),
        PromptCase("highest_total", "Which invoice has the highest total due, and what is that amount? Answer in one sentence.", ["INV-1002", "$565.00"]),
        PromptCase("quote_ink", "Quote the line item from invoice INV-1001 that mentions ink, then name the source file.", ["Ink Cartridge (Black)", "invoice_INV-1001.txt"]),
        PromptCase("summary", "Summarize the indexed invoice set in 3 bullets and mention at least two vendors by name.", ["Alpine Office Supplies", "BrightPath Logistics"]),
    ]
    return {case.name: case for case in cases}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_state_path(run_id: str) -> Path:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"agent_file_search_stepwise_{run_id}.state.json"


def _state_path(args: argparse.Namespace) -> Path:
    if args.state:
        return Path(args.state)
    return _default_state_path(_run_id())


def _log_path_for_state(state_path: Path) -> Path:
    return state_path.with_suffix(".log")


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


def _object_to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_object_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _object_to_plain(v) for k, v in value.items()}
    for attr_name in ("model_dump", "to_dict"):
        method = getattr(value, attr_name, None)
        if callable(method):
            try:
                return _object_to_plain(method())
            except Exception:  # noqa: BLE001
                pass
    if hasattr(value, "__dict__"):
        return {key: _object_to_plain(val) for key, val in vars(value).items() if not key.startswith("_") and not callable(val)}
    return str(value)


def _extract_annotations_from_item(item: Any) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    urls: list[str] = []
    files: list[str] = []
    annotations: list[dict[str, Any]] = []
    if getattr(item, "type", None) != "message":
        return urls, files, annotations
    for content in getattr(item, "content", []) or []:
        if getattr(content, "type", None) != "output_text":
            continue
        text_value = getattr(content, "text", None)
        for ann in getattr(content, "annotations", []) or []:
            ann_type = str(getattr(ann, "type", "") or "")
            ann_dict = _object_to_plain(ann)
            if not isinstance(ann_dict, dict):
                ann_dict = {"raw": ann_dict}
            ann_dict.setdefault("type", ann_type)
            ann_dict["text_excerpt"] = text_value
            annotations.append(ann_dict)

            url = getattr(ann, "url", None) or ann_dict.get("url")
            if isinstance(url, str) and url:
                urls.append(url)

            file_name = (
                getattr(ann, "filename", None)
                or ann_dict.get("filename")
                or ann_dict.get("file_name")
                or ann_dict.get("document_name")
            )
            if isinstance(file_name, str) and file_name:
                files.append(file_name)
    return urls, files, annotations


def _extract_file_search_results_from_item(item: Any) -> list[dict[str, Any]]:
    if getattr(item, "type", None) != "file_search_call":
        return []
    results = getattr(item, "results", None) or []
    return [_object_to_plain(result) for result in results]


def _normalize_annotations(citation_annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uniq_annotations: list[dict[str, Any]] = []
    seen_ann: set[tuple[Any, ...]] = set()
    for ann in citation_annotations:
        key = (
            ann.get("type"),
            ann.get("url"),
            ann.get("filename"),
            ann.get("file_name"),
            ann.get("title"),
            ann.get("start_index"),
            ann.get("end_index"),
        )
        if key in seen_ann:
            continue
        seen_ann.add(key)
        uniq_annotations.append(ann)
    return uniq_annotations


def _marker_matches_output(marker: str, output_text: str) -> bool:
    if marker in output_text:
        return True
    if marker.startswith("$") and marker[1:] in output_text:
        return True
    return False


def _load_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        raise SystemExit(f"Vector store cache not found: {cache_path}. Run vector_store_index.py first.")
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _load_state(path: Path) -> ProbeState:
    return ProbeState(**json.loads(path.read_text(encoding="utf-8")))


def _save_state(path: Path, state: ProbeState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def _record_success(
    state: ProbeState,
    command: str,
    prompt: str,
    response: Any,
    headers: dict[str, str],
    latency_ms: int,
    case_name: str | None = None,
    expected_markers: list[str] | None = None,
) -> None:
    output_text = _output_text(response)
    usage = _usage_snapshot(response)
    output_item_types: list[str] = []
    file_search_results: list[dict[str, Any]] = []
    citation_urls: list[str] = []
    citation_files: list[str] = []
    citation_annotations: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        output_item_types.append(str(getattr(item, "type", "")))
        file_search_results.extend(_extract_file_search_results_from_item(item))
        urls, files, annotations = _extract_annotations_from_item(item)
        citation_urls.extend(urls)
        citation_files.extend(files)
        citation_annotations.extend(annotations)

    matched_markers: list[str] = []
    expectation_met: bool | None = None
    if expected_markers is not None:
        matched_markers = [marker for marker in expected_markers if _marker_matches_output(marker, output_text)]
        expectation_met = len(matched_markers) == len(expected_markers)

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
                file_search_results=file_search_results,
                citation_count=len(_normalize_annotations(citation_annotations)),
                citation_urls=sorted(set(citation_urls)),
                citation_files=sorted(set(citation_files)),
                citation_annotations=_normalize_annotations(citation_annotations),
                case_name=case_name,
                expected_markers=expected_markers or [],
                matched_markers=matched_markers,
                expectation_met=expectation_met,
                apim_request_id=headers.get("apim-request-id"),
                x_request_id=headers.get("x-request-id"),
                x_ms_region=headers.get("x-ms-region"),
            )
        )
    )


def _record_failure(state: ProbeState, command: str, prompt: str, headers: dict[str, str], latency_ms: int, err: Exception, case_name: str | None = None, expected_markers: list[str] | None = None) -> None:
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
                expected_markers=expected_markers or [],
                expectation_met=False if expected_markers is not None else None,
                apim_request_id=headers.get("apim-request-id"),
                x_request_id=headers.get("x-request-id"),
                x_ms_region=headers.get("x-ms-region"),
                error_type=type(err).__name__,
                error_message=str(err),
            )
        )
    )


def main() -> None:
    args = parse_args()
    state_path = _state_path(args)
    log_path = _log_path_for_state(state_path)
    _setup_logging(args.log_level, log_path)
    cfg = load_config()
    cache_path = Path(args.cache_path)
    case_map = _build_cases()

    if state_path.exists():
        state = _load_state(state_path)
    else:
        cache = _load_cache(cache_path)
        vector_store_id = str(cache["vector_store_id"])
        state = ProbeState(
            run_id=_run_id(),
            created_at_utc=_utc_now(),
            project_endpoint=cfg.project_endpoint,
            mode=args.mode,
            model=args.model,
            vector_store_id=vector_store_id,
            sample_corpus=cache.get("sample_corpus"),
            cache_path=str(cache_path),
        )
        _save_state(state_path, state)

    logger.info("state=%s command=%s mode=%s model=%s vector_store_id=%s", state_path, args.command, state.mode, state.model, state.vector_store_id)

    recorder = HeaderRecorder()
    with DefaultAzureCredential() as credential, AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client:
        with project_client.get_openai_client() as openai_client:
            _attach_hooks(openai_client, recorder)

            if args.command == "create-agent":
                agent_name = f"{cfg.agent_name_prefix}-files-step-{_run_id().lower()}"
                agent = project_client.agents.create_version(
                    agent_name=agent_name,
                    definition=PromptAgentDefinition(
                        model=state.model,
                        instructions=(
                            "You are a retrieval assistant. Use the attached file search tool to answer only from indexed files. "
                            "When naming a source, use the actual file name if available. If asked for a narrow value, answer narrowly."
                        ),
                        tools=[FileSearchTool(vector_store_ids=[state.vector_store_id])],
                    ),
                )
                state.agent_name = agent.name
                state.agent_version = str(agent.version)
                _save_state(state_path, state)
                logger.info("created agent_name=%s agent_version=%s", state.agent_name, state.agent_version)
                return

            if args.command == "create-conversation":
                if not state.agent_name:
                    raise SystemExit("State file has no agent yet. Run create-agent first.")
                conversation = openai_client.conversations.create()
                state.conversation_id = conversation.id
                _save_state(state_path, state)
                logger.info("created conversation_id=%s", state.conversation_id)
                return

            if args.command == "show-state":
                print(json.dumps(asdict(state), indent=2))
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
                    case = case_map[case_name]
                    prompt = case.prompt
                    expected_markers = case.expected_markers
                else:
                    case_name = None
                    prompt = args.message
                    expected_markers = None

                max_output_tokens = args.max_output_tokens
                tool_choice = args.tool_choice

                logger.info("START command=%s case=%s conversation_id=%s", args.command, case_name or "-", state.conversation_id)
                response, headers, latency_ms, err = _call_with_capture(
                    recorder,
                    lambda: openai_client.responses.create(
                        conversation=state.conversation_id,
                        input=prompt,
                        include=["file_search_call.results"],
                        max_output_tokens=max_output_tokens,
                        tool_choice=tool_choice,
                        extra_body={"agent_reference": {"name": state.agent_name, "type": "agent_reference"}},
                    ),
                )
                if err is not None:
                    _record_failure(state, args.command, prompt, headers, latency_ms, err, case_name=case_name, expected_markers=expected_markers)
                    _save_state(state_path, state)
                    logger.warning("FAIL command=%s case=%s latency_ms=%s error=%s", args.command, case_name or "-", latency_ms, err)
                    raise SystemExit(1)

                _record_success(state, args.command, prompt, response, headers, latency_ms, case_name=case_name, expected_markers=expected_markers)
                state.last_response_id = getattr(response, "id", None)
                _save_state(state_path, state)

                last = state.records[-1]
                logger.info(
                    "DONE  command=%s case=%s total_tokens=%s citations=%s expectation=%s",
                    args.command,
                    case_name or "-",
                    ((last.get("usage") or {}).get("total_tokens")),
                    last.get("citation_count"),
                    last.get("expectation_met"),
                )
                return

            if args.command == "cleanup":
                if args.delete_conversation and state.conversation_id:
                    openai_client.conversations.delete(state.conversation_id)
                    logger.info("deleted conversation_id=%s", state.conversation_id)
                    state.conversation_id = None
                if args.delete_agent and state.agent_name and state.agent_version:
                    project_client.agents.delete_version(agent_name=state.agent_name, agent_version=state.agent_version)
                    logger.info("deleted agent_name=%s agent_version=%s", state.agent_name, state.agent_version)
                    state.agent_name = None
                    state.agent_version = None
                _save_state(state_path, state)
                return


if __name__ == "__main__":
    main()
