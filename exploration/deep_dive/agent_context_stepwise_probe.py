#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition
from azure.identity import DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("agent_context_stepwise_probe")


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
    recall_label: str | None = None
    recall_expected_answer: str | None = None
    recall_observed_answer: str | None = None
    recall_present: bool | None = None
    apim_request_id: str | None = None
    x_request_id: str | None = None
    x_ms_region: str | None = None
    x_ratelimit_limit_tokens: str | None = None
    x_ratelimit_remaining_tokens: str | None = None
    x_ratelimit_reset_tokens: str | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class ProbeState:
    run_id: str
    created_at_utc: str
    project_endpoint: str
    model: str
    agent_name: str | None = None
    agent_version: str | None = None
    conversation_id: str | None = None
    turn_index: int = 0
    stored_labels: list[str] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)


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
    p = argparse.ArgumentParser(description="Stepwise Prompt-agent probe that preserves remote traces until explicit cleanup.")
    p.add_argument("--state", default="", help="Path to state JSON file. Defaults to output/agent_context_stepwise_<run_id>.state.json")
    p.add_argument("--model", default="gpt-5.1", help="Model/deployment name.")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("create-agent", help="Create a remote prompt agent and persist local state.")
    sub.add_parser("create-conversation", help="Create a remote conversation and persist it in state.")

    p_stuff = sub.add_parser("stuff", help="Send one stuffing turn to the remote agent conversation.")
    p_stuff.add_argument("--block-chars", type=int, default=6000)
    p_stuff.add_argument("--max-output-tokens", type=int, default=120)
    p_stuff.add_argument("--truncation", choices=["omit", "auto", "disabled"], default="omit")

    p_recall = sub.add_parser("recall", help="Ask a binary yes/no recall question about a label.")
    p_recall.add_argument("--label", default="", help="Override label. Defaults to the most recently stuffed label.")
    p_recall.add_argument("--max-output-tokens", type=int, default=40)
    p_recall.add_argument("--truncation", choices=["omit", "auto", "disabled"], default="omit")

    p_send = sub.add_parser("send", help="Send an arbitrary message to the remote agent conversation.")
    p_send.add_argument("--message", required=True)
    p_send.add_argument("--max-output-tokens", type=int, default=220)
    p_send.add_argument("--truncation", choices=["omit", "auto", "disabled"], default="omit")

    sub.add_parser("show-state", help="Print current local state.")
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
    return out_dir / f"agent_context_stepwise_{run_id}.state.json"


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


def _make_stuffing_prompt(turn_index: int, label: str, block_chars: int) -> str:
    filler = f"[BEGIN-{label}] " + ("X" * max(0, block_chars - len(label) - 10)) + f" [END-{label}]"
    return (
        f"Turn {turn_index}. Store this exact marker for later recall: {label}.\n"
        "Do not summarize the body. Just reply with: STORED <label>.\n\n"
        f"{filler}"
    )


def _make_recall_prompt(label: str) -> str:
    return (
        f"In the immediately preceding stuffing turn, was the exact marker label `{label}` stored? "
        "Answer with exactly one token: YES or NO."
    )


def _normalize_yes_no(text: str) -> str | None:
    stripped = text.strip().upper()
    if stripped.startswith("YES"):
        return "YES"
    if stripped.startswith("NO"):
        return "NO"
    return None


def _load_state(path: Path) -> ProbeState:
    return ProbeState(**json.loads(path.read_text(encoding="utf-8")))


def _save_state(path: Path, state: ProbeState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def _state_path(args: argparse.Namespace) -> Path:
    if args.state:
        return Path(args.state)
    return _default_state_path(_run_id())


def _log_path_for_state(state_path: Path) -> Path:
    return state_path.with_suffix(".log")


def _record_success(
    state: ProbeState,
    command: str,
    prompt: str,
    response: Any,
    headers: dict[str, str],
    latency_ms: int,
    recall_label: str | None = None,
    recall_expected_answer: str | None = None,
    recall_observed_answer: str | None = None,
    recall_present: bool | None = None,
) -> None:
    text = _output_text(response)
    usage = _usage_snapshot(response)
    state.records.append(
        asdict(
            StepRecord(
                timestamp_utc=_utc_now(),
                command=command,
                status_code=headers.get(":status") and int(headers[":status"]) or None,
                success=True,
                latency_ms=latency_ms,
                prompt_chars=len(prompt),
                output_chars=len(text),
                usage=usage,
                conversation_id=state.conversation_id,
                output_preview=text[:220] if text else None,
                recall_label=recall_label,
                recall_expected_answer=recall_expected_answer,
                recall_observed_answer=recall_observed_answer,
                recall_present=recall_present,
                apim_request_id=headers.get("apim-request-id"),
                x_request_id=headers.get("x-request-id"),
                x_ms_region=headers.get("x-ms-region"),
                x_ratelimit_limit_tokens=headers.get("x-ratelimit-limit-tokens"),
                x_ratelimit_remaining_tokens=headers.get("x-ratelimit-remaining-tokens"),
                x_ratelimit_reset_tokens=headers.get("x-ratelimit-reset-tokens"),
            )
        )
    )


def _record_failure(state: ProbeState, command: str, prompt: str, latency_ms: int, headers: dict[str, str], err: Exception) -> None:
    state.records.append(
        asdict(
            StepRecord(
                timestamp_utc=_utc_now(),
                command=command,
                status_code=None,
                success=False,
                latency_ms=latency_ms,
                prompt_chars=len(prompt),
                output_chars=0,
                usage=UsageSnapshot(),
                conversation_id=state.conversation_id,
                apim_request_id=headers.get("apim-request-id"),
                x_request_id=headers.get("x-request-id"),
                x_ms_region=headers.get("x-ms-region"),
                x_ratelimit_limit_tokens=headers.get("x-ratelimit-limit-tokens"),
                x_ratelimit_remaining_tokens=headers.get("x-ratelimit-remaining-tokens"),
                x_ratelimit_reset_tokens=headers.get("x-ratelimit-reset-tokens"),
                error_type=type(err).__name__,
                error_message=str(err),
            )
        )
    )


def _responses_kwargs(state: ProbeState, prompt: str, max_output_tokens: int, truncation: str) -> dict[str, Any]:
    if not state.agent_name or not state.conversation_id:
        raise SystemExit("State file is missing remote agent or conversation. Run create-agent and create-conversation first.")
    kwargs: dict[str, Any] = {
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "conversation": state.conversation_id,
        "extra_body": {"agent_reference": {"name": state.agent_name, "type": "agent_reference"}},
    }
    if truncation != "omit":
        kwargs["truncation"] = truncation
    return kwargs


def main() -> None:
    args = parse_args()
    cfg = load_config()
    state_path = _state_path(args)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_exists = state_path.exists()

    if args.command == "create-agent" and not state_exists:
        state = ProbeState(
            run_id=state_path.stem.replace(".state", ""),
            created_at_utc=_utc_now(),
            project_endpoint=cfg.project_endpoint,
            model=args.model,
        )
        _save_state(state_path, state)
    elif state_exists:
        state = _load_state(state_path)
    else:
        raise SystemExit(f"State file not found: {state_path}")

    log_path = _log_path_for_state(state_path)
    _setup_logging(args.log_level, log_path)
    logger.info("state=%s command=%s", state_path, args.command)

    recorder = HeaderRecorder()
    with DefaultAzureCredential() as credential, AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client:
        with project_client.get_openai_client() as openai_client:
            _attach_hooks(openai_client, recorder)

            if args.command == "create-agent":
                agent_name = f"{cfg.agent_name_prefix}-ctxstep-{_run_id().lower()}"
                agent = project_client.agents.create_version(
                    agent_name=agent_name,
                    definition=PromptAgentDefinition(
                        model=state.model,
                        instructions=(
                            "You are a prompt agent used to study remote conversation traces and context handling. "
                            "When asked to store a marker, do not summarize the body. "
                            "When asked a binary recall question, answer with exactly YES or NO."
                        ),
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

            if args.command == "stuff":
                state.turn_index += 1
                label = f"CTXMARK-{state.turn_index:03d}"
                prompt = _make_stuffing_prompt(state.turn_index, label, args.block_chars)
                kwargs = _responses_kwargs(state, prompt, args.max_output_tokens, args.truncation)
                logger.info("START command=stuff turn=%s label=%s conversation_id=%s", state.turn_index, label, state.conversation_id)
                response, headers, latency_ms, err = _call_with_capture(recorder, lambda: openai_client.responses.create(**kwargs))
                if err is not None:
                    _record_failure(state, "stuff", prompt, latency_ms, headers, err)
                    _save_state(state_path, state)
                    logger.warning("FAIL command=stuff turn=%s latency_ms=%s error=%s", state.turn_index, latency_ms, err)
                    raise SystemExit(1)
                _record_success(state, "stuff", prompt, response, headers, latency_ms, recall_label=label)
                state.stored_labels.append(label)
                _save_state(state_path, state)
                logger.info("DONE  command=stuff turn=%s total_tokens=%s", state.turn_index, _usage_snapshot(response).total_tokens)
                return

            if args.command == "recall":
                label = args.label or (state.stored_labels[-1] if state.stored_labels else "")
                if not label:
                    raise SystemExit("No label available. Run stuff first or pass --label.")
                prompt = _make_recall_prompt(label)
                kwargs = _responses_kwargs(state, prompt, args.max_output_tokens, args.truncation)
                logger.info("START command=recall label=%s conversation_id=%s", label, state.conversation_id)
                response, headers, latency_ms, err = _call_with_capture(recorder, lambda: openai_client.responses.create(**kwargs))
                if err is not None:
                    _record_failure(state, "recall", prompt, latency_ms, headers, err)
                    _save_state(state_path, state)
                    logger.warning("FAIL command=recall label=%s latency_ms=%s error=%s", label, latency_ms, err)
                    raise SystemExit(1)
                text = _output_text(response)
                observed = _normalize_yes_no(text)
                _record_success(
                    state,
                    "recall",
                    prompt,
                    response,
                    headers,
                    latency_ms,
                    recall_label=label,
                    recall_expected_answer="YES",
                    recall_observed_answer=observed,
                    recall_present=(observed == "YES"),
                )
                _save_state(state_path, state)
                logger.info(
                    "DONE  command=recall label=%s total_tokens=%s observed=%s",
                    label,
                    _usage_snapshot(response).total_tokens,
                    observed,
                )
                return

            if args.command == "send":
                prompt = args.message
                kwargs = _responses_kwargs(state, prompt, args.max_output_tokens, args.truncation)
                logger.info("START command=send conversation_id=%s", state.conversation_id)
                response, headers, latency_ms, err = _call_with_capture(recorder, lambda: openai_client.responses.create(**kwargs))
                if err is not None:
                    _record_failure(state, "send", prompt, latency_ms, headers, err)
                    _save_state(state_path, state)
                    logger.warning("FAIL command=send latency_ms=%s error=%s", latency_ms, err)
                    raise SystemExit(1)
                _record_success(state, "send", prompt, response, headers, latency_ms)
                _save_state(state_path, state)
                logger.info("DONE  command=send total_tokens=%s", _usage_snapshot(response).total_tokens)
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
