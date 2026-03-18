#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import FunctionTool, PromptAgentDefinition
from azure.identity import DefaultAzureCredential
from openai.types.responses.response_input_param import FunctionCallOutput

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("agent_large_tool_payload_probe")


@dataclass
class UsageSnapshot:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class StepRecord:
    turn_index: int
    phase: str
    status_code: int | None
    success: bool
    latency_ms: int
    payload_chars: int
    output_chars: int
    usage: UsageSnapshot
    conversation_id: str | None
    recall_label: str | None = None
    recall_present: bool | None = None
    output_preview: str | None = None
    apim_request_id: str | None = None
    x_request_id: str | None = None
    x_ms_region: str | None = None
    error_type: str | None = None
    error_message: str | None = None


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
    p = argparse.ArgumentParser(description="Probe large function-tool payload behavior inside a Prompt agent conversation.")
    p.add_argument("--model", default="gpt-5.1", help="Model/deployment name.")
    p.add_argument("--payload-chars", type=int, default=12000, help="Characters returned by the tool each turn.")
    p.add_argument("--max-turns", type=int, default=30, help="Maximum number of tool turns.")
    p.add_argument("--max-output-tokens", type=int, default=160)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--sleep-seconds", type=float, default=0.0)
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


def _deployment_names(project_client: AIProjectClient) -> list[str]:
    return sorted(
        name
        for deployment in project_client.deployments.list()
        for name in [getattr(deployment, "name", None)]
        if isinstance(name, str) and name
    )


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


def _make_payload(label: str, payload_chars: int) -> str:
    head = f"[BEGIN-{label}]"
    tail = f"[END-{label}]"
    middle_len = max(0, payload_chars - len(head) - len(tail))
    return head + ("Z" * middle_len) + tail


def _make_initial_prompt(turn_index: int, label: str) -> str:
    return (
        f"Turn {turn_index}. Call the `large_payload` tool with label `{label}` and return exactly `TOOL_CALLED {label}`. "
        "Do not summarize the payload."
    )


def _make_recall_prompt(expected_labels: list[str]) -> str:
    return (
        "List every large-payload marker label you still remember from this conversation, in order, as a comma-separated list. "
        f"The labels introduced so far were: {', '.join(expected_labels)}."
    )


def _record_failure(turn_index: int, phase: str, payload_chars: int, conversation_id: str | None, latency_ms: int, headers: dict[str, str], err: Exception) -> StepRecord:
    return StepRecord(
        turn_index=turn_index,
        phase=phase,
        status_code=None,
        success=False,
        latency_ms=latency_ms,
        payload_chars=payload_chars,
        output_chars=0,
        usage=UsageSnapshot(),
        conversation_id=conversation_id,
        output_preview=None,
        apim_request_id=headers.get("apim-request-id"),
        x_request_id=headers.get("x-request-id"),
        x_ms_region=headers.get("x-ms-region"),
        error_type=type(err).__name__,
        error_message=str(err),
    )


def _write_outputs(run_id: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"agent_large_tool_payload_probe_{run_id}.json"
    md_path = out_dir / f"agent_large_tool_payload_probe_{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# Agent Large Tool Payload Probe ({run_id})")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    for k, v in payload["metadata"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for k, v in payload["summary"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Steps")
    lines.append("")
    lines.append("| Turn | Phase | Success | Status | Payload Chars | Total Tokens | Recall Label | Recall Present | Latency (ms) | Error |")
    lines.append("|---|---|---|---:|---:|---:|---|---|---:|---|")
    for rec in payload["records"]:
        usage = rec["usage"]
        lines.append(
            f"| {rec['turn_index']} | {rec['phase']} | {'yes' if rec['success'] else 'no'} | "
            f"{rec['status_code'] or '-'} | {rec['payload_chars']} | {usage['total_tokens'] or 0} | "
            f"{rec.get('recall_label') or '-'} | {rec.get('recall_present')} | {rec['latency_ms']} | {rec.get('error_type') or '-'} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main() -> None:
    args = parse_args()
    cfg = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "output"
    log_path = out_dir / f"agent_large_tool_payload_probe_{run_id}.log"
    _setup_logging(args.log_level, log_path)

    logger.info("run_id=%s", run_id)
    logger.info("model=%s payload_chars=%s max_turns=%s", args.model, args.payload_chars, args.max_turns)

    payload: dict[str, Any] = {
        "metadata": {
            "project_endpoint": cfg.project_endpoint,
            "model": args.model,
            "agent_kind": "prompt",
            "payload_chars": args.payload_chars,
            "max_turns": args.max_turns,
            "max_output_tokens": args.max_output_tokens,
            "sleep_seconds": args.sleep_seconds,
            "hypothesis": (
                "If large tool outputs are compacted or truncated in conversation state, recall of the latest labels "
                "should degrade before or instead of a hard context-length failure."
            ),
        },
        "records": [],
        "summary": {},
    }

    recorder = HeaderRecorder()
    agent_name = f"{cfg.agent_name_prefix}-largepayload-{run_id.lower()}"
    stored_labels: list[str] = []
    stop_reason = "max_turns_reached"

    with DefaultAzureCredential() as credential, AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client:
        deployed_models = _deployment_names(project_client)
        if args.model not in deployed_models:
            raise SystemExit(
                "Model deployment not found in this project.\n"
                f"requested: {args.model}\n"
                f"available: {', '.join(deployed_models) if deployed_models else '(none)'}"
            )

        with project_client.get_openai_client() as openai_client:
            _attach_hooks(openai_client, recorder)

            function_tool = FunctionTool(
                name="large_payload",
                description="Return a large labeled payload for context stress testing.",
                parameters={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Marker label to embed in the payload."},
                    },
                    "required": ["label"],
                    "additionalProperties": False,
                },
                strict=False,
            )

            agent = project_client.agents.create_version(
                agent_name=agent_name,
                definition=PromptAgentDefinition(
                    model=args.model,
                    instructions=(
                        "You are a prompt agent used to study how large tool outputs affect conversation memory. "
                        "When asked to call the tool, do so, and follow the user formatting instructions exactly."
                    ),
                    tools=[function_tool],
                ),
            )
            conversation = None
            try:
                conversation = openai_client.conversations.create()
                logger.info("agent_name=%s agent_version=%s conversation_id=%s", agent.name, agent.version, conversation.id)

                last_response_id: str | None = None

                for turn_index in range(1, args.max_turns + 1):
                    label = f"PAYLOAD-{turn_index:03d}"
                    prompt = _make_initial_prompt(turn_index, label)
                    logger.info("START turn=%s phase=tool_call payload_chars=%s", turn_index, args.payload_chars)
                    first, headers, latency_ms, err = _call_with_capture(
                        recorder,
                        lambda: openai_client.responses.create(
                            input=prompt,
                            max_output_tokens=args.max_output_tokens,
                            **(
                                {
                                    "conversation": conversation.id,
                                    "extra_body": {"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                                }
                                if last_response_id is None
                                else {
                                    "previous_response_id": last_response_id,
                                    "extra_body": {"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                                }
                            ),
                        ),
                    )
                    if err is not None:
                        payload["records"].append(asdict(_record_failure(turn_index, "tool_call", args.payload_chars, conversation.id, latency_ms, headers, err)))
                        stop_reason = "tool_call_error"
                        logger.warning("FAIL turn=%s phase=tool_call latency_ms=%s error=%s", turn_index, latency_ms, err)
                        break

                    pending_outputs: list[FunctionCallOutput] = []
                    for item in getattr(first, "output", []) or []:
                        if getattr(item, "type", None) == "function_call" and getattr(item, "name", None) == "large_payload":
                            args_dict = json.loads(item.arguments)
                            payload_text = _make_payload(args_dict["label"], args.payload_chars)
                            pending_outputs.append(
                                FunctionCallOutput(
                                    type="function_call_output",
                                    call_id=item.call_id,
                                    output=json.dumps({"raw_output": payload_text}),
                                )
                            )

                    if not pending_outputs:
                        payload["records"].append(asdict(_record_failure(turn_index, "tool_output", args.payload_chars, conversation.id, latency_ms, headers, RuntimeError("No function_call emitted by agent."))))
                        stop_reason = "missing_function_call"
                        logger.warning("FAIL turn=%s phase=tool_output error=no function_call emitted", turn_index)
                        break

                    logger.info("START turn=%s phase=tool_output payload_chars=%s", turn_index, args.payload_chars)
                    second, headers, latency_ms, err = _call_with_capture(
                        recorder,
                        lambda: openai_client.responses.create(
                            input=pending_outputs,
                            previous_response_id=first.id,
                            extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                        ),
                    )
                    if err is not None:
                        payload["records"].append(asdict(_record_failure(turn_index, "tool_output", args.payload_chars, conversation.id, latency_ms, headers, err)))
                        stop_reason = "tool_output_error"
                        logger.warning("FAIL turn=%s phase=tool_output latency_ms=%s error=%s", turn_index, latency_ms, err)
                        break

                    text = _output_text(second)
                    usage = _usage_snapshot(second)
                    payload["records"].append(
                        asdict(
                            StepRecord(
                                turn_index=turn_index,
                                phase="tool_output",
                                status_code=recorder.status_code,
                                success=True,
                                latency_ms=latency_ms,
                                payload_chars=args.payload_chars,
                                output_chars=len(text),
                                usage=usage,
                                conversation_id=conversation.id,
                                recall_label=label,
                                output_preview=text[:220] if text else None,
                                apim_request_id=headers.get("apim-request-id"),
                                x_request_id=headers.get("x-request-id"),
                                x_ms_region=headers.get("x-ms-region"),
                            )
                        )
                    )
                    stored_labels.append(label)
                    logger.info(
                        "DONE  turn=%s phase=tool_output status=%s latency_ms=%s total_tokens=%s",
                        turn_index,
                        recorder.status_code,
                        latency_ms,
                        usage.total_tokens,
                    )
                    last_response_id = second.id

                    recall_prompt = _make_recall_prompt(stored_labels)
                    logger.info("START turn=%s phase=recall expected_labels=%s", turn_index, len(stored_labels))
                    third, headers, latency_ms, err = _call_with_capture(
                        recorder,
                        lambda: openai_client.responses.create(
                            input=recall_prompt,
                            max_output_tokens=args.max_output_tokens,
                            previous_response_id=last_response_id,
                            extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                        ),
                    )
                    if err is not None:
                        payload["records"].append(asdict(_record_failure(turn_index, "recall", args.payload_chars, conversation.id, latency_ms, headers, err)))
                        stop_reason = "recall_error"
                        logger.warning("FAIL turn=%s phase=recall latency_ms=%s error=%s", turn_index, latency_ms, err)
                        break

                    recall_text = _output_text(third)
                    usage = _usage_snapshot(third)
                    recall_present = label in recall_text
                    payload["records"].append(
                        asdict(
                            StepRecord(
                                turn_index=turn_index,
                                phase="recall",
                                status_code=recorder.status_code,
                                success=True,
                                latency_ms=latency_ms,
                                payload_chars=args.payload_chars,
                                output_chars=len(recall_text),
                                usage=usage,
                                conversation_id=conversation.id,
                                recall_label=label,
                                recall_present=recall_present,
                                output_preview=recall_text[:220] if recall_text else None,
                                apim_request_id=headers.get("apim-request-id"),
                                x_request_id=headers.get("x-request-id"),
                                x_ms_region=headers.get("x-ms-region"),
                            )
                        )
                    )
                    logger.info(
                        "DONE  turn=%s phase=recall status=%s latency_ms=%s total_tokens=%s recall_present=%s",
                        turn_index,
                        recorder.status_code,
                        latency_ms,
                        usage.total_tokens,
                        recall_present,
                    )
                    last_response_id = third.id

                    if not recall_present:
                        stop_reason = "recall_loss_detected"
                        break
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
            finally:
                if conversation is not None:
                    try:
                        openai_client.conversations.delete(conversation.id)
                    except Exception:  # noqa: BLE001
                        logger.warning("cleanup failed for conversation_id=%s", conversation.id)
                try:
                    project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
                except Exception:  # noqa: BLE001
                    logger.warning("cleanup failed for agent_name=%s agent_version=%s", agent.name, agent.version)

    records = payload["records"]
    success_records = [r for r in records if r["success"]]
    failure_records = [r for r in records if not r["success"]]
    recall_records = [r for r in records if r["phase"] == "recall" and r["success"]]
    first_recall_loss = next((r for r in recall_records if r.get("recall_present") is False), None)
    payload["summary"] = {
        "steps_completed": len(records),
        "successful_steps": len(success_records),
        "failure_steps": len(failure_records),
        "stop_reason": stop_reason,
        "max_total_tokens_seen": max((r["usage"]["total_tokens"] or 0 for r in success_records), default=0),
        "first_recall_loss_turn": first_recall_loss["turn_index"] if first_recall_loss else None,
        "first_failure_turn": failure_records[0]["turn_index"] if failure_records else None,
        "final_recall_label_count": len(stored_labels),
    }

    md_path, json_path = _write_outputs(run_id, payload)
    logger.info("Artifacts markdown=%s json=%s log=%s", md_path, json_path, log_path)


if __name__ == "__main__":
    main()
