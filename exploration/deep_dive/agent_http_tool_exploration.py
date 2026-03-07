#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import load_config
from http_request import http_request


@dataclass
class StepRecord:
    step: str
    status_code: int | None
    latency_ms: int
    success: bool
    error_type: str | None = None
    error_message: str | None = None
    apim_request_id: str | None = None
    x_request_id: str | None = None
    x_ms_region: str | None = None


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
    p = argparse.ArgumentParser(description="Explore Foundry agent + HTTP function tool behavior.")
    p.add_argument("--model", default=cfg.default_model_deployment_name, help="Model/deployment name.")
    p.add_argument(
        "--prompt",
        default=(
            "Use the http_request tool to call https://jsonplaceholder.typicode.com/todos/1 with GET. "
            "Then summarize the title and completed status."
        ),
        help="Prompt sent to the agent.",
    )
    p.add_argument(
        "--max-body-chars",
        type=int,
        default=4000,
        help="Maximum response body length returned by the local tool handler.",
    )
    return p.parse_args()


def _attach_hooks_to_openai_client(openai_client: Any, recorder: HeaderRecorder) -> None:
    inner = getattr(openai_client, "_client", None)
    if inner is not None and hasattr(inner, "event_hooks"):
        inner.event_hooks = {"request": [recorder.on_request], "response": [recorder.on_response]}


def _call_with_capture(step: str, rec: HeaderRecorder, fn: Any) -> tuple[Any | None, StepRecord]:
    rec.reset()
    t0 = time.perf_counter()
    try:
        out = fn()
        latency = int((time.perf_counter() - t0) * 1000)
        sr = StepRecord(
            step=step,
            status_code=rec.status_code,
            latency_ms=latency,
            success=True,
            apim_request_id=rec.headers.get("apim-request-id"),
            x_request_id=rec.headers.get("x-request-id"),
            x_ms_region=rec.headers.get("x-ms-region"),
        )
        return out, sr
    except Exception as exc:  # noqa: BLE001
        latency = int((time.perf_counter() - t0) * 1000)
        resp = getattr(exc, "response", None)
        headers: dict[str, str] = {}
        status = rec.status_code
        if resp is not None:
            status = getattr(resp, "status_code", status)
            try:
                headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
            except Exception:  # noqa: BLE001
                headers = {}
        sr = StepRecord(
            step=step,
            status_code=status,
            latency_ms=latency,
            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
            apim_request_id=headers.get("apim-request-id") or rec.headers.get("apim-request-id"),
            x_request_id=headers.get("x-request-id") or rec.headers.get("x-request-id"),
            x_ms_region=headers.get("x-ms-region") or rec.headers.get("x-ms-region"),
        )
        return None, sr


def _write_outputs(out_dir: Path, run_id: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"agent_http_tool_exploration_{run_id}.json"
    md_path = out_dir / f"agent_http_tool_exploration_{run_id}.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# Agent HTTP Tool Exploration ({run_id})")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- model: {payload['metadata']['model']}")
    lines.append(f"- prompt: {payload['metadata']['prompt']}")
    lines.append("")
    lines.append("## Steps")
    lines.append("")
    lines.append("| Step | Success | Status | Latency (ms) | x-request-id | apim-request-id |")
    lines.append("|---|---|---:|---:|---|---|")
    for s in payload["steps"]:
        lines.append(
            f"| {s['step']} | {'yes' if s['success'] else 'no'} | {s['status_code'] or '-'} | {s['latency_ms']} | {s.get('x_request_id') or '-'} | {s.get('apim_request_id') or '-'} |"
        )
    lines.append("")
    lines.append("## Final Output")
    lines.append("")
    lines.append("```text")
    lines.append(payload.get("final_text") or "")
    lines.append("```")
    lines.append("")
    lines.append("## Tool Calls")
    lines.append("")
    for i, call in enumerate(payload.get("tool_calls", []), start=1):
        lines.append(f"### Call {i}: {call.get('name')}")
        lines.append("")
        lines.append("Arguments:")
        lines.append("```json")
        lines.append(json.dumps(call.get("arguments"), indent=2))
        lines.append("```")
        lines.append("Output:")
        lines.append("```json")
        lines.append(json.dumps(call.get("output"), indent=2))
        lines.append("```")
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main() -> None:
    args = parse_args()
    cfg = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    recorder = HeaderRecorder()
    steps: list[StepRecord] = []
    tool_calls: list[dict[str, Any]] = []
    agent = None
    final_response = None

    agent_name = f"{cfg.agent_name_prefix}-httptool-{run_id.lower()}"
    out_dir = Path(__file__).resolve().parent / "output"

    print(f"run_id={run_id}")
    print(f"model={args.model}")
    print("1) create agent with FunctionTool(http_request)")

    with DefaultAzureCredential() as credential:
        with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client:
            with project_client.get_openai_client() as openai_client:
                _attach_hooks_to_openai_client(openai_client, recorder)

                http_tool = FunctionTool(
                    name="http_request",
                    description="Make an HTTP request and return status, headers, and body payload.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "method": {"type": "string", "description": "HTTP method such as GET or POST"},
                            "url": {"type": "string", "description": "Target URL"},
                            "auth_type": {"type": "string"},
                            "auth_token": {"type": "string"},
                            "auth_env_var": {"type": "string"},
                            "headers": {"type": "object", "additionalProperties": {"type": "string"}},
                            "body": {"type": "string"},
                            "verify_ssl": {"type": "boolean"},
                            "basic_auth_username": {"type": "string"},
                            "basic_auth_password": {"type": "string"},
                        },
                        "required": ["method", "url"],
                        "additionalProperties": False,
                    },
                    # Keep optional parameters optional to match the original http_request.py signature.
                    strict=False,
                )

                agent, step = _call_with_capture(
                    "create_agent",
                    recorder,
                    lambda: project_client.agents.create_version(
                        agent_name=agent_name,
                        definition=PromptAgentDefinition(
                            model=args.model,
                            instructions=(
                                "You are a careful API exploration assistant. "
                                "When needed, call http_request and summarize the response concisely."
                            ),
                            tools=[http_tool],
                        ),
                    ),
                )
                steps.append(step)
                print(f"   status={step.status_code} latency_ms={step.latency_ms} success={step.success}")
                if not step.success or agent is None:
                    payload = {
                        "metadata": {"run_id": run_id, "model": args.model, "prompt": args.prompt},
                        "steps": [asdict(s) for s in steps],
                        "tool_calls": tool_calls,
                        "final_text": None,
                    }
                    md_path, json_path = _write_outputs(out_dir, run_id, payload)
                    print(md_path)
                    print(json_path)
                    return

                print("2) run initial response")
                first, step = _call_with_capture(
                    "run_initial",
                    recorder,
                    lambda: openai_client.responses.create(
                        input=args.prompt,
                        extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                    ),
                )
                steps.append(step)
                print(f"   status={step.status_code} latency_ms={step.latency_ms} success={step.success}")
                if not step.success or first is None:
                    payload = {
                        "metadata": {"run_id": run_id, "model": args.model, "prompt": args.prompt},
                        "steps": [asdict(s) for s in steps],
                        "tool_calls": tool_calls,
                        "final_text": None,
                    }
                    md_path, json_path = _write_outputs(out_dir, run_id, payload)
                    print(md_path)
                    print(json_path)
                    return

                pending_outputs: list[FunctionCallOutput] = []
                for item in getattr(first, "output", []) or []:
                    if getattr(item, "type", None) == "function_call" and getattr(item, "name", None) == "http_request":
                        args_dict = json.loads(item.arguments)
                        # Use the original tool implementation copied from AgentExp.
                        raw_output = http_request(
                            method=args_dict["method"],
                            url=args_dict["url"],
                            auth_type=args_dict.get("auth_type"),
                            auth_token=args_dict.get("auth_token"),
                            auth_env_var=args_dict.get("auth_env_var"),
                            headers=args_dict.get("headers"),
                            body=args_dict.get("body"),
                            verify_ssl=args_dict.get("verify_ssl", True),
                            basic_auth_username=args_dict.get("basic_auth_username"),
                            basic_auth_password=args_dict.get("basic_auth_password"),
                        )
                        # Keep output bounded before sending back to the model.
                        if len(raw_output) > args.max_body_chars:
                            raw_output = raw_output[: args.max_body_chars] + "\n... [truncated by exploration harness]"
                        out = {"raw_output": raw_output}
                        tool_calls.append({"name": "http_request", "arguments": args_dict, "output": out})
                        pending_outputs.append(
                            FunctionCallOutput(
                                type="function_call_output",
                                call_id=item.call_id,
                                output=json.dumps(out),
                            )
                        )

                if pending_outputs:
                    print("3) send function_call_output followup")
                    final_response, step = _call_with_capture(
                        "run_followup",
                        recorder,
                        lambda: openai_client.responses.create(
                            input=pending_outputs,
                            previous_response_id=first.id,
                            extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                        ),
                    )
                    steps.append(step)
                    print(f"   status={step.status_code} latency_ms={step.latency_ms} success={step.success}")
                else:
                    final_response = first

                print("4) cleanup agent")
                _, step = _call_with_capture(
                    "cleanup_agent",
                    recorder,
                    lambda: project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version),
                )
                steps.append(step)
                print(f"   status={step.status_code} latency_ms={step.latency_ms} success={step.success}")

    payload = {
        "metadata": {"run_id": run_id, "model": args.model, "prompt": args.prompt},
        "steps": [asdict(s) for s in steps],
        "tool_calls": tool_calls,
        "final_text": getattr(final_response, "output_text", None),
    }
    md_path, json_path = _write_outputs(out_dir, run_id, payload)
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
