#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import httpx
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    CodeInterpreterTool,
    FunctionTool,
    MCPTool,
    PromptAgentDefinition,
    WebSearchApproximateLocation,
    WebSearchTool,
)
from azure.core.pipeline.policies import SansIOHTTPPolicy
from azure.identity import DefaultAzureCredential
from openai.types.responses.response_input_param import FunctionCallOutput, McpApprovalResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import filter_inference_deployments, load_config

logger = logging.getLogger("validation_harness")


@dataclass
class CallRecord:
    run_id: str
    test_group: str
    test_case: str
    model: str
    prompt: str | None
    deployment_name: str
    endpoint_family: str
    endpoint_url: str
    api_style: str
    code_snippet: str
    attempt: int
    retries_allowed: int
    success: bool
    error_type: str | None
    error_message: str | None
    status_code: int | None
    start_time_utc: str
    end_time_utc: str
    latency_ms: int
    request_method: str | None
    request_url: str | None
    request_headers: dict[str, str]
    response_headers: dict[str, str]
    apim_request_id: str | None
    x_request_id: str | None
    x_ms_region: str | None
    provider_headers: dict[str, str]


class HeaderRecorder:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.last_request_headers: dict[str, str] = {}
        self.last_response_headers: dict[str, str] = {}
        self.last_status_code: int | None = None
        self.last_url: str | None = None
        self.last_method: str | None = None

    def on_httpx_request(self, request: httpx.Request) -> None:
        self.last_request_headers = {k.lower(): v for k, v in request.headers.items()}
        self.last_url = str(request.url)
        self.last_method = request.method

    def on_httpx_response(self, response: httpx.Response) -> None:
        self.last_response_headers = {k.lower(): v for k, v in response.headers.items()}
        self.last_status_code = response.status_code

    def on_azure_request(self, request: Any) -> None:
        headers = getattr(request, "headers", {}) or {}
        self.last_request_headers = {str(k).lower(): str(v) for k, v in headers.items()}
        self.last_url = str(getattr(request, "url", ""))
        self.last_method = str(getattr(request, "method", ""))

    def on_azure_response(self, response: Any) -> None:
        headers = getattr(response, "headers", {}) or {}
        self.last_response_headers = {str(k).lower(): str(v) for k, v in headers.items()}
        self.last_status_code = int(getattr(response, "status_code", 0) or 0) or None


class AzurePipelineRecorderPolicy(SansIOHTTPPolicy):
    def __init__(self, recorder: HeaderRecorder) -> None:
        super().__init__()
        self._recorder = recorder

    def on_request(self, request: Any) -> None:
        self._recorder.on_azure_request(request.http_request)

    def on_response(self, request: Any, response: Any) -> None:
        self._recorder.on_azure_response(response.http_response)


def _status_headers_from_exception(exc: Exception) -> tuple[int | None, dict[str, str]]:
    response = getattr(exc, "response", None)
    if response is None:
        return None, {}

    status = getattr(response, "status_code", None)
    headers = getattr(response, "headers", {}) or {}
    out = {str(k).lower(): str(v) for k, v in headers.items()}
    return status, out


def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted = dict(headers)
    if "authorization" in redacted:
        redacted["authorization"] = "Bearer <REDACTED>"
    return redacted


class ProbeRunner:
    def __init__(
        self,
        run_id: str,
        endpoint_url: str,
        retries_allowed: int,
        recorder: HeaderRecorder,
        records: list[CallRecord],
    ) -> None:
        self.run_id = run_id
        self.endpoint_url = endpoint_url
        self.retries_allowed = retries_allowed
        self.recorder = recorder
        self.records = records

    def call(
        self,
        *,
        test_group: str,
        test_case: str,
        model: str,
        prompt: str | None,
        api_style: str,
        code_snippet: str,
        fn: Callable[[], Any],
    ) -> Any:
        last_exc: Exception | None = None

        for attempt in range(1, self.retries_allowed + 2):
            logger.info(
                "START test_group=%s test_case=%s model=%s api_style=%s attempt=%s/%s",
                test_group,
                test_case,
                model,
                api_style,
                attempt,
                self.retries_allowed + 1,
            )
            self.recorder.clear()
            start = datetime.now(timezone.utc)
            t0 = time.perf_counter()
            success = False
            err_type = None
            err_msg = None
            result = None

            try:
                result = fn()
                success = True
                return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                err_type = type(exc).__name__
                err_msg = str(exc)
            finally:
                end = datetime.now(timezone.utc)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                response_headers = dict(self.recorder.last_response_headers)
                status_code = self.recorder.last_status_code
                if not response_headers and last_exc is not None:
                    status_from_exc, headers_from_exc = _status_headers_from_exception(last_exc)
                    response_headers = headers_from_exc
                    status_code = status_from_exc
                provider_headers = {
                    k: v
                    for k, v in response_headers.items()
                    if k.startswith("openai-")
                    or k.startswith("x-ratelimit-")
                    or k.startswith("x-ms-")
                    or k.startswith("azure")
                    or k == "apim-request-id"
                    or k == "x-request-id"
                }
                self.records.append(
                    CallRecord(
                        run_id=self.run_id,
                        test_group=test_group,
                        test_case=test_case,
                        model=model,
                        prompt=prompt,
                        deployment_name=model,
                        endpoint_family="foundry_project_openai_bridge",
                        endpoint_url=self.endpoint_url,
                        api_style=api_style,
                        code_snippet=code_snippet,
                        attempt=attempt,
                        retries_allowed=self.retries_allowed,
                        success=success,
                        error_type=err_type,
                        error_message=err_msg,
                        status_code=status_code,
                        start_time_utc=start.isoformat(),
                        end_time_utc=end.isoformat(),
                        latency_ms=latency_ms,
                        request_method=self.recorder.last_method,
                        request_url=self.recorder.last_url,
                        request_headers=_redact_headers(self.recorder.last_request_headers),
                        response_headers=response_headers,
                        apim_request_id=response_headers.get("apim-request-id"),
                        x_request_id=response_headers.get("x-request-id"),
                        x_ms_region=response_headers.get("x-ms-region"),
                        provider_headers=provider_headers,
                    )
                )
                if success:
                    logger.info(
                        "DONE  test_case=%s model=%s status=%s latency_ms=%s apim_request_id=%s x_request_id=%s",
                        test_case,
                        model,
                        status_code,
                        latency_ms,
                        response_headers.get("apim-request-id"),
                        response_headers.get("x-request-id"),
                    )
                else:
                    logger.warning(
                        "FAIL  test_case=%s model=%s status=%s latency_ms=%s error_type=%s error=%s",
                        test_case,
                        model,
                        status_code,
                        latency_ms,
                        err_type,
                        err_msg,
                    )

            if success:
                return result

        assert last_exc is not None
        raise last_exc


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _sanitize_name(value: str, max_len: int = 48) -> str:
    x = re.sub(r"[^a-zA-Z0-9-]", "-", value)
    x = re.sub(r"-+", "-", x).strip("-")
    return (x[:max_len]).strip("-") or "agent"


def _agent_name(prefix: str, model: str, test_case: str) -> str:
    model_short = _sanitize_name(model, max_len=14).lower()
    test_short = _sanitize_name(test_case, max_len=12).lower()
    uid = uuid4().hex[:6]
    base = f"{prefix}-{model_short}-{test_short}-{uid}"
    return _sanitize_name(base, max_len=60)


def run_basic_response(openai_client: Any, probe: ProbeRunner, model: str) -> None:
    prompt = "Return exactly: validation-response-ok"
    snippet = "openai_client.responses.create(model=model, input=prompt, max_output_tokens=50)"
    probe.call(
        test_group="test-1",
        test_case="basic_response",
        model=model,
        prompt=prompt,
        api_style="responses",
        code_snippet=snippet,
        fn=lambda: openai_client.responses.create(model=model, input=prompt, max_output_tokens=50),
    )


def run_agent_lifecycle(project_client: AIProjectClient, openai_client: Any, probe: ProbeRunner, model: str, prefix: str) -> None:
    agent_name = _agent_name(prefix, model, "lifecycle")
    agent = None
    conversation = None

    try:
        snippet = (
            "project_client.agents.create_version(agent_name=agent_name, "
            "definition=PromptAgentDefinition(model=model, instructions=...))"
        )
        agent = probe.call(
            test_group="test-2",
            test_case="agent_lifecycle.create_agent",
            model=model,
            prompt=None,
            api_style="agents",
            code_snippet=snippet,
            fn=lambda: project_client.agents.create_version(
                agent_name=agent_name,
                definition=PromptAgentDefinition(
                    model=model,
                    instructions="You are a concise assistant. Answer directly.",
                ),
            ),
        )

        first_prompt = "What is the size of France in square miles?"
        snippet = "openai_client.conversations.create(items=[{type:'message', role:'user', content:first_prompt}])"
        conversation = probe.call(
            test_group="test-2",
            test_case="agent_lifecycle.create_conversation",
            model=model,
            prompt=first_prompt,
            api_style="conversations",
            code_snippet=snippet,
            fn=lambda: openai_client.conversations.create(
                items=[{"type": "message", "role": "user", "content": first_prompt}]
            ),
        )

        snippet = "openai_client.responses.create(conversation=conversation.id, extra_body={agent_reference...})"
        probe.call(
            test_group="test-2",
            test_case="agent_lifecycle.response_turn_1",
            model=model,
            prompt=first_prompt,
            api_style="responses",
            code_snippet=snippet,
            fn=lambda: openai_client.responses.create(
                conversation=conversation.id,
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            ),
        )

        second_prompt = "And what is the capital city?"
        snippet = "openai_client.conversations.items.create(conversation_id=conversation.id, items=[...])"
        probe.call(
            test_group="test-2",
            test_case="agent_lifecycle.add_user_turn_2",
            model=model,
            prompt=second_prompt,
            api_style="conversations",
            code_snippet=snippet,
            fn=lambda: openai_client.conversations.items.create(
                conversation_id=conversation.id,
                items=[{"type": "message", "role": "user", "content": second_prompt}],
            ),
        )

        snippet = "openai_client.responses.create(conversation=conversation.id, extra_body={agent_reference...})"
        probe.call(
            test_group="test-2",
            test_case="agent_lifecycle.response_turn_2",
            model=model,
            prompt=second_prompt,
            api_style="responses",
            code_snippet=snippet,
            fn=lambda: openai_client.responses.create(
                conversation=conversation.id,
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            ),
        )

    finally:
        if conversation is not None:
            probe.call(
                test_group="test-2",
                test_case="agent_lifecycle.cleanup_conversation",
                model=model,
                prompt=None,
                api_style="conversations",
                code_snippet="openai_client.conversations.delete(conversation_id=conversation.id)",
                fn=lambda: openai_client.conversations.delete(conversation_id=conversation.id),
            )
        if agent is not None:
            probe.call(
                test_group="test-2",
                test_case="agent_lifecycle.cleanup_agent",
                model=model,
                prompt=None,
                api_style="agents",
                code_snippet="project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)",
                fn=lambda: project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version),
            )


def run_tool_code_interpreter(
    project_client: AIProjectClient,
    openai_client: Any,
    probe: ProbeRunner,
    model: str,
    prefix: str,
) -> None:
    agent_name = _agent_name(prefix, model, "code-int")
    agent = None
    conversation = None

    try:
        agent = probe.call(
            test_group="test-3a",
            test_case="tool_code_interpreter.create_agent",
            model=model,
            prompt=None,
            api_style="agents",
            code_snippet="project_client.agents.create_version(... tools=[CodeInterpreterTool()])",
            fn=lambda: project_client.agents.create_version(
                agent_name=agent_name,
                definition=PromptAgentDefinition(
                    model=model,
                    instructions="You are a helpful assistant.",
                    tools=[CodeInterpreterTool()],
                ),
                description="Code interpreter agent for validation.",
            ),
        )

        conversation = probe.call(
            test_group="test-3a",
            test_case="tool_code_interpreter.create_conversation",
            model=model,
            prompt=None,
            api_style="conversations",
            code_snippet="openai_client.conversations.create()",
            fn=lambda: openai_client.conversations.create(),
        )

        prompt = "Generate a 5x5 multiplication table and explain one pattern you observe."
        probe.call(
            test_group="test-3a",
            test_case="tool_code_interpreter.run",
            model=model,
            prompt=prompt,
            api_style="responses",
            code_snippet=(
                "openai_client.responses.create(conversation=conversation.id, input=prompt, "
                "tool_choice='required', extra_body={agent_reference...})"
            ),
            fn=lambda: openai_client.responses.create(
                conversation=conversation.id,
                input=prompt,
                tool_choice="required",
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            ),
        )

    finally:
        if conversation is not None:
            probe.call(
                test_group="test-3a",
                test_case="tool_code_interpreter.cleanup_conversation",
                model=model,
                prompt=None,
                api_style="conversations",
                code_snippet="openai_client.conversations.delete(conversation_id=conversation.id)",
                fn=lambda: openai_client.conversations.delete(conversation_id=conversation.id),
            )
        if agent is not None:
            probe.call(
                test_group="test-3a",
                test_case="tool_code_interpreter.cleanup_agent",
                model=model,
                prompt=None,
                api_style="agents",
                code_snippet="project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)",
                fn=lambda: project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version),
            )


def run_tool_web_search(
    project_client: AIProjectClient,
    openai_client: Any,
    probe: ProbeRunner,
    model: str,
    prefix: str,
) -> None:
    agent_name = _agent_name(prefix, model, "web-search")
    agent = None
    conversation = None

    try:
        tool = WebSearchTool(user_location=WebSearchApproximateLocation(country="US", city="Seattle", region="WA"))

        agent = probe.call(
            test_group="test-3b",
            test_case="tool_web_search.create_agent",
            model=model,
            prompt=None,
            api_style="agents",
            code_snippet="project_client.agents.create_version(... tools=[WebSearchTool(...)])",
            fn=lambda: project_client.agents.create_version(
                agent_name=agent_name,
                definition=PromptAgentDefinition(
                    model=model,
                    instructions="You are a helpful assistant that can search the web.",
                    tools=[tool],
                ),
                description="Web search agent for validation.",
            ),
        )

        conversation = probe.call(
            test_group="test-3b",
            test_case="tool_web_search.create_conversation",
            model=model,
            prompt=None,
            api_style="conversations",
            code_snippet="openai_client.conversations.create()",
            fn=lambda: openai_client.conversations.create(),
        )

        prompt = "Find two recent facts about Microsoft Build and cite sources if available."
        probe.call(
            test_group="test-3b",
            test_case="tool_web_search.run",
            model=model,
            prompt=prompt,
            api_style="responses",
            code_snippet=(
                "openai_client.responses.create(conversation=conversation.id, input=prompt, "
                "tool_choice='required', extra_body={agent_reference...})"
            ),
            fn=lambda: openai_client.responses.create(
                conversation=conversation.id,
                input=prompt,
                tool_choice="required",
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            ),
        )

    finally:
        if conversation is not None:
            probe.call(
                test_group="test-3b",
                test_case="tool_web_search.cleanup_conversation",
                model=model,
                prompt=None,
                api_style="conversations",
                code_snippet="openai_client.conversations.delete(conversation_id=conversation.id)",
                fn=lambda: openai_client.conversations.delete(conversation_id=conversation.id),
            )
        if agent is not None:
            probe.call(
                test_group="test-3b",
                test_case="tool_web_search.cleanup_agent",
                model=model,
                prompt=None,
                api_style="agents",
                code_snippet="project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)",
                fn=lambda: project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version),
            )


def run_tool_mcp(project_client: AIProjectClient, openai_client: Any, probe: ProbeRunner, model: str, prefix: str) -> None:
    agent_name = _agent_name(prefix, model, "mcp")
    agent = None

    try:
        mcp_tool = MCPTool(
            server_label="api-specs",
            server_url="https://gitmcp.io/Azure/azure-rest-api-specs",
            require_approval="always",
        )

        agent = probe.call(
            test_group="test-3c",
            test_case="tool_mcp.create_agent",
            model=model,
            prompt=None,
            api_style="agents",
            code_snippet="project_client.agents.create_version(... tools=[MCPTool(...)])",
            fn=lambda: project_client.agents.create_version(
                agent_name=agent_name,
                definition=PromptAgentDefinition(
                    model=model,
                    instructions="You are a helpful agent that can use MCP tools.",
                    tools=[mcp_tool],
                ),
            ),
        )

        prompt = "Summarize the Azure REST API specs readme in 3 bullets."
        first = probe.call(
            test_group="test-3c",
            test_case="tool_mcp.run_initial",
            model=model,
            prompt=prompt,
            api_style="responses",
            code_snippet="openai_client.responses.create(input=prompt, extra_body={agent_reference...})",
            fn=lambda: openai_client.responses.create(
                input=prompt,
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            ),
        )

        approvals: list[McpApprovalResponse] = []
        for item in first.output:
            if item.type == "mcp_approval_request" and item.server_label == "api-specs" and item.id:
                approvals.append(
                    McpApprovalResponse(
                        type="mcp_approval_response",
                        approve=True,
                        approval_request_id=item.id,
                    )
                )

        probe.call(
            test_group="test-3c",
            test_case="tool_mcp.approval_followup",
            model=model,
            prompt="[mcp approval responses]",
            api_style="responses",
            code_snippet="openai_client.responses.create(input=approvals, previous_response_id=first.id, extra_body={agent_reference...})",
            fn=lambda: openai_client.responses.create(
                input=approvals,
                previous_response_id=first.id,
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            ),
        )

    finally:
        if agent is not None:
            probe.call(
                test_group="test-3c",
                test_case="tool_mcp.cleanup_agent",
                model=model,
                prompt=None,
                api_style="agents",
                code_snippet="project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)",
                fn=lambda: project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version),
            )


def run_tool_function(
    project_client: AIProjectClient,
    openai_client: Any,
    probe: ProbeRunner,
    model: str,
    prefix: str,
) -> None:
    agent_name = _agent_name(prefix, model, "function")
    agent = None

    def get_horoscope(sign: str) -> str:
        return f"{sign}: Next Tuesday you will befriend a baby otter."

    try:
        tool = FunctionTool(
            name="get_horoscope",
            parameters={
                "type": "object",
                "properties": {
                    "sign": {
                        "type": "string",
                        "description": "An astrological sign like Taurus or Aquarius",
                    }
                },
                "required": ["sign"],
                "additionalProperties": False,
            },
            description="Get today's horoscope for an astrological sign.",
            strict=True,
        )

        agent = probe.call(
            test_group="test-3d",
            test_case="tool_function.create_agent",
            model=model,
            prompt=None,
            api_style="agents",
            code_snippet="project_client.agents.create_version(... tools=[FunctionTool(...)])",
            fn=lambda: project_client.agents.create_version(
                agent_name=agent_name,
                definition=PromptAgentDefinition(
                    model=model,
                    instructions="You are a helpful assistant that can use function tools.",
                    tools=[tool],
                ),
            ),
        )

        prompt = "What is my horoscope? I am an Aquarius."
        first = probe.call(
            test_group="test-3d",
            test_case="tool_function.run_initial",
            model=model,
            prompt=prompt,
            api_style="responses",
            code_snippet="openai_client.responses.create(input=prompt, extra_body={agent_reference...})",
            fn=lambda: openai_client.responses.create(
                input=prompt,
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            ),
        )

        outputs: list[FunctionCallOutput] = []
        for item in first.output:
            if item.type == "function_call" and item.name == "get_horoscope":
                args = json.loads(item.arguments)
                horoscope = get_horoscope(**args)
                outputs.append(
                    FunctionCallOutput(
                        type="function_call_output",
                        call_id=item.call_id,
                        output=json.dumps({"horoscope": horoscope}),
                    )
                )

        probe.call(
            test_group="test-3d",
            test_case="tool_function.followup",
            model=model,
            prompt="[function outputs]",
            api_style="responses",
            code_snippet="openai_client.responses.create(input=outputs, previous_response_id=first.id, extra_body={agent_reference...})",
            fn=lambda: openai_client.responses.create(
                input=outputs,
                previous_response_id=first.id,
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            ),
        )

    finally:
        if agent is not None:
            probe.call(
                test_group="test-3d",
                test_case="tool_function.cleanup_agent",
                model=model,
                prompt=None,
                api_style="agents",
                code_snippet="project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)",
                fn=lambda: project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version),
            )


def _attach_hooks_to_openai_client(openai_client: Any, recorder: HeaderRecorder) -> None:
    inner = getattr(openai_client, "_client", None)
    if inner is not None and hasattr(inner, "event_hooks"):
        inner.event_hooks = {"request": [recorder.on_httpx_request], "response": [recorder.on_httpx_response]}


def _write_outputs(out_dir: Path, records: list[CallRecord], run_metadata: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "run_log.json"
    md_path = out_dir / "run_log.md"
    tsv_path = out_dir / "summary.tsv"

    payload = {
        "metadata": run_metadata,
        "records": [asdict(r) for r in records],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# Agents v2 Validation Run Log ({run_metadata['run_id']})")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append(f"- Python: {run_metadata['python_version']}")
    lines.append(f"- azure-ai-projects: {run_metadata['azure_ai_projects_version']}")
    lines.append(f"- openai: {run_metadata['openai_version']}")
    lines.append(f"- azure-identity: {run_metadata['azure_identity_version']}")
    lines.append(f"- Endpoint: {run_metadata['endpoint_url']}")
    lines.append(f"- Default model: {run_metadata['default_model']}")
    lines.append(f"- Models tested: {', '.join(run_metadata['models_tested'])}")
    lines.append("")
    lines.append("## Calls")
    lines.append("")
    lines.append("| Group | Case | Model | Success | Status | Latency (ms) | apim-request-id | x-request-id | x-ms-region |")
    lines.append("|---|---|---|---|---:|---:|---|---|---|")

    for r in records:
        lines.append(
            f"| {r.test_group} | {r.test_case} | {r.model} | {'yes' if r.success else 'no'} | {r.status_code or '-'} | {r.latency_ms} | {r.apim_request_id or '-'} | {r.x_request_id or '-'} | {r.x_ms_region or '-'} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tsv_lines = [
        "test_group\ttest_case\tmodel\tsuccess\tstatus_code\tlatency_ms\tapim_request_id\tx_request_id\tx_ms_region"
    ]
    for r in records:
        tsv_lines.append(
            "\t".join(
                [
                    r.test_group,
                    r.test_case,
                    r.model,
                    str(r.success),
                    str(r.status_code or ""),
                    str(r.latency_ms),
                    r.apim_request_id or "",
                    r.x_request_id or "",
                    r.x_ms_region or "",
                ]
            )
        )
    tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Foundry Agents v2 validation with metadata capture.")
    parser.add_argument("--retries", type=int, default=0, help="Retries per API call after initial attempt.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional explicit run id (UTC timestamp string). If omitted, current UTC timestamp is used.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Optional explicit output directory. Defaults to validation/results/<run_id>.",
    )
    parser.add_argument(
        "--default-model-only",
        action="store_true",
        help="Run tests only for AZURE_AI_MODEL_DEPLOYMENT_NAME.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
    cfg = load_config()
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger.info("Validation run started run_id=%s", run_id)
    logger.info("Project endpoint=%s", cfg.project_endpoint)

    recorder = HeaderRecorder()
    records: list[CallRecord] = []
    failed_cases: list[str] = []

    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(
            endpoint=cfg.project_endpoint,
            credential=credential,
            per_call_policies=[AzurePipelineRecorderPolicy(recorder)],
        ) as project_client,
        project_client.get_openai_client() as openai_client,
    ):
        _attach_hooks_to_openai_client(openai_client, recorder)

        deployments = list(project_client.deployments.list())
        inference_deployments, embedding_deployments = filter_inference_deployments(deployments)
        all_models = sorted([d.name for d in inference_deployments])
        default_model = cfg.default_model_deployment_name
        if default_model not in all_models:
            all_models.insert(0, default_model)

        if args.default_model_only:
            models_to_test = [default_model]
        else:
            non_default = [m for m in all_models if m != default_model]
            models_to_test = [default_model] + non_default
        if embedding_deployments:
            skipped = ", ".join(sorted(d.name for d in embedding_deployments))
            logger.info("Skipping embedding deployments for validation model loops: %s", skipped)
        logger.info("Models selected=%s", ", ".join(models_to_test))

        probe = ProbeRunner(
            run_id=run_id,
            endpoint_url=cfg.project_endpoint.rstrip("/") + "/openai",
            retries_allowed=max(0, args.retries),
            recorder=recorder,
            records=records,
        )

        for model in models_to_test:
            logger.info("MODEL START model=%s", model)
            cases: list[tuple[str, Callable[[], None]]] = [
                ("test-1.basic_response", lambda m=model: run_basic_response(openai_client, probe, m)),
                (
                    "test-2.agent_lifecycle",
                    lambda m=model: run_agent_lifecycle(
                        project_client, openai_client, probe, m, cfg.agent_name_prefix
                    ),
                ),
                (
                    "test-3a.tool_code_interpreter",
                    lambda m=model: run_tool_code_interpreter(
                        project_client, openai_client, probe, m, cfg.agent_name_prefix
                    ),
                ),
                (
                    "test-3b.tool_web_search",
                    lambda m=model: run_tool_web_search(project_client, openai_client, probe, m, cfg.agent_name_prefix),
                ),
                ("test-3c.tool_mcp", lambda m=model: run_tool_mcp(project_client, openai_client, probe, m, cfg.agent_name_prefix)),
                (
                    "test-3d.tool_function",
                    lambda m=model: run_tool_function(project_client, openai_client, probe, m, cfg.agent_name_prefix),
                ),
            ]

            for case_name, case_fn in cases:
                try:
                    case_fn()
                except Exception as exc:  # noqa: BLE001
                    failed_cases.append(f"{model}:{case_name}:{type(exc).__name__}:{exc}")
                    logger.warning("%s %s failed: %s: %s", model, case_name, type(exc).__name__, exc)
            logger.info("MODEL END model=%s", model)

    out_dir = Path(args.out_dir) if args.out_dir else (Path("validation/results") / run_id)
    run_metadata = {
        "run_id": run_id,
        "python_version": platform.python_version(),
        "azure_ai_projects_version": _package_version("azure-ai-projects"),
        "openai_version": _package_version("openai"),
        "azure_identity_version": _package_version("azure-identity"),
        "endpoint_url": cfg.project_endpoint,
        "default_model": cfg.default_model_deployment_name,
        "models_tested": sorted(list({r.model for r in records})),
        "failed_cases": failed_cases,
    }
    _write_outputs(out_dir, records, run_metadata)
    logger.info("Validation run completed run_id=%s failed_cases=%s", run_id, len(failed_cases))
    logger.info("Artifacts markdown=%s json=%s tsv=%s", out_dir / "run_log.md", out_dir / "run_log.json", out_dir / "summary.tsv")

    print(f"Run completed: {run_id}")
    print(f"Markdown: {out_dir / 'run_log.md'}")
    print(f"JSON: {out_dir / 'run_log.json'}")
    print(f"TSV: {out_dir / 'summary.tsv'}")
    print(f"Failed cases: {len(failed_cases)}")


if __name__ == "__main__":
    main()
