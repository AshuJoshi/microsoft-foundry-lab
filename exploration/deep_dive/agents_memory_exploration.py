#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
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
    MemorySearchOptions,
    MemorySearchPreviewTool,
    MemoryStoreDefaultDefinition,
    MemoryStoreDefaultOptions,
    PromptAgentDefinition,
)
from azure.core.pipeline.policies import SansIOHTTPPolicy
from azure.identity import DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import filter_inference_deployments, load_config

logger = logging.getLogger("agents_memory_exploration")


@dataclass
class StepRecord:
    run_id: str
    model: str
    embedding_model: str
    phase: str
    step: str
    success: bool
    status_code: int | None
    latency_ms: int
    apim_request_id: str | None
    x_request_id: str | None
    x_ms_region: str | None
    request_method: str | None
    request_url: str | None
    error_type: str | None
    error_message: str | None


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


class Runner:
    def __init__(self, run_id: str, recorder: HeaderRecorder, records: list[StepRecord], model: str, embedding_model: str) -> None:
        self.run_id = run_id
        self.recorder = recorder
        self.records = records
        self.model = model
        self.embedding_model = embedding_model

    def step(self, phase: str, step: str, fn: Callable[[], Any]) -> Any:
        self.recorder.clear()
        t0 = time.perf_counter()
        status: int | None = None
        err_type = None
        err_msg = None
        success = False

        logger.info("START phase=%s step=%s model=%s", phase, step, self.model)
        try:
            out = fn()
            success = True
            return out
        except Exception as exc:  # noqa: BLE001
            err_type = type(exc).__name__
            err_msg = str(exc)
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    status = getattr(response, "status_code", None)
                    headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
                    self.recorder.last_response_headers = headers
                except Exception:  # noqa: BLE001
                    pass
            raise
        finally:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            if status is None:
                status = self.recorder.last_status_code
            resp_h = self.recorder.last_response_headers
            self.records.append(
                StepRecord(
                    run_id=self.run_id,
                    model=self.model,
                    embedding_model=self.embedding_model,
                    phase=phase,
                    step=step,
                    success=success,
                    status_code=status,
                    latency_ms=latency_ms,
                    apim_request_id=resp_h.get("apim-request-id"),
                    x_request_id=resp_h.get("x-request-id"),
                    x_ms_region=resp_h.get("x-ms-region"),
                    request_method=self.recorder.last_method,
                    request_url=self.recorder.last_url,
                    error_type=err_type,
                    error_message=err_msg,
                )
            )
            if success:
                logger.info("DONE  phase=%s step=%s status=%s latency_ms=%s", phase, step, status, latency_ms)
            else:
                logger.warning("FAIL  phase=%s step=%s status=%s latency_ms=%s error=%s", phase, step, status, latency_ms, err_msg)


def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def _attach_openai_hooks(openai_client: Any, recorder: HeaderRecorder) -> None:
    inner = getattr(openai_client, "_client", None)
    if inner is not None and hasattr(inner, "event_hooks"):
        inner.event_hooks = {"request": [recorder.on_httpx_request], "response": [recorder.on_httpx_response]}


def _slug(value: str, n: int = 18) -> str:
    x = re.sub(r"[^a-zA-Z0-9-]", "-", value).strip("-").lower()
    return x[:n] or "model"


def _resolve_embedding_model(cli_value: str | None, embedding_deployment_names: list[str]) -> str:
    if cli_value:
        return cli_value

    if embedding_deployment_names:
        return sorted(embedding_deployment_names)[0]

    raise RuntimeError(
        "No embedding model deployment detected. Pass --embedding-model or set MEMORY_EMBEDDING_MODEL_DEPLOYMENT_NAME."
    )


def _surface_methods(obj: Any, prefix: str) -> list[str]:
    methods = []
    for n in dir(obj):
        if n.startswith("_"):
            continue
        v = getattr(obj, n, None)
        if callable(v):
            methods.append(f"{prefix}.{n}")
    return sorted(methods)


def run_model_flow(
    run_id: str,
    project_client: AIProjectClient,
    openai_client: Any,
    recorder: HeaderRecorder,
    model: str,
    embedding_model: str,
    scope: str,
    records: list[StepRecord],
) -> None:
    runner = Runner(run_id=run_id, recorder=recorder, records=records, model=model, embedding_model=embedding_model)

    mem_store = f"mem-{_slug(model)}-{uuid4().hex[:6]}"
    agent_name = f"agent-{_slug(model)}-{uuid4().hex[:6]}"
    conv1 = None
    conv2 = None
    agent = None

    try:
        options = MemoryStoreDefaultOptions(
            user_profile_enabled=True,
            chat_summary_enabled=True,
            user_profile_details=(
                "Store stable preferences and avoid sensitive data such as credentials and precise location."
            ),
        )
        definition = MemoryStoreDefaultDefinition(
            chat_model=model,
            embedding_model=embedding_model,
            options=options,
        )

        runner.step(
            "memory_store",
            "create",
            lambda: project_client.beta.memory_stores.create(
                name=mem_store,
                definition=definition,
                description=f"Memory exploration for {model}",
            ),
        )

        runner.step("memory_store", "get", lambda: project_client.beta.memory_stores.get(name=mem_store))
        runner.step("memory_store", "list", lambda: list(project_client.beta.memory_stores.list(limit=100)))
        runner.step(
            "memory_store",
            "update",
            lambda: project_client.beta.memory_stores.update(name=mem_store, description=f"Updated memory store for {model}"),
        )

        # Direct memory API path.
        update_poller = runner.step(
            "memory_api",
            "begin_update_memories",
            lambda: project_client.beta.memory_stores.begin_update_memories(
                name=mem_store,
                scope=scope,
                items=[{"role": "user", "type": "message", "content": "I prefer dark roast coffee."}],
                update_delay=0,
            ),
        )
        runner.step("memory_api", "update_memories.result", lambda: update_poller.result())

        runner.step(
            "memory_api",
            "search_memories.static",
            lambda: project_client.beta.memory_stores.search_memories(name=mem_store, scope=scope),
        )

        runner.step(
            "memory_api",
            "search_memories.contextual",
            lambda: project_client.beta.memory_stores.search_memories(
                name=mem_store,
                scope=scope,
                items=[{"role": "user", "type": "message", "content": "What are my coffee preferences?"}],
                options=MemorySearchOptions(max_memories=5),
            ),
        )

        # Memory tool on agent path.
        tool = MemorySearchPreviewTool(
            memory_store_name=mem_store,
            scope=scope,
            update_delay=0,
            search_options=MemorySearchOptions(max_memories=5),
        )

        agent = runner.step(
            "agent_memory",
            "create_agent",
            lambda: project_client.agents.create_version(
                agent_name=agent_name,
                definition=PromptAgentDefinition(
                    model=model,
                    instructions="You are a helpful assistant that remembers user preferences.",
                    tools=[tool],
                ),
                description="Agent + memory exploration",
            ),
        )

        conv1 = runner.step("agent_memory", "create_conversation_1", lambda: openai_client.conversations.create())
        runner.step(
            "agent_memory",
            "response_turn_1",
            lambda: openai_client.responses.create(
                conversation=conv1.id,
                input="My favorite coffee is dark roast.",
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            ),
        )

        # Give service a small buffer for memory update materialization.
        runner.step("agent_memory", "sleep_for_memory_materialization", lambda: time.sleep(2))

        conv2 = runner.step("agent_memory", "create_conversation_2", lambda: openai_client.conversations.create())
        runner.step(
            "agent_memory",
            "response_turn_2",
            lambda: openai_client.responses.create(
                conversation=conv2.id,
                input="What coffee do I usually prefer?",
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            ),
        )

    finally:
        if conv1 is not None:
            try:
                runner.step("cleanup", "delete_conversation_1", lambda: openai_client.conversations.delete(conversation_id=conv1.id))
            except Exception:  # noqa: BLE001
                pass
        if conv2 is not None:
            try:
                runner.step("cleanup", "delete_conversation_2", lambda: openai_client.conversations.delete(conversation_id=conv2.id))
            except Exception:  # noqa: BLE001
                pass
        if agent is not None:
            try:
                runner.step(
                    "cleanup",
                    "delete_agent",
                    lambda: project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version),
                )
            except Exception:  # noqa: BLE001
                pass
        # Clean scoped memories before deleting memory store.
        try:
            runner.step("cleanup", "delete_scope", lambda: project_client.beta.memory_stores.delete_scope(name=mem_store, scope=scope))
        except Exception:  # noqa: BLE001
            pass
        try:
            runner.step("cleanup", "delete_memory_store", lambda: project_client.beta.memory_stores.delete(name=mem_store))
        except Exception:  # noqa: BLE001
            pass


def _write_reports(run_id: str, records: list[StepRecord], out_dir: Path, metadata: dict[str, Any]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"agents_memory_exploration_{run_id}.json"
    md_path = out_dir / f"agents_memory_exploration_{run_id}.md"

    payload = {"metadata": metadata, "records": [asdict(r) for r in records]}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"# Agents + Memory Exploration ({run_id})")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    for k, v in metadata.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("| Model | Phase | Step | Success | Status | Latency (ms) | apim-request-id | x-request-id |")
    lines.append("|---|---|---|---|---:|---:|---|---|")
    for r in records:
        lines.append(
            f"| {r.model} | {r.phase} | {r.step} | {'yes' if r.success else 'no'} | {r.status_code or '-'} | {r.latency_ms} | {r.apim_request_id or '-'} | {r.x_request_id or '-'} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep dive exploration for Foundry agents + memory surfaces.")
    p.add_argument("--all-models", action="store_true", help="Run against all deployments. Default runs default model only.")
    p.add_argument("--embedding-model", type=str, default="", help="Embedding deployment name for memory store.")
    p.add_argument("--scope", type=str, default="user_123", help="Memory scope key.")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
    cfg = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    records: list[StepRecord] = []
    recorder = HeaderRecorder()

    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(
            endpoint=cfg.project_endpoint,
            credential=credential,
            per_call_policies=[AzurePipelineRecorderPolicy(recorder)],
        ) as project_client,
        project_client.get_openai_client() as openai_client,
    ):
        _attach_openai_hooks(openai_client, recorder)

        deployments = list(project_client.deployments.list())
        inference_deployments, embedding_deployments = filter_inference_deployments(deployments)
        embedding_deployment_names = [d.name for d in embedding_deployments]
        default_model = cfg.default_model_deployment_name
        embedding_model = _resolve_embedding_model(
            args.embedding_model or (
                os.environ.get("MEMORY_EMBEDDING_MODEL_DEPLOYMENT_NAME", "")
            ) or None,
            embedding_deployment_names,
        )

        if args.all_models:
            models = sorted(d.name for d in inference_deployments)
            if default_model in models:
                models.remove(default_model)
                models.insert(0, default_model)
        else:
            models = [default_model]
        if embedding_deployment_names:
            logger.info(
                "Embedding deployments detected (excluded from chat/agent model loop): %s",
                ", ".join(sorted(embedding_deployment_names)),
            )

        # Surface introspection notes.
        agent_methods = _surface_methods(project_client.agents, "agents")
        memory_methods = _surface_methods(project_client.beta.memory_stores, "beta.memory_stores")

        for model in models:
            logger.info("MODEL START model=%s embedding_model=%s", model, embedding_model)
            try:
                run_model_flow(
                    run_id=run_id,
                    project_client=project_client,
                    openai_client=openai_client,
                    recorder=recorder,
                    model=model,
                    embedding_model=embedding_model,
                    scope=args.scope,
                    records=records,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("MODEL FAILED model=%s err=%s: %s", model, type(exc).__name__, exc)
            logger.info("MODEL END model=%s", model)

    out_dir = Path(__file__).resolve().parent / "output"
    metadata = {
        "run_id": run_id,
        "project_endpoint": cfg.project_endpoint,
        "default_model": cfg.default_model_deployment_name,
        "embedding_model": embedding_model,
        "models_requested": ", ".join(models),
        "scope": args.scope,
        "agent_methods_count": len(agent_methods),
        "memory_methods_count": len(memory_methods),
        "agent_methods_sample": agent_methods[:20],
        "memory_methods_sample": memory_methods[:20],
        "doc_concepts": "docs/source-scrapes/foundrydocs/What is Memory? - Microsoft Foundry.md",
        "doc_howto": "docs/source-scrapes/foundrydocs/Create and Use Memory - Microsoft Foundry.md",
    }
    md_path, json_path = _write_reports(run_id, records, out_dir, metadata)

    print(f"Run completed: {run_id}")
    print(f"Markdown: {md_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
