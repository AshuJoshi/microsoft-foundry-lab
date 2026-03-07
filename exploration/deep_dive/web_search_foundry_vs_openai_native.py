#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, WebSearchApproximateLocation, WebSearchTool
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config


@dataclass
class ProbeCase:
    name: str
    prompt: str


@dataclass
class CaseResult:
    engine: str
    case_name: str
    success: bool
    status_code: int | None
    latency_ms: int
    citation_count: int
    citation_urls: list[str]
    observed_urls: list[str]
    date_mentions: list[str]
    output_preview: str | None
    error_type: str | None
    error_message: str | None
    apim_request_id: str | None
    x_request_id: str | None
    x_ms_region: str | None


class HeaderRecorder:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
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
        description="Compare Foundry SDK web search tool path vs OpenAI native web_search path on the same cases."
    )
    p.add_argument("--model", default=cfg.default_model_deployment_name)
    p.add_argument(
        "--topic",
        default="NVIDIA quarterly earnings and guidance",
        help="Search topic substituted into case templates.",
    )
    p.add_argument(
        "--cases-file",
        default=str(Path(__file__).resolve().parent / "cases" / "web_search_foundry_vs_openai_native.json"),
        help="JSON file containing reusable case templates.",
    )
    p.add_argument("--days-window", type=int, default=30)
    p.add_argument("--tool-choice", choices=["auto", "required"], default="required")
    p.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--country", default="US")
    p.add_argument("--region", default="WA")
    p.add_argument("--city", default="Seattle")
    return p.parse_args()


def _attach_hooks(openai_client: Any, rec: HeaderRecorder) -> None:
    inner = getattr(openai_client, "_client", None)
    if inner is not None and hasattr(inner, "event_hooks"):
        inner.event_hooks = {"request": [rec.on_request], "response": [rec.on_response]}


def _extract_citations_from_item(item: Any) -> list[str]:
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


def _extract_date_mentions(text: str) -> list[str]:
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b",
    ]
    found: list[str] = []
    for pat in patterns:
        found.extend(re.findall(pat, text, flags=re.IGNORECASE))
    seen: set[str] = set()
    out: list[str] = []
    for d in found:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _extract_urls_from_text(text: str) -> list[str]:
    # Basic URL capture from model text output (fallback when annotations are sparse).
    raw = re.findall(r"https?://[^\s\)\]\}>,\"']+", text or "", flags=re.IGNORECASE)
    seen: set[str] = set()
    out: list[str] = []
    for u in raw:
        u2 = u.rstrip(".,;:!?)")
        if u2 and u2 not in seen:
            seen.add(u2)
            out.append(u2)
    return out


def _normalize_url(url: str) -> str:
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        path = (p.path or "").rstrip("/")
        if host.startswith("www."):
            host = host[4:]
        return f"{host}{path}"
    except Exception:  # noqa: BLE001
        return url.lower().strip()


def _run_foundry_case(
    *,
    project_client: AIProjectClient,
    openai_client: Any,
    recorder: HeaderRecorder,
    model: str,
    prompt: str,
    case_name: str,
    tool_choice: str,
    stream: bool,
    location: WebSearchApproximateLocation,
    agent_name_prefix: str,
) -> CaseResult:
    agent = None
    conversation = None
    recorder.clear()
    t0 = time.perf_counter()
    output_text = ""
    citation_urls: list[str] = []
    try:
        run_stamp = datetime.now(timezone.utc).strftime("%H%M%S")
        agent_name = f"{agent_name_prefix}-ws-{case_name}-{run_stamp}".lower().replace("_", "-")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model,
                instructions="You are a web research assistant. Use web search and cite sources.",
                tools=[WebSearchTool(user_location=location)],
            ),
            description="Foundry web search comparison agent.",
        )
        conversation = openai_client.conversations.create()

        if stream:
            s = openai_client.responses.create(
                stream=True,
                conversation=conversation.id,
                input=prompt,
                tool_choice=tool_choice,
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            )
            for event in s:
                if event.type == "response.output_text.delta":
                    output_text += event.delta or ""
                elif event.type == "response.output_item.done":
                    citation_urls.extend(_extract_citations_from_item(event.item))
                elif event.type == "response.completed" and not output_text:
                    output_text = event.response.output_text or ""
        else:
            resp = openai_client.responses.create(
                conversation=conversation.id,
                input=prompt,
                tool_choice=tool_choice,
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            )
            output_text = resp.output_text or ""
            for item in getattr(resp, "output", []) or []:
                citation_urls.extend(_extract_citations_from_item(item))

        latency = int((time.perf_counter() - t0) * 1000)
        urls = sorted(set(citation_urls))
        observed_urls = sorted({_normalize_url(u) for u in (urls + _extract_urls_from_text(output_text)) if u})
        return CaseResult(
            engine="foundry_sdk_web_search_preview_path",
            case_name=case_name,
            success=True,
            status_code=recorder.status_code,
            latency_ms=latency,
            citation_count=len(urls),
            citation_urls=urls,
            observed_urls=observed_urls,
            date_mentions=_extract_date_mentions(output_text),
            output_preview=output_text[:500] if output_text else None,
            error_type=None,
            error_message=None,
            apim_request_id=recorder.headers.get("apim-request-id"),
            x_request_id=recorder.headers.get("x-request-id"),
            x_ms_region=recorder.headers.get("x-ms-region"),
        )
    except Exception as exc:  # noqa: BLE001
        latency = int((time.perf_counter() - t0) * 1000)
        resp = getattr(exc, "response", None)
        status = recorder.status_code
        headers = recorder.headers
        if resp is not None:
            status = getattr(resp, "status_code", status)
            try:
                headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
            except Exception:  # noqa: BLE001
                pass
        return CaseResult(
            engine="foundry_sdk_web_search_preview_path",
            case_name=case_name,
            success=False,
            status_code=status,
            latency_ms=latency,
            citation_count=0,
            citation_urls=[],
            observed_urls=[],
            date_mentions=[],
            output_preview=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
            apim_request_id=headers.get("apim-request-id"),
            x_request_id=headers.get("x-request-id"),
            x_ms_region=headers.get("x-ms-region"),
        )
    finally:
        if conversation is not None:
            try:
                openai_client.conversations.delete(conversation_id=conversation.id)
            except Exception:  # noqa: BLE001
                pass
        if agent is not None:
            try:
                project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
            except Exception:  # noqa: BLE001
                pass


def _run_openai_native_case(
    *,
    client: OpenAI,
    recorder: HeaderRecorder,
    model: str,
    prompt: str,
    case_name: str,
    tool_choice: str,
    stream: bool,
) -> CaseResult:
    recorder.clear()
    t0 = time.perf_counter()
    output_text = ""
    citation_urls: list[str] = []
    try:
        if stream:
            s = client.responses.create(
                stream=True,
                model=model,
                input=prompt,
                tool_choice=tool_choice,
                tools=[{"type": "web_search"}],
            )
            for event in s:
                if event.type == "response.output_text.delta":
                    output_text += event.delta or ""
                elif event.type == "response.output_item.done":
                    citation_urls.extend(_extract_citations_from_item(event.item))
                elif event.type == "response.completed" and not output_text:
                    output_text = event.response.output_text or ""
        else:
            resp = client.responses.create(
                model=model,
                input=prompt,
                tool_choice=tool_choice,
                tools=[{"type": "web_search"}],
            )
            output_text = resp.output_text or ""
            for item in getattr(resp, "output", []) or []:
                citation_urls.extend(_extract_citations_from_item(item))

        latency = int((time.perf_counter() - t0) * 1000)
        urls = sorted(set(citation_urls))
        observed_urls = sorted({_normalize_url(u) for u in (urls + _extract_urls_from_text(output_text)) if u})
        return CaseResult(
            engine="openai_native_web_search",
            case_name=case_name,
            success=True,
            status_code=recorder.status_code,
            latency_ms=latency,
            citation_count=len(urls),
            citation_urls=urls,
            observed_urls=observed_urls,
            date_mentions=_extract_date_mentions(output_text),
            output_preview=output_text[:500] if output_text else None,
            error_type=None,
            error_message=None,
            apim_request_id=recorder.headers.get("apim-request-id"),
            x_request_id=recorder.headers.get("x-request-id"),
            x_ms_region=recorder.headers.get("x-ms-region"),
        )
    except Exception as exc:  # noqa: BLE001
        latency = int((time.perf_counter() - t0) * 1000)
        resp = getattr(exc, "response", None)
        status = recorder.status_code
        headers = recorder.headers
        if resp is not None:
            status = getattr(resp, "status_code", status)
            try:
                headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
            except Exception:  # noqa: BLE001
                pass
        return CaseResult(
            engine="openai_native_web_search",
            case_name=case_name,
            success=False,
            status_code=status,
            latency_ms=latency,
            citation_count=0,
            citation_urls=[],
            observed_urls=[],
            date_mentions=[],
            output_preview=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
            apim_request_id=headers.get("apim-request-id"),
            x_request_id=headers.get("x-request-id"),
            x_ms_region=headers.get("x-ms-region"),
        )


def _write_reports(run_id: str, out_dir: Path, metadata: dict[str, Any], rows: list[CaseResult]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"web_search_foundry_vs_openai_native_{run_id}.json"
    md_path = out_dir / f"web_search_foundry_vs_openai_native_{run_id}.md"
    json_path.write_text(
        json.dumps({"metadata": metadata, "results": [asdict(r) for r in rows]}, indent=2),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append(f"# Web Search: Foundry SDK Path vs OpenAI Native ({run_id})")
    lines.append("")
    for k, v in metadata.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("| Engine | Case | Success | Status | Latency (ms) | Citations | Date Mentions |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r.engine} | {r.case_name} | {'yes' if r.success else 'no'} | {r.status_code or '-'} | {r.latency_ms} | {r.citation_count} | {len(r.date_mentions)} |"
        )
    lines.append("")
    lines.append("## Per-Case URL Overlap")
    lines.append("")
    lines.append("| Case | Foundry URLs | OpenAI URLs | Overlap | Foundry-only | OpenAI-only |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    case_names = sorted({r.case_name for r in rows})
    for case in case_names:
        foundry = next((r for r in rows if r.case_name == case and r.engine == "foundry_sdk_web_search_preview_path"), None)
        native = next((r for r in rows if r.case_name == case and r.engine == "openai_native_web_search"), None)
        f_urls = set(foundry.observed_urls if foundry and foundry.success else [])
        n_urls = set(native.observed_urls if native and native.success else [])
        overlap = f_urls.intersection(n_urls)
        f_only = f_urls - n_urls
        n_only = n_urls - f_urls
        lines.append(f"| {case} | {len(f_urls)} | {len(n_urls)} | {len(overlap)} | {len(f_only)} | {len(n_only)} |")
        if overlap:
            lines.append(f"  overlap_urls: {', '.join(sorted(overlap)[:6])}")
        if f_only:
            lines.append(f"  foundry_only_urls: {', '.join(sorted(f_only)[:6])}")
        if n_only:
            lines.append(f"  openai_only_urls: {', '.join(sorted(n_only)[:6])}")

    lines.append("")
    lines.append("## Raw Results")
    lines.append("")
    for r in rows:
        lines.append(f"### {r.engine} / {r.case_name}")
        lines.append("")
        lines.append(f"- success: {r.success}")
        lines.append(f"- status_code: {r.status_code}")
        lines.append(f"- citation_count: {r.citation_count}")
        lines.append(f"- observed_url_count: {len(r.observed_urls)}")
        if r.error_message:
            lines.append(f"- error: {r.error_message}")
        if r.citation_urls:
            lines.append("- citation_urls:")
            for u in r.citation_urls[:8]:
                lines.append(f"  - {u}")
        lines.append("```text")
        lines.append(r.output_preview or "")
        lines.append("```")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def _load_cases(cases_file: Path, topic: str, since_date: str, days_window: int) -> list[ProbeCase]:
    if not cases_file.exists():
        raise FileNotFoundError(f"Cases file not found: {cases_file}")
    payload = json.loads(cases_file.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", [])
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"No cases found in {cases_file}")

    out: list[ProbeCase] = []
    for c in raw_cases:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "")).strip()
        template = str(c.get("prompt_template", "")).strip()
        if not name or not template:
            continue
        prompt = template.format(topic=topic, since_date=since_date, days_window=days_window)
        out.append(ProbeCase(name=name, prompt=prompt))
    if not out:
        raise ValueError(f"No valid case entries in {cases_file}")
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    since_date = (datetime.now(timezone.utc) - timedelta(days=args.days_window)).date().isoformat()
    location = WebSearchApproximateLocation(country=args.country, city=args.city, region=args.region)
    cases_file = Path(args.cases_file)
    cases = _load_cases(
        cases_file=cases_file,
        topic=args.topic,
        since_date=since_date,
        days_window=args.days_window,
    )
    print(f"run_id={run_id}")
    print(f"model={args.model}")
    print(f"topic={args.topic}")
    print(f"cases_file={cases_file}")
    print(f"case_count={len(cases)}")

    results: list[CaseResult] = []
    with DefaultAzureCredential() as credential:
        with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client:
            with project_client.get_openai_client() as foundry_openai:
                foundry_recorder = HeaderRecorder()
                _attach_hooks(foundry_openai, foundry_recorder)
                for case in cases:
                    print(f"foundry case={case.name}")
                    row = _run_foundry_case(
                        project_client=project_client,
                        openai_client=foundry_openai,
                        recorder=foundry_recorder,
                        model=args.model,
                        prompt=case.prompt,
                        case_name=case.name,
                        tool_choice=args.tool_choice,
                        stream=args.stream,
                        location=location,
                        agent_name_prefix=cfg.agent_name_prefix,
                    )
                    results.append(row)
                    print(
                        f"  success={row.success} status={row.status_code} latency_ms={row.latency_ms} citations={row.citation_count}"
                    )

        # OpenAI native web_search path over AOAI resource endpoint.
        native_recorder = HeaderRecorder()
        with OpenAI(
            base_url=f"https://{cfg.resource_name}.openai.azure.com/openai/v1/",
            api_key=get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default"),
            http_client=httpx.Client(
                event_hooks={"request": [native_recorder.on_request], "response": [native_recorder.on_response]}
            ),
        ) as openai_client:
            for case in cases:
                print(f"openai_native case={case.name}")
                row = _run_openai_native_case(
                    client=openai_client,
                    recorder=native_recorder,
                    model=args.model,
                    prompt=case.prompt,
                    case_name=case.name,
                    tool_choice=args.tool_choice,
                    stream=args.stream,
                )
                results.append(row)
                print(
                    f"  success={row.success} status={row.status_code} latency_ms={row.latency_ms} citations={row.citation_count}"
                )

    metadata = {
        "run_id": run_id,
        "model": args.model,
        "tool_choice": args.tool_choice,
        "stream": args.stream,
        "topic": args.topic,
        "days_window": args.days_window,
        "since_date": since_date,
        "cases_file": str(cases_file),
        "case_count": len(cases),
        "foundry_project_endpoint": cfg.project_endpoint,
        "openai_native_base_url": f"https://{cfg.resource_name}.openai.azure.com/openai/v1/",
    }
    out_dir = Path(__file__).resolve().parent / "output"
    md_path, json_path = _write_reports(run_id, out_dir, metadata, results)
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
