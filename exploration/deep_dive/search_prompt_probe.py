#!/usr/bin/env python3
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
from openai import OpenAI
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("search_prompt_probe")


@dataclass
class PromptCase:
    name: str
    prompt: str


@dataclass
class ProbeRecord:
    endpoint: str
    case_name: str
    run_index: int
    success: bool
    status_code: int | None
    latency_ms: int
    model: str
    output_preview: str | None
    citation_count: int
    citation_urls: list[str]
    mentioned_dates: list[str]
    usage: dict[str, Any] | None
    error_type: str | None
    error_code: str | None
    error_message: str | None
    response_headers: dict[str, str] | None


class HeaderRecorder:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.last_request_headers: dict[str, str] = {}
        self.last_response_headers: dict[str, str] = {}
        self.last_status_code: int | None = None

    def on_request(self, request: httpx.Request) -> None:
        self.last_request_headers = {k.lower(): v for k, v in request.headers.items()}

    def on_response(self, response: httpx.Response) -> None:
        self.last_response_headers = {k.lower(): v for k, v in response.headers.items()}
        self.last_status_code = response.status_code


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Direct prompt-based probe for web_search_preview on Responses API.")
    p.add_argument("--model", default=cfg.default_model_deployment_name, help="Deployment/model name to test.")
    p.add_argument("--topic", default="Microsoft Foundry announcements", help="Topic substituted into prompt cases.")
    p.add_argument("--days-window", type=int, default=30, help="Date window for time-bounded prompts.")
    p.add_argument("--runs", type=int, default=1, help="Runs per case per endpoint.")
    p.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    p.add_argument("--timeout-seconds", type=float, default=45.0, help="Per-request timeout.")
    p.add_argument("--max-retries", type=int, default=1, help="OpenAI client retry count.")
    p.add_argument(
        "--cases",
        default="all",
        help="Comma-separated case set from: baseline,recent_window,strict_dates. Default: all.",
    )
    p.add_argument(
        "--paths",
        default="aoai_responses,project_responses",
        help="Comma-separated path set from: aoai_responses,project_responses.",
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


def _make_http_client(recorder: HeaderRecorder, timeout_seconds: float) -> httpx.Client:
    return httpx.Client(
        timeout=timeout_seconds,
        event_hooks={"request": [recorder.on_request], "response": [recorder.on_response]},
    )


def _attach_hooks_to_openai_client(openai_client: Any, recorder: HeaderRecorder) -> None:
    inner = getattr(openai_client, "_client", None)
    if inner is not None and hasattr(inner, "event_hooks"):
        inner.event_hooks = {"request": [recorder.on_request], "response": [recorder.on_response]}


def _make_aoai_client(cfg: Any, credential: Any, recorder: HeaderRecorder, timeout_seconds: float, max_retries: int) -> OpenAI:
    return OpenAI(
        base_url=f"https://{cfg.resource_name}.openai.azure.com/openai/v1/",
        api_key=get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default"),
        timeout=timeout_seconds,
        max_retries=max_retries,
        http_client=_make_http_client(recorder, timeout_seconds),
    )


def _make_project_client(project_client: AIProjectClient, recorder: HeaderRecorder, timeout_seconds: float, max_retries: int) -> OpenAI:
    client = project_client.get_openai_client(timeout=timeout_seconds, max_retries=max_retries)
    _attach_hooks_to_openai_client(client, recorder)
    return client


def _extract_citation_urls(resp: Any) -> list[str]:
    urls: list[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) != "output_text":
                continue
            for ann in getattr(content, "annotations", []) or []:
                if getattr(ann, "type", None) == "url_citation" and getattr(ann, "url", None):
                    urls.append(str(ann.url))
    return sorted(set(urls))


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


def _usage_dict(obj: Any) -> dict[str, Any] | None:
    usage = getattr(obj, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "as_dict"):
        return usage.as_dict()
    return None


def _err_fields(exc: Exception) -> tuple[str, str | None, str, int | None, dict[str, str]]:
    err_type = type(exc).__name__
    err_code = None
    message = str(exc)
    status = None
    headers: dict[str, str] = {}
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        try:
            headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
        except Exception:  # noqa: BLE001
            headers = {}
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(err, dict):
            if err.get("code") is not None:
                err_code = str(err.get("code"))
            if err.get("message"):
                message = str(err.get("message"))
    return err_type, err_code, message, status, headers


def _build_cases(topic: str, days_window: int) -> list[PromptCase]:
    since_date = (datetime.now(timezone.utc) - timedelta(days=days_window)).date().isoformat()
    return [
        PromptCase(
            "baseline",
            f"Find two recent updates about {topic}. Cite sources with links.",
        ),
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


def _run_case(client: OpenAI, recorder: HeaderRecorder, model: str, case: PromptCase, endpoint: str, run_index: int) -> ProbeRecord:
    t0 = time.perf_counter()
    recorder.clear()
    try:
        resp = client.responses.create(
            model=model,
            input=case.prompt,
            tools=[{"type": "web_search_preview"}],
            tool_choice="required",
            max_output_tokens=500,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        output_text = resp.output_text or ""
        return ProbeRecord(
            endpoint=endpoint,
            case_name=case.name,
            run_index=run_index,
            success=True,
            status_code=recorder.last_status_code or 200,
            latency_ms=latency_ms,
            model=getattr(resp, "model", model),
            output_preview=output_text[:500] if output_text else None,
            citation_count=len(_extract_citation_urls(resp)),
            citation_urls=_extract_citation_urls(resp),
            mentioned_dates=_extract_dates(output_text),
            usage=_usage_dict(resp),
            error_type=None,
            error_code=None,
            error_message=None,
            response_headers=recorder.last_response_headers,
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - t0) * 1000)
        err_type, err_code, err_message, status, headers = _err_fields(exc)
        return ProbeRecord(
            endpoint=endpoint,
            case_name=case.name,
            run_index=run_index,
            success=False,
            status_code=status or recorder.last_status_code,
            latency_ms=latency_ms,
            model=model,
            output_preview=None,
            citation_count=0,
            citation_urls=[],
            mentioned_dates=[],
            usage=None,
            error_type=err_type,
            error_code=err_code,
            error_message=err_message,
            response_headers=recorder.last_response_headers or headers,
        )


def _write_outputs(run_id: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"search_prompt_probe_{run_id}.json"
    md_path = out_dir / f"search_prompt_probe_{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# Search Prompt Probe ({run_id})")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for k, v in payload["metadata"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Endpoint | Case | Run | Success | Status | Latency (ms) | Served Model | Citations | Detail |")
    lines.append("|---|---|---:|---|---:|---:|---|---:|---|")
    for rec in payload["records"]:
        detail = rec["output_preview"] if rec["success"] else f"{rec['error_type']}: {rec['error_message']}"
        detail = (detail or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {rec['endpoint']} | {rec['case_name']} | {rec['run_index']} | "
            f"{'yes' if rec['success'] else 'no'} | {rec['status_code'] or '-'} | {rec['latency_ms']} | "
            f"{rec['model'] or '-'} | {rec['citation_count']} | {detail[:160]} |"
        )
    lines.append("")
    lines.append("## Failure Headers")
    lines.append("")
    lines.append("| Endpoint | Case | Run | Status | APIM Request ID | x-request-id | x-ms-region |")
    lines.append("|---|---|---:|---:|---|---|---|")
    for rec in payload["records"]:
        if rec["success"]:
            continue
        headers = rec.get("response_headers") or {}
        lines.append(
            f"| {rec['endpoint']} | {rec['case_name']} | {rec['run_index']} | {rec['status_code'] or '-'} | "
            f"{headers.get('apim-request-id', '-')} | {headers.get('x-request-id', '-')} | "
            f"{headers.get('x-ms-region', '-')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main() -> None:
    args = parse_args()
    cfg = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "output"
    log_path = out_dir / f"search_prompt_probe_{run_id}.log"
    _setup_logging(args.log_level, log_path)

    all_cases = _build_cases(args.topic, args.days_window)
    if args.cases.strip().lower() == "all":
        selected_cases = all_cases
    else:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        selected_cases = [case for case in all_cases if case.name in wanted]
        if not selected_cases:
            raise SystemExit("No valid cases selected. Choose from: baseline, recent_window, strict_dates")

    wanted_paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    valid_paths = {"aoai_responses", "project_responses"}
    unknown_paths = [p for p in wanted_paths if p not in valid_paths]
    if unknown_paths:
        raise SystemExit(f"Unknown path(s): {unknown_paths}. Choose from: {sorted(valid_paths)}")

    total_calls = len(wanted_paths) * len(selected_cases) * args.runs
    progress = 0

    logger.info("run_id=%s", run_id)
    logger.info("model=%s", args.model)
    logger.info("paths=%s", wanted_paths)
    logger.info("runs=%s", args.runs)
    logger.info("cases=%s", [c.name for c in selected_cases])

    metadata = {
        "model": args.model,
        "topic": args.topic,
        "days_window": args.days_window,
        "runs_per_case": args.runs,
        "timeout_seconds": args.timeout_seconds,
        "max_retries": args.max_retries,
        "paths": wanted_paths,
        "cases": [c.name for c in selected_cases],
        "project_endpoint": cfg.project_endpoint,
        "aoai_endpoint": f"https://{cfg.resource_name}.openai.azure.com/openai/v1/",
    }

    records: list[ProbeRecord] = []
    with DefaultAzureCredential() as credential, AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client:
        aoai_recorder = HeaderRecorder()
        project_recorder = HeaderRecorder()
        clients = {
            "aoai_responses": _make_aoai_client(cfg, credential, aoai_recorder, args.timeout_seconds, args.max_retries),
            "project_responses": _make_project_client(project_client, project_recorder, args.timeout_seconds, args.max_retries),
        }
        recorders = {"aoai_responses": aoai_recorder, "project_responses": project_recorder}

        for endpoint in wanted_paths:
            client = clients[endpoint]
            recorder = recorders[endpoint]
            for case in selected_cases:
                for run_index in range(1, args.runs + 1):
                    progress += 1
                    logger.info(
                        "START progress=%s/%s endpoint=%s case=%s run=%s",
                        progress,
                        total_calls,
                        endpoint,
                        case.name,
                        run_index,
                    )
                    rec = _run_case(client, recorder, args.model, case, endpoint, run_index)
                    records.append(rec)
                    if rec.success:
                        logger.info(
                            "DONE  progress=%s/%s endpoint=%s case=%s run=%s status=%s latency_ms=%s citations=%s served_model=%s",
                            progress,
                            total_calls,
                            endpoint,
                            case.name,
                            run_index,
                            rec.status_code,
                            rec.latency_ms,
                            rec.citation_count,
                            rec.model,
                        )
                    else:
                        logger.warning(
                            "FAIL  progress=%s/%s endpoint=%s case=%s run=%s status=%s latency_ms=%s error=%s",
                            progress,
                            total_calls,
                            endpoint,
                            case.name,
                            run_index,
                            rec.status_code,
                            rec.latency_ms,
                            rec.error_message,
                        )

    payload = {"metadata": metadata, "records": [asdict(r) for r in records]}
    md_path, json_path = _write_outputs(run_id, payload)
    logger.info("Artifacts markdown=%s json=%s log=%s", md_path, json_path, log_path)


if __name__ == "__main__":
    main()
