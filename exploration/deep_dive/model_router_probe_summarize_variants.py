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
from openai import OpenAI
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("model_router_probe")


PROMPT_CASES = [
    ("echo", "Reply with exactly: model-router-ok"),
    ("summarize", "Summarize why least-privilege access matters in exactly 3 bullets."),
    (
        "summarize_alt",
        "Summarize the benefits of least-privilege access in exactly 2 short paragraphs. Keep it under 120 words.",
    ),
    (
        "summarize_exec",
        "Write a concise executive summary explaining why least-privilege access matters. Use exactly 4 sentences.",
    ),
    ("code", "Write a Python function that merges two sorted lists."),
    ("reasoning", "You have 9 coins and one is counterfeit and heavier. What is the minimum number of weighings needed?"),
]


@dataclass
class ProbeRecord:
    endpoint: str
    api_style: str
    case_name: str
    run_index: int
    success: bool
    status_code: int | None
    latency_ms: int
    router_deployment: str
    served_model: str | None
    output_preview: str | None
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
    p = argparse.ArgumentParser(description="Probe Model Router behavior across endpoint/API combinations.")
    p.add_argument("--model", default="model-router", help="Router deployment name.")
    p.add_argument("--runs", type=int, default=3, help="Runs per prompt case.")
    p.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    p.add_argument("--timeout-seconds", type=float, default=45.0, help="Per-request timeout.")
    p.add_argument("--max-retries", type=int, default=1, help="OpenAI client retry count.")
    p.add_argument(
        "--cases",
        default="all",
        help=(
            "Comma-separated case set from: "
            "echo,summarize,summarize_alt,summarize_exec,code,reasoning. Default: all."
        ),
    )
    p.add_argument(
        "--paths",
        default="aoai_chat,project_responses",
        help="Comma-separated path set from: aoai_chat, project_responses. Unsupported paths are omitted by default.",
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


def _usage_dict(obj: Any) -> dict[str, Any] | None:
    usage = getattr(obj, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "as_dict"):
        return usage.as_dict()
    return None


def _run_chat(client: OpenAI, recorder: HeaderRecorder, model: str, prompt: str) -> tuple[bool, dict[str, Any]]:
    t0 = time.perf_counter()
    recorder.clear()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return True, {
            "status_code": recorder.last_status_code or 200,
            "latency_ms": latency_ms,
            "served_model": getattr(resp, "model", None),
            "output_preview": (resp.choices[0].message.content if resp.choices else "")[:220],
            "usage": _usage_dict(resp),
            "response_headers": recorder.last_response_headers,
        }
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - t0) * 1000)
        et, ec, em, sc, headers = _err_fields(exc)
        return False, {
            "status_code": sc,
            "latency_ms": latency_ms,
            "served_model": None,
            "output_preview": None,
            "usage": None,
            "error_type": et,
            "error_code": ec,
            "error_message": em,
            "response_headers": recorder.last_response_headers or headers,
        }


def _run_responses(client: OpenAI, recorder: HeaderRecorder, model: str, prompt: str) -> tuple[bool, dict[str, Any]]:
    t0 = time.perf_counter()
    recorder.clear()
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=300,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return True, {
            "status_code": recorder.last_status_code or 200,
            "latency_ms": latency_ms,
            "served_model": getattr(resp, "model", None),
            "output_preview": (resp.output_text or "")[:220],
            "usage": _usage_dict(resp),
            "response_headers": recorder.last_response_headers,
        }
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - t0) * 1000)
        et, ec, em, sc, headers = _err_fields(exc)
        return False, {
            "status_code": sc,
            "latency_ms": latency_ms,
            "served_model": None,
            "output_preview": None,
            "usage": None,
            "error_type": et,
            "error_code": ec,
            "error_message": em,
            "response_headers": recorder.last_response_headers or headers,
        }


def _write_outputs(run_id: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"model_router_probe_{run_id}.json"
    md_path = out_dir / f"model_router_probe_{run_id}.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# Model Router Probe ({run_id})")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for k, v in payload["metadata"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Endpoint | API Style | Case | Run | Success | Status | Latency (ms) | Served Model | Detail |")
    lines.append("|---|---|---|---:|---|---:|---:|---|---|")
    for rec in payload["records"]:
        detail = rec["output_preview"] if rec["success"] else f"{rec['error_type']}: {rec['error_message']}"
        detail = (detail or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {rec['endpoint']} | {rec['api_style']} | {rec['case_name']} | {rec['run_index']} | "
            f"{'yes' if rec['success'] else 'no'} | {rec['status_code'] or '-'} | {rec['latency_ms']} | "
            f"{rec['served_model'] or '-'} | {detail[:140]} |"
        )
    lines.append("")

    lines.append("## Summary by Path")
    lines.append("")
    lines.append("| Endpoint | API Style | Successes | Failures | Served Models Seen |")
    lines.append("|---|---|---:|---:|---|")
    for row in payload["summary"]:
        models = ", ".join(row["served_models_seen"]) if row["served_models_seen"] else "-"
        lines.append(
            f"| {row['endpoint']} | {row['api_style']} | {row['successes']} | {row['failures']} | {models} |"
        )
    lines.append("")

    lines.append("## Failure Headers")
    lines.append("")
    lines.append("| Endpoint | API Style | Case | Run | Status | Selected Model Header | Selected Version Header |")
    lines.append("|---|---|---|---:|---:|---|---|")
    for rec in payload["records"]:
        if rec["success"]:
            continue
        headers = rec.get("response_headers") or {}
        lines.append(
            f"| {rec['endpoint']} | {rec['api_style']} | {rec['case_name']} | {rec['run_index']} | "
            f"{rec['status_code'] or '-'} | {headers.get('x-model-router-selected-model', '-')} | "
            f"{headers.get('x-model-router-selected-model-version', '-')} |"
        )
    lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main() -> None:
    args = parse_args()
    cfg = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"model_router_probe_{run_id}.log"
    _setup_logging(args.log_level, log_path)
    wanted_paths = {p.strip() for p in args.paths.split(",") if p.strip()}
    if args.cases.strip().lower() == "all":
        selected_cases = PROMPT_CASES
    else:
        wanted_cases = {c.strip() for c in args.cases.split(",") if c.strip()}
        selected_cases = [case for case in PROMPT_CASES if case[0] in wanted_cases]
        if not selected_cases:
            raise SystemExit(
                "No valid cases selected.\n"
                "Choose from: echo, summarize, summarize_alt, summarize_exec, code, reasoning"
            )

    logger.info("run_id=%s", run_id)
    logger.info("model=%s", args.model)
    logger.info("paths=%s", sorted(wanted_paths))
    logger.info("runs=%s", args.runs)
    logger.info("cases=%s", [name for name, _ in selected_cases])

    records: list[ProbeRecord] = []
    total_calls = len(wanted_paths) * len(selected_cases) * args.runs
    completed = 0
    with DefaultAzureCredential() as credential:
        with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client:
            aoai_recorder = HeaderRecorder()
            project_recorder = HeaderRecorder()
            aoai_client = _make_aoai_client(cfg, credential, aoai_recorder, args.timeout_seconds, args.max_retries)
            project_openai_client = _make_project_client(project_client, project_recorder, args.timeout_seconds, args.max_retries)

            path_defs: list[tuple[str, str, OpenAI, HeaderRecorder, Any]] = [
                ("aoai", "chat", aoai_client, aoai_recorder, _run_chat),
                ("project", "responses", project_openai_client, project_recorder, _run_responses),
            ]

            for endpoint, api_style, client, recorder, fn in path_defs:
                path_key = f"{endpoint}_{api_style}"
                if path_key not in wanted_paths:
                    continue
                for case_name, prompt in selected_cases:
                    for run_index in range(1, args.runs + 1):
                        logger.info(
                            "START progress=%s/%s endpoint=%s api_style=%s case=%s run=%s",
                            completed + 1,
                            total_calls,
                            endpoint,
                            api_style,
                            case_name,
                            run_index,
                        )
                        ok, data = fn(client, recorder, args.model, prompt)
                        records.append(
                            ProbeRecord(
                                endpoint=endpoint,
                                api_style=api_style,
                                case_name=case_name,
                                run_index=run_index,
                                success=ok,
                                status_code=data.get("status_code"),
                                latency_ms=data["latency_ms"],
                                router_deployment=args.model,
                                served_model=data.get("served_model"),
                                output_preview=data.get("output_preview"),
                                usage=data.get("usage"),
                                error_type=data.get("error_type"),
                                error_code=data.get("error_code"),
                                error_message=data.get("error_message"),
                                response_headers=data.get("response_headers"),
                            )
                        )
                        completed += 1
                        if ok:
                            logger.info(
                                "DONE  progress=%s/%s endpoint=%s api_style=%s case=%s run=%s status=%s latency_ms=%s served_model=%s",
                                completed,
                                total_calls,
                                endpoint,
                                api_style,
                                case_name,
                                run_index,
                                data.get("status_code"),
                                data["latency_ms"],
                                data.get("served_model"),
                            )
                        else:
                            logger.warning(
                                "FAIL  progress=%s/%s endpoint=%s api_style=%s case=%s run=%s status=%s latency_ms=%s error=%s",
                                completed,
                                total_calls,
                                endpoint,
                                api_style,
                                case_name,
                                run_index,
                                data.get("status_code"),
                                data["latency_ms"],
                                data.get("error_message"),
                            )

    summary_rows: list[dict[str, Any]] = []
    for endpoint, api_style in sorted({(r.endpoint, r.api_style) for r in records}):
        subset = [r for r in records if r.endpoint == endpoint and r.api_style == api_style]
        summary_rows.append(
            {
                "endpoint": endpoint,
                "api_style": api_style,
                "successes": sum(1 for r in subset if r.success),
                "failures": sum(1 for r in subset if not r.success),
                "served_models_seen": sorted({r.served_model for r in subset if r.served_model}),
            }
        )

    payload = {
        "metadata": {
            "router_deployment": args.model,
            "project_endpoint": cfg.project_endpoint,
            "aoai_endpoint": f"https://{cfg.resource_name}.openai.azure.com/openai/v1/",
            "runs_per_case": args.runs,
            "timeout_seconds": args.timeout_seconds,
            "max_retries": args.max_retries,
            "paths": sorted(wanted_paths),
            "prompt_cases": [name for name, _ in selected_cases],
        },
        "records": [asdict(r) for r in records],
        "summary": summary_rows,
    }
    md_path, json_path = _write_outputs(run_id, payload)
    logger.info("Artifacts markdown=%s json=%s log=%s", md_path, json_path, log_path)
    print(md_path)
    print(json_path)
    print(log_path)


if __name__ == "__main__":
    main()
