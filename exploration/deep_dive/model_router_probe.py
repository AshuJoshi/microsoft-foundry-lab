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

from openai import OpenAI
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config


PROMPT_CASES = [
    ("echo", "Reply with exactly: model-router-ok"),
    ("summarize", "Summarize why least-privilege access matters in exactly 3 bullets."),
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe Model Router behavior across endpoint/API combinations.")
    p.add_argument("--model", default="model-router", help="Router deployment name.")
    p.add_argument("--runs", type=int, default=3, help="Runs per prompt case.")
    p.add_argument(
        "--cases",
        default="all",
        help="Comma-separated case set from: echo,summarize,code,reasoning. Default: all.",
    )
    p.add_argument(
        "--paths",
        default="aoai_chat,project_responses",
        help="Comma-separated path set from: aoai_chat, project_responses. Unsupported paths are omitted by default.",
    )
    return p.parse_args()


def _make_aoai_client(cfg: Any, credential: Any) -> OpenAI:
    return OpenAI(
        base_url=f"https://{cfg.resource_name}.openai.azure.com/openai/v1/",
        api_key=get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default"),
    )


def _make_project_client(project_client: AIProjectClient) -> OpenAI:
    return project_client.get_openai_client()


def _err_fields(exc: Exception) -> tuple[str, str | None, str, int | None]:
    err_type = type(exc).__name__
    err_code = None
    message = str(exc)
    status = None
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(err, dict):
            if err.get("code") is not None:
                err_code = str(err.get("code"))
            if err.get("message"):
                message = str(err.get("message"))
    return err_type, err_code, message, status


def _usage_dict(obj: Any) -> dict[str, Any] | None:
    usage = getattr(obj, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "as_dict"):
        return usage.as_dict()
    return None


def _run_chat(client: OpenAI, model: str, prompt: str) -> tuple[bool, dict[str, Any]]:
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return True, {
            "status_code": 200,
            "latency_ms": latency_ms,
            "served_model": getattr(resp, "model", None),
            "output_preview": (resp.choices[0].message.content if resp.choices else "")[:220],
            "usage": _usage_dict(resp),
        }
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - t0) * 1000)
        et, ec, em, sc = _err_fields(exc)
        return False, {
            "status_code": sc,
            "latency_ms": latency_ms,
            "served_model": None,
            "output_preview": None,
            "usage": None,
            "error_type": et,
            "error_code": ec,
            "error_message": em,
        }


def _run_responses(client: OpenAI, model: str, prompt: str) -> tuple[bool, dict[str, Any]]:
    t0 = time.perf_counter()
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=300,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return True, {
            "status_code": 200,
            "latency_ms": latency_ms,
            "served_model": getattr(resp, "model", None),
            "output_preview": (resp.output_text or "")[:220],
            "usage": _usage_dict(resp),
        }
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - t0) * 1000)
        et, ec, em, sc = _err_fields(exc)
        return False, {
            "status_code": sc,
            "latency_ms": latency_ms,
            "served_model": None,
            "output_preview": None,
            "usage": None,
            "error_type": et,
            "error_code": ec,
            "error_message": em,
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

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main() -> None:
    args = parse_args()
    cfg = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    wanted_paths = {p.strip() for p in args.paths.split(",") if p.strip()}
    if args.cases.strip().lower() == "all":
        selected_cases = PROMPT_CASES
    else:
        wanted_cases = {c.strip() for c in args.cases.split(",") if c.strip()}
        selected_cases = [case for case in PROMPT_CASES if case[0] in wanted_cases]
        if not selected_cases:
            raise SystemExit(
                "No valid cases selected.\n"
                "Choose from: echo, summarize, code, reasoning"
            )

    print(f"run_id={run_id}")
    print(f"model={args.model}")
    print(f"paths={sorted(wanted_paths)}")
    print(f"runs={args.runs}")
    print(f"cases={[name for name, _ in selected_cases]}")

    records: list[ProbeRecord] = []
    with DefaultAzureCredential() as credential:
        with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client:
            aoai_client = _make_aoai_client(cfg, credential)
            project_openai_client = _make_project_client(project_client)

            path_defs: list[tuple[str, str, OpenAI, Any]] = [
                ("aoai", "chat", aoai_client, _run_chat),
                ("project", "responses", project_openai_client, _run_responses),
            ]

            for endpoint, api_style, client, fn in path_defs:
                path_key = f"{endpoint}_{api_style}"
                if path_key not in wanted_paths:
                    continue
                for case_name, prompt in selected_cases:
                    for run_index in range(1, args.runs + 1):
                        ok, data = fn(client, args.model, prompt)
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
                            )
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
            "paths": sorted(wanted_paths),
            "prompt_cases": [name for name, _ in selected_cases],
        },
        "records": [asdict(r) for r in records],
        "summary": summary_rows,
    }
    md_path, json_path = _write_outputs(run_id, payload)
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
