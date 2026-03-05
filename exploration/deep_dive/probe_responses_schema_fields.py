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
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config


@dataclass
class ProbeRecord:
    run_id: str
    endpoint_label: str
    endpoint_url: str
    auth_scope: str
    model: str
    case_name: str
    success: bool
    status_code: int | None
    latency_ms: int
    error_type: str | None
    error_code: str | None
    error_param: str | None
    error_message: str | None
    apim_request_id: str | None
    x_request_id: str | None
    x_ms_region: str | None
    request_url: str | None
    skipped: bool
    skip_reason: str | None
    payload: dict[str, Any]


class HeaderRecorder:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.request_url: str | None = None
        self.response_headers: dict[str, str] = {}
        self.status_code: int | None = None

    def on_request(self, request: httpx.Request) -> None:
        self.request_url = str(request.url)

    def on_response(self, response: httpx.Response) -> None:
        self.response_headers = {k.lower(): v for k, v in response.headers.items()}
        self.status_code = response.status_code


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Differential probe for Responses API field acceptance.")
    p.add_argument("--model", default=cfg.default_model_deployment_name, help="Deployment/model to use for probes.")
    p.add_argument(
        "--models",
        default="",
        help="Comma-separated model/deployment names to test. Overrides --model when set.",
    )
    p.add_argument(
        "--all-models",
        action="store_true",
        help="Discover inference deployments from project and test all of them.",
    )
    p.add_argument(
        "--endpoint-mode",
        "--endpoint",
        dest="endpoint_mode",
        choices=["all", "project_bridge", "resource_openai_v1", "resource_services_v1"],
        default="all",
        help="Which endpoint families to test.",
    )
    p.add_argument("--prompt", default="Reply with exactly: schema-probe-ok", help="Prompt input for all cases.")
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "output"),
        help="Output directory for JSON/Markdown reports.",
    )
    p.add_argument(
        "--skip-direct-responses-unsupported",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip direct openai/v1 endpoint probes for models whose deployment metadata indicates "
            "Responses API is unsupported."
        ),
    )
    return p.parse_args()


def _error_fields(exc: Exception) -> tuple[str, str | None, str | None, str]:
    et = type(exc).__name__
    ec = None
    ep = None
    msg = str(exc)

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(err, dict):
            if err.get("code") is not None:
                ec = str(err.get("code"))
            if err.get("param") is not None:
                ep = str(err.get("param"))
            if err.get("message"):
                msg = str(err.get("message"))
    return et, ec, ep, msg


def _attach_hooks_to_openai_client(openai_client: Any, recorder: HeaderRecorder) -> None:
    inner = getattr(openai_client, "_client", None)
    if inner is not None and hasattr(inner, "event_hooks"):
        inner.event_hooks = {"request": [recorder.on_request], "response": [recorder.on_response]}


def _mk_cases(prompt: str) -> list[tuple[str, dict[str, Any]]]:
    # Keep payloads small and focused so failures isolate schema acceptance, not task complexity.
    return [
        ("baseline", {"input": prompt, "max_output_tokens": 30}),
        ("temperature", {"input": prompt, "max_output_tokens": 30, "temperature": 0}),
        ("top_p", {"input": prompt, "max_output_tokens": 30, "top_p": 1}),
        ("metadata", {"input": prompt, "max_output_tokens": 30, "metadata": {"probe": "responses-schema"}}),
        ("reasoning_effort_none", {"input": prompt, "max_output_tokens": 30, "reasoning": {"effort": "none"}}),
        ("reasoning_effort_low", {"input": prompt, "max_output_tokens": 30, "reasoning": {"effort": "low"}}),
        ("text_format_text", {"input": prompt, "max_output_tokens": 30, "text": {"format": {"type": "text"}}}),
        ("text_verbosity_low", {"input": prompt, "max_output_tokens": 30, "text": {"verbosity": "low"}}),
        ("truncation_disabled", {"input": prompt, "max_output_tokens": 30, "truncation": "disabled"}),
        ("store_false", {"input": prompt, "max_output_tokens": 30, "store": False}),
    ]


def _looks_like_responses_supported(deployment: Any) -> bool | None:
    data: dict[str, Any] = {}
    if hasattr(deployment, "as_dict") and callable(getattr(deployment, "as_dict")):
        try:
            data = deployment.as_dict()
        except Exception:  # noqa: BLE001
            data = {}
    blob = json.dumps(data, ensure_ascii=False).lower() if data else ""

    # Strong negative hints first.
    if blob and "chat completions" in blob and "responses" not in blob:
        return False

    if blob:
        positive_hints = [
            "responses",
            "response api",
            "/responses",
        ]
        for hint in positive_hints:
            if hint in blob:
                return True

    return None


def _resolve_models(cfg: Any, args: argparse.Namespace) -> tuple[list[str], dict[str, bool | None]]:
    explicit_models: list[str] = []
    if args.models.strip():
        explicit_models = [m.strip() for m in args.models.split(",") if m.strip()]

    model_cap_hints: dict[str, bool | None] = {}
    discovered_models: list[str] = []

    if args.all_models or not explicit_models:
        with DefaultAzureCredential() as credential:
            with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client:
                deployments = list(project_client.deployments.list())
        for d in deployments:
            name = getattr(d, "name", None)
            if not name:
                continue
            # Exclude embeddings for this probe surface.
            if "embedding" in name.lower():
                continue
            discovered_models.append(name)
            model_cap_hints[name] = _looks_like_responses_supported(d)

    if args.all_models:
        models = discovered_models
    elif explicit_models:
        models = explicit_models
    else:
        models = [args.model]

    # Deduplicate while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for m in models:
        if m not in seen:
            deduped.append(m)
            seen.add(m)

    for m in deduped:
        model_cap_hints.setdefault(m, None)

    return deduped, model_cap_hints


def _run_case(
    run_id: str,
    endpoint_label: str,
    endpoint_url: str,
    auth_scope: str,
    client: OpenAI,
    recorder: HeaderRecorder,
    model: str,
    case_name: str,
    payload: dict[str, Any],
) -> ProbeRecord:
    recorder.clear()
    body = {"model": model, **payload}
    t0 = time.perf_counter()
    status = None
    err_type = None
    err_code = None
    err_param = None
    err_msg = None
    success = False

    try:
        client.responses.create(**body)
        success = True
    except Exception as exc:  # noqa: BLE001
        err_type, err_code, err_param, err_msg = _error_fields(exc)
        resp = getattr(exc, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)
            try:
                recorder.response_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
            except Exception:  # noqa: BLE001
                pass

    latency = int((time.perf_counter() - t0) * 1000)
    if status is None:
        status = recorder.status_code

    h = recorder.response_headers
    return ProbeRecord(
        run_id=run_id,
        endpoint_label=endpoint_label,
        endpoint_url=endpoint_url,
        auth_scope=auth_scope,
        model=model,
        case_name=case_name,
        success=success,
        status_code=status,
        latency_ms=latency,
        error_type=err_type,
        error_code=err_code,
        error_param=err_param,
        error_message=err_msg,
        apim_request_id=h.get("apim-request-id"),
        x_request_id=h.get("x-request-id"),
        x_ms_region=h.get("x-ms-region"),
        request_url=recorder.request_url,
        skipped=False,
        skip_reason=None,
        payload=body,
    )


def _skipped_case(
    run_id: str,
    endpoint_label: str,
    endpoint_url: str,
    auth_scope: str,
    model: str,
    case_name: str,
    payload: dict[str, Any],
    reason: str,
) -> ProbeRecord:
    body = {"model": model, **payload}
    return ProbeRecord(
        run_id=run_id,
        endpoint_label=endpoint_label,
        endpoint_url=endpoint_url,
        auth_scope=auth_scope,
        model=model,
        case_name=case_name,
        success=False,
        status_code=None,
        latency_ms=0,
        error_type=None,
        error_code=None,
        error_param=None,
        error_message=None,
        apim_request_id=None,
        x_request_id=None,
        x_ms_region=None,
        request_url=None,
        skipped=True,
        skip_reason=reason,
        payload=body,
    )


def _write_reports(run_id: str, records: list[ProbeRecord], out_dir: Path, metadata: dict[str, Any]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"probe_responses_schema_fields_{run_id}.json"
    md_path = out_dir / f"probe_responses_schema_fields_{run_id}.md"

    json_path.write_text(
        json.dumps({"metadata": metadata, "records": [asdict(r) for r in records]}, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append(f"# Responses Schema Probe ({run_id})")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    for k, v in metadata.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("| Model | Endpoint | Case | Success | Status | Error Type | Error Param | Latency (ms) | Skipped | apim-request-id |")
    lines.append("|---|---|---|---|---:|---|---|---:|---|---|")
    for r in records:
        lines.append(
            f"| {r.model} | {r.endpoint_label} | {r.case_name} | {'yes' if r.success else 'no'} | {r.status_code or '-'} | {r.error_type or '-'} | {r.error_param or '-'} | {r.latency_ms} | {'yes' if r.skipped else 'no'} | {r.apim_request_id or '-'} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def _print_progress(r: ProbeRecord) -> None:
    if r.skipped:
        print(f"SKIP  model={r.model} endpoint={r.endpoint_label} case={r.case_name} reason={r.skip_reason}")
        return
    status = r.status_code if r.status_code is not None else "-"
    if r.success:
        print(f"PASS  model={r.model} endpoint={r.endpoint_label} case={r.case_name} status={status} latency_ms={r.latency_ms}")
    else:
        print(
            "FAIL  "
            f"model={r.model} endpoint={r.endpoint_label} case={r.case_name} "
            f"status={status} error_type={r.error_type or '-'} error_param={r.error_param or '-'} latency_ms={r.latency_ms}"
        )


def _print_summary(records: list[ProbeRecord]) -> None:
    key_stats: dict[tuple[str, str], dict[str, int]] = {}
    for r in records:
        key = (r.model, r.endpoint_label)
        s = key_stats.setdefault(key, {"pass": 0, "fail": 0, "skip": 0})
        if r.skipped:
            s["skip"] += 1
        elif r.success:
            s["pass"] += 1
        else:
            s["fail"] += 1

    print("")
    print("Summary by model+endpoint:")
    for (model, endpoint), s in sorted(key_stats.items()):
        print(f"  {model:28} {endpoint:22} pass={s['pass']:2d} fail={s['fail']:2d} skip={s['skip']:2d}")


def main() -> None:
    args = parse_args()
    cfg = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cases = _mk_cases(args.prompt)
    models, model_cap_hints = _resolve_models(cfg, args)
    records: list[ProbeRecord] = []

    endpoint_cases = []
    if args.endpoint_mode in {"all", "project_bridge"}:
        endpoint_cases.append(
            {
                "label": "project_bridge",
                "url": cfg.project_endpoint.rstrip("/") + "/openai/v1/",
                "scope": "https://ai.azure.com/.default",
                "kind": "project_bridge",
            }
        )
    if args.endpoint_mode in {"all", "resource_openai_v1"}:
        endpoint_cases.append(
            {
                "label": "resource_openai_v1",
                "url": f"https://{cfg.resource_name}.openai.azure.com/openai/v1/",
                "scope": "https://cognitiveservices.azure.com/.default",
                "kind": "direct_openai",
            }
        )
    if args.endpoint_mode in {"all", "resource_services_v1"}:
        endpoint_cases.append(
            {
                "label": "resource_services_v1",
                "url": f"https://{cfg.resource_name}.services.ai.azure.com/openai/v1/",
                "scope": "https://cognitiveservices.azure.com/.default",
                "kind": "direct_openai",
            }
        )

    with DefaultAzureCredential() as credential:
        for ep in endpoint_cases:
            recorder = HeaderRecorder()
            if ep["kind"] == "project_bridge":
                with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client:
                    with project_client.get_openai_client() as client:
                        _attach_hooks_to_openai_client(client, recorder)
                        for model in models:
                            for case_name, payload in cases:
                                rec = _run_case(
                                    run_id=run_id,
                                    endpoint_label=ep["label"],
                                    endpoint_url=ep["url"],
                                    auth_scope=ep["scope"],
                                    client=client,
                                    recorder=recorder,
                                    model=model,
                                    case_name=case_name,
                                    payload=payload,
                                )
                                records.append(rec)
                                _print_progress(rec)
            else:
                provider = get_bearer_token_provider(credential, ep["scope"])
                with OpenAI(
                    base_url=ep["url"],
                    api_key=provider,
                    http_client=httpx.Client(event_hooks={"request": [recorder.on_request], "response": [recorder.on_response]}),
                ) as client:
                    for model in models:
                        if args.skip_direct_responses_unsupported and model_cap_hints.get(model) is False:
                            reason = "deployment metadata indicates chat-completions-only support"
                            for case_name, payload in cases:
                                rec = _skipped_case(
                                    run_id=run_id,
                                    endpoint_label=ep["label"],
                                    endpoint_url=ep["url"],
                                    auth_scope=ep["scope"],
                                    model=model,
                                    case_name=case_name,
                                    payload=payload,
                                    reason=reason,
                                )
                                records.append(rec)
                                _print_progress(rec)
                            continue

                        for case_name, payload in cases:
                            rec = _run_case(
                                run_id=run_id,
                                endpoint_label=ep["label"],
                                endpoint_url=ep["url"],
                                auth_scope=ep["scope"],
                                client=client,
                                recorder=recorder,
                                model=model,
                                case_name=case_name,
                                payload=payload,
                            )
                            records.append(rec)
                            _print_progress(rec)

    out_dir = Path(args.out_dir)
    metadata = {
        "run_id": run_id,
        "models": ", ".join(models),
        "project_endpoint": cfg.project_endpoint,
        "endpoint_mode": args.endpoint_mode,
        "skip_direct_responses_unsupported": args.skip_direct_responses_unsupported,
        "cases": ", ".join([name for name, _ in cases]),
        "note": "Exploration-only differential schema acceptance probe for responses.create",
    }
    md_path, json_path = _write_reports(run_id, records, out_dir, metadata)
    _print_summary(records)
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
