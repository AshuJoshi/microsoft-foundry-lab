import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI, AzureOpenAI

from config import load_config


@dataclass
class ProbeResult:
    client_type: str
    endpoint_type: str
    endpoint_url: str
    auth_scope: str
    api_style: str
    api_version: str | None
    model: str
    success: bool
    status_code: int | None
    latency_ms: int
    output_preview: str | None
    error_type: str | None
    error_code: str | None
    error_message: str | None
    request_headers: dict[str, str]
    response_headers: dict[str, str]


class HeaderRecorder:
    def __init__(self) -> None:
        self.last_request_headers: dict[str, str] = {}
        self.last_response_headers: dict[str, str] = {}
        self.last_status_code: int | None = None
        self.last_url: str | None = None

    def on_request(self, request: httpx.Request) -> None:
        self.last_request_headers = {k.lower(): v for k, v in request.headers.items()}
        self.last_url = str(request.url)

    def on_response(self, response: httpx.Response) -> None:
        self.last_response_headers = {k.lower(): v for k, v in response.headers.items()}
        self.last_status_code = response.status_code

    def clear(self) -> None:
        self.last_request_headers = {}
        self.last_response_headers = {}
        self.last_status_code = None
        self.last_url = None


def _error_fields(exc: Exception) -> tuple[str, str | None, str, int | None, dict[str, str]]:
    err_type = type(exc).__name__
    err_code = None
    msg = str(exc)
    status = None
    resp_headers: dict[str, str] = {}

    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        try:
            resp_headers = {k.lower(): v for k, v in response.headers.items()}
        except Exception:  # noqa: BLE001
            resp_headers = {}

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(err, dict):
            if err.get("code") is not None:
                err_code = str(err.get("code"))
            if err.get("message"):
                msg = str(err.get("message"))

    return err_type, err_code, msg, status, resp_headers


def _call_responses(client: Any, model: str, prompt: str) -> tuple[bool, str | None, Exception | None]:
    try:
        r = client.responses.create(model=model, input=prompt, max_output_tokens=30)
        return True, (r.output_text or "")[:160], None
    except Exception as exc:  # noqa: BLE001
        return False, None, exc


def _call_chat(client: Any, model: str, prompt: str) -> tuple[bool, str | None, Exception | None]:
    attempts = [
        {"max_tokens": 30, "temperature": 0},
        {"max_completion_tokens": 30, "temperature": 0},
    ]
    last_exc: Exception | None = None
    for p in attempts:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **p,
            )
            txt = r.choices[0].message.content if r.choices else ""
            return True, (txt or "")[:160], None
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    return False, None, last_exc


def _make_http_client(rec: HeaderRecorder) -> httpx.Client:
    return httpx.Client(event_hooks={"request": [rec.on_request], "response": [rec.on_response]})


def _attach_hooks_to_openai_client(openai_client: Any, rec: HeaderRecorder) -> None:
    # openai.OpenAI keeps an internal SyncHttpxClientWrapper at `_client` which
    # exposes event_hooks compatible with httpx.
    inner = getattr(openai_client, "_client", None)
    if inner is not None and hasattr(inner, "event_hooks"):
        inner.event_hooks = {"request": [rec.on_request], "response": [rec.on_response]}


def _build_openai_client(url: str, scope: str, api_version: str | None, cred: DefaultAzureCredential, rec: HeaderRecorder) -> OpenAI:
    provider = get_bearer_token_provider(cred, scope)
    kwargs: dict[str, Any] = {
        "base_url": url,
        "api_key": provider,
        "http_client": _make_http_client(rec),
    }
    if api_version:
        kwargs["default_query"] = {"api-version": api_version}
    return OpenAI(**kwargs)


def _build_azureopenai_client(url: str, scope: str, api_version: str | None, cred: DefaultAzureCredential, rec: HeaderRecorder) -> AzureOpenAI:
    provider = get_bearer_token_provider(cred, scope)
    if api_version is None:
        api_version = "2025-05-01-preview"
    return AzureOpenAI(
        base_url=url,
        api_version=api_version,
        azure_ad_token_provider=provider,
        http_client=_make_http_client(rec),
    )


def _build_foundry_bridge(cfg_endpoint: str, cred: DefaultAzureCredential, rec: HeaderRecorder) -> tuple[AIProjectClient, OpenAI]:
    project = AIProjectClient(endpoint=cfg_endpoint, credential=cred)
    client = project.get_openai_client()
    _attach_hooks_to_openai_client(client, rec)
    return project, client


def main() -> None:
    cfg = load_config()
    credential = DefaultAzureCredential()
    prompt = "Reply with exactly: matrix-ok"

    with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project:
        models = [d.name for d in project.deployments.list()]

    endpoint_cases = [
        {
            "endpoint_type": "foundry_project",
            "url": cfg.project_endpoint.rstrip("/") + "/openai",
            "scope": "https://ai.azure.com/.default",
            "api_versions": ["2025-11-15-preview"],
        },
        {
            "endpoint_type": "azure_openai_resource_v1",
            "url": f"https://{cfg.resource_name}.openai.azure.com/openai/v1/",
            "scope": "https://cognitiveservices.azure.com/.default",
            "api_versions": [None],
        },
    ]

    client_types = ["foundry_sdk_bridge", "openai", "azureopenai"]
    api_styles: list[tuple[str, Callable[[Any, str, str], tuple[bool, str | None, Exception | None]]]] = [
        ("responses", _call_responses),
        ("chat.completions", _call_chat),
    ]

    results: list[ProbeResult] = []

    for ep in endpoint_cases:
        for api_version in ep["api_versions"]:
            for client_type in client_types:
                if client_type == "foundry_sdk_bridge" and ep["endpoint_type"] != "foundry_project":
                    continue

                rec = HeaderRecorder()
                project_obj: AIProjectClient | None = None
                client: Any

                try:
                    if client_type == "foundry_sdk_bridge":
                        project_obj, client = _build_foundry_bridge(cfg.project_endpoint, credential, rec)
                    elif client_type == "openai":
                        client = _build_openai_client(ep["url"], ep["scope"], api_version, credential, rec)
                    else:
                        client = _build_azureopenai_client(ep["url"], ep["scope"], api_version, credential, rec)

                    for model in models:
                        for api_style, fn in api_styles:
                            rec.clear()
                            t0 = time.perf_counter()
                            ok, output, exc = fn(client, model, prompt)
                            latency = int((time.perf_counter() - t0) * 1000)

                            if ok:
                                results.append(
                                    ProbeResult(
                                        client_type=client_type,
                                        endpoint_type=ep["endpoint_type"],
                                        endpoint_url=ep["url"],
                                        auth_scope=ep["scope"],
                                        api_style=api_style,
                                        api_version=api_version,
                                        model=model,
                                        success=True,
                                        status_code=rec.last_status_code,
                                        latency_ms=latency,
                                        output_preview=output,
                                        error_type=None,
                                        error_code=None,
                                        error_message=None,
                                        request_headers=rec.last_request_headers,
                                        response_headers=rec.last_response_headers,
                                    )
                                )
                            else:
                                assert exc is not None
                                et, ec, em, sc, err_headers = _error_fields(exc)
                                results.append(
                                    ProbeResult(
                                        client_type=client_type,
                                        endpoint_type=ep["endpoint_type"],
                                        endpoint_url=ep["url"],
                                        auth_scope=ep["scope"],
                                        api_style=api_style,
                                        api_version=api_version,
                                        model=model,
                                        success=False,
                                        status_code=rec.last_status_code or sc,
                                        latency_ms=latency,
                                        output_preview=None,
                                        error_type=et,
                                        error_code=ec,
                                        error_message=em,
                                        request_headers=rec.last_request_headers,
                                        response_headers=rec.last_response_headers or err_headers,
                                    )
                                )
                finally:
                    try:
                        if project_obj is not None:
                            project_obj.close()
                    except Exception:  # noqa: BLE001
                        pass

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"compatibility_matrix_clients_headers_{now}.json"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")

    md_path = out_dir / f"compatibility_matrix_clients_headers_{now}.md"
    lines = []
    lines.append(f"# Client + Endpoint + API Compatibility Matrix ({now})")
    lines.append("")
    lines.append("| Client | Endpoint | API Version | API Style | Model | Success | Status | Latency (ms) | Detail |")
    lines.append("|---|---|---|---|---|---|---:|---:|---|")
    for r in results:
        detail = r.output_preview if r.success else f"{r.error_type}: {r.error_message or ''}"
        detail = (detail or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {r.client_type} | {r.endpoint_type} | {r.api_version or '-'} | {r.api_style} | {r.model} | {'yes' if r.success else 'no'} | {r.status_code or '-'} | {r.latency_ms} | {detail[:160]} |"
        )

    lines.append("")
    lines.append("## Header Presence Summary")
    lines.append("")

    header_keys = [
        "apim-request-id",
        "x-ms-region",
        "openai-project",
        "openai-processing-ms",
        "x-ratelimit-limit-tokens",
        "x-request-id",
        "api-supported-versions",
    ]

    lines.append("| Client | Endpoint | API Style | Header | Present Count | Total |")
    lines.append("|---|---|---|---|---:|---:|")

    grouped: dict[tuple[str, str, str], list[ProbeResult]] = {}
    for r in results:
        key = (r.client_type, r.endpoint_type, r.api_style)
        grouped.setdefault(key, []).append(r)

    for (client_type, endpoint_type, api_style), rows in sorted(grouped.items()):
        total = len(rows)
        for h in header_keys:
            present = sum(1 for row in rows if h in row.response_headers)
            lines.append(f"| {client_type} | {endpoint_type} | {api_style} | {h} | {present} | {total} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    latest_json = out_dir / "compatibility_matrix_clients_headers_latest.json"
    latest_md = out_dir / "compatibility_matrix_clients_headers_latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(str(md_path))
    print(str(json_path))


if __name__ == "__main__":
    main()
