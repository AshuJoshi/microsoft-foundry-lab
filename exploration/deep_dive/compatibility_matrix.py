import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from config import load_config


@dataclass
class CaseResult:
    endpoint_label: str
    endpoint_url: str
    auth_scope: str
    api_style: str
    api_version: str | None
    model: str
    success: bool
    latency_ms: int
    output_preview: str | None
    error_type: str | None
    error_code: str | None
    error_message: str | None


def _err_fields(exc: Exception) -> tuple[str, str | None, str]:
    err_type = type(exc).__name__
    err_code = None
    message = str(exc)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        e = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(e, dict):
            if e.get("code") is not None:
                err_code = str(e.get("code"))
            if e.get("message"):
                message = str(e.get("message"))
    return err_type, err_code, message


def _run_responses(client: OpenAI, model: str, prompt: str) -> tuple[bool, str | None, str | None, str | None, int]:
    t0 = time.perf_counter()
    try:
        resp = client.responses.create(model=model, input=prompt, max_output_tokens=30)
        latency = int((time.perf_counter() - t0) * 1000)
        return True, (resp.output_text or "")[:120], None, None, latency
    except Exception as exc:  # noqa: BLE001
        latency = int((time.perf_counter() - t0) * 1000)
        et, ec, em = _err_fields(exc)
        return False, None, f"{et}", ec, latency if em is None else latency


def _run_chat(client: OpenAI, model: str, prompt: str) -> tuple[bool, str | None, str | None, str | None, int, str | None]:
    t0 = time.perf_counter()
    last_exc: Exception | None = None

    # Try max_tokens first, then max_completion_tokens for models that require it.
    attempts: list[dict[str, Any]] = [
        {"max_tokens": 30, "temperature": 0},
        {"max_completion_tokens": 30, "temperature": 0},
    ]

    for params in attempts:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **params,
            )
            latency = int((time.perf_counter() - t0) * 1000)
            msg = None
            if resp.choices:
                msg = resp.choices[0].message.content
            return True, (msg or "")[:120], None, None, latency, json.dumps(params)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

    latency = int((time.perf_counter() - t0) * 1000)
    assert last_exc is not None
    et, ec, em = _err_fields(last_exc)
    return False, None, et, ec, latency, em


def _mk_client(endpoint_url: str, scope: str, api_version: str | None, credential: DefaultAzureCredential) -> OpenAI:
    token_provider = get_bearer_token_provider(credential, scope)
    kwargs: dict[str, Any] = {
        "base_url": endpoint_url,
        "api_key": token_provider,
    }
    if api_version:
        kwargs["default_query"] = {"api-version": api_version}
    return OpenAI(**kwargs)


def main() -> None:
    cfg = load_config()
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    credential = DefaultAzureCredential()

    with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project:
        models = [d.name for d in project.deployments.list()]

    project_endpoint = cfg.project_endpoint.rstrip("/") + "/openai"
    aoai_endpoint = f"https://{cfg.resource_name}.openai.azure.com/openai/v1/"

    endpoint_cases = [
        {
            "label": "foundry_project",
            "url": project_endpoint,
            "scope": "https://ai.azure.com/.default",
            "api_versions": ["2025-11-15-preview", "2025-05-15-preview"],
        },
        {
            "label": "azure_openai_resource",
            "url": aoai_endpoint,
            "scope": "https://cognitiveservices.azure.com/.default",
            "api_versions": [None],
        },
    ]

    prompt = os.getenv("MATRIX_TEST_PROMPT", "Reply with exactly: matrix-ok")

    results: list[CaseResult] = []

    for ep in endpoint_cases:
        for api_version in ep["api_versions"]:
            client = _mk_client(ep["url"], ep["scope"], api_version, credential)
            for model in models:
                ok, out, et, ec, lat = _run_responses(client, model, prompt)
                err_msg = None
                if not ok:
                    # Re-run to capture message in a consistent way
                    try:
                        client.responses.create(model=model, input=prompt, max_output_tokens=30)
                    except Exception as exc:  # noqa: BLE001
                        _, _, err_msg = _err_fields(exc)
                results.append(
                    CaseResult(
                        endpoint_label=ep["label"],
                        endpoint_url=ep["url"],
                        auth_scope=ep["scope"],
                        api_style="responses",
                        api_version=api_version,
                        model=model,
                        success=ok,
                        latency_ms=lat,
                        output_preview=out,
                        error_type=et,
                        error_code=ec,
                        error_message=err_msg,
                    )
                )

                okc, outc, etc, ecc, latc, extra = _run_chat(client, model, prompt)
                results.append(
                    CaseResult(
                        endpoint_label=ep["label"],
                        endpoint_url=ep["url"],
                        auth_scope=ep["scope"],
                        api_style="chat.completions",
                        api_version=api_version,
                        model=model,
                        success=okc,
                        latency_ms=latc,
                        output_preview=outc,
                        error_type=etc,
                        error_code=ecc,
                        error_message=extra if not okc else None,
                    )
                )

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"compatibility_matrix_{now}.json"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")

    md_path = out_dir / f"compatibility_matrix_{now}.md"
    lines = []
    lines.append(f"# Compatibility Matrix ({now})")
    lines.append("")
    lines.append(f"Models tested: {', '.join(models)}")
    lines.append("")
    lines.append("| Endpoint | API Version | API Style | Model | Success | Latency (ms) | Detail |")
    lines.append("|---|---|---|---|---|---:|---|")
    for r in results:
        detail = r.output_preview if r.success else f"{r.error_type}: {r.error_message or ''}"
        detail = (detail or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {r.endpoint_label} | {r.api_version or '-'} | {r.api_style} | {r.model} | {'yes' if r.success else 'no'} | {r.latency_ms} | {detail[:140]} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    latest_md = out_dir / "compatibility_matrix_latest.md"
    latest_json = out_dir / "compatibility_matrix_latest.json"
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(str(md_path))
    print(str(json_path))


if __name__ == "__main__":
    main()
