import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import filter_inference_deployments, load_config

SAMPLES = 6
PROMPT = "Reply with exactly: latency-ok"
FOUNDRY_API_VERSION = "2025-11-15-preview"


@dataclass
class Sample:
    endpoint: str
    endpoint_url: str
    api_style: str
    api_version: str | None
    model: str
    model_family: str
    idx: int
    success: bool
    latency_ms: int
    error_type: str | None
    error_message: str | None


@dataclass
class Summary:
    endpoint: str
    endpoint_url: str
    api_style: str
    api_version: str | None
    model: str
    model_family: str
    runs: int
    successes: int
    failures: int
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None


def _fmt_latency(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "-"


def _model_family(name: str) -> str:
    n = name.lower()
    if n.startswith("gpt-") or n.startswith("o"):
        return "openai"
    return "non_openai"


def _call_responses(client: OpenAI, model: str) -> tuple[bool, int, str | None, str | None]:
    t0 = time.perf_counter()
    try:
        client.responses.create(model=model, input=PROMPT, max_output_tokens=30)
        dt = int((time.perf_counter() - t0) * 1000)
        return True, dt, None, None
    except Exception as exc:  # noqa: BLE001
        dt = int((time.perf_counter() - t0) * 1000)
        return False, dt, type(exc).__name__, str(exc)


def _call_chat(client: OpenAI, model: str) -> tuple[bool, int, str | None, str | None]:
    t0 = time.perf_counter()
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=30,
            temperature=0,
        )
        dt = int((time.perf_counter() - t0) * 1000)
        return True, dt, None, None
    except Exception as exc:  # noqa: BLE001
        dt = int((time.perf_counter() - t0) * 1000)
        return False, dt, type(exc).__name__, str(exc)


def _percentile(vals: list[int], p: float) -> float:
    if not vals:
        return float("nan")
    vals = sorted(vals)
    k = (len(vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return float(vals[f])
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return float(d0 + d1)


def main() -> None:
    cfg = load_config()
    cred = DefaultAzureCredential()
    with AIProjectClient(endpoint=cfg.project_endpoint, credential=cred) as project:
        deployments = list(project.deployments.list())
    inference_deployments, _ = filter_inference_deployments(deployments)
    all_models = [d.name for d in inference_deployments if getattr(d, "name", None)]
    # Use simple naming heuristic for grouping in report text only.
    openai_models = [m for m in all_models if m.lower().startswith("gpt-") or m.lower().startswith("o")]
    non_openai_models = [m for m in all_models if m not in openai_models]

    foundry_client = OpenAI(
        base_url=cfg.project_endpoint.rstrip("/") + "/openai",
        api_key=get_bearer_token_provider(cred, "https://ai.azure.com/.default"),
        default_query={"api-version": FOUNDRY_API_VERSION},
    )

    aoai_client = OpenAI(
        base_url=f"https://{cfg.resource_name}.openai.azure.com/openai/v1/",
        api_key=get_bearer_token_provider(cred, "https://cognitiveservices.azure.com/.default"),
    )
    foundry_endpoint_url = cfg.project_endpoint.rstrip("/") + "/openai"
    aoai_endpoint_url = f"https://{cfg.resource_name}.openai.azure.com/openai/v1/"

    samples: list[Sample] = []

    # Model-family-specific comparison:
    # - OpenAI models: Foundry responses vs AOAI responses
    # - Non-OpenAI models: Foundry responses vs AOAI chat.completions
    for model in all_models:
        family = _model_family(model)
        for i in range(1, SAMPLES + 1):
            ok, lat, et, em = _call_responses(foundry_client, model)
            samples.append(
                Sample(
                    "foundry_project",
                    foundry_endpoint_url,
                    "responses",
                    FOUNDRY_API_VERSION,
                    model,
                    family,
                    i,
                    ok,
                    lat,
                    et,
                    em,
                )
            )

        for i in range(1, SAMPLES + 1):
            if family == "openai":
                ok, lat, et, em = _call_responses(aoai_client, model)
                api_style = "responses"
            else:
                ok, lat, et, em = _call_chat(aoai_client, model)
                api_style = "chat.completions"
            samples.append(
                Sample(
                    "azure_openai_resource_v1",
                    aoai_endpoint_url,
                    api_style,
                    "v1",
                    model,
                    family,
                    i,
                    ok,
                    lat,
                    et,
                    em,
                )
            )

    summaries: list[Summary] = []
    keys = sorted({(s.endpoint, s.endpoint_url, s.api_style, s.api_version, s.model, s.model_family) for s in samples})
    for endpoint, endpoint_url, api_style, api_version, model, family in keys:
        rows = [
            s
            for s in samples
            if s.endpoint == endpoint
            and s.endpoint_url == endpoint_url
            and s.api_style == api_style
            and s.api_version == api_version
            and s.model == model
        ]
        ok_vals = [s.latency_ms for s in rows if s.success]
        summaries.append(
            Summary(
                endpoint=endpoint,
                endpoint_url=endpoint_url,
                api_style=api_style,
                api_version=api_version,
                model=model,
                model_family=family,
                runs=len(rows),
                successes=len(ok_vals),
                failures=len(rows) - len(ok_vals),
                mean_ms=(statistics.mean(ok_vals) if ok_vals else None),
                p50_ms=(_percentile(ok_vals, 0.5) if ok_vals else None),
                p95_ms=(_percentile(ok_vals, 0.95) if ok_vals else None),
            )
        )

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"latency_non_openai_{now}.json"
    md_path = out_dir / f"latency_non_openai_{now}.md"

    json_path.write_text(
        json.dumps({"samples": [asdict(s) for s in samples], "summaries": [asdict(s) for s in summaries]}, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append(f"# Latency Benchmark ({now})")
    lines.append("")
    lines.append(f"Samples per combination: {SAMPLES}")
    lines.append(f"All inference models discovered: {', '.join(all_models)}")
    lines.append(f"Heuristic OpenAI group: {', '.join(openai_models) if openai_models else '-'}")
    lines.append(f"Heuristic Non-OpenAI group: {', '.join(non_openai_models) if non_openai_models else '-'}")
    lines.append("")
    lines.append("## Endpoints")
    lines.append("")
    lines.append(f"- foundry_project base_url: `{foundry_endpoint_url}` api_version: `{FOUNDRY_API_VERSION}`")
    lines.append(f"- azure_openai_resource_v1 base_url: `{aoai_endpoint_url}` api_version: `v1`")
    lines.append("")
    lines.append("## Model Pivot (Side-by-Side)")
    lines.append("")
    lines.append("| Model | Family | Foundry Responses (ok/runs) | Foundry P50 | Foundry P95 | AOAI Path | AOAI (ok/runs) | AOAI P50 | AOAI P95 |")
    lines.append("|---|---|---|---:|---:|---|---|---:|---:|")

    summary_map: dict[tuple[str, str, str], Summary] = {
        (s.endpoint, s.api_style, s.model): s for s in summaries
    }
    model_order = sorted({s.model for s in summaries})

    for model in model_order:
        family = _model_family(model)
        fr = summary_map.get(("foundry_project", "responses", model))
        aoai_style = "responses" if family == "openai" else "chat.completions"
        ao = summary_map.get(("azure_openai_resource_v1", aoai_style, model))

        fr_ok = f"{fr.successes}/{fr.runs}" if fr else "-"
        ao_ok = f"{ao.successes}/{ao.runs}" if ao else "-"

        lines.append(
            "| "
            + f"{model} | "
            + f"{family} | "
            + f"{fr_ok} | {_fmt_latency(fr.p50_ms if fr else None)} | {_fmt_latency(fr.p95_ms if fr else None)} | "
            + f"{aoai_style} | {ao_ok} | {_fmt_latency(ao.p50_ms if ao else None)} | {_fmt_latency(ao.p95_ms if ao else None)} |"
        )

    lines.append("")
    lines.append("| Endpoint | Base URL | API Version | API Style | Model | Family | Runs | Successes | Failures | Mean (ms) | P50 (ms) | P95 (ms) |")
    lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        mean = _fmt_latency(s.mean_ms)
        p50 = _fmt_latency(s.p50_ms)
        p95 = _fmt_latency(s.p95_ms)
        lines.append(
            f"| {s.endpoint} | {s.endpoint_url} | {s.api_version or '-'} | {s.api_style} | {s.model} | {s.model_family} | {s.runs} | {s.successes} | {s.failures} | {mean} | {p50} | {p95} |"
        )

    lines.append("")
    lines.append("## Failure Samples")
    lines.append("")
    lines.append("| Endpoint | API Version | API Style | Model | Family | Run | Error Type | Error Message |")
    lines.append("|---|---|---|---|---|---:|---|---|")
    for s in samples:
        if not s.success:
            msg = (s.error_message or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {s.endpoint} | {s.api_version or '-'} | {s.api_style} | {s.model} | {s.model_family} | {s.idx} | {s.error_type or '-'} | {msg[:160]} |"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out_dir / "latency_non_openai_latest.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "latency_non_openai_latest.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
