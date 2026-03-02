import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config


NON_OPENAI_MODELS = [
    "DeepSeek-V3.2",
    "grok-4",
    "Kimi-K2.5",
    "Mistral-Large-3",
]

SAMPLES = 6
PROMPT = "Reply with exactly: latency-ok"


@dataclass
class Sample:
    endpoint: str
    api_style: str
    model: str
    idx: int
    success: bool
    latency_ms: int
    error_type: str | None
    error_message: str | None


@dataclass
class Summary:
    endpoint: str
    api_style: str
    model: str
    runs: int
    successes: int
    failures: int
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None


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

    foundry_client = OpenAI(
        base_url=cfg.project_endpoint.rstrip("/") + "/openai",
        api_key=get_bearer_token_provider(cred, "https://ai.azure.com/.default"),
        default_query={"api-version": "2025-11-15-preview"},
    )

    aoai_client = OpenAI(
        base_url=f"https://{cfg.resource_name}.openai.azure.com/openai/v1/",
        api_key=get_bearer_token_provider(cred, "https://cognitiveservices.azure.com/.default"),
    )

    samples: list[Sample] = []

    for model in NON_OPENAI_MODELS:
        for i in range(1, SAMPLES + 1):
            ok, lat, et, em = _call_responses(foundry_client, model)
            samples.append(Sample("foundry_project", "responses", model, i, ok, lat, et, em))

    for model in NON_OPENAI_MODELS:
        for i in range(1, SAMPLES + 1):
            ok, lat, et, em = _call_chat(aoai_client, model)
            samples.append(Sample("azure_openai_resource_v1", "chat.completions", model, i, ok, lat, et, em))

    summaries: list[Summary] = []
    for endpoint, api_style in [
        ("foundry_project", "responses"),
        ("azure_openai_resource_v1", "chat.completions"),
    ]:
        for model in NON_OPENAI_MODELS:
            rows = [s for s in samples if s.endpoint == endpoint and s.api_style == api_style and s.model == model]
            ok_vals = [s.latency_ms for s in rows if s.success]
            summaries.append(
                Summary(
                    endpoint=endpoint,
                    api_style=api_style,
                    model=model,
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
    lines.append(f"# Non-OpenAI Latency Benchmark ({now})")
    lines.append("")
    lines.append(f"Samples per combination: {SAMPLES}")
    lines.append("")
    lines.append("| Endpoint | API Style | Model | Runs | Successes | Failures | Mean (ms) | P50 (ms) | P95 (ms) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        mean = f"{s.mean_ms:.1f}" if s.mean_ms is not None else "-"
        p50 = f"{s.p50_ms:.1f}" if s.p50_ms is not None else "-"
        p95 = f"{s.p95_ms:.1f}" if s.p95_ms is not None else "-"
        lines.append(
            f"| {s.endpoint} | {s.api_style} | {s.model} | {s.runs} | {s.successes} | {s.failures} | {mean} | {p50} | {p95} |"
        )

    lines.append("")
    lines.append("## Failure Samples")
    lines.append("")
    lines.append("| Endpoint | API Style | Model | Run | Error Type | Error Message |")
    lines.append("|---|---|---|---:|---|---|")
    for s in samples:
        if not s.success:
            msg = (s.error_message or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {s.endpoint} | {s.api_style} | {s.model} | {s.idx} | {s.error_type or '-'} | {msg[:160]} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out_dir / "latency_non_openai_latest.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "latency_non_openai_latest.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
