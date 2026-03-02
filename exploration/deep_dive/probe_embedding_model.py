#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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


class HeaderRecorder:
    def __init__(self) -> None:
        self.request_headers: dict[str, str] = {}
        self.response_headers: dict[str, str] = {}
        self.request_url: str | None = None
        self.request_method: str | None = None
        self.status_code: int | None = None

    def on_request(self, request: httpx.Request) -> None:
        self.request_headers = {k.lower(): v for k, v in request.headers.items()}
        self.request_url = str(request.url)
        self.request_method = request.method

    def on_response(self, response: httpx.Response) -> None:
        self.response_headers = {k.lower(): v for k, v in response.headers.items()}
        self.status_code = response.status_code


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe embedding deployment with a single embeddings.create() call.")
    p.add_argument(
        "--model",
        default=(
            os.getenv("MEMORY_EMBEDDING_MODEL_DEPLOYMENT_NAME")
            or os.getenv("AZURE_AI_EMBEDDING_MODEL_DEPLOYMENT_NAME")
            or "text-embedding-3-small"
        ),
        help="Embedding deployment name in Foundry project.",
    )
    p.add_argument(
        "--input",
        default="Foundry embedding probe text.",
        help="Input text for embeddings call.",
    )
    p.add_argument(
        "--mode",
        default="all",
        choices=["all", "project_bridge", "resource_openai_v1", "resource_services_v1"],
        help="Endpoint/client family to probe. Default: all.",
    )
    return p.parse_args()


def _summarize_error(exc: Exception) -> dict[str, Any]:
    out: dict[str, Any] = {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            out["status_code"] = response.status_code
            out["response_headers"] = {str(k).lower(): str(v) for k, v in response.headers.items()}
            body_text = response.text
            if body_text:
                out["response_text"] = body_text[:2000]
        except Exception:  # noqa: BLE001
            pass
    return out


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        try:
            return _to_jsonable(value.model_dump())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return _to_jsonable(value.to_dict())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "as_dict") and callable(getattr(value, "as_dict")):
        try:
            return _to_jsonable(value.as_dict())
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _run_single_probe(
    *,
    run_id: str,
    mode: str,
    model: str,
    text: str,
    cfg: Any,
) -> dict[str, Any]:
    rec = HeaderRecorder()
    started = time.perf_counter()
    result: dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "project_endpoint": cfg.project_endpoint,
        "model": model,
        "input_preview": text[:120],
        "success": False,
    }
    try:
        with DefaultAzureCredential() as credential:
            if mode == "project_bridge":
                with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client:
                    with project_client.get_openai_client() as openai_client:
                        inner = getattr(openai_client, "_client", None)
                        if inner is not None and hasattr(inner, "event_hooks"):
                            inner.event_hooks = {"request": [rec.on_request], "response": [rec.on_response]}
                        response = openai_client.embeddings.create(model=model, input=text)
            else:
                if mode == "resource_openai_v1":
                    base_url = f"https://{cfg.resource_name}.openai.azure.com/openai/v1/"
                elif mode == "resource_services_v1":
                    base_url = f"https://{cfg.resource_name}.services.ai.azure.com/openai/v1/"
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
                provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
                client = OpenAI(
                    base_url=base_url,
                    api_key=provider,
                    http_client=httpx.Client(
                        event_hooks={"request": [rec.on_request], "response": [rec.on_response]},
                    ),
                )
                response = client.embeddings.create(model=model, input=text)

        latency_ms = int((time.perf_counter() - started) * 1000)
        dims = None
        if getattr(response, "data", None):
            embedding = getattr(response.data[0], "embedding", None)
            if embedding is not None:
                dims = len(embedding)

        result.update(
            {
                "success": True,
                "latency_ms": latency_ms,
                "status_code": rec.status_code,
                "base_url_or_endpoint": rec.request_url.rsplit("/embeddings", 1)[0] if rec.request_url else None,
                "request_method": rec.request_method,
                "request_url": rec.request_url,
                "apim_request_id": rec.response_headers.get("apim-request-id"),
                "x_request_id": rec.response_headers.get("x-request-id"),
                "x_ms_region": rec.response_headers.get("x-ms-region"),
                "response_model": getattr(response, "model", None),
                "embedding_dimensions": dims,
                "usage": _to_jsonable(getattr(response, "usage", None)),
            }
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - started) * 1000)
        result.update(
            {
                "latency_ms": latency_ms,
                "status_code": rec.status_code,
                "base_url_or_endpoint": rec.request_url.rsplit("/embeddings", 1)[0] if rec.request_url else None,
                "request_method": rec.request_method,
                "request_url": rec.request_url,
                "apim_request_id": rec.response_headers.get("apim-request-id"),
                "x_request_id": rec.response_headers.get("x-request-id"),
                "x_ms_region": rec.response_headers.get("x-ms-region"),
                "error": _summarize_error(exc),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    cfg = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.mode == "all":
        modes = ["project_bridge", "resource_openai_v1", "resource_services_v1"]
    else:
        modes = [args.mode]

    results = [
        _run_single_probe(run_id=run_id, mode=mode, model=args.model, text=args.input, cfg=cfg)
        for mode in modes
    ]

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"embedding_probe_{run_id}.json"
    out_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    print(json.dumps({"results": results}, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
