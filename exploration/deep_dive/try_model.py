#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Try an OpenAI or non-OpenAI deployment with a sensible default API path.")
    p.add_argument("--model", required=True, help="Deployment name, e.g. gpt-4.1 or Mistral-Large-3")
    p.add_argument("--prompt", default="Reply with exactly: try-model-ok", help="Prompt to send.")
    p.add_argument(
        "--mode",
        choices=["auto", "responses", "chat"],
        default="auto",
        help="API mode. auto = responses for OpenAI family, chat for non-OpenAI family.",
    )
    p.add_argument(
        "--endpoint",
        choices=["auto", "project", "aoai"],
        default="auto",
        help="Endpoint family. auto = project for responses, aoai for chat.",
    )
    return p.parse_args()


def _model_family(model: str) -> str:
    m = model.lower()
    if m.startswith("gpt-") or m.startswith("o"):
        return "openai"
    return "non_openai"


def _make_project_openai_client(project_client: AIProjectClient) -> OpenAI:
    return project_client.get_openai_client()


def _make_aoai_client(cfg: Any, credential: Any) -> OpenAI:
    return OpenAI(
        base_url=f"https://{cfg.resource_name}.openai.azure.com/openai/v1/",
        api_key=get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default"),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config()
    family = _model_family(args.model)

    if args.mode == "auto":
        api_mode = "responses" if family == "openai" else "chat"
    else:
        api_mode = args.mode

    if args.endpoint == "auto":
        endpoint_mode = "project" if api_mode == "responses" else "aoai"
    else:
        endpoint_mode = args.endpoint

    print(f"model={args.model}")
    print(f"family={family}")
    print(f"api_mode={api_mode}")
    print(f"endpoint_mode={endpoint_mode}")

    with DefaultAzureCredential() as credential:
        with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client:
            client = _make_project_openai_client(project_client) if endpoint_mode == "project" else _make_aoai_client(cfg, credential)

            if api_mode == "responses":
                response = client.responses.create(
                    model=args.model,
                    input=args.prompt,
                    max_output_tokens=200,
                )
                payload = {
                    "id": response.id,
                    "model": response.model,
                    "endpoint_mode": endpoint_mode,
                    "api_mode": api_mode,
                    "output_text": response.output_text,
                    "usage": response.usage.model_dump() if getattr(response, "usage", None) else None,
                }
            else:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": args.prompt}],
                    max_tokens=200,
                    temperature=0,
                )
                payload = {
                    "id": response.id,
                    "model": response.model,
                    "endpoint_mode": endpoint_mode,
                    "api_mode": api_mode,
                    "message": response.choices[0].message.content if response.choices else None,
                    "usage": response.usage.model_dump() if getattr(response, "usage", None) else None,
                }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
