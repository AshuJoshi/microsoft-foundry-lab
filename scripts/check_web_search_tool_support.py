#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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

logger = logging.getLogger("check_web_search_tool_support")


class HeaderRecorder:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.last_status_code: int | None = None
        self.last_response_headers: dict[str, str] = {}

    def on_request(self, request: httpx.Request) -> None:
        _ = request

    def on_response(self, response: httpx.Response) -> None:
        self.last_status_code = response.status_code
        self.last_response_headers = {k.lower(): v for k, v in response.headers.items()}


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description='Try direct web_search tool against Azure Responses paths.')
    p.add_argument('--model', default=cfg.default_model_deployment_name)
    p.add_argument('--path', choices=['project_responses', 'aoai_responses'], default='project_responses')
    p.add_argument('--tool-type', choices=['web_search', 'web_search_preview'], default='web_search')
    p.add_argument('--country', default='US')
    p.add_argument('--region', default='WA')
    p.add_argument('--city', default='Seattle')
    p.add_argument('--search-context-size', default='')
    p.add_argument('--prompt', default='Find one recent Microsoft Foundry update and cite the source.')
    p.add_argument('--timeout-seconds', type=float, default=45.0)
    p.add_argument('--max-retries', type=int, default=0)
    p.add_argument('--log-level', default='INFO')
    return p.parse_args()


def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _make_http_client(recorder: HeaderRecorder, timeout_seconds: float) -> httpx.Client:
    return httpx.Client(
        timeout=timeout_seconds,
        event_hooks={'request': [recorder.on_request], 'response': [recorder.on_response]},
    )


def _attach_hooks_to_openai_client(openai_client: Any, recorder: HeaderRecorder) -> None:
    inner = getattr(openai_client, '_client', None)
    if inner is not None and hasattr(inner, 'event_hooks'):
        inner.event_hooks = {'request': [recorder.on_request], 'response': [recorder.on_response]}


def _make_aoai_client(cfg: Any, credential: Any, recorder: HeaderRecorder, timeout_seconds: float, max_retries: int) -> OpenAI:
    return OpenAI(
        base_url=f'https://{cfg.resource_name}.openai.azure.com/openai/v1/',
        api_key=get_bearer_token_provider(credential, 'https://cognitiveservices.azure.com/.default'),
        timeout=timeout_seconds,
        max_retries=max_retries,
        http_client=_make_http_client(recorder, timeout_seconds),
    )


def _make_project_client(project_client: AIProjectClient, recorder: HeaderRecorder, timeout_seconds: float, max_retries: int) -> OpenAI:
    client = project_client.get_openai_client(timeout=timeout_seconds, max_retries=max_retries)
    _attach_hooks_to_openai_client(client, recorder)
    return client


def _usage_dict(obj: Any) -> dict[str, Any] | None:
    usage = getattr(obj, 'usage', None)
    if usage is None:
        return None
    if hasattr(usage, 'model_dump'):
        return usage.model_dump()
    if hasattr(usage, 'as_dict'):
        return usage.as_dict()
    return None


def main() -> None:
    args = parse_args()
    cfg = load_config()
    _setup_logging(args.log_level)

    tool_spec: dict[str, Any] = {
        'type': args.tool_type,
        'user_location': {
            'type': 'approximate',
            'country': args.country,
            'region': args.region,
            'city': args.city,
        },
    }
    if args.search_context_size:
        tool_spec['search_context_size'] = args.search_context_size

    logger.info('model=%s', args.model)
    logger.info('path=%s', args.path)
    logger.info('tool_type=%s', args.tool_type)
    logger.info('tool_spec=%s', tool_spec)

    recorder = HeaderRecorder()
    with DefaultAzureCredential() as credential, AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client:
        if args.path == 'project_responses':
            client = _make_project_client(project_client, recorder, args.timeout_seconds, args.max_retries)
        else:
            client = _make_aoai_client(cfg, credential, recorder, args.timeout_seconds, args.max_retries)

        t0 = time.perf_counter()
        try:
            resp = client.responses.create(
                model=args.model,
                input=args.prompt,
                tools=[tool_spec],
                tool_choice='required',
                max_output_tokens=300,
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            result = {
                'success': True,
                'status_code': recorder.last_status_code,
                'latency_ms': latency_ms,
                'requested_model': args.model,
                'served_model': getattr(resp, 'model', None),
                'output_text': getattr(resp, 'output_text', None),
                'usage': _usage_dict(resp),
                'response_headers': recorder.last_response_headers,
                'output_item_types': [getattr(item, 'type', None) for item in getattr(resp, 'output', []) or []],
            }
            print(json.dumps(result, indent=2, default=str))
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - t0) * 1000)
            response = getattr(exc, 'response', None)
            headers: dict[str, str] = recorder.last_response_headers.copy()
            status = recorder.last_status_code
            body = getattr(exc, 'body', None)
            if response is not None:
                status = getattr(response, 'status_code', status)
                try:
                    headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
                except Exception:  # noqa: BLE001
                    pass
            result = {
                'success': False,
                'status_code': status,
                'latency_ms': latency_ms,
                'requested_model': args.model,
                'tool_type': args.tool_type,
                'error_type': type(exc).__name__,
                'error_message': str(exc),
                'error_body': body,
                'response_headers': headers,
            }
            print(json.dumps(result, indent=2, default=str))
            raise SystemExit(1)


if __name__ == '__main__':
    main()
