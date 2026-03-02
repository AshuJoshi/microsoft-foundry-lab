# AIProjectClient Pipeline Instrumentation

Date: 2026-03-02
Scope: `validation/scripts/run_agents_v2_validation.py`

## Problem

The validation harness originally captured HTTP metadata from OpenAI client hooks only.

- Calls through `openai_client.responses.*` and `openai_client.conversations.*` had status/headers.
- Calls through `project_client.agents.*` (for example `create_version`, `delete_version`) often showed empty:
  - `status_code`
  - `apim-request-id`
  - `x-request-id`
  - `x-ms-region`

Root cause: `AIProjectClient` uses Azure SDK pipeline transport, not the OpenAI client's internal `httpx` event hooks.

## What Changed

Added Azure SDK pipeline instrumentation in addition to existing OpenAI hook instrumentation.

1. Added `AzurePipelineRecorderPolicy(SansIOHTTPPolicy)`.
2. Injected policy into `AIProjectClient(..., per_call_policies=[...])`.
3. Extended recorder with two paths:
   - `on_httpx_*` for OpenAI client traffic
   - `on_azure_*` for AIProjectClient traffic
4. Added exception fallback extraction for status/headers.

## Before

Only OpenAI client hooks:

```python
def _attach_hooks_to_openai_client(openai_client, recorder):
    inner = getattr(openai_client, "_client", None)
    if inner is not None and hasattr(inner, "event_hooks"):
        inner.event_hooks = {"request": [recorder.on_request], "response": [recorder.on_response]}

with (
    DefaultAzureCredential() as credential,
    AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client,
    project_client.get_openai_client() as openai_client,
):
    _attach_hooks_to_openai_client(openai_client, recorder)
```

## After

OpenAI + Azure pipeline capture:

```python
class AzurePipelineRecorderPolicy(SansIOHTTPPolicy):
    def __init__(self, recorder):
        self._recorder = recorder
    def on_request(self, request):
        self._recorder.on_azure_request(request.http_request)
    def on_response(self, request, response):
        self._recorder.on_azure_response(response.http_response)

def _attach_hooks_to_openai_client(openai_client, recorder):
    inner = getattr(openai_client, "_client", None)
    if inner is not None and hasattr(inner, "event_hooks"):
        inner.event_hooks = {
            "request": [recorder.on_httpx_request],
            "response": [recorder.on_httpx_response],
        }

with (
    DefaultAzureCredential() as credential,
    AIProjectClient(
        endpoint=cfg.project_endpoint,
        credential=credential,
        per_call_policies=[AzurePipelineRecorderPolicy(recorder)],
    ) as project_client,
    project_client.get_openai_client() as openai_client,
):
    _attach_hooks_to_openai_client(openai_client, recorder)
```

## Verification

Smoke run confirmed previously blank agent rows now include metadata:

- `agent_lifecycle.create_agent`: `status_code=200`, request IDs present
- `agent_lifecycle.cleanup_agent`: `status_code=200`, request IDs present
- same pattern for `tool_* create_agent/cleanup_agent`

Run reference: `validation/results/20260302T145621Z/summary.tsv`

## Relationship: AIProjectClient vs OpenAI Client

- `AIProjectClient` is the Foundry SDK client for project operations (`agents`, `deployments`, etc.).
- `project_client.get_openai_client()` returns an OpenAI-compatible client configured for the same Foundry project.
- They share project context/auth, but use different HTTP stacks:
  - OpenAI client path (httpx hooks)
  - Azure pipeline path (policy hooks)

Instrumentation now covers both paths.
