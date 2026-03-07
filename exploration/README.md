# Exploration

This area contains exploratory work, probes, and architecture artifacts.

## Subfolders

- `exploration/deep_dive/`: runnable probes and scripts.

## Deep Dive Scripts

- `exploration/deep_dive/list_models.py`: list deployments through `azure-ai-projects`.
- `exploration/deep_dive/trace_openai_requests.py`: low-level request tracing for responses calls.
- `exploration/deep_dive/trace_chat_completions.py`: low-level request tracing for chat completions.
- `exploration/deep_dive/map_endpoints.py`: inspect generated SDK endpoint routes.
- `exploration/deep_dive/probe_project_surfaces.py`: inspect deployments/connections/agents/memory.
- `exploration/deep_dive/compare_foundry_vs_openai_client.py`: compare Foundry bridge vs direct client usage.
- `exploration/deep_dive/compatibility_matrix.py`: endpoint/API/model compatibility matrix.
- `exploration/deep_dive/compatibility_matrix_clients_headers.py`: compatibility matrix with header capture.
- `exploration/deep_dive/latency_non_openai.py`: latency benchmark for all discovered inference models, comparing model-family-specific endpoint/API paths.
- `exploration/deep_dive/probe_responses_schema_fields.py`: probes `responses.create` field acceptance across endpoint families.
  - Supports multi-model runs, live PASS/FAIL console progress, and optional skipping of direct Responses probes for chat-only models.
  - Case definitions:
    - `baseline`: minimal payload (`input`, `max_output_tokens`).
    - `temperature`: adds `temperature=0`.
    - `top_p`: adds `top_p=1`.
    - `metadata`: adds a small metadata object.
    - `reasoning_effort_none`: adds `reasoning={"effort":"none"}`.
    - `reasoning_effort_low`: adds `reasoning={"effort":"low"}`.
    - `text_format_text`: adds `text={"format":{"type":"text"}}`.
    - `text_verbosity_low`: adds `text={"verbosity":"low"}`.
    - `truncation_disabled`: adds `truncation="disabled"`.
    - `store_false`: adds `store=False`.
- `exploration/deep_dive/agent_http_tool_exploration.py`: creates a temporary Foundry agent with a local `http_request` function tool handler and captures tool-call behavior.
- `exploration/deep_dive/http_request.py`: original HTTP tool source copied from prior `AgentExp` work and reused by `agent_http_tool_exploration.py`.
- `exploration/deep_dive/list_sdk_tools.py`: enumerates tool-related model classes exported by `azure.ai.projects.models`.
- `exploration/deep_dive/agents_memory_exploration.py`: explores agent+memory behavior with SDK and request metadata capture.
- `exploration/deep_dive/probe_embedding_model.py`: probes embedding deployment behavior across endpoint modes.
- `exploration/deep_dive/run_memory_docs_sample.py`: runs a docs-aligned memory sample workflow end-to-end.
- `exploration/deep_dive/web_search_foundry_vs_openai_native.py`: compares Foundry SDK web-search-tool path (preview behavior) against OpenAI native `web_search` and computes per-case URL overlap/differences.
- `exploration/deep_dive/cases/web_search_foundry_vs_openai_native.json`: reusable case templates for web-search result comparison (`{topic}`, `{since_date}`, `{days_window}`).

## Environment Inputs

All deep-dive scripts load from repo-root `config.py`, which reads `.env`.

- `AZURE_AI_PROJECT_ENDPOINT` (or `FOUNDRY_PROJECT_ENDPOINT`)
- `AZURE_AI_MODEL_DEPLOYMENT_NAME`
- `FOUNDRY_RESOURCE_NAME`
- `FOUNDRY_PROJECT_NAME`
- `AGENT_NAME_PREFIX` (preferred)
- `BUGBASH_AGENT_NAME_PREFIX` (legacy fallback)

## Common Commands

```bash
uv run exploration/deep_dive/list_models.py
uv run exploration/deep_dive/trace_openai_requests.py
uv run exploration/deep_dive/probe_project_surfaces.py
uv run exploration/deep_dive/compare_foundry_vs_openai_client.py
uv run exploration/deep_dive/compatibility_matrix.py
uv run exploration/deep_dive/compatibility_matrix_clients_headers.py
uv run exploration/deep_dive/latency_non_openai.py
uv run exploration/deep_dive/map_endpoints.py
uv run exploration/deep_dive/probe_responses_schema_fields.py --model gpt-5.2 --endpoint all --out-dir exploration/deep_dive/output
uv run exploration/deep_dive/probe_responses_schema_fields.py --models gpt-5.2,grok-4,Kimi-K2.5 --endpoint all
uv run exploration/deep_dive/probe_responses_schema_fields.py --all-models --endpoint all
uv run exploration/deep_dive/agent_http_tool_exploration.py --model gpt-5-mini
uv run exploration/deep_dive/list_sdk_tools.py
uv run exploration/deep_dive/agents_memory_exploration.py --model gpt-5-mini
uv run exploration/deep_dive/probe_embedding_model.py --model text-embedding-3-small --mode all
uv run exploration/deep_dive/run_memory_docs_sample.py --chat-model gpt-5-mini --embedding-model text-embedding-3-small --wait-seconds 5
uv run exploration/deep_dive/web_search_foundry_vs_openai_native.py --model gpt-5-mini --tool-choice required --no-stream
uv run exploration/deep_dive/web_search_foundry_vs_openai_native.py --model gpt-5-mini --topic "NVIDIA quarterly earnings and guidance" --days-window 14 --cases-file exploration/deep_dive/cases/web_search_foundry_vs_openai_native.json --tool-choice required --no-stream
```

## Outputs

Generated deep-dive outputs go to `exploration/deep_dive/output/` (ignored by git).
