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
- `exploration/deep_dive/latency_non_openai.py`: latency benchmark for non-OpenAI models.
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
```

## Outputs

Generated deep-dive outputs go to `exploration/deep_dive/output/` (ignored by git).
