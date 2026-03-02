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
```

## Outputs

Generated deep-dive outputs go to `exploration/deep_dive/output/` (ignored by git).
