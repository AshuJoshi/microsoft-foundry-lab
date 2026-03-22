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
- `exploration/deep_dive/try_model.py`: quick interactive probe for a deployment. In `auto` mode it uses `responses` on the project endpoint for OpenAI models and `chat.completions` on the AOAI endpoint for non-OpenAI models.
- `exploration/deep_dive/model_router_probe.py`: dedicated probe for the `model-router` deployment across the supported paths (`aoai + chat.completions` and `project + responses`), including the concrete backend model returned on each run.
- `exploration/deep_dive/list_sdk_tools.py`: enumerates tool-related model classes exported by `azure.ai.projects.models`.
- `exploration/deep_dive/agents_memory_exploration.py`: explores agent+memory behavior with SDK and request metadata capture.
- `exploration/deep_dive/probe_embedding_model.py`: probes embedding deployment behavior across endpoint modes.
- `exploration/deep_dive/run_memory_docs_sample.py`: runs a docs-aligned memory sample workflow end-to-end.
- `exploration/deep_dive/web_search_foundry_vs_openai_native.py`: compares Foundry SDK web-search-tool path against OpenAI native `web_search` and computes per-case URL overlap/differences.
- `exploration/deep_dive/cases/web_search_foundry_vs_openai_native.json`: reusable case templates for web-search result comparison (`{topic}`, `{since_date}`, `{days_window}`).
- `exploration/deep_dive/search_prompt_probe.py`: direct Responses API probe for `web_search_preview`, comparing `aoai_responses` and `project_responses` with citation/date extraction, annotation capture, output item types, and optional location/context-size tool parameters.
- `exploration/deep_dive/search_agent_probe.py`: agent-based probe for Foundry `WebSearchTool` using a temporary Prompt agent and remote conversation, with citation annotation capture and optional streaming.
- `exploration/deep_dive/search_bing_grounding_probe.py`: agent-based probe for `BingGroundingTool` using a project `GroundingWithBingSearch` connection, including citation annotation capture and Bing-specific options such as `market`, `set_lang`, `count`, and `freshness`.
- `exploration/deep_dive/list_search_tool_resources.py`: inspects project connections and flags search-related resources/configuration visible through the SDK.
- `exploration/deep_dive/agent_context_limit_probe.py`: batch probe for growing a single Prompt-agent conversation with repeated stuffing/recall turns to study context pressure, throttling, and recall behavior.
- `exploration/deep_dive/agent_large_tool_payload_probe.py`: Prompt-agent probe that uses a large local tool payload to stress the remote conversation with oversized tool outputs instead of plain user-message stuffing.
- `exploration/deep_dive/agent_context_stepwise_probe.py`: stepwise Prompt-agent trace probe that preserves the same remote agent and conversation across separate invocations for portal trace inspection and stateful conversation experiments.

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
uv run exploration/deep_dive/try_model.py --model gpt-4.1 --prompt "Reply exactly: openai-ok"
uv run exploration/deep_dive/try_model.py --model Mistral-Large-3 --prompt "Reply exactly: non-openai-ok"
uv run exploration/deep_dive/model_router_probe.py --model model-router --runs 2
uv run exploration/deep_dive/list_sdk_tools.py
uv run exploration/deep_dive/agents_memory_exploration.py --model gpt-5-mini
uv run exploration/deep_dive/probe_embedding_model.py --model text-embedding-3-small --mode all
uv run exploration/deep_dive/run_memory_docs_sample.py --chat-model gpt-5-mini --embedding-model text-embedding-3-small --wait-seconds 5
uv run exploration/deep_dive/web_search_foundry_vs_openai_native.py --model gpt-5-mini --tool-choice required --no-stream
uv run exploration/deep_dive/web_search_foundry_vs_openai_native.py --model gpt-5-mini --topic "NVIDIA quarterly earnings and guidance" --days-window 14 --cases-file exploration/deep_dive/cases/web_search_foundry_vs_openai_native.json --tool-choice required --no-stream
uv run exploration/deep_dive/search_prompt_probe.py --model gpt-5.4 --runs 1 --log-level INFO
uv run exploration/deep_dive/search_agent_probe.py --model gpt-5.4 --runs 1 --log-level INFO
uv run exploration/deep_dive/search_agent_probe.py --model model-router --runs 1 --country US --region WA --city Seattle --log-level INFO
uv run exploration/deep_dive/search_bing_grounding_probe.py --model gpt-5.1 --runs 1 --market en-US --set-lang en-US --count 5 --log-level INFO
uv run exploration/deep_dive/list_search_tool_resources.py --log-level INFO
uv run exploration/deep_dive/agent_context_limit_probe.py --model gpt-5.1 --target-input-tokens 272000 --block-chars 6000 --max-turns 30 --max-output-tokens 120 --sleep-seconds 3 --log-level INFO
uv run exploration/deep_dive/agent_large_tool_payload_probe.py --model gpt-5.1 --payload-chars 150000 --max-turns 3 --max-output-tokens 120 --sleep-seconds 2 --log-level INFO
uv run exploration/deep_dive/agent_context_stepwise_probe.py --state exploration/deep_dive/output/run_001/ctxstep.state.json --model gpt-5.1 create-agent
uv run exploration/deep_dive/agent_context_stepwise_probe.py --state exploration/deep_dive/output/run_001/ctxstep.state.json create-conversation
uv run exploration/deep_dive/agent_context_stepwise_probe.py --state exploration/deep_dive/output/run_001/ctxstep.state.json stuff --block-chars 6000 --truncation auto
uv run exploration/deep_dive/agent_context_stepwise_probe.py --state exploration/deep_dive/output/run_001/ctxstep.state.json recall --truncation auto
uv run exploration/deep_dive/agent_context_stepwise_probe.py --state exploration/deep_dive/output/run_001/ctxstep.state.json cleanup --delete-conversation --delete-agent
```

## Outputs

Generated deep-dive outputs go to `exploration/deep_dive/output/` (ignored by git).
