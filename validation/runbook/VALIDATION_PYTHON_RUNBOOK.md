# Agents v2 Validation: Python Runbook

Date: 2026-03-02

## Scope

Python validation execution using a consolidated harness built on top of the SDK sample flows:

- `validation/scripts/run_agents_v2_validation.py`
- `validation/scripts/run_python_validation.sh`
- Source samples remain in `validation/samples/python/azure-ai-projects/samples/`.

## Preflight Checklist

1. Azure auth
- `az login`
- `az account show`

2. Required `.env` at repo root
- `AZURE_AI_PROJECT_ENDPOINT`
- `AZURE_AI_MODEL_DEPLOYMENT_NAME` (`gpt-5-mini` default for this validation run)

3. Recommended in `.env`
- `AGENT_NAME_PREFIX` (preferred, portal-friendly naming prefix)
- `BUGBASH_AGENT_NAME_PREFIX` (legacy fallback)
- `FOUNDRY_RESOURCE_NAME`
- `FOUNDRY_PROJECT_NAME`

4. Package sanity
- `uv run python -c "import azure.ai.projects, openai, azure.identity, dotenv; print('ok')"`
- `uv run python -c "import azure.ai.projects as p; print(getattr(p,'__version__','unknown'))"`

## Execution Order (implemented by harness)

1. Responses basic
- Plain `responses.create()` call.

2. Agent basic lifecycle
- Create agent version
- Create conversation
- Multi-turn responses
- Cleanup

3. Tools
- Code Interpreter
- Web Search
- MCP
- Function calling

4. Model matrix sequence
- Run all tests on default model first.
- Then rerun full suite for all other deployed models.

## One-command Runner

```bash
bash validation/scripts/run_python_validation.sh
```

Outputs:

- `validation/results/<run_id>/harness.log`
- `validation/results/<run_id>/run_log.md`
- `validation/results/<run_id>/run_log.json`
- `validation/results/<run_id>/summary.tsv`

Optional (faster smoke pass):

```bash
bash validation/scripts/run_python_validation.sh --default-model-only
```

## Notes

- Web Search may incur extra costs and has data-boundary implications.
- Current known compatibility from latest full run:
  - `tool_web_search` returns `400` for: `DeepSeek-V3.2`, `Kimi-K2.5`, `Mistral-Large-3`, `grok-4`, `grok-4-1-fast-non-reasoning`.
  - Intermittent `500` observed for:
    - `Kimi-K2.5` with `tool_mcp`
    - `grok-4-1-fast-non-reasoning` with `tool_code_interpreter`
- Harness records request/response metadata per API call:
  - status code
  - `apim-request-id`
  - `x-request-id`
  - `x-ms-region`
  - provider/model headers (`openai-*`, `x-ratelimit-*`, `x-ms-*`, `azure*`)
- Harness records reproducibility context per call:
  - prompt
  - deployment name
  - endpoint family
  - API style
  - exact code snippet
- Harness records timing:
  - start/end timestamps (UTC)
  - latency in ms
  - attempt index and retry budget

## Failure Report Checklist

1. Script path / test case (`validation/scripts/run_agents_v2_validation.py` + `test_case`)
2. Exact command run
3. Environment summary (SDK version, Python version, region)
4. Full error/traceback
5. Request IDs and timestamp (UTC)
6. Expected vs actual behavior
