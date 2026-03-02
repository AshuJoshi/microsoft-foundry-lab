# Foundry SDK Explorations And Validation

This repository has two active tracks:

- `exploration/`: ad hoc and deep-dive research scripts/reports.
- `validation/`: repeatable regression-style validation harness.

## Folder Guides

- `exploration/README.md`
- `validation/README.md`

## Prerequisites

1. [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)
2. [Azure & Foundry Setup](https://learn.microsoft.com/en-us/azure/ai-foundry/tutorials/quickstart-create-foundry-resources?view=foundry&tabs=azurecli): set up Azure account, create Foundry resource/project, and deploy models.
3. Authenticate locally:

```bash
az login
```

## Quickstart

### 1) Install dependencies

```bash
uv sync --locked
```

### 2) Create `.env`

```bash
cat > .env <<'EOF'
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_RESOURCE_GROUP=<your-resource-group>
FOUNDRY_RESOURCE_NAME=<your-foundry-resource-name>
FOUNDRY_PROJECT_NAME=<your-project-name>
AZURE_AI_PROJECT_ENDPOINT=https://<your-foundry-resource-name>.services.ai.azure.com/api/projects/<your-project-name>
AZURE_AI_MODEL_DEPLOYMENT_NAME=gpt-5-mini
AGENT_NAME_PREFIX=ValidationAgent
EOF
```

### 3) Exploration

```bash
uv run exploration/deep_dive/list_models.py
uv run exploration/deep_dive/trace_openai_requests.py
```

### 4) Validation Smoke Run

```bash
bash validation/scripts/run_python_validation.sh --default-model-only --retries 1
```
