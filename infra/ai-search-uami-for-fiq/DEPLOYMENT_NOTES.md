# AI Search UAMI For Foundry IQ Deployment Notes

This note captures the validated deployment sequence for the managed-identity Foundry IQ path.

It is intentionally separate from:
- `infra/ai-search-existing-project/DEPLOYMENT_NOTES.md`

because this path adds one more layer:
- a user-assigned managed identity for Search outbound model access during knowledge-base operations

## Goal

Use the existing validated Search baseline and add the extra identity plumbing needed for Foundry IQ without relying on API keys.

Prerequisites already in place:
- an existing Foundry project
- an existing Azure AI Search service
- Search indexes already created and populated
- the direct Azure AI Search path already validated

In this repo, those prerequisites came from:
- `infra/ai-search-existing-project/`
- `exploration/deep_dive/ai_search_index_setup.py`
- the direct Azure AI Search probe flow

## Working Sequence

### 1. Create the UAMI and assign Foundry / OpenAI RBAC

The separate infra module creates:
- a user-assigned managed identity
- role assignments on the Foundry / OpenAI account:
  - `Cognitive Services OpenAI User`
  - `Azure AI User`
  - `Cognitive Services User`

Use a local parameters file:

```bash
cp infra/ai-search-uami-for-fiq/main.parameters.example.json infra/ai-search-uami-for-fiq/main.parameters.json
```

For the validated environment, the filled values were:
- `location = westus2`
- `userAssignedIdentityName = fj-fiq-search-uami`
- `searchServiceName = fj-ai-search-01`
- `openAiAccountResourceGroup = AgenticExperiments`
- `openAiAccountName = FoundryJourney`

Deploy:

```bash
az deployment group create \
  --resource-group AgenticExperimentsSearch \
  --template-file infra/ai-search-uami-for-fiq/main.bicep \
  --parameters @infra/ai-search-uami-for-fiq/main.parameters.json
```

Validated outputs:
- `userAssignedIdentityName = fj-fiq-search-uami`
- `userAssignedIdentityPrincipalId = 8d7ae723-ba02-4733-b269-dfe639900bc0`
- `userAssignedIdentityClientId = 9a341dbf-61ab-4e78-b91b-0803c3d6e5b1`
- `userAssignedIdentityResourceId = /subscriptions/c3bf4bdb-861c-441a-b6da-dc206e1d2dc4/resourceGroups/AgenticExperimentsSearch/providers/Microsoft.ManagedIdentity/userAssignedIdentities/fj-fiq-search-uami`

## Important Deviation From The First Attempt

The original plan was to attach the UAMI to the Search service directly from Bicep by updating the Search resource identity.

That failed with:

```text
Resource identity type 'SystemAssigned, UserAssigned' is invalid. Supported identity types are: None,SystemAssigned.
```

However, the Azure portal showed that:
- the Search service **does** expose a `User assigned` identity tab
- the control plane supports adding a UAMI through the Search Identity UI

This led to the working conclusion:
- the full ARM/Bicep resource update path was not the reliable control-plane path in this environment
- the Search Management Update API was the correct attach mechanism

## 2. Attach the UAMI to the Search service

Use the companion script:

```bash
uv run infra/ai-search-uami-for-fiq/attach_search_uami.py \
  --subscription-id c3bf4bdb-861c-441a-b6da-dc206e1d2dc4 \
  --search-resource-group AgenticExperimentsSearch \
  --search-service-name fj-ai-search-01 \
  --uami-resource-id /subscriptions/c3bf4bdb-861c-441a-b6da-dc206e1d2dc4/resourceGroups/AgenticExperimentsSearch/providers/Microsoft.ManagedIdentity/userAssignedIdentities/fj-fiq-search-uami
```

Validated artifact:
- `exploration/deep_dive/output/attach_search_uami_20260429T034036Z.json`

Validated result:
- `success = true`
- `existing_identity_type = SystemAssigned`
- `new_identity_type = SystemAssigned, UserAssigned`
- `status_code = 200`

This is the key operational nuance:
- **Bicep creates the UAMI and RBAC**
- **Search Management Update/PATCH attaches the UAMI to Search**

## 3. Create knowledge sources and the knowledge base

The Search-side knowledge sources were straightforward.

Command:

```bash
uv run exploration/deep_dive/fiq_knowledge_setup.py \
  --auth-mode managed-identity \
  --uami-resource-id /subscriptions/c3bf4bdb-861c-441a-b6da-dc206e1d2dc4/resourceGroups/AgenticExperimentsSearch/providers/Microsoft.ManagedIdentity/userAssignedIdentities/fj-fiq-search-uami \
  --knowledge-base-name zava-agentic-kb \
  --log-level INFO
```

### Required environment values

- `AZURE_SEARCH_SERVICE_ENDPOINT`
- `AZURE_SEARCH_ADMIN_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_CHATGPT_DEPLOYMENT`
- `AZURE_OPENAI_CHATGPT_MODEL_NAME`
- `FIQ_SEARCH_UAMI_RESOURCE_ID` or `--uami-resource-id`

Not required in this mode:
- `AZURE_OPENAI_KEY`

### Important payload detail

The working managed-identity payload shape is:

```json
"authIdentity": {
  "@odata.type": "#Microsoft.Azure.Search.DataUserAssignedIdentity",
  "userAssignedIdentity": "<uami-resource-id>"
}
```

Not:
- `resourceId`

That field-name distinction mattered.

### Model-name constraint

The Search knowledge-base API rejects unsupported `modelName` values even if the Foundry project itself has a broader model surface.

The rejected error that surfaced during validation was:

```text
Unsupported model type in Knowledge Base Model Configuration.
ModelName must be either gpt-4o, gpt-4o-mini, gpt-4.1-nano, gpt-4.1-mini, gpt-4.1, gpt-5, gpt-5-mini, gpt-5-nano.
```

The validated working configuration used:
- `AZURE_OPENAI_CHATGPT_DEPLOYMENT = gpt-5`
- `AZURE_OPENAI_CHATGPT_MODEL_NAME = gpt-5`

Validated artifact:
- `exploration/deep_dive/output/fiq_knowledge_setup_20260429T035333Z.json`

Validated result:
- `hrdocs-knowledge-source = 204`
- `healthdocs-knowledge-source = 204`
- `zava-agentic-kb = 201`

## 4. Create the Foundry project connection to the KB MCP endpoint

Command:

```bash
uv run exploration/deep_dive/fiq_project_connection_setup.py \
  --project-connection-name fiq-knowledge-base \
  --log-level INFO
```

Validated artifact:
- `exploration/deep_dive/output/fiq_project_connection_setup_20260429T035440Z.json`

Validated connection settings:
- `authType = ProjectManagedIdentity`
- `category = RemoteTool`
- `audience = https://search.azure.com/`
- target:
  - `https://fj-ai-search-01.search.windows.net/knowledgebases/zava-agentic-kb/mcp?api-version=2025-11-01-preview`

## 5. Run the Foundry IQ MCP-backed agent probe

Command:

```bash
uv run exploration/deep_dive/agent_foundry_iq_probe.py \
  --model gpt-5 \
  --cases hr_policy,health_fact,cross_source_compare,unknown \
  --project-connection-name fiq-knowledge-base \
  --runs 1 \
  --log-level INFO
```

Validated artifact:
- `exploration/deep_dive/output/agent_foundry_iq_probe_20260429T035550Z.json`

Validated outcomes:
- `hr_policy` -> `4 weeks`
- `health_fact` -> `$30`
- `cross_source_compare` -> `Innovator of the Month` and `$35`
- `unknown` -> `I don't know`

Observed MCP-backed tool sequence:
- `mcp_list_tools`
- `reasoning`
- `mcp_call`
- `reasoning`
- `message`

## What This Established

The managed-identity Foundry IQ path is now proven in this repo.

The working control-plane split is:

1. AI Search prerequisite baseline
2. UAMI creation + Foundry/OpenAI RBAC through Bicep
3. Search-service UAMI attach through Search Management Update/PATCH
4. Search knowledge base creation using `authIdentity`
5. Foundry MCP project connection using `ProjectManagedIdentity`
6. MCP-backed Prompt agent validation

## Relationship To The AI Search Baseline

The AI Search prerequisite still matters.

This FIQ path assumes the earlier direct AI Search work already succeeded:
- Search service created
- Search indexes created and populated
- Search and Foundry identities already aligned enough for the Search baseline

This note therefore does **not** replace:
- `infra/ai-search-existing-project/DEPLOYMENT_NOTES.md`

It extends it with the managed-identity FIQ-specific layer.
