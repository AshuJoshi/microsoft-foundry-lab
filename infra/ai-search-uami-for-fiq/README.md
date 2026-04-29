# AI Search UAMI For Foundry IQ

This module creates a **user-assigned managed identity** for the Foundry IQ knowledge-base path and grants it outbound model-access roles on an existing Azure OpenAI / Foundry account. A companion script then associates that UAMI with the existing Azure AI Search service using the Search Management **Update** API.

Use this when:
- your OpenAI / Foundry account has `disableLocalAuth = true`
- you do **not** want to use an API key in the Search knowledge-base model configuration
- you want the Search service to reach the model through a user-assigned managed identity

## What It Creates

1. A user-assigned managed identity
2. OpenAI / Foundry account-scoped RBAC for the UAMI:
- `Cognitive Services OpenAI User`
- `Azure AI User`
- `Cognitive Services User`

## What It Does Not Do

- create a Search service
- create Search indexes
- create knowledge sources
- create a knowledge base
- create the Foundry MCP connection
- patch `fiq_knowledge_setup.py`

Those steps stay separate.

## Intended Baseline

This module is designed to work against a Search service that was created by:
- `infra/ai-search-existing-project/`

That matters because the companion attach step needs to preserve the Search service identity state without disrupting the working direct AI Search baseline.

## Why This Exists

The current Search knowledge-base preview API supports model configuration with:
- `apiKey`
- or `authIdentity`

For `authIdentity`, current docs describe a **user-assigned managed identity** resource assigned to the Search service.

That makes this module the prerequisite for a managed-identity FIQ flow.

## Inputs

- `userAssignedIdentityName`
- `searchServiceName`
- `openAiAccountResourceGroup`
- `openAiAccountName`

Get the Foundry / OpenAI account identity target:

```bash
az resource list \
  --resource-type Microsoft.CognitiveServices/accounts \
  --query "[].{name:name,resourceGroup:resourceGroup}" \
  -o table
```

## Deployment

Copy the example parameters file if needed:

```bash
cp infra/ai-search-uami-for-fiq/main.parameters.example.json infra/ai-search-uami-for-fiq/main.parameters.json
```

Deploy into the resource group where you want the UAMI to live:

```bash
az deployment group create \
  --resource-group <uami_resource_group> \
  --template-file infra/ai-search-uami-for-fiq/main.bicep \
  --parameters @infra/ai-search-uami-for-fiq/main.parameters.json
```

Then attach the created UAMI to the existing Search service using the companion script:

```bash
uv run infra/ai-search-uami-for-fiq/attach_search_uami.py \
  --subscription-id <subscription_id> \
  --search-resource-group <search_resource_group> \
  --search-service-name <search_service_name> \
  --uami-resource-id <uami_resource_id>
```

## Outputs

Important outputs:
- `userAssignedIdentityName`
- `userAssignedIdentityResourceId`
- `userAssignedIdentityPrincipalId`
- `userAssignedIdentityClientId`
- `searchServiceResourceId`
- `roleAssignmentIds`

The `userAssignedIdentityResourceId` is the critical handoff into the knowledge-base creation script.

## Verification

After deployment, verify that:
- the UAMI exists
- the Foundry / OpenAI account has the three expected RBAC assignments for the UAMI principal

After the attach script, also verify that:
- the Search service identity type is now `SystemAssigned, UserAssigned`
- the UAMI resource id appears under `identity.userAssignedIdentities`

Example checks:

```bash
az identity show \
  --resource-group <uami_resource_group> \
  --name <uami_name> \
  --query "{id:id,principalId:principalId,clientId:clientId}" \
  -o json
```

```bash
az resource show \
  --resource-group <search_resource_group> \
  --resource-type Microsoft.Search/searchServices \
  --name <search_service_name> \
  --api-version 2023-11-01 \
  --query "{identityType:identity.type,userAssignedIdentities:identity.userAssignedIdentities}" \
  -o json
```

## Next Step

After deployment and attach, the FIQ knowledge-base creation flow should use:
- `azureOpenAIParameters.authIdentity`

with the `userAssignedIdentityResourceId` output from this module.
