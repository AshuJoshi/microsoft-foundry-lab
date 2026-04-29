# AI Search For An Existing Foundry Project

This module creates an Azure AI Search service and the RBAC required to use it with an existing Foundry project.

It does **not**:
- create the Foundry account
- create the Foundry project
- create the search indexes
- upload the sample data

Use this when you already have:
- a Foundry project
- one or more model deployments
- an Azure OpenAI / Foundry account you want the search vectorizer to use

## What It Creates

1. Azure AI Search service
2. Search-service-scoped role assignments for:
- your user
- the existing Foundry project managed identity
3. Azure OpenAI account-scoped role assignments for the Search managed identity

Those OpenAI roles mirror the workshop sample:
- `Cognitive Services OpenAI User`
- `Azure AI User`
- `Cognitive Services User`

## Why These RBAC Assignments Exist

### On the search service

Your user gets:
- `Search Service Contributor`
- `Search Index Data Reader`
- `Search Index Data Contributor`

The existing Foundry project managed identity gets:
- `Search Index Data Reader`
- `Search Index Data Contributor`
- `Search Service Contributor`

Those project roles are important for two paths:
- direct `AzureAISearchTool` access with an AAD/keyless project connection
- later Foundry IQ MCP access when the project connection uses `ProjectManagedIdentity`

### On the Azure OpenAI / Foundry account

The Search service managed identity gets the roles needed by the vectorizer workflow used by the workshop index definition.

## Inputs You Need

- `searchServiceName`
- `userObjectId`
- `foundryProjectPrincipalId`
- `openAiAccountName`
- `openAiAccountResourceGroup`

## How To Get The Values

### Current user object id

```bash
az ad signed-in-user show --query id -o tsv
```

### Existing Foundry project principal id

You need the managed identity principal id of the existing Foundry project resource.

If you know the project ARM resource id:

```bash
az resource show --ids <project_resource_id> --api-version 2025-04-01-preview --query identity.principalId -o tsv
```

### Existing OpenAI / Foundry account name and resource group

If your repo `.env` has the project endpoint but not the account resource group, inspect the Azure resource directly:

```bash
az resource list --resource-type Microsoft.CognitiveServices/accounts --query "[].{name:name,resourceGroup:resourceGroup}" -o table
```

## Deployment

Create a parameters file from the example:

```bash
cp infra/ai-search-existing-project/main.parameters.example.json infra/ai-search-existing-project/main.parameters.json
```

Then deploy into the resource group where you want the Search service to live:

```bash
az deployment group create \
  --resource-group <search_resource_group> \
  --template-file infra/ai-search-existing-project/main.bicep \
  --parameters @infra/ai-search-existing-project/main.parameters.json
```

## Outputs

The deployment outputs:
- `searchServiceName`
- `searchServiceEndpoint`
- `searchServicePrincipalId`
- `searchResourceId`
- `openAiAccountResourceId`
- `roleAssignmentIds`

Save the output JSON somewhere. The `roleAssignmentIds` are useful for teardown.

## Teardown / Undo

Yes, you should plan undo up front.

Deleting the resource group that contains the Search service is **not sufficient** by itself, because the OpenAI account-scoped role assignments are extension resources on the existing OpenAI / Foundry account.

### Step 1: delete the Search service resource group

If the Search service lives in a dedicated resource group, delete that group:

```bash
az group delete --name <search_resource_group> --yes --no-wait
```

This removes:
- the Search service
- the search-service-scoped role assignments

### Step 2: delete the OpenAI account-scoped role assignments

Use the deployment outputs from `roleAssignmentIds`:

```bash
az role assignment delete --ids <searchServiceOpenAiUserRoleAssignmentId>
az role assignment delete --ids <searchServiceAzureAiUserRoleAssignmentId>
az role assignment delete --ids <searchServiceCognitiveServicesUserRoleAssignmentId>
```

If you lost the outputs, list assignments for the deleted Search service principal before deleting the group, or query by principal id:

```bash
az role assignment list --assignee <search_service_principal_id> --all -o table
```

## Next Step After Deployment

This module only gives you the service and RBAC.

After deployment:

1. set:
- `AZURE_SEARCH_SERVICE_ENDPOINT`
- `AZURE_SEARCH_ADMIN_KEY`
- `AZURE_OPENAI_ENDPOINT`

2. create the indexes and upload data:

```bash
uv run exploration/deep_dive/ai_search_index_setup.py --indexes hrdocs,healthdocs --log-level INFO
```

3. create the direct Azure AI Search project connection

Recommended first:
- use `AAD`
- your Foundry project managed identity already has the Search RBAC needed for the direct tool path:
  - `Search Index Data Reader`
  - `Search Index Data Contributor`
  - `Search Service Contributor`
- this avoids storing the Search admin key in the project connection

```bash
uv run scripts/ai_search_project_connection_setup.py --connection-name ai-search-direct --auth-type AAD
```

4. run the direct tool probe:

```bash
uv run exploration/deep_dive/agent_ai_search_probe.py --model gpt-5.1 --cases vacation_senior,mental_health_copay,unknown --project-connection-name ai-search-direct --runs 1 --log-level INFO
```

5. then move to the Foundry IQ flow.

## If You Deployed An Earlier Version Of This Module

An earlier version of this module only granted the Foundry project managed identity:
- `Search Index Data Reader`

That is not enough for the direct `AzureAISearchTool` AAD path.

If you already deployed that earlier version, patch the Search service RBAC with:

```bash
az role assignment create \
  --assignee-object-id <foundry_project_principal_id> \
  --assignee-principal-type ServicePrincipal \
  --role "Search Service Contributor" \
  --scope <search_service_resource_id>
```

```bash
az role assignment create \
  --assignee-object-id <foundry_project_principal_id> \
  --assignee-principal-type ServicePrincipal \
  --role "Search Index Data Contributor" \
  --scope <search_service_resource_id>
```
