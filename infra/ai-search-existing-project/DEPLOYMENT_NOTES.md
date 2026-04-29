# AI Search Existing-Project Deployment Notes

This note captures the practical deployment flow for `infra/ai-search-existing-project/` and explains the identity model and the Bicep design choices that came up during implementation.

## Goal

Use an **existing Foundry project** and add:
- a new Azure AI Search service
- the RBAC required for the Search service and the Foundry project to work together

This module does **not**:
- create a Foundry account
- create a Foundry project
- create indexes
- upload corpus data
- create the direct Azure AI Search Foundry connection

Those later steps are handled separately in the repo.

## Command Sequence

### 1. Identify the existing Foundry project managed identity

Get the Foundry project principal id:

```bash
az resource show \
  --ids /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<account>/projects/<project> \
  --api-version 2025-04-01-preview \
  --query identity.principalId -o tsv
```

Example output:

```text
00000000-0000-0000-0000-000000000000
```

That value is the **managed identity principal id** of the Foundry project.

### 2. Get the signed-in user object id

```bash
az ad signed-in-user show --query id -o tsv
```

Example output:

```text
11111111-1111-1111-1111-111111111111
```

That value is your Entra user object id and is used for the Search RBAC assignments.

### 3. Find the Azure OpenAI / Foundry account name and resource group

```bash
az resource list \
  --resource-type Microsoft.CognitiveServices/accounts \
  --query "[].{name:name,resourceGroup:resourceGroup}" -o table
```

Example output:

```text
MyFoundryAccount  MyFoundryResourceGroup
```

These become:
- `openAiAccountName`
- `openAiAccountResourceGroup`

### 4. Create a resource group for Search

Recommended:
- keep Search in a separate resource group from Foundry
- easier cleanup
- lower blast radius

Example:

```bash
az group create --name MySearchResourceGroup --location westus2
```

### 5. Fill the parameters file

Copy the example if needed:

```bash
cp infra/ai-search-existing-project/main.parameters.example.json infra/ai-search-existing-project/main.parameters.json
```

Then set real values for:
- `location`
- `searchServiceName`
- `userObjectId`
- `foundryProjectPrincipalId`
- `openAiAccountName`
- `openAiAccountResourceGroup`

### 6. Deploy the Bicep

```bash
az deployment group create \
  --resource-group <search_resource_group> \
  --template-file infra/ai-search-existing-project/main.bicep \
  --parameters @infra/ai-search-existing-project/main.parameters.json
```

If you intentionally edited `main.parameters.example.json` with real values, that also works, but it is better to keep `example` as an example and use `main.parameters.json` for live values.

### 7. After infrastructure succeeds

Set the runtime values you need for the next repo steps:
- `AZURE_SEARCH_SERVICE_ENDPOINT`
- `AZURE_SEARCH_ADMIN_KEY`
- `AZURE_OPENAI_ENDPOINT`

Get the Search admin key:

```bash
az search admin-key show \
  --resource-group <search_resource_group> \
  --service-name <search_service_name> \
  --query primaryKey -o tsv
```

Suggested `.env` additions after a successful deployment:

```dotenv
AZURE_SEARCH_SERVICE_ENDPOINT=https://fj-ai-search-01.search.windows.net
AZURE_SEARCH_ADMIN_KEY=<search-admin-key>
AZURE_OPENAI_ENDPOINT=<existing-foundry-or-openai-endpoint>
```

If you paste a live admin key into a shared terminal transcript, treat that as exposed and rotate it later if needed.

Then create indexes and upload the sample data:

```bash
uv run exploration/deep_dive/ai_search_index_setup.py --indexes hrdocs,healthdocs --log-level INFO
```

### 8. What is still missing after this module

This module does **not** create the direct Foundry project connection needed by `AzureAISearchTool`.

So after:
- Search service creation
- RBAC assignment
- index creation

the next missing control-plane step is:
- create a direct Azure AI Search project connection for the Foundry project

That is separate from the Foundry IQ MCP connection flow.

In this repo, that helper is:

```bash
uv run scripts/ai_search_project_connection_setup.py --connection-name ai-search-direct --auth-type AAD
```

Recommended first:
- use `AAD`
- the Foundry project managed identity should have these Search roles for the direct tool path:
  - `Search Index Data Reader`
  - `Search Index Data Contributor`
  - `Search Service Contributor`
- this avoids putting the Search admin key into the project connection

When this succeeds, it writes:
- `exploration/deep_dive/output/ai_search_project_connection_setup_<run_id>.json`
- `exploration/deep_dive/output/ai_search_project_connection_setup_<run_id>.md`
- `exploration/deep_dive/output/ai_search_project_connection_cache.json`

For key-based auth instead:

```bash
uv run scripts/ai_search_project_connection_setup.py --connection-name ai-search-direct --auth-type ApiKey
```

The helper uses the current Foundry project from `.env` and creates a `CognitiveSearch` project connection against the Search endpoint.

### 9. Run the direct Azure AI Search probe

After the connection exists, validate the direct tool path:

```bash
uv run exploration/deep_dive/agent_ai_search_probe.py --model gpt-5.1 --cases vacation_senior,mental_health_copay,unknown --project-connection-name ai-search-direct --runs 1 --log-level INFO
```

This proves:
- the Search service exists
- the indexes exist
- the Foundry project connection works
- `AzureAISearchTool` can retrieve from the indexed corpus

## Identity Model

Three identities matter here.

### 1. Your user identity

This is your signed-in Entra user.

Used for:
- deploying resources
- creating and populating indexes
- direct Search administration

Roles assigned on the Search service:
- `Search Service Contributor`
- `Search Index Data Reader`
- `Search Index Data Contributor`

### 2. The Foundry project managed identity

This belongs to the existing Foundry project resource.

Used for:
- later Search-backed access from the Foundry project side
- especially relevant for Foundry IQ MCP access using `ProjectManagedIdentity`

Role assigned on the Search service:
- `Search Index Data Reader`
- `Search Index Data Contributor`
- `Search Service Contributor`

### 3. The Azure AI Search managed identity

This belongs to the newly created Search service.

Used for:
- the vectorizer workflow against the Azure OpenAI / Foundry account

Roles assigned on the Azure OpenAI / Foundry account:
- `Cognitive Services OpenAI User`
- `Azure AI User`
- `Cognitive Services User`

Those assignments mirror the workshop sample.

## Why The Bicep Needed A Module

The first version of the Bicep tried to create everything in one resource-group-scoped file.

That failed with:

```text
BCP139: A resource's scope must match the scope of the Bicep file for it to be deployable.
You must use modules to deploy resources to a different scope.
```

Why:
- the Search service is created in the deployment resource group
- but the Azure OpenAI / Foundry account already exists in a **different resource group**
- the three OpenAI-account-scoped role assignments are extension resources on that existing account

So the fix was:
- keep the Search service and Search-scoped RBAC in `main.bicep`
- move the OpenAI-account role assignments into a module:
  - `infra/ai-search-existing-project/modules/openai-role-assignments.bicep`
- deploy that module with:
  - `scope: resourceGroup(openAiAccountResourceGroup)`

Important:
- this was a **cross-resource-group scope** issue
- not a region issue

## Region Capacity Issue

Separately, a deployment attempt in `eastus2` failed with:

```text
InsufficientResourcesAvailable
The region 'eastus2' is currently out of the resources required to provision new services.
```

That is an Azure capacity issue, not a Bicep design issue.

The fix is simply:
- choose a different region
- rerun the deployment

Example fallback region:
- `westus2`

In one repo run, `eastus2` failed with capacity and `westus2` succeeded.

## Direct AzureAISearchTool AAD Troubleshooting

In this repo run, the first direct `AzureAISearchTool` probe failed with:

```text
Access denied. Check your permissions or managed identity access to the search service.
```

Root cause:
- the Search service and direct project connection existed
- the indexes existed
- but the Foundry project managed identity only had `Search Index Data Reader`

That role alone was not sufficient for the direct `AzureAISearchTool` AAD/keyless path.

The working fix was to grant the Foundry project managed identity:
- `Search Service Contributor`
- `Search Index Data Contributor`

Example patch commands:

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

After applying those roles, wait a few minutes for RBAC propagation before rerunning:

```bash
uv run exploration/deep_dive/agent_ai_search_probe.py --model gpt-5.1 --cases vacation_senior,mental_health_copay,unknown --project-connection-name ai-search-direct --runs 1 --log-level INFO
```

## az vs azd

### `az`

We are using `az` here because:
- you already have an existing Foundry project
- we only need a focused infrastructure add-on
- `az deployment group create` is enough for this module

Commands used:

```bash
az ad signed-in-user show --query id -o tsv
az resource list --resource-type Microsoft.CognitiveServices/accounts --query "[].{name:name,resourceGroup:resourceGroup}" -o table
az resource show --ids <project_resource_id> --api-version 2025-04-01-preview --query identity.principalId -o tsv
az group create --name <rg> --location <location>
az deployment group create --resource-group <rg> --template-file infra/ai-search-existing-project/main.bicep --parameters @infra/ai-search-existing-project/main.parameters.json
```

### `azd`

The workshop sample uses `azd` for the full environment because it provisions:
- Azure AI Search
- Foundry account/project
- model deployments
- post-provision hooks

That is useful for full greenfield setup, but for this repo we are intentionally **not** reprovisioning Foundry, so `azd` is not the best fit for this narrower task.

## Teardown / Undo

Deleting the Search resource group is not enough by itself, because the OpenAI-account-scoped role assignments live on the existing OpenAI / Foundry account.

Teardown steps:

### 1. Delete the Search resource group

```bash
az group delete --name AgenticExperimentsSearch --yes --no-wait
```

This removes:
- the Search service
- the Search-service-scoped role assignments

### 2. Delete the OpenAI-account-scoped role assignments

Use the deployment outputs from `roleAssignmentIds`:

```bash
az role assignment delete --ids <searchServiceOpenAiUserRoleAssignmentId>
az role assignment delete --ids <searchServiceAzureAiUserRoleAssignmentId>
az role assignment delete --ids <searchServiceCognitiveServicesUserRoleAssignmentId>
```

If you lose those values, query by the Search service principal id:

```bash
az role assignment list --assignee <search_service_principal_id> --all -o table
```

## What Comes Next

After this infra layer succeeds:

1. create Search indexes:
- `exploration/deep_dive/ai_search_index_setup.py`

2. create or verify the direct Azure AI Search project connection

3. run the direct tool probe:
- `exploration/deep_dive/agent_ai_search_probe.py`

4. only then move on to the Foundry IQ knowledge-base flow
