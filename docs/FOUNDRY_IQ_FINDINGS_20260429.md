# Foundry IQ Findings 2026-04-29

This note captures the validated Foundry IQ path in this repo after the direct Azure AI Search baseline was already working.

## Summary

The Foundry IQ path is now working end-to-end for this repo:

1. Azure AI Search indexes exist and are populated
2. Azure AI Search knowledge sources can be created over those indexes
3. A Search knowledge base can be created
4. The knowledge base exposes an MCP endpoint
5. A Foundry project connection can target that MCP endpoint with `ProjectManagedIdentity`
6. A Prompt agent using `MCPTool` can answer correctly through the knowledge base

Validated artifacts:
- `exploration/deep_dive/output/fiq_knowledge_setup_20260429T035333Z.json`
- `exploration/deep_dive/output/fiq_project_connection_setup_20260429T035440Z.json`
- `exploration/deep_dive/output/agent_foundry_iq_probe_20260429T035550Z.json`

## Managed Identity Findings

### 1. UAMI creation and RBAC can be handled declaratively

The separate infra module works for:
- creating the user-assigned managed identity
- assigning the Foundry / OpenAI account roles needed for outbound model access

Validated outputs:
- `userAssignedIdentityName = fj-fiq-search-uami`
- `userAssignedIdentityPrincipalId = 8d7ae723-ba02-4733-b269-dfe639900bc0`
- `userAssignedIdentityClientId = 9a341dbf-61ab-4e78-b91b-0803c3d6e5b1`

### 2. Search-service UAMI association needed the Search Management Update API

The original attempt to associate the UAMI through a full Search ARM/Bicep resource update failed with:
- `Resource identity type 'SystemAssigned, UserAssigned' is invalid`

However, the Search Management Update API path worked:
- `infra/ai-search-uami-for-fiq/attach_search_uami.py`

Validated artifact:
- `exploration/deep_dive/output/attach_search_uami_20260429T034036Z.json`

Validated result:
- `existing_identity_type = SystemAssigned`
- `new_identity_type = SystemAssigned, UserAssigned`
- `success = true`

This is the important control-plane finding:
- the safe working split is:
  - Bicep for UAMI creation and AOAI/Foundry RBAC
  - Search Management Update/PATCH for Search-service UAMI attachment

## Knowledge Base Findings

### 1. Knowledge sources worked cleanly

Validated knowledge sources:
- `hrdocs-knowledge-source`
- `healthdocs-knowledge-source`

Observed status:
- first create: `201`
- later idempotent updates: `204`

### 2. Managed-identity knowledge-base model config works

`exploration/deep_dive/fiq_knowledge_setup.py` was patched to support:
- `--auth-mode managed-identity`

The working model config uses:
- `azureOpenAIParameters.authIdentity`
- `@odata.type = #Microsoft.Azure.Search.DataUserAssignedIdentity`
- `userAssignedIdentity = <UAMI resource id>`

This means the Search knowledge base can use a user-assigned managed identity for outbound model access instead of an API key.

### 3. The MCP endpoint was created as expected

Validated MCP endpoint:
- `https://fj-ai-search-01.search.windows.net/knowledgebases/zava-agentic-kb/mcp?api-version=2025-11-01-preview`

## Foundry Project Connection Findings

The project connection for the MCP endpoint worked with:
- `authType = ProjectManagedIdentity`
- `category = RemoteTool`
- `audience = https://search.azure.com/`

Validated artifact:
- `exploration/deep_dive/output/fiq_project_connection_setup_20260429T035440Z.json`

Validated connection:
- name: `fiq-knowledge-base`
- target: KB MCP endpoint above

## Agent Probe Findings

The MCP-backed Foundry IQ probe worked end-to-end.

Validated artifact:
- `exploration/deep_dive/output/agent_foundry_iq_probe_20260429T035550Z.json`

### Case results

1. `hr_policy`
- answer: `4 weeks`
- citation count: `1`
- expectation met: `true`

2. `health_fact`
- answer: `$30`
- citation count: `1`
- expectation met: `true`

3. `cross_source_compare`
- answer contained:
  - `Innovator of the Month`
  - `$35`
- citation count: `2`
- expectation met: `true`

4. `unknown`
- answer: `I don't know`
- citation count: `0`
- expectation met: `true`

### Tool-shape evidence

The successful FIQ run used MCP-backed retrieval rather than direct AI Search tool calls.

Observed output item sequence:
- `mcp_list_tools`
- `reasoning`
- `mcp_call`
- `reasoning`
- `message`

That confirms the retrieval path is:
- Foundry agent -> MCPTool -> Search knowledge base MCP endpoint

## Citation Findings Relative To Direct Azure AI Search

Compared with the earlier direct `AzureAISearchTool` baseline, the FIQ path showed more consistent grounded citation behavior in the validated cases:

- `hr_policy`: citation present
- `health_fact`: citation present
- `cross_source_compare`: two citations present
- `unknown`: no citation, which is appropriate

The citations still point at the MCP endpoint / doc labels rather than polished final source URLs, but the provenance behavior is more consistent than the direct AI Search baseline that was investigated earlier.

## Final Conclusion

The repo now has two validated retrieval paths:

1. Direct Azure AI Search
- external Search index
- direct Foundry project connection
- `AzureAISearchTool`

2. Foundry IQ
- Search indexes
- knowledge sources
- knowledge base
- MCP endpoint
- project connection
- `MCPTool`

The key architectural takeaway is:
- Foundry IQ is not just “Azure AI Search with a different name”
- it is a layered MCP-backed retrieval system built on top of Azure AI Search assets
