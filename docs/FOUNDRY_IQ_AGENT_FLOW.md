# Foundry IQ Agent Flow

This note captures the practical flow implemented from the workshop sample under `samplecode/fiq/`.

## End-To-End Path

1. Create or refresh Azure AI Search indexes
- example indexes:
  - `hrdocs`
  - `healthdocs`

2. Create Azure AI Search knowledge sources
- one knowledge source per index

3. Create an Azure AI Search knowledge base
- current workshop pattern uses:
  - `outputMode = extractiveData`

4. Use the knowledge base MCP endpoint
- format:
  - `{search_endpoint}/knowledgebases/{knowledge_base_name}/mcp?api-version=2025-11-01-preview`

5. Create a Foundry project connection to that MCP endpoint
- current workshop pattern uses:
  - `authType = ProjectManagedIdentity`
  - `category = RemoteTool`
  - `audience = https://search.azure.com/`

6. Create a Foundry Prompt agent with `MCPTool`
- restrict the MCP surface to:
  - `knowledge_base_retrieve`

7. Run the agent through project `responses.create(...)`
- use a conversation
- use `extra_body={"agent_reference": ...}`

## Why This Is Not `AzureAISearchTool`

`AzureAISearchTool`:
- attaches a single Azure AI Search index resource directly to the agent

Foundry IQ path:
- builds a knowledge base on top of one or more Azure AI Search knowledge sources
- exposes the knowledge base as MCP
- the agent consumes the MCP endpoint through `MCPTool`

So the Foundry IQ path is:
- broader
- more layered
- more explicit about knowledge-base retrieval orchestration

## Why This Is Not `FileSearchTool`

`FileSearchTool`:
- uses runtime `/files`
- uses runtime `vector_stores`
- uses managed vector-store-backed retrieval

Foundry IQ:
- does not depend on runtime `/files`
- does not depend on runtime `vector_stores`
- uses Azure AI Search and MCP

## Practical Repo Mapping

Phase 1:
- `exploration/deep_dive/ai_search_index_setup.py`
- `exploration/deep_dive/agent_ai_search_probe.py`

Phase 2:
- `exploration/deep_dive/fiq_knowledge_setup.py`
- `exploration/deep_dive/fiq_project_connection_setup.py`
- `exploration/deep_dive/agent_foundry_iq_probe.py`

This keeps the learning path explicit:
- direct Azure AI Search tool first
- Foundry IQ on top of that second

## Validated Managed-Identity Path

The managed-identity path that worked in this repo is:

1. Create a user-assigned managed identity
2. Grant that UAMI outbound Foundry / OpenAI RBAC
3. Attach the UAMI to the Search service
4. Create the Search knowledge base using:
   - `azureOpenAIParameters.authIdentity`
   - `#Microsoft.Azure.Search.DataUserAssignedIdentity`
5. Create the Foundry project connection to the KB MCP endpoint
6. Run the agent through `MCPTool`

Important implementation detail:
- UAMI creation + RBAC worked well as separate infra
- Search-service UAMI association worked through the Search Management Update API helper
- the original full Search ARM/Bicep identity update path was not the working control-plane path in this environment

Validated run artifacts:
- `exploration/deep_dive/output/fiq_knowledge_setup_20260429T035333Z.json`
- `exploration/deep_dive/output/fiq_project_connection_setup_20260429T035440Z.json`
- `exploration/deep_dive/output/agent_foundry_iq_probe_20260429T035550Z.json`
