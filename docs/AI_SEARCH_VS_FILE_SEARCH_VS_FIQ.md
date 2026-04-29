# AI Search Vs. File Search Vs. Foundry IQ

This note separates three retrieval surfaces that look similar in the portal but are not the same runtime path.

## FileSearchTool

What it uses:
- Foundry/OpenAI runtime root `/files`
- runtime `vector_stores`
- `FileSearchTool(vector_store_ids=[...])`

How data gets in:
- upload files into the project runtime
- create a vector store
- attach the vector store to a Prompt agent

What it is best for:
- project-local file retrieval
- quick controlled corpora
- simple demonstrations

What we observed already:
- portal `Data > Datasets` aligns closely with durable runtime `/files`
- portal `Knowledge > Foundry IQ > Indexes` showed managed vector-store-backed indexes
- deleting a vector store does not delete the underlying runtime file assets

Important boundary:
- these managed indexes are specific to the File Search path
- they are not the same thing as external Azure AI Search indexes

## AzureAISearchTool

What it uses:
- an external Azure AI Search index
- a Foundry project connection that points at the Azure AI Search resource
- `AzureAISearchTool` with `AISearchIndexResource`

How data gets in:
- create and populate indexes through Azure AI Search APIs or SDKs
- do not use runtime `/files` or `vector_stores`

What it is best for:
- existing enterprise search indexes
- direct retrieval over Azure AI Search without the Foundry IQ knowledge-base layer

Important boundary:
- the data/index lifecycle belongs to Azure AI Search
- Foundry Agent Service only consumes the configured search index through the tool surface

## Foundry IQ

What it uses:
- Azure AI Search indexes underneath
- Azure AI Search knowledge sources
- Azure AI Search knowledge bases
- an MCP endpoint automatically exposed by the knowledge base
- a Foundry project connection to that MCP endpoint
- `MCPTool` on the agent side

How data gets in:
- build and populate Azure AI Search indexes first
- create knowledge sources over those indexes
- create a knowledge base over the knowledge sources

What it is best for:
- agentic retrieval over one or more knowledge sources
- MCP-based consumption by agents and tools
- richer retrieval orchestration than a single direct search index attachment

Important boundary:
- this is not `AzureAISearchTool`
- the agent consumes the knowledge base through MCP, not through a direct AI Search tool binding

## Layering Summary

1. `FileSearchTool`
- runtime file assets
- runtime vector stores
- Foundry/OpenAI project runtime path

2. `AzureAISearchTool`
- Azure AI Search index
- Foundry agent uses a direct tool binding to the external search index

3. Foundry IQ
- Azure AI Search indexes -> knowledge sources -> knowledge base -> MCP endpoint
- Foundry agent uses `MCPTool` through a project connection

## Why The Portal Can Be Confusing

The portal uses similar language for:
- datasets
- files
- indexes
- Foundry IQ
- AI Search

Those labels span multiple layers:
- project runtime assets
- managed vector-store indexes
- external Azure AI Search indexes
- knowledge-base abstractions over Azure AI Search

The correct mental model is to ask:
- where is the underlying data stored?
- who owns the index lifecycle?
- which agent tool consumes it?

That is the distinction this repo now uses:
- `FileSearchTool` != `AzureAISearchTool` != Foundry IQ
