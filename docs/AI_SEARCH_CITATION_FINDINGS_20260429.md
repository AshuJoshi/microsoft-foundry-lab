# Azure AI Search Citation Findings (2026-04-29)

## Scope

This note captures what we found while validating the direct `AzureAISearchTool` path against the workshop-derived `hrdocs` and `healthdocs` indexes.

Relevant runs:
- `exploration/deep_dive/output/agent_ai_search_probe_20260429T003451Z.json`
- `exploration/deep_dive/output/agent_ai_search_probe_20260429T003746Z.json`
- `exploration/deep_dive/output/agent_ai_search_probe_20260429T004702Z.json`

Models tested:
- `gpt-5.1`
- `gpt-5.4`

## What Worked

Both runs passed the deterministic content checks:

- `vacation_senior` -> `4 weeks`
- `mental_health_copay` -> `$30`
- `unknown` -> `I don't know`

The output shape clearly showed the direct tool path was active:
- `azure_ai_search_call`
- `azure_ai_search_call_output`
- `message`

So retrieval was working even when citations were absent.

## What Was Missing

Both runs recorded:
- `citation_count = 0`
- `citation_urls = []`
- `citation_annotations = []`

This was consistent across:
- `gpt-5.1`
- `gpt-5.4`
- both indexes

So the missing citations were not model-specific.

## Root Cause

The original sample index schema had:
- `blob_path`

But it did **not** have an obvious citation-friendly source field such as:
- a retrievable source URL
- a retrievable source title

The direct probe only extracts message-level `url_citation` annotations. Since none were returned, the most likely explanation is that the index schema did not provide the source metadata the tool needed to surface linkable citations.

This aligns with current Learn guidance for the Azure AI Search tool, which recommends a retrievable source URL field so citations can include a link.

## Why This Nuance Matters

This is the key lesson from the experiment:

> In Azure AI Search agent grounding, "retrieval works" and "citations appear" are related but not identical outcomes.

That distinction matters because it is easy to draw the wrong conclusion from a successful answer that lacks citations.

### 1. Correct answers do not prove citation plumbing

The direct probe already proved:
- retrieval succeeded
- the answers were correct

But it did **not** prove:
- that message-level citation metadata was being surfaced

So a successful grounded answer is not enough to conclude that provenance rendering is configured correctly.

### 2. Citation output depends on index shape

If an index only exposes:
- content fields
- internal identifiers like `blob_path`

then the tool can still:
- retrieve relevant chunks
- synthesize the correct answer

while still failing to emit useful link-style citations.

The likely reason is simple:
- the response layer has no clear retrievable URL/title metadata to attach as message-level citations

### 3. Source fields are part of retrieval product design

This is not just an indexing detail.

Fields such as:
- `source_url`
- `source_title`

affect what the agent can return to the user as provenance.

So source metadata should be treated as part of the retrieval experience design, not just as optional storage decoration.

### 4. This kind of issue causes false conclusions

Before the schema patch, someone could have concluded:

> `AzureAISearchTool` does not support citations.

That would have been too strong.

The more accurate conclusion was:

> The current index schema did not appear to support message-level linkable citations.

That is a materially different diagnosis.

The first suggests:
- product/tool limitation

The second suggests:
- schema/configuration limitation

### 5. The validation has to separate three layers

This experiment only makes sense if the test distinguishes:

1. Retrieval correctness
- Did the tool retrieve enough relevant information to answer correctly?

2. Tool invocation shape
- Did the response clearly show the expected tool path, such as:
  - `azure_ai_search_call`
  - `azure_ai_search_call_output`

3. Provenance / citation rendering
- Did the final message include citation annotations and URLs?

The earlier probe runs gave us evidence for:
- Layer 1
- Layer 2

The schema patch was specifically about:
- Layer 3

### Operational takeaway

When grounded output lacks citations, do **not** immediately conclude:
- model issue
- tool failure
- SDK bug

Inspect:
- index schema
- source metadata fields
- final response annotation shape

This is the practical reasoning pattern you want to keep in mind when comparing:
- File Search
- Azure AI Search
- Foundry IQ

and when distinguishing:
- retrieval success
- source-attribution success
- product behavior
- schema/configuration behavior

## Changes Applied

To test this hypothesis, the sample data and shared index definition were patched.

### Index schema changes

Updated:
- `exploration/sample_data/fiq/index-data/index.json`

Added retrievable fields:
- `source_url`
- `source_title`

### Sample corpus changes

Updated:
- `exploration/sample_data/fiq/index-data/hrdocs-exported.jsonl`
- `exploration/sample_data/fiq/index-data/healthdocs-exported.jsonl`

Each record now includes:
- `source_url`
- `source_title`

Current mapping:
- `source_title` is derived from the file name in `blob_path`
- `source_url` is a deterministic placeholder URL based on corpus + file name

Examples:
- `https://example.com/hrdocs/employee_handbook.pdf`
- `https://example.com/healthdocs/Northwind_Health_Plus_Benefits_Details.pdf`

## Important Interpretation

Before the schema patch:

> Zero message-level citations did **not** mean Azure AI Search retrieval failed.

It meant:
- the tool call succeeded
- the answer quality was correct
- but the index schema likely did not support linkable citation output

That is the same general pattern we saw earlier with file-search provenance:
- successful retrieval does not guarantee message-level citation metadata

The difference here is that for direct `AzureAISearchTool`, the missing source URL/title fields looked like the strongest concrete reason citations were absent.

## Why The Rebuild Was Necessary

Updating local files alone was not enough.

The index schema and document payloads stored in Azure AI Search had already been created, so the remote indexes needed to be rebuilt to apply:
- the new `source_url` field
- the new `source_title` field
- the updated JSONL document records

That is why the rebuild step mattered:

```bash
uv run exploration/deep_dive/ai_search_index_setup.py --indexes hrdocs,healthdocs --log-level INFO
```

Without that rebuild:
- the repo would contain the patched schema and sample data
- but the live Search service would still be serving the old index shape

So the rebuild is part of the experiment, not just a maintenance step.

## Next Validation Step

Rebuild the indexes with the patched schema and corpus:

```bash
uv run exploration/deep_dive/ai_search_index_setup.py --indexes hrdocs,healthdocs --log-level INFO
```

Then rerun the direct probe:

```bash
uv run exploration/deep_dive/agent_ai_search_probe.py --model gpt-5.4 --cases vacation_senior,mental_health_copay,unknown --project-connection-name ai-search-direct --runs 1 --log-level INFO
```

Success criteria for the rerun:
- answers still correct
- `citation_count` becomes non-zero for at least the grounded non-`unknown` cases
- `citation_urls` point to the new `source_url` values

## Post-Patch Rerun Result

After rebuilding the indexes with the patched schema and corpus, the direct probe was rerun with:
- `gpt-5.4`

Artifact:
- `exploration/deep_dive/output/agent_ai_search_probe_20260429T004702Z.json`

### What changed

The rerun showed a real behavior change:
- `hrdocs / vacation_senior` now produced a message-level citation
- `citation_count` increased from `0` to `1` for that case

This is the strongest evidence that the schema patch mattered.

### What did not fully improve

Citation behavior remained inconsistent:
- `hrdocs / vacation_senior` -> citation present
- `healthdocs / mental_health_copay` -> still `citation_count = 0`
- `unknown` cases -> still no citations, which is acceptable

So the correct updated conclusion is:

> Citation-friendly source fields can unlock message-level citations, but citation rendering can still be selective or inconsistent across grounded cases.

### Why this still matters

The post-patch rerun validated the architectural hypothesis:
- index shape influences citation behavior

But it also showed that:
- adding source fields is necessary
- adding source fields is not always sufficient for uniform citation output

That means there is still another layer of product behavior beyond raw schema presence.

## What The Current Learn Docs Actually Say

Current Learn guidance for the direct Azure AI Search tool does **not** require literal field names such as:
- `source_url`
- `source_title`

What it does require is the capability:
- a retrievable field that contains a source URL
- optionally a title field

The practical implication is:
- the exact field names can vary
- but the index must expose retrievable source metadata if you want linkable citations

This matters because the experiment was testing for the presence of source metadata capability, not for hard-coded field-name magic.

The docs also say the index should include:
- at least one retrievable text field containing the content to cite
- a retrievable field containing a source URL

That lines up with what we changed in the sample schema.

## Query Planning vs Direct Tool Retrieval

This experiment also clarifies a second important distinction:

### Direct `AzureAISearchTool`

The direct tool path:
- queries an Azure AI Search index
- grounds the response in indexed content
- can produce inline citations when the index exposes suitable source metadata

But this path is **not** the same as Azure AI Search agentic retrieval / Foundry IQ query planning.

### Agentic retrieval / Foundry IQ

In Azure AI Search agentic retrieval and Foundry IQ:
- a supported LLM performs a reasoning step
- the LLM breaks a complex query into focused subqueries
- the system executes and merges those subqueries

This is the "query planning" layer described in current Azure AI Search agentic retrieval docs.

So the clean distinction is:

- direct `AzureAISearchTool`:
  - grounding over an existing search index
  - no separate documented LLM-based multi-query planning layer in the flow we tested

- Foundry IQ / Azure AI Search agentic retrieval:
  - knowledge-base path
  - explicit LLM-based query planning and subquery decomposition

That means if you are asking:

> "Does Azure AI Search use GPT-style models for query planning?"

the answer is:
- **not for the direct `AzureAISearchTool` path we just validated**
- **yes for agentic retrieval / Foundry IQ knowledge-base flows**
