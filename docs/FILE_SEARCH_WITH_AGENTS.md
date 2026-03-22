# File Search With Agents

This document explains the file-search exploration workflow in this repo, what part of the stack each script exercises, and how to interpret the results.

## What This Workflow Tests

This workflow tests **Foundry Agent Service file search** in the **unpublished project-scoped agent path**.

It does **not** test:
- published agent applications
- direct non-agent Responses file-search usage
- web-search or Bing-grounding tools

It specifically tests:
1. creating and reusing a **vector store**
2. uploading files into that vector store
3. attaching `FileSearchTool` to a **Prompt agent**
4. invoking that agent through the project runtime
5. inspecting both:
   - the model answer
   - the retrieved file-search results returned by the runtime

## Architecture

There are two layers in the current Python implementation:

1. **Foundry SDK (`azure-ai-projects`)**
- used for agent management
- used to construct `FileSearchTool`
- used to create and delete Prompt agents

2. **OpenAI-compatible project runtime client**
- obtained from `project_client.get_openai_client()`
- used for:
  - `vector_stores.create(...)`
  - `vector_stores.files.upload_and_poll(...)`
  - `conversations.create(...)`
  - `responses.create(...)`

This is still the **Foundry project runtime** because the OpenAI-compatible client is pointed at the Foundry project endpoint:

`https://<resource>.services.ai.azure.com/api/projects/<project>/openai/v1/...`

So the runtime path is:
- Foundry-managed
- project-scoped
- stateful if you reuse a conversation
- OpenAI-compatible at the protocol layer

## Scripts

### 1. Build Or Reuse A Vector Store

`exploration/deep_dive/vector_store_index.py`

Purpose:
- create a vector store
- upload files into it
- cache the `vector_store_id`

Supports:
- explicit file paths via `--files`
- tracked sample corpora via `--sample-corpus`

Current tracked sample corpus:
- `invoices`

Example:

```bash
uv run exploration/deep_dive/vector_store_index.py --sample-corpus invoices --log-level INFO
```

Cache file:

`exploration/deep_dive/output/file_search_vector_store.json`

That cache stores:
- `vector_store_id`
- `vector_store_name`
- `sample_corpus`
- uploaded file paths
- project endpoint

### 2. Inspect The Vector Store

`exploration/deep_dive/vector_store_inspect.py`

Purpose:
- read the cached `vector_store_id`
- retrieve vector store metadata
- list file entries attached to the vector store

Example:

```bash
uv run exploration/deep_dive/vector_store_inspect.py --log-level INFO
```

Use this to confirm:
- the vector store exists
- the expected files are attached

### 3. Probe File Search Through An Agent

`exploration/deep_dive/agent_file_search_probe.py`

Purpose:
- create a temporary Prompt agent
- attach `FileSearchTool(vector_store_ids=[...])`
- create a temporary conversation
- run deterministic file-search test cases
- persist artifacts
- clean up the temporary agent and conversation

This script **does not delete the vector store**.

Example:

```bash
uv run exploration/deep_dive/agent_file_search_probe.py --model gpt-5.1 --cases vendor,total_due,highest_total --runs 1 --log-level INFO
```

### 4. Delete The Vector Store

`exploration/deep_dive/vector_store_delete.py`

Purpose:
- delete the cached vector store
- optionally remove the local cache file

Example:

```bash
uv run exploration/deep_dive/vector_store_delete.py --yes --log-level INFO
```

## Tracked Sample Corpus

Tracked sample files live under:

`exploration/sample_data/invoices/`

These files were chosen because they are easy to validate:
- stable invoice IDs
- stable PO numbers
- clear vendor names
- exact amounts
- exact due dates
- short quotable lines

Examples:
- `INV-1002` total due is `$565.00`
- `INV-1004` vendor is `Northwind IT Services`
- `INV-1001` includes the line item `Ink Cartridge (Black)`

This makes the probe deterministic enough to validate retrieval quality without relying on vague prompts.

## Current Probe Cases

`agent_file_search_probe.py` currently includes:

- `ids`
  - list invoice IDs and PO numbers
- `vendor`
  - vendor lookup for `INV-1004`
- `total_due`
  - amount lookup for `INV-1002`
- `highest_total`
  - cross-file comparison
- `quote_ink`
  - targeted quote plus source file
- `summary`
  - corpus-wide summary

Each case has:
- expected marker strings
- matched marker tracking
- `expectation_met` in the output artifact

## What The Response Looks Like

The runtime response contains two relevant output items:

1. `message`
- the final natural-language answer

2. `file_search_call`
- the tool-call output item for file search

The most important finding from this exploration is:

> file-source provenance is not guaranteed to appear as a `message` citation annotation on every response shape.

Instead, source information may show up in **two different places**.

### A. Message Annotations

In some cases, the message output includes file citation annotations such as:
- `file_citation`
- `container_file_citation`

Example observed:
- `quote_ink`
  - returned a `file_citation`
  - with:
    - `filename = invoice_INV-1001.txt`
    - `file_id = ...`

### B. File Search Tool Results

The runtime also exposes a dedicated file-search tool-call item:
- `type = "file_search_call"`

That item can include:
- `results`

Each result contains fields like:
- `file_id`
- `filename`
- `score`
- `text`

This source information is only returned if the request includes:

```python
include=["file_search_call.results"]
```

That is now enabled in `agent_file_search_probe.py`.

## Why Earlier Runs Seemed To Have "No Citations"

Earlier runs only inspected URL-style annotations.

That was incomplete for file search.

What actually happened:
- retrieval was working
- but message-level citations were absent for some answer shapes
- and the authoritative retrieved-source information was sitting in:
  - `file_search_call.results`

So:
- absence of `message` citations is **not enough** to conclude file search failed
- for file search, the reliable provenance signal is:
  - `file_search_call.results`

## What To Look For In Artifacts

JSON and markdown artifacts are written under:

`exploration/deep_dive/output/`

Current artifact prefix:

`agent_file_search_probe_<run_id>`

Key fields to inspect:

- `expectation_met`
- `matched_markers`
- `citation_annotations`
- `citation_files`
- `file_search_results`
- `output_item_types`

Interpretation:

- `expectation_met = true`
  - the answer included the expected exact markers
- `citation_annotations`
  - message-level citations, when present
- `file_search_results`
  - retrieved files/chunks returned by the tool-call item
- `output_item_types`
  - should include:
    - `file_search_call`
    - `message`

## Current Findings

From the invoice sample runs:

1. File search is working through the agent path.
- exact invoice lookup works
- cross-file comparison works
- quote extraction works

2. Message-level file citations are inconsistent.
- present for some prompts, such as `quote_ink`
- absent for others, such as `vendor` and `total_due`

3. `file_search_call.results` is the reliable source-provenance channel.
- it shows retrieved filenames
- scores
- file IDs
- retrieved text

So the working rule is:

> For file search, treat `file_search_call.results` as the primary retrieval provenance signal. Treat message-level file citation annotations as optional/additional metadata.

## Recommended Usage Sequence

1. Create or reuse the vector store:

```bash
uv run exploration/deep_dive/vector_store_index.py --sample-corpus invoices --log-level INFO
```

2. Inspect the vector store:

```bash
uv run exploration/deep_dive/vector_store_inspect.py --log-level INFO
```

3. Run targeted validation:

```bash
uv run exploration/deep_dive/agent_file_search_probe.py --model gpt-5.1 --cases vendor,total_due,highest_total --runs 1 --log-level INFO
```

4. Run quote/source validation:

```bash
uv run exploration/deep_dive/agent_file_search_probe.py --model gpt-5.1 --cases quote_ink,ids,summary --runs 1 --log-level INFO
```

5. Delete the vector store when finished:

```bash
uv run exploration/deep_dive/vector_store_delete.py --yes --log-level INFO
```

## Commit Scope

The intended commit set for this workflow is:

- `exploration/sample_data/invoices/invoice_INV-1001.txt`
- `exploration/sample_data/invoices/invoice_INV-1002.txt`
- `exploration/sample_data/invoices/invoice_INV-1003.txt`
- `exploration/sample_data/invoices/invoice_INV-1004.txt`
- `exploration/sample_data/invoices/invoice_INV-1005.txt`
- `exploration/deep_dive/vector_store_index.py`
- `exploration/deep_dive/vector_store_inspect.py`
- `exploration/deep_dive/vector_store_delete.py`
- `exploration/deep_dive/agent_file_search_probe.py`
- `exploration/README.md`

Optional:
- this document

