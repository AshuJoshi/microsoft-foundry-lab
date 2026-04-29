#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("ai_search_index_setup")

DEFAULT_DATA_DIR = REPO_ROOT / "exploration" / "sample_data" / "fiq" / "index-data"
DEFAULT_INDEX_DEF = DEFAULT_DATA_DIR / "index.json"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "output" / "ai_search_indexes_cache.json"


@dataclass
class IndexSetupRecord:
    index_name: str
    records_file: str
    success: bool
    document_count: int
    error_type: str | None = None
    error_message: str | None = None


INDEX_FILES = {
    "hrdocs": "hrdocs-exported.jsonl",
    "healthdocs": "healthdocs-exported.jsonl",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create or update Azure AI Search indexes and upload the FIQ sample corpora.")
    p.add_argument("--indexes", default="hrdocs,healthdocs", help="Comma-separated Azure AI Search index names to build.")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing index.json and exported JSONL files.")
    p.add_argument("--index-definition", default=str(DEFAULT_INDEX_DEF), help="Path to the Azure AI Search index definition JSON file.")
    p.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH), help="Path to write the latest index cache JSON.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _setup_logging(level_name: str, log_path: Path) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _write_outputs(run_id: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"ai_search_index_setup_{run_id}.json"
    md_path = out_dir / f"ai_search_index_setup_{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# AI Search Index Setup ({run_id})")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for k, v in payload["metadata"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Index | Records File | Success | Documents | Error |")
    lines.append("|---|---|---|---:|---|")
    for rec in payload["records"]:
        err = "" if rec["success"] else f"{rec['error_type']}: {rec['error_message']}"
        err = err.replace("|", "\\|")
        lines.append(f"| {rec['index_name']} | {rec['records_file']} | {'yes' if rec['success'] else 'no'} | {rec['document_count']} | {err} |")
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def _load_records(records_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:  # noqa: PERF203
                raise ValueError(f"Invalid JSON in {records_path.name} at line {line_num}: {exc}") from exc
    return records


async def _create_or_update_index(index_client: SearchIndexClient, index_definition_path: Path, index_name: str, azure_openai_endpoint: str) -> None:
    index_data = json.loads(index_definition_path.read_text(encoding="utf-8"))
    index = SearchIndex.deserialize(index_data)
    index.name = index_name
    vector_search = getattr(index, "vector_search", None)
    vectorizers = getattr(vector_search, "vectorizers", None) if vector_search else None
    if vectorizers:
        try:
            vectorizers[0].parameters.resource_url = azure_openai_endpoint
        except Exception:  # noqa: BLE001
            logger.warning("Could not patch vectorizer resource_url for index=%s", index_name)
    await index_client.create_or_update_index(index)


async def _upload_documents(search_client: SearchClient, records: list[dict[str, Any]], batch_size: int = 100) -> int:
    uploaded = 0
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        await search_client.upload_documents(documents=chunk)
        uploaded += len(chunk)
    return uploaded


async def _run_index(index_name: str, records_file: str, index_definition_path: Path, data_dir: Path, endpoint: str, credential: AzureKeyCredential, azure_openai_endpoint: str) -> IndexSetupRecord:
    records_path = data_dir / records_file
    try:
        if not records_path.exists():
            raise FileNotFoundError(f"Records file not found: {records_path}")
        if not index_definition_path.exists():
            raise FileNotFoundError(f"Index definition not found: {index_definition_path}")
        records = _load_records(records_path)
        async with SearchIndexClient(endpoint=endpoint, credential=credential) as index_client:
            await _create_or_update_index(index_client, index_definition_path, index_name, azure_openai_endpoint)
        async with SearchClient(endpoint=endpoint, index_name=index_name, credential=credential) as search_client:
            uploaded = await _upload_documents(search_client, records)
        return IndexSetupRecord(index_name=index_name, records_file=records_file, success=True, document_count=uploaded)
    except Exception as exc:  # noqa: BLE001
        return IndexSetupRecord(
            index_name=index_name,
            records_file=records_file,
            success=False,
            document_count=0,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


async def main_async() -> int:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"ai_search_index_setup_{run_id}.log"
    _setup_logging(args.log_level, log_path)
    _ = load_config()

    endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT", "")
    admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY", "")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    if not endpoint or not admin_key or not azure_openai_endpoint:
        raise SystemExit("Missing required env vars: AZURE_SEARCH_SERVICE_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, AZURE_OPENAI_ENDPOINT")

    data_dir = Path(args.data_dir)
    index_definition_path = Path(args.index_definition)
    cache_path = Path(args.cache_path)
    requested = [name.strip() for name in args.indexes.split(",") if name.strip()]
    unknown = [name for name in requested if name not in INDEX_FILES]
    if unknown:
        raise SystemExit(f"Unsupported index names: {', '.join(unknown)}. Valid: {', '.join(sorted(INDEX_FILES))}")

    logger.info("endpoint=%s", endpoint)
    logger.info("indexes=%s", requested)
    logger.info("data_dir=%s", data_dir)

    credential = AzureKeyCredential(admin_key)
    records: list[IndexSetupRecord] = []
    for index_name in requested:
        logger.info("START index=%s", index_name)
        rec = await _run_index(index_name, INDEX_FILES[index_name], index_definition_path, data_dir, endpoint, credential, azure_openai_endpoint)
        records.append(rec)
        if rec.success:
            logger.info("DONE  index=%s documents=%s", rec.index_name, rec.document_count)
        else:
            logger.warning("FAIL  index=%s error=%s", rec.index_name, rec.error_message)

    payload = {
        "metadata": {
            "run_id": run_id,
            "search_endpoint": endpoint,
            "data_dir": str(data_dir),
            "index_definition": str(index_definition_path),
            "azure_openai_endpoint": azure_openai_endpoint,
        },
        "records": [asdict(r) for r in records],
    }
    md_path, json_path = _write_outputs(run_id, payload)

    cache_payload = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "search_endpoint": endpoint,
        "index_definition": str(index_definition_path),
        "indexes": [r.index_name for r in records if r.success],
        "records": [asdict(r) for r in records],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
    logger.info("Artifacts markdown=%s json=%s log=%s cache=%s", md_path, json_path, log_path, cache_path)
    return 0 if all(r.success for r in records) else 1


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
