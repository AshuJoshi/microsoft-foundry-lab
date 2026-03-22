#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("vector_store_inspect")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect a cached vector store used for FileSearchTool experiments.")
    p.add_argument(
        "--cache-path",
        default="exploration/deep_dive/output/file_search_vector_store.json",
        help="Cache file produced by vector_store_index.py",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    for attr_name in ("model_dump", "to_dict"):
        method = getattr(value, attr_name, None)
        if callable(method):
            try:
                return _plain(method())
            except Exception:  # noqa: BLE001
                pass
    if hasattr(value, "__dict__"):
        return {key: _plain(val) for key, val in vars(value).items() if not key.startswith("_") and not callable(val)}
    return str(value)


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        raise SystemExit(f"Cache file not found: {cache_path}")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    vector_store_id = str(cache["vector_store_id"])
    cfg = load_config()

    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):
        vector_store = _plain(openai_client.vector_stores.retrieve(vector_store_id))
        file_entries = [_plain(item) for item in openai_client.vector_stores.files.list(vector_store_id=vector_store_id)]

    payload = {
        "cache_path": str(cache_path),
        "vector_store_id": vector_store_id,
        "sample_corpus": cache.get("sample_corpus"),
        "cached_files": cache.get("files"),
        "vector_store": vector_store,
        "vector_store_files": file_entries,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
