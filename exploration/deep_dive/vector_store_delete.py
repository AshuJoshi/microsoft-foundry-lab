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

logger = logging.getLogger("vector_store_delete")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Delete a cached vector store used for FileSearchTool experiments.")
    p.add_argument(
        "--cache-path",
        default="exploration/deep_dive/output/file_search_vector_store.json",
        help="Cache file produced by vector_store_index.py",
    )
    p.add_argument("--yes", action="store_true", help="Actually delete the vector store.")
    p.add_argument("--keep-cache", action="store_true", help="Do not remove the local cache file after deletion.")
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
    if hasattr(value, "__dict__"):
        return {key: _plain(val) for key, val in vars(value).items() if not key.startswith("_") and not callable(val)}
    return str(value)


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
    if not args.yes:
        raise SystemExit("Refusing to delete without --yes.")

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
        try:
            delete_result = _plain(openai_client.vector_stores.delete(vector_store_id))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Failed to delete vector_store_id={vector_store_id}: {exc}") from exc

    if not args.keep_cache:
        cache_path.unlink(missing_ok=True)

    payload = {
        "vector_store_id": vector_store_id,
        "delete_result": delete_result,
        "cache_removed": not args.keep_cache,
        "cache_path": str(cache_path),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
