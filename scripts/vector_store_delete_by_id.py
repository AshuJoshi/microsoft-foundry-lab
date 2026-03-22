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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("vector_store_delete_by_id")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Delete a Foundry/OpenAI vector store by explicit id.")
    p.add_argument("--vector-store-id", required=True, help="Vector store id, e.g. vs_...")
    p.add_argument("--yes", action="store_true", help="Actually perform the delete.")
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
    if not args.yes:
        raise SystemExit("Refusing to delete without --yes.")

    cfg = load_config()
    logger.info("project_endpoint=%s", cfg.project_endpoint)
    logger.info("vector_store_id=%s", args.vector_store_id)

    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):
        delete_result = _plain(openai_client.vector_stores.delete(args.vector_store_id))

    print(json.dumps({"vector_store_id": args.vector_store_id, "delete_result": delete_result}, indent=2))


if __name__ == "__main__":
    main()
