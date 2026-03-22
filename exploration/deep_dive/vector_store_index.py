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

logger = logging.getLogger("vector_store_index")

SAMPLE_CORPORA: dict[str, list[str]] = {
    "invoices": [
        "exploration/sample_data/invoices/invoice_INV-1001.txt",
        "exploration/sample_data/invoices/invoice_INV-1002.txt",
        "exploration/sample_data/invoices/invoice_INV-1003.txt",
        "exploration/sample_data/invoices/invoice_INV-1004.txt",
        "exploration/sample_data/invoices/invoice_INV-1005.txt",
    ]
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create or reuse a vector store for FileSearchTool experiments.")
    p.add_argument("--name", default="ExplorationFileSearchVectorStore", help="Vector store name when creating a new one.")
    p.add_argument("--files", help="Comma-separated file paths to upload into the vector store.")
    p.add_argument(
        "--sample-corpus",
        choices=sorted(SAMPLE_CORPORA),
        help="Use a tracked sample corpus instead of passing --files.",
    )
    p.add_argument(
        "--cache-path",
        default="exploration/deep_dive/output/file_search_vector_store.json",
        help="Where to cache the vector store id and indexed files.",
    )
    p.add_argument("--force-new", action="store_true", help="Create a new vector store even if the cached one exists.")
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


def _load_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_input_files(files_arg: str | None, sample_corpus: str | None) -> tuple[list[Path], str | None]:
    if files_arg and sample_corpus:
        raise SystemExit("Pass either --files or --sample-corpus, not both.")
    if sample_corpus:
        items = [REPO_ROOT / relpath for relpath in SAMPLE_CORPORA[sample_corpus]]
        return items, sample_corpus
    if files_arg:
        items = [Path(part.strip()) for part in files_arg.split(",") if part.strip()]
        return items, None
    raise SystemExit("Pass --files or --sample-corpus.")


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
    cfg = load_config()
    cache_path = Path(args.cache_path)
    files, sample_corpus = _resolve_input_files(args.files, args.sample_corpus)

    if not files:
        raise SystemExit("No files specified.")
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"Missing file(s): {missing}")

    file_paths = [str(p.resolve()) for p in files]
    logger.info("project_endpoint=%s", cfg.project_endpoint)
    logger.info("cache_path=%s", cache_path)
    logger.info("sample_corpus=%s", sample_corpus or "-")
    logger.info("files=%s", file_paths)

    cache = None if args.force_new else _load_cache(cache_path)

    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):
        if cache and cache.get("vector_store_id"):
            vector_store_id = str(cache["vector_store_id"])
            try:
                openai_client.vector_stores.retrieve(vector_store_id)
                logger.info("Reusing vector_store_id=%s", vector_store_id)
                print(json.dumps(cache, indent=2))
                return
            except Exception:  # noqa: BLE001
                logger.info("Cached vector store id not found; creating a new one")

        logger.info("Creating vector store name=%s", args.name)
        vector_store = openai_client.vector_stores.create(name=args.name)
        logger.info("Created vector_store_id=%s", vector_store.id)

        for idx, file_path in enumerate(files, start=1):
            logger.info("Uploading file %s/%s path=%s", idx, len(files), file_path)
            with open(file_path, "rb") as handle:
                openai_client.vector_stores.files.upload_and_poll(
                    vector_store_id=vector_store.id,
                    file=handle,
                )

        payload = {
            "vector_store_id": vector_store.id,
            "vector_store_name": args.name,
            "sample_corpus": sample_corpus,
            "files": file_paths,
            "project_endpoint": cfg.project_endpoint,
        }
        _save_cache(cache_path, payload)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
