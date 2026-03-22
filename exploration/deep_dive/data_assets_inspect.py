#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("data_assets_inspect")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Inspect project runtime data assets: root /files, vector stores, and the files attached "
            "to the cached vector store."
        )
    )
    p.add_argument(
        "--cache-path",
        default="exploration/deep_dive/output/file_search_vector_store.json",
        help="Optional cache file produced by vector_store_index.py.",
    )
    p.add_argument("--limit-files", type=int, default=200, help="Max number of root runtime files to list.")
    p.add_argument("--limit-vector-stores", type=int, default=100, help="Max number of vector stores to list.")
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


def _load_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_outputs(run_id: str, payload: dict[str, Any], log_path: Path) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"data_assets_inspect_{run_id}.json"
    md_path = out_dir / f"data_assets_inspect_{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# Data Assets Inspect ({run_id})")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- project_endpoint: {payload['project_endpoint']}")
    lines.append(f"- cache_path: {payload['cache_path']}")
    lines.append(f"- cached_vector_store_id: {payload.get('cache', {}).get('vector_store_id', '-') if payload.get('cache') else '-'}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- root runtime files: {payload['root_files_summary']['count']}")
    lines.append(f"- unique filenames: {payload['root_files_summary']['unique_filenames']}")
    lines.append(f"- vector stores: {payload['vector_stores_summary']['count']}")
    lines.append("")
    lines.append("## Root Files By Filename")
    lines.append("")
    lines.append("| Filename | Count | Purposes | Statuses |")
    lines.append("|---|---:|---|---|")
    for row in payload["root_files_by_filename"]:
        lines.append(
            f"| `{row['filename']}` | {row['count']} | {', '.join(row['purposes'])} | {', '.join(row['statuses'])} |"
        )
    lines.append("")
    lines.append("## Vector Stores By Name")
    lines.append("")
    lines.append("| Name | Count | IDs |")
    lines.append("|---|---:|---|")
    for row in payload["vector_stores_by_name"]:
        lines.append(f"| `{row['name']}` | {row['count']} | {', '.join(row['ids'])} |")
    lines.append("")
    lines.append("## Cached Vector Store Files")
    lines.append("")
    if not payload["cached_vector_store_files"]:
        lines.append("No cached vector store file entries captured.")
    else:
        lines.append("| File ID | Status | Usage Bytes |")
        lines.append("|---|---|---:|")
        for row in payload["cached_vector_store_files"]:
            lines.append(f"| `{row.get('id', '-')}` | {row.get('status', '-')} | {row.get('usage_bytes', '-') or '-'} |")
    lines.append("")
    lines.append("## Orphan Root Files By Filename")
    lines.append("")
    if not payload["orphans_by_filename"]:
        lines.append("No orphan root files detected.")
    else:
        lines.append("| Filename | Count | File IDs |")
        lines.append("|---|---:|---|")
        for row in payload["orphans_by_filename"]:
            lines.append(f"| `{row['filename']}` | {row['count']} | {', '.join(row['file_ids'])} |")
    lines.append("")
    lines.append("## Vector Store File Coverage")
    lines.append("")
    for row in payload["vector_store_files"]:
        lines.append(f"### `{row['name']}` / `{row['vector_store_id']}`")
        lines.append("")
        if not row["files"]:
            lines.append("No attached files captured.")
            lines.append("")
            continue
        lines.append("| File ID | Status | Usage Bytes |")
        lines.append("|---|---|---:|")
        for file_row in row["files"]:
            lines.append(f"| `{file_row.get('id', '-')}` | {file_row.get('status', '-')} | {file_row.get('usage_bytes', '-') or '-'} |")
        lines.append("")
    lines.append(f"- log: `{log_path}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path


def main() -> None:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"data_assets_inspect_{run_id}.log"
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    cfg = load_config()
    cache_path = Path(args.cache_path)
    cache = _load_cache(cache_path)
    cached_vector_store_id = str(cache.get("vector_store_id")) if cache and cache.get("vector_store_id") else None

    logger.info("project_endpoint=%s", cfg.project_endpoint)
    logger.info("cache_path=%s", cache_path)
    logger.info("cached_vector_store_id=%s", cached_vector_store_id or "-")

    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):
        root_files = list(openai_client.files.list(limit=args.limit_files))
        vector_stores = list(openai_client.vector_stores.list(limit=args.limit_vector_stores))

        cached_vector_store = None
        cached_vector_store_files: list[Any] = []
        if cached_vector_store_id:
            try:
                cached_vector_store = openai_client.vector_stores.retrieve(cached_vector_store_id)
                cached_vector_store_files = list(openai_client.vector_stores.files.list(vector_store_id=cached_vector_store_id, limit=100))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to inspect cached vector store %s: %s", cached_vector_store_id, exc)

        vector_store_file_map: dict[str, list[Any]] = {}
        for vector_store in vector_stores:
            vector_store_id = str(getattr(vector_store, "id", ""))
            if not vector_store_id:
                continue
            try:
                vector_store_file_map[vector_store_id] = list(
                    openai_client.vector_stores.files.list(vector_store_id=vector_store_id, limit=100)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to list files for vector_store_id=%s error=%s", vector_store_id, exc)
                vector_store_file_map[vector_store_id] = []

    root_file_rows = []
    root_file_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in root_files:
        row = {
            "id": getattr(item, "id", None),
            "filename": getattr(item, "filename", None),
            "purpose": getattr(item, "purpose", None),
            "status": getattr(item, "status", None),
            "created_at": getattr(item, "created_at", None),
        }
        root_file_rows.append(row)
        root_file_groups[str(row["filename"])].append(row)

    grouped_files = []
    for filename, items in sorted(root_file_groups.items()):
        grouped_files.append(
            {
                "filename": filename,
                "count": len(items),
                "file_ids": [row["id"] for row in items],
                "purposes": sorted({str(row["purpose"]) for row in items}),
                "statuses": sorted({str(row["status"]) for row in items}),
            }
        )

    vector_store_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    vector_store_rows = []
    for item in vector_stores:
        row = {
            "id": getattr(item, "id", None),
            "name": getattr(item, "name", None),
            "status": getattr(item, "status", None),
            "created_at": getattr(item, "created_at", None),
            "file_counts": _plain(getattr(item, "file_counts", None)),
        }
        vector_store_rows.append(row)
        vector_store_groups[str(row["name"])].append(row)

    vector_store_group_rows = []
    for name, items in sorted(vector_store_groups.items()):
        vector_store_group_rows.append(
            {
                "name": name,
                "count": len(items),
                "ids": [str(item["id"]) for item in items],
            }
        )

    cached_vector_store_file_rows = []
    for item in cached_vector_store_files:
        cached_vector_store_file_rows.append(
            {
                "id": getattr(item, "id", None),
                "status": getattr(item, "status", None),
                "usage_bytes": getattr(item, "usage_bytes", None),
                "last_error": _plain(getattr(item, "last_error", None)),
                "attributes": _plain(getattr(item, "attributes", None)),
            }
        )

    vector_store_file_rows: list[dict[str, Any]] = []
    attached_root_file_ids: set[str] = set()
    for vector_store in vector_store_rows:
        store_id = str(vector_store["id"])
        entries = vector_store_file_map.get(store_id, [])
        file_rows = []
        for item in entries:
            file_id = str(getattr(item, "id", "") or "")
            if file_id:
                attached_root_file_ids.add(file_id)
            file_rows.append(
                {
                    "id": getattr(item, "id", None),
                    "status": getattr(item, "status", None),
                    "usage_bytes": getattr(item, "usage_bytes", None),
                    "last_error": _plain(getattr(item, "last_error", None)),
                    "attributes": _plain(getattr(item, "attributes", None)),
                }
            )
        vector_store_file_rows.append(
            {
                "vector_store_id": store_id,
                "name": vector_store["name"],
                "file_count": len(file_rows),
                "files": file_rows,
            }
        )

    orphan_root_files = [row for row in root_file_rows if str(row["id"]) not in attached_root_file_ids]
    orphan_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in orphan_root_files:
        orphan_groups[str(row["filename"])].append(row)
    orphan_group_rows = []
    for filename, items in sorted(orphan_groups.items()):
        orphan_group_rows.append(
            {
                "filename": filename,
                "count": len(items),
                "file_ids": [row["id"] for row in items],
            }
        )

    payload = {
        "project_endpoint": cfg.project_endpoint,
        "cache_path": str(cache_path),
        "cache": cache,
        "root_files_summary": {
            "count": len(root_file_rows),
            "unique_filenames": len(grouped_files),
        },
        "root_files_by_filename": grouped_files,
        "root_files": root_file_rows,
        "vector_stores_summary": {
            "count": len(vector_store_rows),
        },
        "vector_stores_by_name": vector_store_group_rows,
        "vector_stores": vector_store_rows,
        "vector_store_files": vector_store_file_rows,
        "cached_vector_store": _plain(cached_vector_store) if cached_vector_store is not None else None,
        "cached_vector_store_files": cached_vector_store_file_rows,
        "attached_root_file_ids": sorted(attached_root_file_ids),
        "orphan_root_files": orphan_root_files,
        "orphans_by_filename": orphan_group_rows,
    }
    md_path, json_path = _write_outputs(run_id, payload, log_path)
    logger.info("Artifacts markdown=%s json=%s log=%s", md_path, json_path, log_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
