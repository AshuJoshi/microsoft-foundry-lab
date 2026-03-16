#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("list_search_tool_resources")


@dataclass
class ConnectionRecord:
    name: str | None
    type: str | None
    target: str | None
    is_default: bool | None
    auth_type: str | None
    search_related: bool
    raw_excerpt: dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect project connections and search-related resource hints.")
    p.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
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


def _to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "as_dict"):
        try:
            return obj.as_dict()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(obj, dict):
        return obj
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def _search_related(raw: dict[str, Any]) -> bool:
    text = json.dumps(raw, default=str).lower()
    needles = [
        "bing",
        "grounding",
        "custom search",
        "custom_search",
        "web search",
        "web_search",
        "search configuration",
        "search_configuration",
    ]
    return any(n in text for n in needles)


def _raw_excerpt(raw: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "id",
        "name",
        "type",
        "target",
        "is_default",
        "isDefault",
        "credentials",
        "metadata",
        "properties",
        "authentication_type",
        "authenticationType",
    ]
    out: dict[str, Any] = {}
    for key in keep:
        if key in raw:
            out[key] = raw[key]
    return out or raw


def _record_connection(conn: Any) -> ConnectionRecord:
    raw = _to_dict(conn)
    return ConnectionRecord(
        name=raw.get("name") or getattr(conn, "name", None),
        type=str(raw.get("type") or getattr(conn, "type", None) or ""),
        target=raw.get("target") or getattr(conn, "target", None),
        is_default=raw.get("isDefault", raw.get("is_default", getattr(conn, "is_default", None))),
        auth_type=raw.get("authenticationType", raw.get("authentication_type", None)),
        search_related=_search_related(raw),
        raw_excerpt=_raw_excerpt(raw),
    )


def _write_outputs(run_id: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"list_search_tool_resources_{run_id}.json"
    md_path = out_dir / f"list_search_tool_resources_{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# Search Tool Resource Discovery ({run_id})")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for k, v in payload["metadata"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Surface Summary")
    lines.append("")
    for k, v in payload["summary"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Connections")
    lines.append("")
    lines.append("| Name | Type | Default | Search Related | Target | Auth Type |")
    lines.append("|---|---|---|---|---|---|")
    for rec in payload["connections"]:
        lines.append(
            f"| {rec['name'] or '-'} | {rec['type'] or '-'} | {rec['is_default']} | "
            f"{'yes' if rec['search_related'] else 'no'} | {rec['target'] or '-'} | {rec['auth_type'] or '-'} |"
        )
    lines.append("")
    lines.append("## Search-Related Connection Excerpts")
    lines.append("")
    related = [c for c in payload["connections"] if c["search_related"]]
    if not related:
        lines.append("No search-related connections detected from current SDK surface and connection payloads.")
        lines.append("")
    else:
        for rec in related:
            lines.append(f"### {rec['name'] or '(unnamed)'}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(rec["raw_excerpt"], indent=2))
            lines.append("```")
            lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main() -> None:
    args = parse_args()
    cfg = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "output"
    log_path = out_dir / f"list_search_tool_resources_{run_id}.log"
    _setup_logging(args.log_level, log_path)

    logger.info("run_id=%s", run_id)
    logger.info("project_endpoint=%s", cfg.project_endpoint)

    with DefaultAzureCredential() as credential, AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project:
        deployments = list(project.deployments.list())
        connections = list(project.connections.list())
        agents = list(project.agents.list())
        logger.info("deployments=%s connections=%s agents=%s", len(deployments), len(connections), len(agents))

        try:
            telemetry_conn = project.telemetry.get_application_insights_connection_string()
            app_insights_present = bool(telemetry_conn)
        except ResourceNotFoundError:
            app_insights_present = False

    connection_records = [_record_connection(c) for c in connections]
    search_related_count = sum(1 for c in connection_records if c.search_related)

    payload = {
        "metadata": {
            "run_id": run_id,
            "project_endpoint": cfg.project_endpoint,
        },
        "summary": {
            "deployment_count": len(deployments),
            "connection_count": len(connection_records),
            "agent_count": len(agents),
            "application_insights_connection_present": app_insights_present,
            "search_related_connection_count": search_related_count,
        },
        "connections": [asdict(c) for c in connection_records],
    }
    md_path, json_path = _write_outputs(run_id, payload)
    logger.info("Artifacts markdown=%s json=%s log=%s", md_path, json_path, log_path)


if __name__ == "__main__":
    main()
