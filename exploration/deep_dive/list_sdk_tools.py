#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import azure.ai.projects.models as models


@dataclass
class ToolSurface:
    class_name: str
    is_preview: bool
    signature: str
    doc_summary: str


def _doc_summary(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    if not doc:
        return ""
    return doc.strip().splitlines()[0][:180]


def main() -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[ToolSurface] = []

    for name in sorted(dir(models)):
        obj = getattr(models, name)
        if not inspect.isclass(obj):
            continue
        if "Tool" not in name:
            continue
        if name.startswith("ToolChoice") or name in {"Tool", "ToolType", "ToolDescription", "ToolProjectConnection"}:
            continue
        try:
            sig = str(inspect.signature(obj))
        except Exception:  # noqa: BLE001
            sig = "(signature unavailable)"
        rows.append(
            ToolSurface(
                class_name=name,
                is_preview=("Preview" in name),
                signature=sig,
                doc_summary=_doc_summary(obj),
            )
        )

    payload = {
        "run_id": run_id,
        "module": "azure.ai.projects.models",
        "tool_count": len(rows),
        "tools": [asdict(r) for r in rows],
    }

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"list_sdk_tools_{run_id}.json"
    md_path = out_dir / f"list_sdk_tools_{run_id}.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# SDK Tool Surfaces ({run_id})")
    lines.append("")
    lines.append(f"- module: `azure.ai.projects.models`")
    lines.append(f"- tool_count: {len(rows)}")
    lines.append("")
    lines.append("| Class | Preview | Constructor |")
    lines.append("|---|---|---|")
    for r in rows:
        lines.append(f"| `{r.class_name}` | {'yes' if r.is_preview else 'no'} | `{r.signature}` |")
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
