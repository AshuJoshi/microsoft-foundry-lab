import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import is_embedding_deployment, load_config


def _collect_paths(obj: Any, path: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            out.append((p, v))
            out.extend(_collect_paths(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            out.append((p, v))
            out.extend(_collect_paths(v, p))
    return out


def _capability_summary(data: dict[str, Any]) -> dict[str, Any]:
    all_paths = _collect_paths(data)
    interesting = []
    for p, v in all_paths:
        lp = p.lower()
        if any(tok in lp for tok in ["capab", "api", "completion", "response", "inference"]):
            if isinstance(v, (str, int, float, bool)) or v is None:
                interesting.append({"path": p, "value": v})
            elif isinstance(v, list):
                interesting.append({"path": p, "value": v[:10]})
            elif isinstance(v, dict):
                interesting.append({"path": p, "value_keys": list(v.keys())[:20]})

    blob = json.dumps(data, ensure_ascii=False).lower() if data else ""
    api_hints = {
        "mentions_responses": "responses" in blob or "/responses" in blob,
        "mentions_chat_completions": "chat completions" in blob or "chat_completions" in blob or "/chat/completions" in blob,
        "mentions_embeddings": "embedding" in blob or "/embeddings" in blob,
    }

    direct_fields = {}
    if "capabilities" in data:
        direct_fields["capabilities"] = data.get("capabilities")
    if isinstance(data.get("properties"), dict) and "capabilities" in data.get("properties", {}):
        direct_fields["properties.capabilities"] = data["properties"]["capabilities"]

    return {
        "api_hints": api_hints,
        "direct_fields": direct_fields,
        "matched_fields": interesting[:60],
    }


def main() -> None:
    cfg = load_config()
    credential = DefaultAzureCredential()

    with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project:
        deployments = list(project.deployments.list())

    rows = []
    for d in deployments:
        data = d.as_dict()
        rows.append(
            {
                "name": d.name,
                "deployment_family": "embedding" if is_embedding_deployment(d) else "inference",
                "capability_summary": _capability_summary(data),
                "raw": data,
            }
        )

    counts = {
        "total": len(rows),
        "inference": sum(1 for r in rows if r["deployment_family"] == "inference"),
        "embedding": sum(1 for r in rows if r["deployment_family"] == "embedding"),
    }

    data = {
        "config": asdict(cfg),
        "counts": counts,
        "deployments": rows,
    }
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
