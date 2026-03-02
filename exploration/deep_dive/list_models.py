import json
from dataclasses import asdict
from pathlib import Path
import sys

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import is_embedding_deployment, load_config


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
