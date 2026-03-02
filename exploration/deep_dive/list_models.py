import json
from dataclasses import asdict

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from config import load_config


def main() -> None:
    cfg = load_config()
    credential = DefaultAzureCredential()

    with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project:
        deployments = list(project.deployments.list())

    data = {
        "config": asdict(cfg),
        "deployments": [d.as_dict() for d in deployments],
    }
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
