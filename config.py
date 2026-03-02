import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass(frozen=True)
class FoundryConfig:
    subscription_id: str
    resource_group: str
    resource_name: str
    project_name: str
    project_endpoint: str
    default_model_deployment_name: str
    agent_name_prefix: str

    @property
    def account_name(self) -> str:
        # Backward-compat alias for older scripts/docs that used "account_name".
        return self.resource_name


REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")


def _default_project_endpoint(resource_name: str, project_name: str) -> str:
    return f"https://{resource_name}.services.ai.azure.com/api/projects/{project_name}"


def load_config() -> FoundryConfig:
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID", "")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP", "")
    resource_name = os.getenv("FOUNDRY_RESOURCE_NAME") or os.getenv("FOUNDRY_ACCOUNT_NAME", "")
    project_name = os.getenv("FOUNDRY_PROJECT_NAME", "")
    project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT") or os.getenv("FOUNDRY_PROJECT_ENDPOINT")

    if not project_endpoint and resource_name and project_name:
        project_endpoint = _default_project_endpoint(resource_name=resource_name, project_name=project_name)

    if not project_endpoint:
        raise ValueError(
            "Missing project endpoint. Set AZURE_AI_PROJECT_ENDPOINT (or FOUNDRY_PROJECT_ENDPOINT), "
            "or set FOUNDRY_RESOURCE_NAME + FOUNDRY_PROJECT_NAME."
        )

    return FoundryConfig(
        subscription_id=subscription_id,
        resource_group=resource_group,
        resource_name=resource_name,
        project_name=project_name,
        project_endpoint=project_endpoint,
        default_model_deployment_name=os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5-mini"),
        agent_name_prefix=os.getenv("AGENT_NAME_PREFIX") or os.getenv("BUGBASH_AGENT_NAME_PREFIX", "ValidationAgent"),
    )


def deployment_as_dict(deployment: Any) -> dict[str, Any]:
    if hasattr(deployment, "as_dict") and callable(getattr(deployment, "as_dict")):
        try:
            return deployment.as_dict()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(deployment, dict):
        return deployment
    return {}


def is_embedding_deployment(deployment: Any) -> bool:
    data = deployment_as_dict(deployment)
    name = (getattr(deployment, "name", None) or data.get("name") or "").lower()
    if "embedding" in name:
        return True

    blob = json.dumps(data, ensure_ascii=False).lower() if data else ""
    # Broad match is acceptable here because embedding deployments are not valid for chat/response tests.
    return "embedding" in blob


def filter_inference_deployments(deployments: list[Any]) -> tuple[list[Any], list[Any]]:
    inference: list[Any] = []
    embedding: list[Any] = []
    for d in deployments:
        if is_embedding_deployment(d):
            embedding.append(d)
        else:
            inference.append(d)
    return inference, embedding
