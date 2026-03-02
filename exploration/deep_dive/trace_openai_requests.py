import json
import os

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from config import load_config


def main() -> None:
    cfg = load_config()
    os.environ.setdefault("AZURE_AI_PROJECTS_CONSOLE_LOGGING", "fals")

    deployment_name = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-5.2")
    prompt = os.getenv("TEST_PROMPT", "Return exactly: sdk-trace-ok")

    credential = DefaultAzureCredential()
    with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project:
        openai_client = project.get_openai_client()
        response = openai_client.responses.create(
            model=deployment_name,
            input=prompt,
            max_output_tokens=30,
        )

    print("\n=== Parsed Response (truncated) ===")
    print(
        json.dumps(
            {
                "id": getattr(response, "id", None),
                "model": getattr(response, "model", None),
                "status": getattr(response, "status", None),
                "output_text": getattr(response, "output_text", None),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
