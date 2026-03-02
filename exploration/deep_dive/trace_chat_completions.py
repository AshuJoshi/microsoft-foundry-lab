import json
import os
import sys
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config


def main() -> None:
    cfg = load_config()
    os.environ.setdefault('AZURE_AI_PROJECTS_CONSOLE_LOGGING', 'true')

    deployment_name = os.getenv('MODEL_DEPLOYMENT_NAME', 'grok-4')
    prompt = os.getenv('TEST_PROMPT', 'Reply with exactly: chat-completions-ok')

    credential = DefaultAzureCredential()
    with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project:
        client = project.get_openai_client()
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=40,
            temperature=0,
        )

    msg = response.choices[0].message.content if response.choices else None
    print('\n=== Parsed Response (truncated) ===')
    print(json.dumps({'id': response.id, 'model': response.model, 'message': msg}, indent=2))


if __name__ == '__main__':
    main()
