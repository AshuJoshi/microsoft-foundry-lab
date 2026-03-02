import json

from openai import OpenAI
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from config import load_config


def main() -> None:
    cfg = load_config()
    # model = 'gpt-5.2'
    # model = 'Kimi-K2.5'
    model = 'DeepSeek-V3.2'
    prompt = 'Reply with exactly: comparison-ok'

    credential = DefaultAzureCredential()

    with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project:
        foundry_openai = project.get_openai_client()
        resp_a = foundry_openai.responses.create(model=model, input=prompt, max_output_tokens=20)

    direct_openai = OpenAI(
        api_key=get_bearer_token_provider(credential, 'https://ai.azure.com/.default'),
        base_url=cfg.project_endpoint.rstrip('/') + '/openai',
        default_query={'api-version': '2025-11-15-preview'},
    )
    resp_b = direct_openai.responses.create(model=model, input=prompt, max_output_tokens=20)

    out = {
        'foundry_client': {
            'id': resp_a.id,
            'model': resp_a.model,
            'status': resp_a.status,
            'output_text': resp_a.output_text,
        },
        'direct_openai_client': {
            'id': resp_b.id,
            'model': resp_b.model,
            'status': resp_b.status,
            'output_text': resp_b.output_text,
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
