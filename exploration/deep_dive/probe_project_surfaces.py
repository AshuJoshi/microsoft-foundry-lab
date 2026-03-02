import json
from collections import Counter

from azure.core.exceptions import ResourceNotFoundError
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from config import load_config


def main() -> None:
    cfg = load_config()
    credential = DefaultAzureCredential()

    with AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project:
        deployments = list(project.deployments.list())
        connections = list(project.connections.list())
        agents = list(project.agents.list())
        memory_stores = list(project.memory_stores.list())

        try:
            app_insights_conn = project.telemetry.get_application_insights_connection_string()
            telemetry = {'application_insights_connection_string_present': bool(app_insights_conn)}
        except ResourceNotFoundError:
            telemetry = {'application_insights_connection_string_present': False}

    connection_types = Counter([str(getattr(c, 'type', 'unknown')) for c in connections])

    output = {
        'counts': {
            'deployments': len(deployments),
            'connections': len(connections),
            'agents': len(agents),
            'memory_stores': len(memory_stores),
        },
        'connection_types': dict(connection_types),
        'agent_names': [a.name for a in agents],
        'memory_store_names': [m.name for m in memory_stores],
        'telemetry': telemetry,
    }
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
