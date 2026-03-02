#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    MemorySearchOptions,
    MemorySearchPreviewTool,
    MemoryStoreDefaultDefinition,
    MemoryStoreDefaultOptions,
    PromptAgentDefinition,
)
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    parser = argparse.ArgumentParser(
        description="Run the Memory docs-style sample flow (minimal scaffolding)."
    )
    parser.add_argument("--chat-model", default=cfg.default_model_deployment_name)
    parser.add_argument(
        "--embedding-model",
        default=(
            os.getenv("MEMORY_EMBEDDING_MODEL_DEPLOYMENT_NAME")
            or os.getenv("AZURE_AI_EMBEDDING_MODEL_DEPLOYMENT_NAME")
            or "text-embedding-3-small"
        ),
    )
    parser.add_argument("--scope", default="user_123")
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=65,
        help="Wait between turns (docs example uses 65s).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on feature-level errors (default is graceful handling).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = load_config()

    memory_store_name = f"sample-mem-{run_id.lower()}"
    agent_name = f"sample-agent-{run_id.lower()}"

    agent = None
    conversation_1 = None
    conversation_2 = None

    print(f"run_id={run_id}")
    print(f"project_endpoint={cfg.project_endpoint}")
    print(f"chat_model={args.chat_model}")
    print(f"embedding_model={args.embedding_model}")
    print(f"scope={args.scope}")
    print()

    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):
        try:
            print("1) Create memory store")
            options = MemoryStoreDefaultOptions(
                chat_summary_enabled=True,
                user_profile_enabled=True,
                user_profile_details=(
                    "Avoid irrelevant or sensitive data, such as age, financials, precise location, and credentials"
                ),
            )
            definition = MemoryStoreDefaultDefinition(
                chat_model=args.chat_model,
                embedding_model=args.embedding_model,
                options=options,
            )
            memory_store = project_client.beta.memory_stores.create(
                name=memory_store_name,
                definition=definition,
                description="Memory store for docs-style sample",
            )
            print(f"   created memory store: {memory_store.name}")

            print("2) Update/list memory store")
            updated_store = project_client.beta.memory_stores.update(
                name=memory_store_name,
                description="Updated description",
            )
            print(f"   updated description: {updated_store.description}")
            stores_list = list(project_client.beta.memory_stores.list())
            print(f"   stores found: {len(stores_list)}")

            print("3) Create agent with memory search tool")
            tool = MemorySearchPreviewTool(
                memory_store_name=memory_store_name,
                scope=args.scope,
                update_delay=1,
            )
            agent = project_client.agents.create_version(
                agent_name=agent_name,
                definition=PromptAgentDefinition(
                    model=args.chat_model,
                    instructions="You are a helpful assistant that answers general questions.",
                    tools=[tool],
                ),
            )
            print(f"   created agent: {agent.name} v{agent.version}")

            print("4) Conversation turn 1 (write preference)")
            conversation_1 = openai_client.conversations.create()
            response_1 = openai_client.responses.create(
                input="I prefer dark roast coffee",
                conversation=conversation_1.id,
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            )
            print(f"   response_1: {getattr(response_1, 'output_text', '')}")

            print(f"5) Waiting {args.wait_seconds}s for memory materialization")
            time.sleep(args.wait_seconds)

            print("6) Conversation turn 2 (recall)")
            conversation_2 = openai_client.conversations.create()
            response_2 = openai_client.responses.create(
                input="Please order my usual coffee",
                conversation=conversation_2.id,
                extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
            )
            print(f"   response_2: {getattr(response_2, 'output_text', '')}")

            print("7) Direct memory update/search APIs")
            try:
                user_message = {
                    "role": "user",
                    "content": "I also like cappuccinos in the afternoon",
                    "type": "message",
                }
                update_poller = project_client.beta.memory_stores.begin_update_memories(
                    name=memory_store_name,
                    scope=args.scope,
                    items=[user_message],
                    update_delay=0,
                )
                update_result = update_poller.result()
                print(f"   update operations: {len(update_result.memory_operations)}")

                query_message = {
                    "role": "user",
                    "content": "What are my coffee preferences?",
                    "type": "message",
                }
                search_response = project_client.beta.memory_stores.search_memories(
                    name=memory_store_name,
                    scope=args.scope,
                    items=[query_message],
                    options=MemorySearchOptions(max_memories=5),
                )
                print(f"   memories found: {len(search_response.memories)}")
            except HttpResponseError as exc:
                print(f"   warning: memory update/search step failed: {exc}")
                if args.strict:
                    raise

        finally:
            print("\nCleanup")
            if conversation_1 is not None:
                try:
                    openai_client.conversations.delete(conversation_id=conversation_1.id)
                    print("   deleted conversation_1")
                except Exception as exc:  # noqa: BLE001
                    print(f"   cleanup warning (conversation_1): {exc}")
            if conversation_2 is not None:
                try:
                    openai_client.conversations.delete(conversation_id=conversation_2.id)
                    print("   deleted conversation_2")
                except Exception as exc:  # noqa: BLE001
                    print(f"   cleanup warning (conversation_2): {exc}")
            if agent is not None:
                try:
                    project_client.agents.delete_version(
                        agent_name=agent.name,
                        agent_version=agent.version,
                    )
                    print("   deleted agent version")
                except Exception as exc:  # noqa: BLE001
                    print(f"   cleanup warning (agent): {exc}")
            try:
                project_client.beta.memory_stores.delete_scope(
                    name=memory_store_name,
                    scope=args.scope,
                )
                print("   deleted memory scope")
            except Exception as exc:  # noqa: BLE001
                print(f"   cleanup warning (delete_scope): {exc}")
            try:
                project_client.beta.memory_stores.delete(name=memory_store_name)
                print("   deleted memory store")
            except Exception as exc:  # noqa: BLE001
                print(f"   cleanup warning (memory_store): {exc}")


if __name__ == "__main__":
    main()
