#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition
from azure.identity import DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("agent_context_limit_probe")


@dataclass
class UsageSnapshot:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class TurnRecord:
    turn_index: int
    phase: str
    status_code: int | None
    success: bool
    latency_ms: int
    prompt_chars: int
    estimated_prompt_tokens: int
    cumulative_estimated_tokens: int
    output_chars: int
    usage: UsageSnapshot
    conversation_id: str | None
    recall_label: str | None = None
    recall_present: bool | None = None
    recall_expected_answer: str | None = None
    recall_observed_answer: str | None = None
    output_preview: str | None = None
    apim_request_id: str | None = None
    x_request_id: str | None = None
    x_ms_region: str | None = None
    x_ratelimit_limit_tokens: str | None = None
    x_ratelimit_remaining_tokens: str | None = None
    x_ratelimit_reset_tokens: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class HeaderRecorder:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.status_code: int | None = None
        self.headers: dict[str, str] = {}

    def on_request(self, request: httpx.Request) -> None:
        _ = request

    def on_response(self, response: httpx.Response) -> None:
        self.status_code = response.status_code
        self.headers = {k.lower(): v for k, v in response.headers.items()}


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Probe prompt-agent context growth, recall, and compaction behavior.")
    p.add_argument("--model", default="gpt-5.1", help="Model/deployment name.")
    p.add_argument("--target-input-tokens", type=int, default=272000, help="Approximate target input-window threshold.")
    p.add_argument("--block-chars", type=int, default=24000, help="Characters added per stuffing turn.")
    p.add_argument("--max-turns", type=int, default=20, help="Maximum stuffing turns before stopping.")
    p.add_argument("--max-output-tokens", type=int, default=220)
    p.add_argument("--truncation", choices=["omit", "auto", "disabled"], default="omit")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--sleep-seconds", type=float, default=0.0)
    return p.parse_args()


def _setup_logging(level_name: str, log_path: Path) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _attach_hooks(openai_client: Any, recorder: HeaderRecorder) -> None:
    inner = getattr(openai_client, "_client", None)
    if inner is not None and hasattr(inner, "event_hooks"):
        inner.event_hooks = {"request": [recorder.on_request], "response": [recorder.on_response]}


def _call_with_capture(recorder: HeaderRecorder, fn: Any) -> tuple[Any | None, dict[str, str], int, Exception | None]:
    recorder.reset()
    t0 = time.perf_counter()
    try:
        out = fn()
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return out, recorder.headers.copy(), latency_ms, None
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - t0) * 1000)
        headers = recorder.headers.copy()
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
            except Exception:  # noqa: BLE001
                pass
        return None, headers, latency_ms, exc


def _deployment_names(project_client: AIProjectClient) -> list[str]:
    return sorted(
        name
        for deployment in project_client.deployments.list()
        for name in [getattr(deployment, "name", None)]
        if isinstance(name, str) and name
    )


def _usage_snapshot(response: Any) -> UsageSnapshot:
    usage = getattr(response, "usage", None)
    if usage is None:
        return UsageSnapshot()
    return UsageSnapshot(
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def _output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    return text if isinstance(text, str) else ""


def _estimate_tokens_from_chars(chars: int) -> int:
    # Deliberately conservative rough estimate; enough for threshold guidance, not billing precision.
    return math.ceil(chars / 4)


def _make_stuffing_prompt(turn_index: int, label: str, block_chars: int) -> str:
    filler = f"[BEGIN-{label}] " + ("X" * max(0, block_chars - len(label) - 10)) + f" [END-{label}]"
    return (
        f"Turn {turn_index}. Store this exact marker for later recall: {label}.\n"
        "Do not summarize the body. Just reply with: STORED <label>.\n\n"
        f"{filler}"
    )


def _make_recall_prompt(label: str) -> str:
    return (
        f"In the immediately preceding stuffing turn, was the exact marker label `{label}` stored? "
        "Answer with exactly one token: YES or NO."
    )


def _normalize_yes_no(text: str) -> str | None:
    stripped = text.strip().upper()
    if stripped.startswith("YES"):
        return "YES"
    if stripped.startswith("NO"):
        return "NO"
    return None


def _record_failure(turn_index: int, phase: str, prompt: str, cumulative_est_tokens: int, conversation_id: str | None, latency_ms: int, headers: dict[str, str], err: Exception) -> TurnRecord:
    return TurnRecord(
        turn_index=turn_index,
        phase=phase,
        status_code=None,
        success=False,
        latency_ms=latency_ms,
        prompt_chars=len(prompt),
        estimated_prompt_tokens=_estimate_tokens_from_chars(len(prompt)),
        cumulative_estimated_tokens=cumulative_est_tokens,
        output_chars=0,
        usage=UsageSnapshot(),
        conversation_id=conversation_id,
        output_preview=None,
        apim_request_id=headers.get("apim-request-id"),
        x_request_id=headers.get("x-request-id"),
        x_ms_region=headers.get("x-ms-region"),
        x_ratelimit_limit_tokens=headers.get("x-ratelimit-limit-tokens"),
        x_ratelimit_remaining_tokens=headers.get("x-ratelimit-remaining-tokens"),
        x_ratelimit_reset_tokens=headers.get("x-ratelimit-reset-tokens"),
        error_type=type(err).__name__,
        error_message=str(err),
    )


def _write_outputs(run_id: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"agent_context_limit_probe_{run_id}.json"
    md_path = out_dir / f"agent_context_limit_probe_{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# Agent Context Limit Probe ({run_id})")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    for k, v in payload["metadata"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for k, v in payload["summary"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Turns")
    lines.append("")
    lines.append("| Turn | Phase | Success | Status | Prompt Chars | Est Prompt Tokens | Cum Est Tokens | Total Tokens | Recall Label | Expected | Observed | Recall Present | Latency (ms) | Error |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---:|---|")
    for rec in payload["records"]:
        usage = rec["usage"]
        lines.append(
            f"| {rec['turn_index']} | {rec['phase']} | {'yes' if rec['success'] else 'no'} | "
            f"{rec['status_code'] or '-'} | {rec['prompt_chars']} | {rec['estimated_prompt_tokens']} | "
                f"{rec['cumulative_estimated_tokens']} | {usage['total_tokens'] or 0} | {rec.get('recall_label') or '-'} | "
                f"{rec.get('recall_expected_answer') or '-'} | {rec.get('recall_observed_answer') or '-'} | "
                f"{rec.get('recall_present')} | {rec['latency_ms']} | {rec.get('error_type') or '-'} |"
            )
    lines.append("")
    lines.append("## Failure Headers")
    lines.append("")
    lines.append("| Turn | Phase | Status | APIM Request ID | x-request-id | x-ms-region | RL Limit Tokens | RL Remaining Tokens | RL Reset Tokens |")
    lines.append("|---|---|---:|---|---|---|---|---|---|")
    for rec in payload["records"]:
        if rec["success"]:
            continue
        lines.append(
            f"| {rec['turn_index']} | {rec['phase']} | {rec['status_code'] or '-'} | "
            f"{rec.get('apim_request_id') or '-'} | {rec.get('x_request_id') or '-'} | "
            f"{rec.get('x_ms_region') or '-'} | {rec.get('x_ratelimit_limit_tokens') or '-'} | "
            f"{rec.get('x_ratelimit_remaining_tokens') or '-'} | {rec.get('x_ratelimit_reset_tokens') or '-'} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main() -> None:
    args = parse_args()
    cfg = load_config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "output"
    log_path = out_dir / f"agent_context_limit_probe_{run_id}.log"
    _setup_logging(args.log_level, log_path)

    logger.info("run_id=%s", run_id)
    logger.info("model=%s", args.model)
    logger.info("target_input_tokens=%s", args.target_input_tokens)
    logger.info("block_chars=%s max_turns=%s truncation=%s", args.block_chars, args.max_turns, args.truncation)

    payload: dict[str, Any] = {
        "metadata": {
            "project_endpoint": cfg.project_endpoint,
            "model": args.model,
            "agent_kind": "prompt",
            "target_input_tokens": args.target_input_tokens,
            "block_chars": args.block_chars,
            "max_turns": args.max_turns,
            "max_output_tokens": args.max_output_tokens,
            "truncation": args.truncation,
            "sleep_seconds": args.sleep_seconds,
            "hypothesis": (
                "If conversation-state loss happens under pressure, a binary yes/no fact check about the immediately "
                "previous stuffing turn should flip before a hard failure appears."
            ),
        },
        "records": [],
        "summary": {},
    }

    recorder = HeaderRecorder()
    agent_name = f"{cfg.agent_name_prefix}-ctxlimit-{run_id.lower()}"
    cumulative_est_tokens = 0
    stored_labels: list[str] = []
    stop_reason = "max_turns_reached"

    with DefaultAzureCredential() as credential, AIProjectClient(endpoint=cfg.project_endpoint, credential=credential) as project_client:
        deployed_models = _deployment_names(project_client)
        if args.model not in deployed_models:
            raise SystemExit(
                "Model deployment not found in this project.\n"
                f"requested: {args.model}\n"
                f"available: {', '.join(deployed_models) if deployed_models else '(none)'}"
            )

        with project_client.get_openai_client() as openai_client:
            _attach_hooks(openai_client, recorder)
            agent = project_client.agents.create_version(
                agent_name=agent_name,
                definition=PromptAgentDefinition(
                    model=args.model,
                    instructions=(
                        "You are a prompt agent used to study long conversation carryover and compaction. "
                        "Follow each instruction exactly. When asked to store a marker, do not summarize the body. "
                        "When asked a yes/no recall question, answer with exactly YES or NO."
                    ),
                ),
            )
            conversation = None
            try:
                conversation = openai_client.conversations.create()
                logger.info("agent_name=%s agent_version=%s conversation_id=%s", agent.name, agent.version, conversation.id)

                for turn_index in range(1, args.max_turns + 1):
                    label = f"CTXMARK-{turn_index:03d}"
                    stuffing_prompt = _make_stuffing_prompt(turn_index, label, args.block_chars)
                    cumulative_est_tokens += _estimate_tokens_from_chars(len(stuffing_prompt))

                    kwargs: dict[str, Any] = {
                        "input": stuffing_prompt,
                        "max_output_tokens": args.max_output_tokens,
                        "conversation": conversation.id,
                        "extra_body": {"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                    }
                    if args.truncation != "omit":
                        kwargs["truncation"] = args.truncation

                    logger.info(
                        "START turn=%s phase=stuff cumulative_est_tokens=%s threshold_pct=%.1f",
                        turn_index,
                        cumulative_est_tokens,
                        100.0 * cumulative_est_tokens / max(1, args.target_input_tokens),
                    )
                    response, headers, latency_ms, err = _call_with_capture(recorder, lambda: openai_client.responses.create(**kwargs))
                    if err is not None:
                        payload["records"].append(
                            asdict(_record_failure(turn_index, "stuff", stuffing_prompt, cumulative_est_tokens, conversation.id, latency_ms, headers, err))
                        )
                        stop_reason = "stuffing_error"
                        logger.warning("FAIL turn=%s phase=stuff latency_ms=%s error=%s", turn_index, latency_ms, err)
                        break

                    text = _output_text(response)
                    usage = _usage_snapshot(response)
                    payload["records"].append(
                        asdict(
                            TurnRecord(
                                turn_index=turn_index,
                                phase="stuff",
                                status_code=recorder.status_code,
                                success=True,
                                latency_ms=latency_ms,
                                prompt_chars=len(stuffing_prompt),
                                estimated_prompt_tokens=_estimate_tokens_from_chars(len(stuffing_prompt)),
                                cumulative_estimated_tokens=cumulative_est_tokens,
                                output_chars=len(text),
                                usage=usage,
                                conversation_id=conversation.id,
                                recall_label=label,
                                output_preview=text[:220] if text else None,
                                apim_request_id=headers.get("apim-request-id"),
                                x_request_id=headers.get("x-request-id"),
                                x_ms_region=headers.get("x-ms-region"),
                                x_ratelimit_limit_tokens=headers.get("x-ratelimit-limit-tokens"),
                                x_ratelimit_remaining_tokens=headers.get("x-ratelimit-remaining-tokens"),
                                x_ratelimit_reset_tokens=headers.get("x-ratelimit-reset-tokens"),
                            )
                        )
                    )
                    stored_labels.append(label)
                    logger.info(
                        "DONE  turn=%s phase=stuff status=%s latency_ms=%s total_tokens=%s",
                        turn_index,
                        recorder.status_code,
                        latency_ms,
                        usage.total_tokens,
                    )

                    recall_prompt = _make_recall_prompt(label)
                    cumulative_est_tokens += _estimate_tokens_from_chars(len(recall_prompt))
                    recall_kwargs = {
                        "input": recall_prompt,
                        "max_output_tokens": args.max_output_tokens,
                        "conversation": conversation.id,
                        "extra_body": {"agent_reference": {"name": agent.name, "type": "agent_reference"}},
                    }
                    if args.truncation != "omit":
                        recall_kwargs["truncation"] = args.truncation

                    logger.info("START turn=%s phase=recall expected_label=%s", turn_index, label)
                    recall_resp, headers, latency_ms, err = _call_with_capture(
                        recorder, lambda: openai_client.responses.create(**recall_kwargs)
                    )
                    if err is not None:
                        payload["records"].append(
                            asdict(_record_failure(turn_index, "recall", recall_prompt, cumulative_est_tokens, conversation.id, latency_ms, headers, err))
                        )
                        stop_reason = "recall_error"
                        logger.warning("FAIL turn=%s phase=recall latency_ms=%s error=%s", turn_index, latency_ms, err)
                        break

                    recall_text = _output_text(recall_resp)
                    usage = _usage_snapshot(recall_resp)
                    observed_answer = _normalize_yes_no(recall_text)
                    recall_present = observed_answer == "YES"
                    payload["records"].append(
                        asdict(
                            TurnRecord(
                                turn_index=turn_index,
                                phase="recall",
                                status_code=recorder.status_code,
                                success=True,
                                latency_ms=latency_ms,
                                prompt_chars=len(recall_prompt),
                                estimated_prompt_tokens=_estimate_tokens_from_chars(len(recall_prompt)),
                                cumulative_estimated_tokens=cumulative_est_tokens,
                                output_chars=len(recall_text),
                                usage=usage,
                                conversation_id=conversation.id,
                                recall_label=label,
                                recall_present=recall_present,
                                recall_expected_answer="YES",
                                recall_observed_answer=observed_answer,
                                output_preview=recall_text[:220] if recall_text else None,
                                apim_request_id=headers.get("apim-request-id"),
                                x_request_id=headers.get("x-request-id"),
                                x_ms_region=headers.get("x-ms-region"),
                                x_ratelimit_limit_tokens=headers.get("x-ratelimit-limit-tokens"),
                                x_ratelimit_remaining_tokens=headers.get("x-ratelimit-remaining-tokens"),
                                x_ratelimit_reset_tokens=headers.get("x-ratelimit-reset-tokens"),
                            )
                        )
                    )
                    logger.info(
                        "DONE  turn=%s phase=recall status=%s latency_ms=%s total_tokens=%s recall_answer=%s recall_present=%s",
                        turn_index,
                        recorder.status_code,
                        latency_ms,
                        usage.total_tokens,
                        observed_answer,
                        recall_present,
                    )

                    if not recall_present:
                        stop_reason = "recall_loss_detected"
                        break
                    if cumulative_est_tokens >= args.target_input_tokens:
                        stop_reason = "target_threshold_reached"
                        break
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
            finally:
                if conversation is not None:
                    try:
                        openai_client.conversations.delete(conversation.id)
                    except Exception:  # noqa: BLE001
                        logger.warning("cleanup failed for conversation_id=%s", conversation.id)
                try:
                    project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
                except Exception:  # noqa: BLE001
                    logger.warning("cleanup failed for agent_name=%s agent_version=%s", agent.name, agent.version)

    records = payload["records"]
    success_records = [r for r in records if r["success"]]
    failure_records = [r for r in records if not r["success"]]
    recall_records = [r for r in records if r["phase"] == "recall" and r["success"]]
    first_recall_loss = next((r for r in recall_records if r.get("recall_present") is False), None)
    payload["summary"] = {
        "turn_steps_completed": len(records),
        "successful_steps": len(success_records),
        "failure_steps": len(failure_records),
        "stop_reason": stop_reason,
        "max_cumulative_estimated_tokens": max((r["cumulative_estimated_tokens"] for r in records), default=0),
        "max_total_tokens_seen": max((r["usage"]["total_tokens"] or 0 for r in success_records), default=0),
        "first_recall_loss_turn": first_recall_loss["turn_index"] if first_recall_loss else None,
        "first_failure_turn": failure_records[0]["turn_index"] if failure_records else None,
        "final_recall_label_count": len(stored_labels),
    }

    md_path, json_path = _write_outputs(run_id, payload)
    logger.info("Artifacts markdown=%s json=%s log=%s", md_path, json_path, log_path)


if __name__ == "__main__":
    main()
