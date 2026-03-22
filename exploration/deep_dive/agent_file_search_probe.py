#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import FileSearchTool, PromptAgentDefinition
from azure.identity import DefaultAzureCredential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config

logger = logging.getLogger("agent_file_search_probe")


@dataclass
class PromptCase:
    name: str
    prompt: str
    expected_markers: list[str] = field(default_factory=list)


@dataclass
class ProbeRecord:
    case_name: str
    run_index: int
    success: bool
    status_code: int | None
    latency_ms: int
    model: str
    agent_name: str
    conversation_id: str | None
    output_preview: str | None
    output_text: str | None
    output_item_types: list[str]
    file_search_results: list[dict[str, Any]]
    citation_count: int
    citation_urls: list[str]
    citation_files: list[str]
    citation_annotations: list[dict[str, Any]]
    mentioned_ids: list[str]
    expected_markers: list[str]
    matched_markers: list[str]
    expectation_met: bool
    error_type: str | None
    error_message: str | None
    response_headers: dict[str, str] | None


class HeaderRecorder:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.last_status_code: int | None = None
        self.last_headers: dict[str, str] = {}

    def on_request(self, request: httpx.Request) -> None:
        _ = request

    def on_response(self, response: httpx.Response) -> None:
        self.last_status_code = response.status_code
        self.last_headers = {k.lower(): v for k, v in response.headers.items()}


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Agent-based probe for Foundry FileSearchTool behavior.")
    p.add_argument("--model", default=cfg.default_model_deployment_name, help="Deployment/model name to test.")
    p.add_argument("--runs", type=int, default=1, help="Runs per case.")
    p.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    p.add_argument(
        "--cases",
        default="all",
        help="Comma-separated case set from: ids,vendor,total_due,highest_total,quote_ink,summary. Default: all.",
    )
    p.add_argument("--tool-choice", choices=["auto", "required"], default="required")
    p.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--cache-path",
        default="exploration/deep_dive/output/file_search_vector_store.json",
        help="Cache file produced by vector_store_index.py",
    )
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


def _attach_hooks(openai_client: Any, rec: HeaderRecorder) -> None:
    inner = getattr(openai_client, "_client", None)
    if inner is not None and hasattr(inner, "event_hooks"):
        inner.event_hooks = {"request": [rec.on_request], "response": [rec.on_response]}


def _object_to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_object_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _object_to_plain(v) for k, v in value.items()}
    for attr_name in ("model_dump", "to_dict"):
        method = getattr(value, attr_name, None)
        if callable(method):
            try:
                return _object_to_plain(method())
            except Exception:  # noqa: BLE001
                pass
    if hasattr(value, "__dict__"):
        return {
            key: _object_to_plain(val)
            for key, val in vars(value).items()
            if not key.startswith("_") and not callable(val)
        }
    return str(value)


def _extract_annotations_from_item(item: Any) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    urls: list[str] = []
    files: list[str] = []
    annotations: list[dict[str, Any]] = []
    if getattr(item, "type", None) != "message":
        return urls, files, annotations
    for content in getattr(item, "content", []) or []:
        if getattr(content, "type", None) != "output_text":
            continue
        text_value = getattr(content, "text", None)
        for ann in getattr(content, "annotations", []) or []:
            ann_type = str(getattr(ann, "type", "") or "")
            ann_dict = _object_to_plain(ann)
            if not isinstance(ann_dict, dict):
                ann_dict = {"raw": ann_dict}
            ann_dict.setdefault("type", ann_type)
            ann_dict["text_excerpt"] = text_value
            annotations.append(ann_dict)

            url = getattr(ann, "url", None) or ann_dict.get("url")
            if isinstance(url, str) and url:
                urls.append(url)

            file_name = (
                getattr(ann, "filename", None)
                or ann_dict.get("filename")
                or ann_dict.get("file_name")
                or ann_dict.get("document_name")
            )
            if isinstance(file_name, str) and file_name:
                files.append(file_name)
    return urls, files, annotations


def _extract_file_search_results_from_item(item: Any) -> list[dict[str, Any]]:
    if getattr(item, "type", None) != "file_search_call":
        return []
    results = getattr(item, "results", None) or []
    normalized: list[dict[str, Any]] = []
    for result in results:
        normalized.append(_object_to_plain(result))
    return normalized


def _extract_doc_ids(text: str) -> list[str]:
    found = re.findall(r"\b[A-Z]{2,10}-\d{2,6}\b", text)
    seen: set[str] = set()
    uniq: list[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def _load_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        raise SystemExit(f"Vector store cache not found: {cache_path}. Run vector_store_index.py first.")
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _build_cases() -> list[PromptCase]:
    return [
        PromptCase(
            "ids",
            "List all invoice IDs and PO numbers found in the indexed invoice files.",
            ["INV-1001", "INV-1005", "PO-7781", "PO-7931"],
        ),
        PromptCase(
            "vendor",
            "What is the vendor for invoice INV-1004? Answer with the vendor name only.",
            ["Northwind IT Services"],
        ),
        PromptCase(
            "total_due",
            "What is the total due for invoice INV-1002? Answer with the amount only.",
            ["$565.00"],
        ),
        PromptCase(
            "highest_total",
            "Which invoice has the highest total due, and what is that amount? Answer in one sentence.",
            ["INV-1002", "$565.00"],
        ),
        PromptCase(
            "quote_ink",
            "Quote the line item from invoice INV-1001 that mentions ink, then name the source file.",
            ["Ink Cartridge (Black)", "invoice_INV-1001.txt"],
        ),
        PromptCase(
            "summary",
            "Summarize the indexed invoice set in 3 bullets and mention at least two vendors by name.",
            ["Alpine Office Supplies", "BrightPath Logistics"],
        ),
    ]


def _normalize_annotations(citation_annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uniq_annotations: list[dict[str, Any]] = []
    seen_ann: set[tuple[Any, ...]] = set()
    for ann in citation_annotations:
        key = (
            ann.get("type"),
            ann.get("url"),
            ann.get("filename"),
            ann.get("file_name"),
            ann.get("title"),
            ann.get("start_index"),
            ann.get("end_index"),
        )
        if key in seen_ann:
            continue
        seen_ann.add(key)
        uniq_annotations.append(ann)
    return uniq_annotations


def _marker_matches_output(marker: str, output_text: str) -> bool:
    if marker in output_text:
        return True
    if marker.startswith("$") and marker[1:] in output_text:
        return True
    return False


def _run_case(
    *,
    openai_client: Any,
    recorder: HeaderRecorder,
    agent_name: str,
    conversation_id: str,
    prompt_case: PromptCase,
    stream: bool,
    tool_choice: str,
    model: str,
    run_index: int,
) -> ProbeRecord:
    recorder.clear()
    t0 = time.perf_counter()
    output_text = ""
    citation_urls: list[str] = []
    citation_files: list[str] = []
    citation_annotations: list[dict[str, Any]] = []
    output_item_types: list[str] = []
    file_search_results: list[dict[str, Any]] = []
    resp_model = model
    try:
        if stream:
            stream_resp = openai_client.responses.create(
                stream=True,
                conversation=conversation_id,
                input=prompt_case.prompt,
                include=["file_search_call.results"],
                tool_choice=tool_choice,
                extra_body={"agent_reference": {"name": agent_name, "type": "agent_reference"}},
            )
            for event in stream_resp:
                if event.type == "response.output_text.delta":
                    output_text += event.delta or ""
                elif event.type == "response.output_item.done":
                    output_item_types.append(str(getattr(event.item, "type", "")))
                    file_search_results.extend(_extract_file_search_results_from_item(event.item))
                    urls, files, anns = _extract_annotations_from_item(event.item)
                    citation_urls.extend(urls)
                    citation_files.extend(files)
                    citation_annotations.extend(anns)
                elif event.type == "response.completed":
                    if not output_text:
                        output_text = event.response.output_text or ""
                    resp_model = getattr(event.response, "model", resp_model)
        else:
            resp = openai_client.responses.create(
                conversation=conversation_id,
                input=prompt_case.prompt,
                include=["file_search_call.results"],
                tool_choice=tool_choice,
                extra_body={"agent_reference": {"name": agent_name, "type": "agent_reference"}},
            )
            resp_model = getattr(resp, "model", resp_model)
            output_text = resp.output_text or ""
            for item in getattr(resp, "output", []) or []:
                output_item_types.append(str(getattr(item, "type", "")))
                file_search_results.extend(_extract_file_search_results_from_item(item))
                urls, files, anns = _extract_annotations_from_item(item)
                citation_urls.extend(urls)
                citation_files.extend(files)
                citation_annotations.extend(anns)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        urls = sorted(set(citation_urls))
        files = sorted(set(citation_files))
        uniq_annotations = _normalize_annotations(citation_annotations)
        matched_markers = [marker for marker in prompt_case.expected_markers if _marker_matches_output(marker, output_text)]
        return ProbeRecord(
            case_name=prompt_case.name,
            run_index=run_index,
            success=True,
            status_code=recorder.last_status_code,
            latency_ms=latency_ms,
            model=resp_model,
            agent_name=agent_name,
            conversation_id=conversation_id,
            output_preview=output_text[:500] if output_text else None,
            output_text=output_text or None,
            output_item_types=output_item_types,
            file_search_results=file_search_results,
            citation_count=len(uniq_annotations),
            citation_urls=urls,
            citation_files=files,
            citation_annotations=uniq_annotations,
            mentioned_ids=_extract_doc_ids(output_text),
            expected_markers=prompt_case.expected_markers,
            matched_markers=matched_markers,
            expectation_met=len(matched_markers) == len(prompt_case.expected_markers),
            error_type=None,
            error_message=None,
            response_headers=recorder.last_headers,
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - t0) * 1000)
        resp = getattr(exc, "response", None)
        status = recorder.last_status_code
        headers: dict[str, str] = {}
        if resp is not None:
            status = getattr(resp, "status_code", status)
            try:
                headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
            except Exception:  # noqa: BLE001
                headers = {}
        return ProbeRecord(
            case_name=prompt_case.name,
            run_index=run_index,
            success=False,
            status_code=status,
            latency_ms=latency_ms,
            model=model,
            agent_name=agent_name,
            conversation_id=conversation_id,
            output_preview=None,
            output_text=None,
            output_item_types=[],
            file_search_results=[],
            citation_count=0,
            citation_urls=[],
            citation_files=[],
            citation_annotations=[],
            mentioned_ids=[],
            expected_markers=prompt_case.expected_markers,
            matched_markers=[],
            expectation_met=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
            response_headers=headers or recorder.last_headers,
        )


def _write_outputs(run_id: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"agent_file_search_probe_{run_id}.json"
    md_path = out_dir / f"agent_file_search_probe_{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# Search File Search Probe ({run_id})")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for k, v in payload["metadata"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Case | Run | Success | Status | Latency (ms) | Expectation | Citations | Mentioned IDs | x-request-id | Detail |")
    lines.append("|---|---:|---|---:|---:|---|---:|---|---|---|")
    for rec in payload["records"]:
        detail = rec["output_preview"] if rec["success"] else f"{rec['error_type']}: {rec['error_message']}"
        detail = (detail or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {rec['case_name']} | {rec['run_index']} | {'yes' if rec['success'] else 'no'} | "
            f"{rec['status_code'] if rec['status_code'] is not None else '-'} | {rec['latency_ms']} | "
            f"{'met' if rec['expectation_met'] else 'miss'} | {rec['citation_count']} | "
            f"{', '.join(rec['mentioned_ids']) or '-'} | "
            f"{(rec.get('response_headers') or {}).get('x-request-id', '-') } | {detail[:160]} |"
        )
    lines.append("")
    lines.append("## Citations")
    lines.append("")
    for rec in payload["records"]:
        lines.append(f"### {rec['case_name']} / run {rec['run_index']}")
        lines.append("")
        if not rec["citation_annotations"]:
            lines.append("No citation annotations captured.")
            lines.append("")
            continue
        for ann in rec["citation_annotations"]:
            lines.append(f"- Type: `{ann.get('type', '')}`")
            if ann.get("filename"):
                lines.append(f"  File: `{ann['filename']}`")
            elif ann.get("file_name"):
                lines.append(f"  File: `{ann['file_name']}`")
            elif ann.get("document_name"):
                lines.append(f"  File: `{ann['document_name']}`")
            if ann.get("url"):
                lines.append(f"  URL: {ann['url']}")
            if ann.get("title"):
                lines.append(f"  Title: {ann['title']}")
        lines.append("")
    lines.append("## File Search Results")
    lines.append("")
    for rec in payload["records"]:
        lines.append(f"### {rec['case_name']} / run {rec['run_index']}")
        lines.append("")
        if not rec["file_search_results"]:
            lines.append("No file search results captured.")
            lines.append("")
            continue
        for result in rec["file_search_results"]:
            lines.append(f"- File: `{result.get('filename', '-')}`")
            if result.get("file_id"):
                lines.append(f"  File ID: `{result['file_id']}`")
            if result.get("score") is not None:
                lines.append(f"  Score: `{result['score']}`")
            if result.get("text"):
                snippet = str(result["text"]).replace("\n", " ")
                lines.append(f"  Text: {snippet[:220]}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path


def main() -> None:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"agent_file_search_probe_{run_id}.log"
    _setup_logging(args.log_level, log_path)

    cache_path = Path(args.cache_path)
    cache = _load_cache(cache_path)
    vector_store_id = str(cache["vector_store_id"])

    case_map = {case.name: case for case in _build_cases()}
    selected_names = list(case_map) if args.cases == "all" else [part.strip() for part in args.cases.split(",") if part.strip()]
    bad = [name for name in selected_names if name not in case_map]
    if bad:
        raise SystemExit(f"Unknown case(s): {bad}")
    cases = [case_map[name] for name in selected_names]

    logger.info("run_id=%s", run_id)
    logger.info("model=%s", args.model)
    logger.info("runs=%s", args.runs)
    logger.info("cases=%s", selected_names)
    logger.info("tool_choice=%s stream=%s", args.tool_choice, args.stream)
    logger.info("vector_store_id=%s", vector_store_id)

    metadata = {
        "model": args.model,
        "runs_per_case": args.runs,
        "tool_choice": args.tool_choice,
        "stream": args.stream,
        "project_endpoint": load_config().project_endpoint,
        "vector_store_id": vector_store_id,
        "cache_path": str(cache_path),
        "sample_corpus": cache.get("sample_corpus"),
        "files": cache.get("files"),
    }
    records: list[dict[str, Any]] = []
    recorder = HeaderRecorder()

    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=load_config().project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):
        _attach_hooks(openai_client, recorder)
        agent = project_client.agents.create_version(
            agent_name=f"ValidationAgent-files-{args.model.replace('.', '-')}-{run_id[-6:].lower()}",
            definition=PromptAgentDefinition(
                model=args.model,
                instructions=(
                    "You are a retrieval assistant. Use the attached file search tool to answer only from indexed files. "
                    "When naming a source, use the actual file name if available."
                ),
                tools=[FileSearchTool(vector_store_ids=[vector_store_id])],
            ),
        )
        logger.info("agent_name=%s agent_version=%s", agent.name, agent.version)

        try:
            total = len(cases) * args.runs
            progress = 0
            for case in cases:
                for run_index in range(1, args.runs + 1):
                    progress += 1
                    conversation = openai_client.conversations.create()
                    logger.info(
                        "START progress=%s/%s case=%s run=%s conversation_id=%s",
                        progress,
                        total,
                        case.name,
                        run_index,
                        conversation.id,
                    )
                    record = _run_case(
                        openai_client=openai_client,
                        recorder=recorder,
                        agent_name=agent.name,
                        conversation_id=conversation.id,
                        prompt_case=case,
                        stream=args.stream,
                        tool_choice=args.tool_choice,
                        model=args.model,
                        run_index=run_index,
                    )
                    logger.info(
                        "DONE  progress=%s/%s case=%s run=%s status=%s latency_ms=%s citations=%s expectation=%s served_model=%s",
                        progress,
                        total,
                        case.name,
                        run_index,
                        record.status_code,
                        record.latency_ms,
                        record.citation_count,
                        "met" if record.expectation_met else "miss",
                        record.model,
                    )
                    records.append(asdict(record))
                    try:
                        openai_client.conversations.delete(conversation.id)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to delete conversation_id=%s error=%s", conversation.id, exc)
        finally:
            try:
                project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
                logger.info("deleted agent_name=%s agent_version=%s", agent.name, agent.version)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to delete agent_name=%s agent_version=%s error=%s", agent.name, agent.version, exc)

    payload = {"metadata": metadata, "records": records}
    md_path, json_path = _write_outputs(run_id, payload)
    logger.info("Artifacts markdown=%s json=%s log=%s", md_path, json_path, log_path)


if __name__ == "__main__":
    main()
