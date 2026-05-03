from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import socket
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "test"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "results"
DEFAULT_KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "data" / "local_knowledge"
DEFAULT_RAG_INDEX_DIR = PROJECT_ROOT / "data" / "local_knowledge_index"
DEFAULT_THRESHOLDS = {
    "intent_accuracy": 0.80,
    "tool_recall": 0.85,
    "forbidden_tool_violation_rate": 0.0,
    "rag_pipeline_accuracy": 0.90,
    "literature_recall@5": 0.60,
    "literature_recall@10": 0.80,
}
TOOL_NODE_MAP = {"RAG": "rag", "WebSearch": "web_search", "scAnalysis": "sc_analysis"}
RAG_PIPELINE_FIELDS = {
    "bm25": "bm25_executed",
    "vector": "vector_executed",
    "rrf": "rrf_executed",
    "bge_reranker": "reranker_executed",
}


def json_dump(data: Any, *, pretty: bool = False) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2 if pretty else None, default=str)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_no} in {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL line {line_no} in {path} must be an object.")
            cases.append(payload)
    return cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_dump(row) + "\n")


def external_error_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [ConnectionError, TimeoutError, OSError]
    try:
        import requests

        types.append(requests.exceptions.RequestException)
    except ImportError:
        pass
    try:
        import httpx

        types.append(httpx.HTTPError)
    except ImportError:
        pass
    try:
        import openai

        types.extend([openai.APIConnectionError, openai.APITimeoutError, openai.APIStatusError])
    except ImportError:
        pass
    return tuple(dict.fromkeys(types))


def load_zh_genes(gene_file: Path, limit: int = 3) -> tuple[list[str], str]:
    if not gene_file.exists():
        return [], f"{gene_file} not found"
    text = gene_file.read_text(encoding="utf-8", errors="ignore")
    zh_genes = list(dict.fromkeys(re.findall(r"\bZH\d{2}G\d{5}\b", text)))
    if zh_genes:
        return zh_genes[:limit], "selected ZH-prefixed genes from gene_id.txt"
    fallback = list(dict.fromkeys(re.findall(r'"([A-Za-z][A-Za-z0-9;._-]{2,})"\s*:', text)))
    return fallback[:limit], "no ZH-prefixed genes found; selected real symbols from gene_id.txt"


def replace_dynamic_values(value: Any, zh_genes: list[str]) -> Any:
    if value == "$ZH_GENES":
        return list(zh_genes)
    if isinstance(value, str):
        return value.replace("<ZH_GENES>", ", ".join(zh_genes))
    if isinstance(value, list):
        return [replace_dynamic_values(item, zh_genes) for item in value]
    if isinstance(value, dict):
        return {key: replace_dynamic_values(item, zh_genes) for key, item in value.items()}
    return value


def load_cases(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    zh_genes, gene_note = load_zh_genes(PROJECT_ROOT / "data" / "gene_id.txt")
    files = [
        "tool_call_cases.jsonl",
        "literature_recall_cases.jsonl",
        "sc_analysis_cases.jsonl",
        "eval_cases.jsonl",
    ]
    cases: list[dict[str, Any]] = []
    for file_name in files:
        for case in read_jsonl(dataset_dir / file_name):
            resolved = replace_dynamic_values(case, zh_genes)
            resolved.setdefault("source_file", file_name)
            cases.append(resolved)
    return cases, {"selected_genes": zh_genes, "gene_selection_note": gene_note}


def socket_open(host: str, port: int, timeout: float = 0.5) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def choose_mode(mode: str) -> str:
    if mode != "auto":
        return mode
    return "api" if socket_open("127.0.0.1", 8000) else "direct"


def parse_sse_events(raw: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_name = ""
    data_lines: list[str] = []
    for raw_line in raw.decode("utf-8", errors="replace").splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                payload = json.loads("\n".join(data_lines))
                events.append({"event": event_name, "data": payload})
            event_name, data_lines = "", []
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if data_lines:
        events.append({"event": event_name, "data": json.loads("\n".join(data_lines))})
    return events


def call_agent_api(case: dict[str, Any], user_id: str, session_id: str) -> dict[str, Any]:
    data = urllib.parse.urlencode({"user_id": user_id, "session_id": session_id, "text": case["query"]}).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:8000/api/agent/submit",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=int(case.get("timeout_s") or 1800)) as response:
        events = parse_sse_events(response.read())
    final_events = [event for event in events if event.get("event") == "final"]
    if not final_events:
        raise RuntimeError(f"API response for {case.get('id')} did not contain a final event.")
    final_payload = final_events[-1]["data"]
    return {"mode": "api", "events": events, "final_payload": final_payload, "trace": build_trace_from_api(final_payload)}


def call_agent_direct(case: dict[str, Any], user_id: str, session_id: str) -> dict[str, Any]:
    from backend.agent import stream_agent
    from backend.util import current_turn_llm_traces, current_turn_steps, current_turn_tool_results

    workspace_settings = dict(case.get("workspace_settings") or {})
    workspace_settings.setdefault("enable_semantic_memory", False)
    workspace_settings.setdefault("multi_query_count", 1)
    events: list[dict[str, Any]] = []
    final_state: dict[str, Any] = {}

    def event_callback(event: dict[str, Any]) -> None:
        events.append(dict(event))

    for event in stream_agent(
        user_input=str(case["query"]),
        user_id=user_id,
        session_id=session_id,
        knowledge_base_path=str(case.get("knowledge_base_path") or DEFAULT_KNOWLEDGE_BASE_PATH),
        rag_index_dir=str(case.get("index_dir") or DEFAULT_RAG_INDEX_DIR),
        workspace_settings=workspace_settings,
        event_callback=event_callback,
    ):
        if isinstance(event, tuple) and len(event) == 2 and event[0] == "final_state":
            final_state = dict(event[1] or {})
    tool_results = current_turn_tool_results(final_state)
    steps = current_turn_steps(final_state)
    if not tool_results and isinstance(final_state.get("tool_results"), dict):
        tool_results = dict(final_state["tool_results"])
    if not steps:
        steps = list(final_state.get("steps") or [])
    trace = build_trace_from_state(final_state, tool_results, steps, len(current_turn_llm_traces(final_state)))
    trace["events"] = events
    return {"mode": "direct", "events": events, "final_state": final_state, "trace": trace}


def build_trace_from_api(payload: dict[str, Any]) -> dict[str, Any]:
    agent = payload.get("agent") if isinstance(payload.get("agent"), dict) else {}
    decision = agent.get("decision") if isinstance(agent.get("decision"), dict) else {}
    state = agent.get("state") if isinstance(agent.get("state"), dict) else {}
    tool_result = payload.get("tool_result") if isinstance(payload.get("tool_result"), dict) else {}
    steps = list(state.get("steps") or [])
    meta = tool_result.get("meta") if isinstance(tool_result.get("meta"), dict) else {}
    rag_trace = dict(meta.get("rag_retrieval_trace") or tool_result.get("retrieval_trace") or {})
    web_meta = dict(meta.get("web_search_meta") or {})
    sc_meta = dict(meta.get("sc_analysis_meta") or {})
    tools_called = tools_from_steps(steps)
    return {
        "intent": decision.get("intent_type") or decision.get("intent") or state.get("intent_type"),
        "tools_called": tools_called,
        "tool_order": tools_called,
        "llm_call_count": int(tool_result.get("metrics", {}).get("llm_call_count") or decision.get("llm_call_count") or 0),
        "rag": {"called": bool(rag_trace or "rag" in tools_called), "retrieval_trace": rag_trace, "eval_trace": tool_result.get("eval_trace") or {}, "final_context_chunks": list(tool_result.get("chunks") or [])},
        "web_search": {"called": bool(web_meta or "web_search" in tools_called), "meta": web_meta},
        "sc_analysis": {"called": bool(sc_meta or "sc_analysis" in tools_called), "params": dict(tool_result.get("analysis_params") or sc_meta.get("function_call_arguments") or {}), "result": dict(tool_result.get("analysis_result") or {})},
        "answer": str(tool_result.get("answer") or ""),
    }


def build_trace_from_state(state: dict[str, Any], tool_results: dict[str, Any], steps: list[str], llm_call_count: int) -> dict[str, Any]:
    rag_result = tool_results.get("rag") if isinstance(tool_results.get("rag"), dict) else {}
    web_result = tool_results.get("web_search") if isinstance(tool_results.get("web_search"), dict) else {}
    sc_result = tool_results.get("sc_analysis") if isinstance(tool_results.get("sc_analysis"), dict) else {}
    rag_meta = rag_result.get("meta") if isinstance(rag_result.get("meta"), dict) else {}
    web_meta = web_result.get("meta") if isinstance(web_result.get("meta"), dict) else {}
    sc_meta = sc_result.get("meta") if isinstance(sc_result.get("meta"), dict) else {}
    return {
        "intent": state.get("intent_type") or state.get("intent"),
        "tools_called": tools_from_steps(steps),
        "tool_order": tools_from_steps(steps, keep_repeats=True),
        "llm_call_count": llm_call_count,
        "rag": {
            "called": bool(rag_result),
            "retrieval_trace": dict(rag_meta.get("retrieval_trace") or rag_result.get("retrieval_trace") or {}),
            "eval_trace": dict(rag_meta.get("eval_trace") or rag_result.get("eval_trace") or {}),
            "final_context_chunks": list(rag_result.get("chunks") or []),
        },
        "web_search": {"called": bool(web_result), "meta": dict(web_meta), "result": dict(web_result)},
        "sc_analysis": {
            "called": bool(sc_result),
            "params": dict(sc_result.get("analysis_params") or sc_meta.get("function_call_arguments") or {}),
            "result": dict(sc_result.get("analysis_result") or {}),
            "meta": dict(sc_meta),
            "pdf_report": sc_result.get("pdf_report"),
            "artifacts": list(sc_result.get("artifacts") or []),
        },
        "answer": str(state.get("final_answer") or ""),
    }


def tools_from_steps(steps: list[Any], keep_repeats: bool = False) -> list[str]:
    tools: list[str] = []
    for step in steps:
        tool = TOOL_NODE_MAP.get(str(step))
        if tool and (keep_repeats or tool not in tools):
            tools.append(tool)
    return tools


def rag_pipeline_actual(trace: dict[str, Any]) -> dict[str, bool]:
    rag = trace.get("rag") if isinstance(trace.get("rag"), dict) else {}
    raw = rag.get("retrieval_trace") if isinstance(rag.get("retrieval_trace"), dict) else {}
    eval_trace = rag.get("eval_trace") if isinstance(rag.get("eval_trace"), dict) else {}
    return {
        "bm25": bool(raw.get("bm25_executed") or eval_trace.get("bm25", {}).get("called")),
        "vector": bool(raw.get("vector_executed") or eval_trace.get("vector", {}).get("called")),
        "rrf": bool(raw.get("rrf_executed") or eval_trace.get("rrf", {}).get("called")),
        "bge_reranker": bool(raw.get("reranker_executed") or eval_trace.get("reranker", {}).get("called")),
    }


def derive_analysis_type(params: dict[str, Any]) -> str:
    if params.get("deep_analysis"):
        return "deep_analysis"
    if params.get("need_gene_corr") or params.get("gene_list") or params.get("target_genes"):
        return "gene_correlation"
    if params.get("need_batch_correction"):
        return "batch_correction"
    return "basic_sc_analysis"


def path_matches(expected: str, actual: str) -> bool:
    if not expected:
        return True
    if not actual:
        return False
    expected_path = (PROJECT_ROOT.parent / expected).resolve() if not Path(expected).is_absolute() else Path(expected).resolve()
    actual_path = Path(actual).expanduser().resolve()
    return actual_path == expected_path or actual_path.name == expected_path.name or str(actual_path).endswith(str(expected_path))


def compare_params(expected: dict[str, Any], trace: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    if not expected:
        return True, {}, []
    sc = trace.get("sc_analysis") if isinstance(trace.get("sc_analysis"), dict) else {}
    params = dict(sc.get("params") or {})
    result = dict(sc.get("result") or {})
    actual = dict(params)
    actual.update({key: result.get(key) for key in ("has_batch", "batch_key", "target_genes_for_corr") if key in result})
    failures: list[str] = []
    if "input_file" in expected and not path_matches(str(expected["input_file"]), str(params.get("h5ad_path") or sc.get("meta", {}).get("input_h5ad") or "")):
        failures.append("input_file mismatch")
    if "analysis_type" in expected and str(expected["analysis_type"]) != derive_analysis_type(params):
        failures.append(f"analysis_type mismatch: expected {expected['analysis_type']}, got {derive_analysis_type(params)}")
    if expected.get("need_batch_correction") is not None and bool(params.get("need_batch_correction")) != bool(expected["need_batch_correction"]):
        failures.append("need_batch_correction mismatch")
    expected_genes = expected.get("target_genes")
    if expected_genes:
        actual_genes = list(params.get("gene_list") or result.get("target_genes_for_corr") or params.get("target_genes") or [])
        missing = [gene for gene in expected_genes if gene not in actual_genes]
        if missing:
            failures.append(f"missing target_genes: {missing}")
    if expected.get("deep_analysis") is not None and bool(params.get("deep_analysis")) != bool(expected["deep_analysis"]):
        failures.append("deep_analysis mismatch")
    return not failures, actual, failures


def evaluate_answer_constraints(case: dict[str, Any], trace: dict[str, Any]) -> tuple[bool, list[str]]:
    answer = str(trace.get("answer") or "")
    normalized = answer.casefold()
    failures: list[str] = []
    for expected in case.get("answer_must_include") or []:
        if str(expected).casefold() not in normalized:
            failures.append(f"answer missing required text: {expected}")
    for forbidden in case.get("answer_must_not_include") or []:
        if str(forbidden).casefold() in normalized:
            failures.append(f"answer contains forbidden text: {forbidden}")
    any_group = [str(item) for item in case.get("answer_any_include") or []]
    if any_group and not any(item.casefold() in normalized for item in any_group):
        failures.append(f"answer missing any accepted text: {any_group}")
    return not failures, failures


def evaluate_tool_case(case: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    expected_tools = set(case.get("expected_tools") or [])
    forbidden_tools = set(case.get("forbidden_tools") or [])
    actual_tools = set(trace.get("tools_called") or [])
    actual_order = list(trace.get("tool_order") or trace.get("tools_called") or [])
    criteria = dict(case.get("pass_criteria") or {})
    failures: list[str] = []
    intent_ok = not criteria.get("intent_must_match", True) or trace.get("intent") == case.get("expected_intent")
    if not intent_ok:
        failures.append(f"intent mismatch: expected {case.get('expected_intent')}, got {trace.get('intent')}")
    tool_recall_ok = expected_tools.issubset(actual_tools)
    if criteria.get("all_expected_tools_called", True) and not tool_recall_ok:
        failures.append(f"missing tools: {sorted(expected_tools - actual_tools)}")
    exact_match_ok = actual_tools == expected_tools
    forbidden_hit = forbidden_tools & actual_tools
    if forbidden_hit:
        failures.append(f"forbidden tools called: {sorted(forbidden_hit)}")
    pipeline_expected = dict(case.get("expected_rag_pipeline") or case.get("must_use_rag_pipeline") or {})
    pipeline_actual = rag_pipeline_actual(trace)
    pipeline_ok = True
    if pipeline_expected:
        for key, expected in pipeline_expected.items():
            if bool(pipeline_actual.get(key)) != bool(expected):
                pipeline_ok = False
                failures.append(f"rag pipeline mismatch: {key} expected {expected}, got {pipeline_actual.get(key)}")
    params_ok, actual_params, param_failures = compare_params(dict(case.get("expected_params") or {}), trace)
    failures.extend(param_failures)
    order_ok = True
    if criteria.get("rag_called_before_web"):
        order_ok = "rag" in actual_order and "web_search" in actual_order and actual_order.index("rag") < actual_order.index("web_search")
        if not order_ok:
            failures.append(f"call order mismatch: expected rag before web_search, got {actual_order}")
    if criteria.get("web_called_after_low_confidence"):
        web_meta = trace.get("web_search", {}).get("meta", {}) if isinstance(trace.get("web_search"), dict) else {}
        if not bool(web_meta.get("triggered_after_low_confidence_rag")) and "web_search" in actual_tools:
            failures.append("web_search was not marked as triggered_after_low_confidence_rag")
    llm_ok = int(trace.get("llm_call_count") or 0) <= int(case.get("max_llm_calls") or 8)
    if not llm_ok:
        failures.append(f"llm_call_count exceeded limit: {trace.get('llm_call_count')}")
    answer_ok, answer_failures = evaluate_answer_constraints(case, trace)
    failures.extend(answer_failures)
    return {
        "passed": not failures,
        "failures": failures,
        "metrics": {
            "intent_ok": intent_ok,
            "tool_exact_match_ok": exact_match_ok,
            "tool_recall_ok": tool_recall_ok,
            "tool_precision_ok": not bool(actual_tools - expected_tools - forbidden_tools),
            "forbidden_violation": bool(forbidden_hit),
            "rag_pipeline_ok": pipeline_ok if pipeline_expected else None,
            "parameter_ok": params_ok if case.get("expected_params") else None,
            "call_order_ok": order_ok if criteria.get("rag_called_before_web") else None,
            "llm_call_count_ok": llm_ok,
            "answer_constraints_ok": answer_ok if any(case.get(key) for key in ("answer_must_include", "answer_must_not_include", "answer_any_include")) else None,
        },
        "expected": {"tools": sorted(expected_tools), "params": case.get("expected_params") or {}, "rag_pipeline": pipeline_expected},
        "actual": {"tools": sorted(actual_tools), "params": actual_params, "rag_pipeline": pipeline_actual, "tool_order": actual_order},
    }


def run_literature_case(case: dict[str, Any]) -> dict[str, Any]:
    from backend.tools.RAG import run_rag

    result = run_rag(
        query=str(case["query"]),
        knowledge_base_path=str(case.get("knowledge_base_path") or DEFAULT_KNOWLEDGE_BASE_PATH),
        index_dir=str(case.get("index_dir") or DEFAULT_RAG_INDEX_DIR),
        multi_query_count=case.get("multi_query_count") or 1,
    )
    eval_trace = dict(result.get("eval_trace") or result.get("meta", {}).get("eval_trace") or {})
    rag_trace = dict(result.get("retrieval_trace") or result.get("meta", {}).get("retrieval_trace") or {})
    trace = {
        "intent": "professional_qa",
        "tools_called": ["rag"],
        "tool_order": ["rag"],
        "llm_call_count": 0,
        "rag": {
            "called": True,
            "retrieval_trace": rag_trace,
            "eval_trace": eval_trace,
            "final_context_chunks": list(eval_trace.get("final_context_chunks") or result.get("chunks") or []),
        },
        "web_search": {"called": False},
        "sc_analysis": {"called": False, "params": {}},
        "answer": str(result.get("answer") or ""),
    }
    evaluation = evaluate_literature_case(case, trace)
    return {"trace": trace, "evaluation": evaluation}


def result_matches_gold(item: dict[str, Any], case: dict[str, Any]) -> tuple[bool, list[str]]:
    text = " ".join(str(item.get(key) or "") for key in ("chunk_id", "doc_id", "title", "source_path", "text")).lower()
    reasons: list[str] = []
    for chunk_id in case.get("gold_chunk_ids") or []:
        if str(item.get("chunk_id")) == str(chunk_id):
            reasons.append(f"chunk:{chunk_id}")
    for doc_id in case.get("gold_doc_ids") or []:
        if str(item.get("doc_id")) == str(doc_id):
            reasons.append(f"doc:{doc_id}")
    for title in case.get("gold_titles") or []:
        if str(title).lower() in text:
            reasons.append(f"title:{title}")
    for keyword in case.get("gold_keywords") or []:
        if str(keyword).lower() in text:
            reasons.append(f"keyword:{keyword}")
    for pattern in case.get("gold_text_patterns") or []:
        if re.search(str(pattern), text, flags=re.IGNORECASE):
            reasons.append(f"pattern:{pattern}")
    return bool(reasons), reasons


def relevance_vector(results: list[dict[str, Any]], case: dict[str, Any]) -> list[int]:
    return [1 if result_matches_gold(item, case)[0] else 0 for item in results]


def dcg(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances[:k], start=1))


def evaluate_literature_case(case: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    rag = trace.get("rag") if isinstance(trace.get("rag"), dict) else {}
    eval_trace = rag.get("eval_trace") if isinstance(rag.get("eval_trace"), dict) else {}
    final_chunks = list(eval_trace.get("final_context_chunks") or rag.get("final_context_chunks") or [])
    relevances = relevance_vector(final_chunks, case)
    first_hit = next((idx for idx, rel in enumerate(relevances, start=1) if rel), None)
    ideal = sorted(relevances, reverse=True)
    metrics = {
        "recall@1": 1.0 if any(relevances[:1]) else 0.0,
        "recall@3": 1.0 if any(relevances[:3]) else 0.0,
        "recall@5": 1.0 if any(relevances[:5]) else 0.0,
        "recall@10": 1.0 if any(relevances[:10]) else 0.0,
        "mrr": 1.0 / first_hit if first_hit else 0.0,
        "ndcg@5": dcg(relevances, 5) / max(dcg(ideal, 5), 1e-12),
        "ndcg@10": dcg(relevances, 10) / max(dcg(ideal, 10), 1e-12),
        "source_hit": 1.0 if any(relevances) else 0.0,
    }
    pipeline_expected = dict(case.get("must_use_rag_pipeline") or {})
    pipeline_actual = rag_pipeline_actual(trace)
    failures: list[str] = []
    for key, expected in pipeline_expected.items():
        if bool(pipeline_actual.get(key)) != bool(expected):
            failures.append(f"pipeline mismatch: {key}")
    if not any(relevances):
        failures.append("no final top chunk matched gold labels")
    expected_confidence = str(case.get("expected_rag_confidence") or "").lower()
    confidence_result = str(rag.get("retrieval_trace", {}).get("confidence_result") or "").lower()
    if expected_confidence == "high" and confidence_result not in {"sufficient", "high"}:
        failures.append(f"confidence mismatch: expected high/sufficient, got {confidence_result}")
    if expected_confidence == "low" and confidence_result not in {"insufficient", "low"}:
        failures.append(f"confidence mismatch: expected low/insufficient, got {confidence_result}")
    hit_details = []
    for rank, item in enumerate(final_chunks, start=1):
        matched, reasons = result_matches_gold(item, case)
        if matched:
            hit_details.append({"rank": rank, "chunk_id": item.get("chunk_id"), "doc_id": item.get("doc_id"), "title": item.get("title"), "reasons": reasons})
    return {
        "passed": not failures,
        "failures": failures,
        "metrics": metrics,
        "expected": {
            "gold_doc_ids": case.get("gold_doc_ids") or [],
            "gold_chunk_ids": case.get("gold_chunk_ids") or [],
            "gold_titles": case.get("gold_titles") or [],
            "gold_keywords": case.get("gold_keywords") or [],
            "rag_pipeline": pipeline_expected,
        },
        "actual": {
            "rag_pipeline": pipeline_actual,
            "hit_details": hit_details,
            "top_chunks": [{"rank": idx, "chunk_id": item.get("chunk_id"), "doc_id": item.get("doc_id"), "title": item.get("title")} for idx, item in enumerate(final_chunks[:10], start=1)],
        },
    }


def run_agent_case(case: dict[str, Any], mode: str, user_id: str) -> dict[str, Any]:
    session_id = f"eval_{case['id']}_{uuid4().hex[:8]}"
    if mode == "api":
        payload = call_agent_api(case, user_id, session_id)
    else:
        payload = call_agent_direct(case, user_id, session_id)
    trace = payload["trace"]
    return {"trace": trace, "evaluation": evaluate_tool_case(case, trace)}


def run_conversation_case(case: dict[str, Any], mode: str, user_id: str) -> dict[str, Any]:
    session_id = f"eval_{case['id']}_{uuid4().hex[:8]}"
    traces: list[dict[str, Any]] = []
    turn_results: list[dict[str, Any]] = []
    base_settings = dict(case.get("workspace_settings") or {})
    for index, turn in enumerate(case.get("turns") or [], start=1):
        turn_case = dict(case)
        turn_case.update(turn)
        turn_case["query"] = turn.get("query") or turn.get("user") or ""
        workspace_settings = dict(base_settings)
        workspace_settings.update(dict(turn.get("workspace_settings") or {}))
        turn_case["workspace_settings"] = workspace_settings
        if mode == "api":
            payload = call_agent_api(turn_case, user_id, session_id)
        else:
            payload = call_agent_direct(turn_case, user_id, session_id)
        trace = payload["trace"]
        traces.append(trace)
        turn_results.append({"turn": index, "query": turn_case["query"], "trace": trace})
    if not traces:
        raise ValueError(f"Conversation case {case.get('id')} has no turns.")
    final_case = dict(case)
    final_case["query"] = (case.get("turns") or [{}])[-1].get("query", "")
    if "expected_final_tools" in case:
        final_case["expected_tools"] = case.get("expected_final_tools") or []
    evaluation = evaluate_tool_case(final_case, traces[-1])
    evaluation["turn_results"] = turn_results
    return {"trace": {"turns": traces, **traces[-1]}, "evaluation": evaluation}


def run_concurrency_case(case: dict[str, Any], mode: str, user_id: str) -> dict[str, Any]:
    requests = list(case.get("requests") or [])
    if not requests:
        raise ValueError(f"Concurrency case {case.get('id')} has no requests.")
    if mode == "direct":
        from backend.agent import get_graph

        get_graph()
    max_workers = max(1, min(int(case.get("max_workers") or len(requests)), len(requests)))
    shared_user = bool(case.get("shared_user", True))
    base_settings = dict(case.get("workspace_settings") or {})
    started_at = time.perf_counter()

    def run_one(index: int, request: dict[str, Any]) -> dict[str, Any]:
        request_case = dict(case)
        request_case.update(request)
        request_case["id"] = request.get("id") or f"{case.get('id')}_request_{index}"
        request_case["query"] = request.get("query") or request.get("user") or ""
        workspace_settings = dict(base_settings)
        workspace_settings.update(dict(request.get("workspace_settings") or {}))
        request_case["workspace_settings"] = workspace_settings
        request_user_id = user_id if shared_user else f"{user_id}_{index}"
        session_id = str(request.get("session_id") or f"eval_{case['id']}_{index}_{uuid4().hex[:8]}")
        request_started = time.perf_counter()
        payload = call_agent_api(request_case, request_user_id, session_id) if mode == "api" else call_agent_direct(request_case, request_user_id, session_id)
        evaluation = evaluate_tool_case(request_case, payload["trace"])
        return {
            "index": index,
            "id": request_case["id"],
            "query": request_case["query"],
            "session_id": session_id,
            "elapsed_ms": elapsed_ms(request_started),
            "passed": bool(evaluation.get("passed")),
            "evaluation": evaluation,
            "trace": payload["trace"],
        }

    request_results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(run_one, index, request): index for index, request in enumerate(requests, start=1)}
        for future in concurrent.futures.as_completed(future_to_index):
            request_results.append(future.result())
    request_results.sort(key=lambda item: int(item.get("index") or 0))
    failures: list[str] = []
    for item in request_results:
        for failure in item.get("evaluation", {}).get("failures") or []:
            failures.append(f"{item.get('id')}: {failure}")
    unique_sessions_ok = len({item.get("session_id") for item in request_results}) == len(request_results)
    if bool(case.get("require_unique_sessions", True)) and not unique_sessions_ok:
        failures.append("concurrent requests did not use unique sessions")
    trace = {"requests": [item.get("trace") for item in request_results], "elapsed_ms": elapsed_ms(started_at), "max_workers": max_workers}
    return {
        "trace": trace,
        "evaluation": {
            "passed": not failures,
            "failures": failures,
            "metrics": {
                "request_count": len(request_results),
                "request_pass_rate": mean([float(item.get("passed", False)) for item in request_results]),
                "unique_sessions_ok": unique_sessions_ok,
                "elapsed_ms": trace["elapsed_ms"],
                "max_workers": max_workers,
            },
            "request_results": request_results,
        },
    }


def summarize_results(results: list[dict[str, Any]], metadata: dict[str, Any], mode: str) -> dict[str, Any]:
    tool_results = [row for row in results if row.get("category") in {"tool_call", "sc_analysis", "conversation"}]
    lit_results = [row for row in results if row.get("category") == "literature_recall"]
    concurrency_results = [row for row in results if row.get("category") == "concurrency"]
    tool_metrics = aggregate_tool_metrics(tool_results)
    literature_metrics = aggregate_literature_metrics(lit_results)
    concurrency_metrics = aggregate_concurrency_metrics(concurrency_results)
    threshold_status = evaluate_thresholds(tool_metrics, literature_metrics)
    failed = [row for row in results if not row.get("passed")]
    return {
        "mode": mode,
        "case_count": len(results),
        "passed_count": len(results) - len(failed),
        "failed_count": len(failed),
        "failed_case_ids": [row.get("id") for row in failed],
        "metadata": metadata,
        "tool_call": tool_metrics,
        "literature_recall": literature_metrics,
        "concurrency": concurrency_metrics,
        "thresholds": DEFAULT_THRESHOLDS,
        "threshold_status": threshold_status,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def aggregate_tool_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row.get("evaluation", {}).get("metrics", {}) for row in rows]
    expected_total = sum(len(row.get("evaluation", {}).get("expected", {}).get("tools", [])) for row in rows)
    actual_total = sum(len(row.get("evaluation", {}).get("actual", {}).get("tools", [])) for row in rows)
    hit_total = sum(len(set(row.get("evaluation", {}).get("expected", {}).get("tools", [])) & set(row.get("evaluation", {}).get("actual", {}).get("tools", []))) for row in rows)
    rag_pipeline_values = [float(item["rag_pipeline_ok"]) for item in metrics if item.get("rag_pipeline_ok") is not None]
    parameter_values = [float(item["parameter_ok"]) for item in metrics if item.get("parameter_ok") is not None]
    order_values = [float(item["call_order_ok"]) for item in metrics if item.get("call_order_ok") is not None]
    answer_values = [float(item["answer_constraints_ok"]) for item in metrics if item.get("answer_constraints_ok") is not None]
    return {
        "case_count": len(rows),
        "intent_accuracy": mean([float(item.get("intent_ok", False)) for item in metrics]),
        "tool_exact_match_accuracy": mean([float(item.get("tool_exact_match_ok", False)) for item in metrics]),
        "tool_recall": hit_total / expected_total if expected_total else 0.0,
        "tool_precision": hit_total / actual_total if actual_total else 0.0,
        "forbidden_tool_violation_rate": mean([float(item.get("forbidden_violation", False)) for item in metrics]),
        "rag_pipeline_accuracy": mean(rag_pipeline_values),
        "parameter_accuracy": mean(parameter_values),
        "call_order_accuracy": mean(order_values),
        "answer_constraint_accuracy": mean(answer_values),
        "llm_call_count_violation_count": sum(1 for item in metrics if not item.get("llm_call_count_ok", True)),
    }


def aggregate_literature_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row.get("evaluation", {}).get("metrics", {}) for row in rows]
    return {
        "case_count": len(rows),
        "recall@1": mean([float(item.get("recall@1", 0.0)) for item in metrics]),
        "recall@3": mean([float(item.get("recall@3", 0.0)) for item in metrics]),
        "recall@5": mean([float(item.get("recall@5", 0.0)) for item in metrics]),
        "recall@10": mean([float(item.get("recall@10", 0.0)) for item in metrics]),
        "mrr": mean([float(item.get("mrr", 0.0)) for item in metrics]),
        "ndcg@5": mean([float(item.get("ndcg@5", 0.0)) for item in metrics]),
        "ndcg@10": mean([float(item.get("ndcg@10", 0.0)) for item in metrics]),
        "source_hit_rate": mean([float(item.get("source_hit", 0.0)) for item in metrics]),
        "pipeline_coverage": {
            "bm25": mean([float(row.get("evaluation", {}).get("actual", {}).get("rag_pipeline", {}).get("bm25", False)) for row in rows]),
            "vector": mean([float(row.get("evaluation", {}).get("actual", {}).get("rag_pipeline", {}).get("vector", False)) for row in rows]),
            "rrf": mean([float(row.get("evaluation", {}).get("actual", {}).get("rag_pipeline", {}).get("rrf", False)) for row in rows]),
            "bge_reranker": mean([float(row.get("evaluation", {}).get("actual", {}).get("rag_pipeline", {}).get("bge_reranker", False)) for row in rows]),
        },
    }


def aggregate_concurrency_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row.get("evaluation", {}).get("metrics", {}) for row in rows]
    request_counts = [int(item.get("request_count") or 0) for item in metrics]
    weighted_pass = sum(float(item.get("request_pass_rate", 0.0)) * int(item.get("request_count") or 0) for item in metrics)
    total_requests = sum(request_counts)
    elapsed_values = [float(item.get("elapsed_ms", 0.0)) for item in metrics]
    case_performance: list[dict[str, Any]] = []
    request_latencies: list[float] = []
    for row in rows:
        row_metrics = row.get("evaluation", {}).get("metrics", {})
        request_results = row.get("evaluation", {}).get("request_results", [])
        latencies = [float(item.get("elapsed_ms", 0.0)) for item in request_results]
        request_latencies.extend(latencies)
        elapsed = float(row_metrics.get("elapsed_ms", 0.0))
        request_count = int(row_metrics.get("request_count") or len(request_results))
        slowest = max(request_results, key=lambda item: float(item.get("elapsed_ms", 0.0)), default={})
        case_performance.append({
            "id": row.get("id"),
            "passed": bool(row.get("passed")),
            "request_count": request_count,
            "max_workers": int(row_metrics.get("max_workers") or 0),
            "elapsed_ms": elapsed,
            "throughput_rps": request_count / (elapsed / 1000.0) if elapsed else 0.0,
            "request_latency_ms": {
                "avg": mean(latencies),
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "max": max(latencies, default=0.0),
            },
            "slowest_request": {
                "id": slowest.get("id"),
                "elapsed_ms": float(slowest.get("elapsed_ms", 0.0)) if slowest else 0.0,
                "passed": bool(slowest.get("passed", False)) if slowest else False,
            },
        })
    total_elapsed_ms = sum(elapsed_values)
    return {
        "case_count": len(rows),
        "request_count": total_requests,
        "case_pass_rate": mean([float(row.get("passed", False)) for row in rows]),
        "request_pass_rate": weighted_pass / total_requests if total_requests else 0.0,
        "unique_sessions_rate": mean([float(item.get("unique_sessions_ok", False)) for item in metrics]),
        "total_elapsed_ms": total_elapsed_ms,
        "avg_case_elapsed_ms": mean(elapsed_values),
        "max_elapsed_ms": max([float(item.get("elapsed_ms", 0.0)) for item in metrics], default=0.0),
        "aggregate_throughput_rps": total_requests / (total_elapsed_ms / 1000.0) if total_elapsed_ms else 0.0,
        "request_latency_ms": {
            "avg": mean(request_latencies),
            "p50": percentile(request_latencies, 0.50),
            "p95": percentile(request_latencies, 0.95),
            "max": max(request_latencies, default=0.0),
        },
        "case_performance": case_performance,
    }


def evaluate_thresholds(tool_metrics: dict[str, Any], literature_metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "intent_accuracy": float(tool_metrics.get("intent_accuracy", 0.0)) >= DEFAULT_THRESHOLDS["intent_accuracy"],
        "tool_recall": float(tool_metrics.get("tool_recall", 0.0)) >= DEFAULT_THRESHOLDS["tool_recall"],
        "forbidden_tool_violation_rate": float(tool_metrics.get("forbidden_tool_violation_rate", 1.0)) <= DEFAULT_THRESHOLDS["forbidden_tool_violation_rate"],
        "rag_pipeline_accuracy": float(tool_metrics.get("rag_pipeline_accuracy", 0.0)) >= DEFAULT_THRESHOLDS["rag_pipeline_accuracy"],
        "literature_recall@5": float(literature_metrics.get("recall@5", 0.0)) >= DEFAULT_THRESHOLDS["literature_recall@5"],
        "literature_recall@10": float(literature_metrics.get("recall@10", 0.0)) >= DEFAULT_THRESHOLDS["literature_recall@10"],
    }


def write_summary_md(path: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    failed = [row for row in results if not row.get("passed")]
    lines = [
        "# Agent Evaluation Summary",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Cases: {summary['case_count']}",
        f"- Passed: {summary['passed_count']}",
        f"- Failed: {summary['failed_count']}",
        f"- Generated at: {summary['generated_at']}",
        "",
        "## Tool Calling",
    ]
    for key, value in summary["tool_call"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Literature Recall"])
    for key, value in summary["literature_recall"].items():
        lines.append(f"- {key}: {json_dump(value) if isinstance(value, dict) else value}")
    lines.extend(["", "## Concurrency"])
    for key, value in summary["concurrency"].items():
        if key == "case_performance":
            continue
        lines.append(f"- {key}: {json_dump(value) if isinstance(value, dict) else value}")
    if summary["concurrency"].get("case_performance"):
        lines.extend([
            "",
            "### Concurrency Performance",
            "| case | passed | requests | workers | wall_ms | throughput_rps | avg_req_ms | p50_req_ms | p95_req_ms | max_req_ms | slowest_request |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for item in summary["concurrency"]["case_performance"]:
            latency = item.get("request_latency_ms", {})
            slowest = item.get("slowest_request", {})
            slowest_text = f"{slowest.get('id')} ({slowest.get('elapsed_ms', 0.0):.0f} ms)"
            lines.append(
                f"| {item.get('id')} | {item.get('passed')} | {item.get('request_count')} | "
                f"{item.get('max_workers')} | {item.get('elapsed_ms', 0.0):.0f} | "
                f"{item.get('throughput_rps', 0.0):.3f} | {latency.get('avg', 0.0):.0f} | "
                f"{latency.get('p50', 0.0):.0f} | {latency.get('p95', 0.0):.0f} | "
                f"{latency.get('max', 0.0):.0f} | {slowest_text} |"
            )
    lines.extend(["", "## Thresholds"])
    for key, value in summary["threshold_status"].items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    lines.extend(["", "## Failed Cases"])
    if not failed:
        lines.append("- None")
    for row in failed:
        lines.append(f"- {row.get('id')}: {'; '.join(row.get('evaluation', {}).get('failures', row.get('failures', [])))}")
    lines.extend([
        "",
        "## Notes",
        "- Literature gold labels are built from local `data/local_knowledge_index/chunks.jsonl` metadata where available.",
        "- If doc/chunk/title labels are missing, the evaluator falls back to keyword or text-pattern matching and records that limitation in case results.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = choose_mode(args.mode)
    cases, metadata = load_cases(dataset_dir)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.get("id") in wanted]
    user_id = f"eval_{int(time.time())}_{uuid4().hex[:6]}"
    metadata.update({"user_id": user_id, "dataset_dir": str(dataset_dir), "output_dir": str(output_dir)})
    results: list[dict[str, Any]] = []
    external_errors = external_error_types()
    for index, case in enumerate(cases, start=1):
        category = str(case.get("category") or "")
        base = {"id": case.get("id"), "category": category, "query": case.get("query"), "source_file": case.get("source_file")}
        print(f"[{index}/{len(cases)}] {case.get('id')} ({category})", flush=True)
        try:
            if category == "literature_recall":
                payload = run_literature_case(case)
            elif category == "conversation":
                payload = run_conversation_case(case, mode, user_id)
            elif category == "concurrency":
                payload = run_concurrency_case(case, mode, user_id)
            elif category in {"tool_call", "sc_analysis"}:
                payload = run_agent_case(case, mode, user_id)
            else:
                raise ValueError(f"Unsupported case category: {category}")
            evaluation = payload["evaluation"]
            results.append({**base, "passed": bool(evaluation.get("passed")), "evaluation": evaluation, "eval_trace": payload["trace"]})
        except external_errors as exc:
            results.append({**base, "passed": False, "evaluation": {"passed": False, "failures": [f"external service error: {type(exc).__name__}: {exc}"]}, "eval_trace": {}})
    summary = summarize_results(results, metadata, mode)
    write_jsonl(output_dir / "case_results.jsonl", results)
    (output_dir / "summary.json").write_text(json_dump(summary, pretty=True) + "\n", encoding="utf-8")
    write_summary_md(output_dir / "summary.md", summary, results)
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Wrote {output_dir / 'case_results.jsonl'}")
    if args.no_fail_on_threshold:
        return 0
    return 0 if all(summary["threshold_status"].values()) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate agent tool-calling and literature recall behavior.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mode", choices=["auto", "api", "direct"], default="auto")
    parser.add_argument("--case-id", action="append", help="Run only selected case id. Can be provided multiple times.")
    parser.add_argument("--no-fail-on-threshold", action="store_true", help="Always exit 0 after writing reports.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_all(parse_args()))
