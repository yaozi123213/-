"""LangGraph execution state machine for Hermes Phase 1 integration."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class ExecutionState(TypedDict, total=False):
    session_id: str
    input_text: str
    output_text: str
    status: str
    retries: int
    error: str


MAX_RETRIES = 1


def _execute(state: ExecutionState) -> ExecutionState:
    text = state.get("input_text", "")
    if not text:
        return {"status": "degraded", "error": "empty-input"}
    return {
        "status": "ok",
        "output_text": text,
    }


def _retry_or_degrade(state: ExecutionState) -> ExecutionState:
    retries = state.get("retries", 0)
    if state.get("status") == "ok":
        return state
    if retries < MAX_RETRIES:
        return {"retries": retries + 1, "status": "retry"}
    return {"status": "degraded"}


def _human_in_the_loop(state: ExecutionState) -> ExecutionState:
    if state.get("status") != "degraded":
        return state
    return {
        "status": "human_review_required",
    }


def _route_after_execute(state: ExecutionState) -> str:
    return "retry_or_degrade"


def _route_after_retry(state: ExecutionState) -> str:
    if state.get("status") == "retry":
        return "execute"
    if state.get("status") == "degraded":
        return "human_in_the_loop"
    return END


def build_execution_graph() -> StateGraph:
    graph = StateGraph(ExecutionState)
    graph.add_node("execute", _execute)
    graph.add_node("retry_or_degrade", _retry_or_degrade)
    graph.add_node("human_in_the_loop", _human_in_the_loop)

    graph.add_edge(START, "execute")
    graph.add_conditional_edges("execute", _route_after_execute)
    graph.add_conditional_edges("retry_or_degrade", _route_after_retry)
    graph.add_edge("human_in_the_loop", END)

    return graph
