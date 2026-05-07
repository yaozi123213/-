"""LangGraph-style execution state machine for Hermes Phase 1."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TypedDict

from hermes.langgraph_bridge import InMemoryCheckpointStore


class ExecutionState(TypedDict, total=False):
    session_id: str
    input_text: str
    output_text: str
    status: str
    retries: int
    error: str


MAX_RETRIES = 1


def execute_step(state: ExecutionState) -> ExecutionState:
    text = state.get("input_text", "")
    if not text:
        state["status"] = "degraded"
        state["error"] = "empty-input"
        return state
    state["status"] = "ok"
    state["output_text"] = text
    state.pop("error", None)
    return state


def retry_or_degrade_step(state: ExecutionState) -> ExecutionState:
    retries = state.get("retries", 0)
    if state.get("status") == "ok":
        return state
    if retries < MAX_RETRIES:
        state["retries"] = retries + 1
        state["status"] = "retry"
        return state
    state["status"] = "degraded"
    return state


def human_in_the_loop_step(state: ExecutionState) -> ExecutionState:
    if state.get("status") == "degraded":
        state["status"] = "human_review_required"
    return state


def run_execution_flow(
    initial_state: ExecutionState,
    checkpoint_store: InMemoryCheckpointStore,
    session_id: str,
) -> ExecutionState:
    """Run the minimal Phase 1 execution flow with checkpoint snapshots."""

    state: ExecutionState = deepcopy(initial_state)

    state = execute_step(state)
    checkpoint_store.save(session_id, state)

    state = retry_or_degrade_step(state)
    checkpoint_store.save(session_id, state)

    if state.get("status") == "retry":
        state = execute_step(state)
        checkpoint_store.save(session_id, state)
        state = retry_or_degrade_step(state)
        checkpoint_store.save(session_id, state)

    if state.get("status") == "degraded":
        state = human_in_the_loop_step(state)
        checkpoint_store.save(session_id, state)

    return deepcopy(state)
