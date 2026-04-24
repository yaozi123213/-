"""Minimal session runner wiring normalized Hermes input to execution flow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from hermes.execution_graph import run_execution_flow
from hermes.langgraph_bridge import InMemoryCheckpointStore, LangGraphBridge


@dataclass(frozen=True)
class SessionRunner:
    """Phase 1 session runner.

    Hermes remains the control-plane and passes only execution input here.
    """

    bridge: LangGraphBridge

    @classmethod
    def create(cls) -> "SessionRunner":
        return cls(bridge=LangGraphBridge(checkpoint_store=InMemoryCheckpointStore()))

    def run(self, session_id: str, input_text: str) -> dict[str, Any]:
        initial_state = {
            "session_id": session_id,
            "input_text": input_text,
            "status": "pending",
            "retries": 0,
        }
        result = self.bridge.invoke(session_id, initial_state, run_execution_flow)
        # Never return references into internal store/control-plane objects.
        return deepcopy(result)

    def latest_snapshot(self, session_id: str) -> dict[str, Any] | None:
        return self.bridge.checkpoint_store.latest(session_id)
