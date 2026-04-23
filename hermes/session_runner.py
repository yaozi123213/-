"""Minimal session runner wiring Hermes session calls to LangGraph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hermes.execution_graph import build_execution_graph
from hermes.langgraph_bridge import LangGraphBridge


@dataclass
class SessionRunner:
    """LangGraph-backed runner.

    Hermes is still expected to resolve profile/memory/gateway/tools before
    invoking this runner with normalized input.
    """

    graph: Any

    @classmethod
    def from_bridge(cls) -> "SessionRunner":
        bridge = LangGraphBridge(graph_factory=build_execution_graph)
        return cls(graph=bridge.compile())

    def run(self, session_id: str, input_text: str) -> dict[str, Any]:
        config = {"configurable": {"thread_id": session_id}}
        initial_state = {
            "session_id": session_id,
            "input_text": input_text,
            "status": "pending",
            "retries": 0,
        }
        result = self.graph.invoke(initial_state, config=config)
        return dict(result)
