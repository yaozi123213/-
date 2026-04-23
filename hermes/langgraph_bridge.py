"""Phase 1 bridge between Hermes control-plane and LangGraph execution-plane.

Hermes remains responsible for profile/memory/gateway/tool registration and passes
resolved execution inputs into this bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph


StateFactory = Callable[[], StateGraph]


@dataclass(frozen=True)
class LangGraphBridge:
    """Builds and compiles LangGraph with sync durability semantics.

    The compiled graph owns execution state as the source-of-truth.
    """

    graph_factory: StateFactory

    def compile(self) -> Any:
        graph = self.graph_factory()
        # Phase 1 durability requirement: synchronous checkpointer.
        checkpointer = MemorySaver()
        return graph.compile(checkpointer=checkpointer)
