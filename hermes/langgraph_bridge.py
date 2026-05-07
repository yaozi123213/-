"""Phase 1 bridge: Hermes control-plane calls into execution-plane flow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InMemoryCheckpointStore:
    """Sync durability store that snapshots state by value, not by reference."""

    _snapshots: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def save(self, session_id: str, state: dict[str, Any]) -> dict[str, Any]:
        snapshot = deepcopy(state)
        self._snapshots.setdefault(session_id, []).append(snapshot)
        return deepcopy(snapshot)

    def latest(self, session_id: str) -> dict[str, Any] | None:
        history = self._snapshots.get(session_id)
        if not history:
            return None
        return deepcopy(history[-1])

    def history(self, session_id: str) -> list[dict[str, Any]]:
        return deepcopy(self._snapshots.get(session_id, []))


@dataclass
class LangGraphBridge:
    """Execution-plane facade with sync checkpoints around execution flow."""

    checkpoint_store: InMemoryCheckpointStore

    def invoke(self, session_id: str, initial_state: dict[str, Any], flow: Any) -> dict[str, Any]:
        """Run flow and checkpoint start/end snapshots.

        Hermes control-plane state is intentionally not handled here.
        """

        start_state = deepcopy(initial_state)
        self.checkpoint_store.save(session_id, start_state)
        final_state = flow(deepcopy(start_state), self.checkpoint_store, session_id)
        self.checkpoint_store.save(session_id, final_state)
        return deepcopy(final_state)
