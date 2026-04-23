from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExecutionState:
    session_id: str
    input_text: str
    attempts: int = 0
    status: str = "received"
    output: str | None = None
    requires_human: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryCheckpointStore:
    """Phase 1 sync durability store.

    This intentionally blocks until write is completed, making the last
    execution state in this store the source of truth for runtime state.
    """

    def __init__(self) -> None:
        self._states: dict[str, ExecutionState] = {}

    def save_sync(self, state: ExecutionState) -> None:
        self._states[state.session_id] = state

    def load(self, session_id: str) -> ExecutionState | None:
        return self._states.get(session_id)


class HermesExecutionGraph:
    """Minimal state machine for Phase 1.

    Execution-plane concerns:
    - state transitions
    - sync checkpoint writes
    - retry/degrade/human-in-the-loop paths
    """

    def __init__(self, checkpoint_store: InMemoryCheckpointStore, max_retries: int = 1) -> None:
        self.checkpoint_store = checkpoint_store
        self.max_retries = max_retries

    def run(self, state: ExecutionState) -> ExecutionState:
        state.status = "running"
        self.checkpoint_store.save_sync(state)

        while True:
            state.attempts += 1
            try:
                state.output = self._execute(state.input_text)
                state.status = "completed"
                state.requires_human = False
                self.checkpoint_store.save_sync(state)
                return state
            except Exception as exc:  # noqa: BLE001
                state.metadata["last_error"] = str(exc)
                if state.attempts <= self.max_retries:
                    state.status = "retrying"
                    self.checkpoint_store.save_sync(state)
                    continue
                state.status = "degraded"
                state.requires_human = True
                self.checkpoint_store.save_sync(state)
                return state

    @staticmethod
    def _execute(input_text: str) -> str:
        if input_text.startswith("fail"):
            raise RuntimeError("forced execution failure")
        return f"executed:{input_text}"
