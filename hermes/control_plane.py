from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class HermesControlPlane:
    """Owns Hermes control-plane concerns.

    Phase 1 scope keeps these responsibilities out of the execution graph:
    profile, memory, gateway, and tools registration.
    """

    profile: dict[str, Any]
    memory: dict[str, Any] = field(default_factory=dict)
    gateway: str = "internal"
    tools: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def register_tool(self, name: str, tool: Callable[..., Any]) -> None:
        self.tools[name] = tool
