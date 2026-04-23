from dataclasses import dataclass

from hermes.control_plane import HermesControlPlane
from hermes.langgraph_bridge import LangGraphBridge


@dataclass(slots=True)
class SessionRunner:
    """Minimal runner integration for Phase 1.

    Hermes remains control-plane owner, while execution state comes from
    LangGraph-backed bridge.
    """

    control_plane: HermesControlPlane
    bridge: LangGraphBridge

    def run_session(self, session_id: str, text: str) -> dict[str, object]:
        state = self.bridge.invoke(session_id=session_id, text=text)
        return {
            "session_id": state.session_id,
            "status": state.status,
            "output": state.output,
            "requires_human": state.requires_human,
            "profile": self.control_plane.profile,
            "gateway": self.control_plane.gateway,
        }
