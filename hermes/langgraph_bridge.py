from hermes.execution_graph import ExecutionState, HermesExecutionGraph


class LangGraphBridge:
    """Thin bridge between Hermes session flow and LangGraph execution plane."""

    def __init__(self, graph: HermesExecutionGraph) -> None:
        self.graph = graph

    def invoke(self, session_id: str, text: str) -> ExecutionState:
        state = ExecutionState(session_id=session_id, input_text=text)
        return self.graph.run(state)
