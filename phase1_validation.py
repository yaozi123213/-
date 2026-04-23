from hermes.control_plane import HermesControlPlane
from hermes.execution_graph import HermesExecutionGraph, InMemoryCheckpointStore
from hermes.langgraph_bridge import LangGraphBridge
from hermes.session_runner import SessionRunner


store = InMemoryCheckpointStore()
graph = HermesExecutionGraph(store, max_retries=1)
bridge = LangGraphBridge(graph)
control = HermesControlPlane(profile={"name": "tester"}, gateway="internal")
runner = SessionRunner(control_plane=control, bridge=bridge)

ok = runner.run_session("s1", "hello")
assert ok["status"] == "completed"
assert ok["output"] == "executed:hello"

bad = runner.run_session("s2", "fail-now")
assert bad["status"] == "degraded"
assert bad["requires_human"] is True

print("phase1 validation passed")
