import unittest

from hermes.execution_graph import run_execution_flow
from hermes.langgraph_bridge import InMemoryCheckpointStore, LangGraphBridge
from hermes.session_runner import SessionRunner


class Phase1IntegrationTests(unittest.TestCase):
    def test_checkpoint_store_returns_copies(self) -> None:
        store = InMemoryCheckpointStore()
        original = {"status": "ok", "nested": {"k": "v"}}

        saved = store.save("s1", original)
        saved["nested"]["k"] = "mutated"

        latest = store.latest("s1")
        self.assertEqual(latest["nested"]["k"], "v")

    def test_bridge_does_not_expose_mutable_references(self) -> None:
        store = InMemoryCheckpointStore()
        bridge = LangGraphBridge(store)
        initial = {"session_id": "s2", "input_text": "hello", "retries": 0, "status": "pending"}

        result = bridge.invoke("s2", initial, run_execution_flow)
        result["status"] = "tampered"

        latest = store.latest("s2")
        self.assertNotEqual(latest["status"], "tampered")

    def test_session_runner_success(self) -> None:
        runner = SessionRunner.create()
        result = runner.run("s3", "echo")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["output_text"], "echo")

    def test_session_runner_degrade_then_human_review(self) -> None:
        runner = SessionRunner.create()
        result = runner.run("s4", "")

        self.assertEqual(result["status"], "human_review_required")
        self.assertEqual(result["error"], "empty-input")


if __name__ == "__main__":
    unittest.main()
