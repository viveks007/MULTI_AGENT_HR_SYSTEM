"""Basic agent tests."""

from agents.supervisor import SupervisorAgent

def test_supervisor_runs():
    s = SupervisorAgent()
    assert s.run() == "supervisor-run"
