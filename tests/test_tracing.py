"""P7.1 — coordinator run config.

The run-config helper names + tags a coordinator run so its LangSmith trace is
findable under the project. Env activation is ``configure_langsmith_env``'s job
(covered in test_config.py), not this module's — so these tests pin only the
config shape.
"""
from __future__ import annotations

from app.orchestrator.tracing import coordinator_run_config


def test_run_config_names_and_tags_the_run() -> None:
    """A stable run name + the dry-run tag + run_id metadata so the trace is
    findable under the project."""
    config = coordinator_run_config(run_id="abc123", dry_run=True)
    assert config["run_name"] == "coordinator-abc123"
    assert "orchestrator" in config["tags"]
    assert "dry_run" in config["tags"]
    assert config["metadata"] == {"run_id": "abc123", "dry_run": True}


def test_run_config_tags_live_when_not_dry_run() -> None:
    """A real run is tagged ``live``, not ``dry_run`` — so traces can be
    filtered to the runs that actually spent tokens."""
    config = coordinator_run_config(run_id="r", dry_run=False)
    assert "live" in config["tags"]
    assert "dry_run" not in config["tags"]
    assert config["metadata"]["dry_run"] is False
