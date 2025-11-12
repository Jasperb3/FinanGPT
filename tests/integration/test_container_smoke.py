"""Smoke tests for the dependency injection container."""

from __future__ import annotations

from textwrap import dedent

from finangpt.shared.config import load_config
from finangpt.shared.dependency_injection import Container


def test_analysis_orchestrator_can_be_resolved(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    conversation_db = tmp_path / "conversations.db"
    yaml_path.write_text(
        dedent(
            f"""
            database:
              duckdb:
                path: ":memory:"
              conversation:
                path: "{conversation_db}"
            cache:
              redis_url: memory://
            """
        )
    )

    monkeypatch.setenv("FINANGPT_CONFIG_FILE", str(yaml_path))
    load_config(reload=True)

    container = Container()
    orchestrator = container.analysis_orchestrator()

    assert orchestrator is not None
