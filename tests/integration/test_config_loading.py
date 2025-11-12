"""Integration tests for the typed configuration loader."""

from __future__ import annotations

from textwrap import dedent

from finangpt.shared.config import load_config


def test_env_overrides_yaml(tmp_path, monkeypatch):
    yaml_contents = dedent(
        """
        analysis:
          max_query_steps: 4
          v2_analysis_enabled: false
        """
    )
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml_contents)

    monkeypatch.setenv("FINANGPT_CONFIG_FILE", str(yaml_path))

    config = load_config(reload=True)
    assert config.analysis.max_query_steps == 4
    assert config.analysis.v2_analysis_enabled is False

    monkeypatch.setenv("FINANGPT_ANALYSIS__MAX_QUERY_STEPS", "9")
    config = load_config(reload=True)
    assert config.analysis.max_query_steps == 9

    monkeypatch.setenv("FINANGPT_V2_ANALYSIS", "true")
    config = load_config(reload=True)
    assert config.analysis.v2_analysis_enabled is True

    monkeypatch.delenv("FINANGPT_ANALYSIS__MAX_QUERY_STEPS", raising=False)
    monkeypatch.delenv("FINANGPT_V2_ANALYSIS", raising=False)
