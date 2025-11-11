from __future__ import annotations

import os
import time

from src.utils import paths


class DummyConfig:
    def __init__(self, data=None):
        self._data = data or {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    @property
    def duckdb_path(self):
        return self._data.get('duckdb_path', 'financial_data.duckdb')


def test_data_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANGPT_DATA_DIR", str(tmp_path / "custom"))
    monkeypatch.setattr(paths, "load_config", lambda: DummyConfig({}))
    paths.reset_data_paths_cache_for_tests()

    data_dir = paths.get_data_dir()
    assert data_dir == (tmp_path / "custom")


def test_duckdb_path_migrates_legacy_file(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("FINANGPT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(paths, "load_config", lambda: DummyConfig({}))
    paths.reset_data_paths_cache_for_tests()

    legacy = tmp_path / "financial_data.duckdb"
    legacy.write_text("legacy")

    dest = paths.get_duckdb_path()
    assert dest.exists()
    assert dest.read_text() == "legacy"


def test_chart_retention_prunes(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANGPT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(paths, "load_config", lambda: DummyConfig({}))
    paths.reset_data_paths_cache_for_tests()

    charts_dir = paths.get_charts_dir()
    for i in range(5):
        file = charts_dir / f"chart_{i}.png"
        file.write_text("x")
        os.utime(file, (time.time() + i, time.time() + i))

    paths.prune_old_charts(limit=2)

    remaining = list(charts_dir.glob("*.png"))
    assert len(remaining) == 2
