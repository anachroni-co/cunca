"""Isolated tests for config.config_manager module."""

import pytest
from config.config_manager import ConfigManager


class TestConfigManager:
    def test_init(self, tmp_path):
        cm = ConfigManager(config_dir=str(tmp_path))
        assert cm.configs == {}

    def test_load_config_toml(self, tmp_path):
        toml_content = b'[model]\nname = "test"\nhidden_size = 768\n'
        (tmp_path / "model.toml").write_bytes(toml_content)
        cm = ConfigManager(config_dir=str(tmp_path))
        cfg = cm.load_config("model")
        assert cfg["model"]["name"] == "test"
        assert cfg["model"]["hidden_size"] == 768

    def test_get_config_loaded(self, tmp_path):
        (tmp_path / "a.toml").write_bytes(b'x = 1\n')
        cm = ConfigManager(config_dir=str(tmp_path))
        cm.load_config("a")
        assert cm.get_config("a") == {"x": 1}

    def test_get_config_not_loaded(self, tmp_path):
        cm = ConfigManager(config_dir=str(tmp_path))
        assert cm.get_config("missing") is None

    def test_get_value_nested(self, tmp_path):
        (tmp_path / "c.toml").write_bytes(b'[a]\n[a.b]\nc = 42\n')
        cm = ConfigManager(config_dir=str(tmp_path))
        cm.load_config("c")
        assert cm.get_value("c", "a.b.c") == 42

    def test_get_value_default(self, tmp_path):
        cm = ConfigManager(config_dir=str(tmp_path))
        assert cm.get_value("missing", "x.y", default="fallback") == "fallback"

    def test_load_nonexistent_raises(self, tmp_path):
        cm = ConfigManager(config_dir=str(tmp_path))
        with pytest.raises(Exception):
            cm.load_config("nonexistent")

    def test_validate_config_pass(self, tmp_path):
        (tmp_path / "v.toml").write_bytes(b'name = "x"\ncount = 5\n')
        cm = ConfigManager(config_dir=str(tmp_path))
        cm.load_config("v")
        schema = {"name": str, "count": int}
        assert cm.validate_config("v", schema) is True

    def test_validate_config_missing_key(self, tmp_path):
        (tmp_path / "v.toml").write_bytes(b'name = "x"\n')
        cm = ConfigManager(config_dir=str(tmp_path))
        cm.load_config("v")
        schema = {"name": str, "missing_key": int}
        assert cm.validate_config("v", schema) is False

    def test_validate_config_wrong_type(self, tmp_path):
        (tmp_path / "v.toml").write_bytes(b'name = 123\n')
        cm = ConfigManager(config_dir=str(tmp_path))
        cm.load_config("v")
        schema = {"name": str}
        assert cm.validate_config("v", schema) is False

    def test_validate_not_loaded(self, tmp_path):
        cm = ConfigManager(config_dir=str(tmp_path))
        assert cm.validate_config("nope", {"x": str}) is False
