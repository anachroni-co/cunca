"""Isolated tests for core.config module."""

import json
import pytest
from core.config import (
    Config,
    ModelConfig,
    RouterConfig,
    ModularModelConfig,
    distributed_jit,
    model_sharded_jit,
    batch_sharded_jit,
    create_unified_mesh,
    load_config,
)


class TestConfig:
    def test_init_defaults(self):
        c = Config()
        assert c.settings == {}
        assert c.config_file is None
        assert c.environment == "development"

    def test_init_with_kwargs(self):
        c = Config(model_name="test", batch_size=16)
        assert c.get("model_name") == "test"
        assert c.get("batch_size") == 16

    def test_get_default(self):
        c = Config()
        assert c.get("missing", 42) == 42

    def test_set(self):
        c = Config()
        c.set("lr", 0.01)
        assert c.get("lr") == 0.01

    def test_get_all_settings_returns_copy(self):
        c = Config(x=1)
        s = c.get_all_settings()
        s["x"] = 999
        assert c.get("x") == 1

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "cfg.json")
        c = Config(name="test", value=123)
        assert c.save_to_file(path) is True

        c2 = Config()
        assert c2.load_from_file(path) is True
        assert c2.get("name") == "test"
        assert c2.get("value") == 123

    def test_load_nonexistent(self):
        c = Config()
        assert c.load_from_file("/nonexistent/path.json") is False

    def test_validate_config_pass(self):
        c = Config(model_name="x", max_length=512, temperature=0.7)
        assert c.validate_config() is True

    def test_validate_config_fail(self):
        c = Config(model_name="x")
        assert c.validate_config() is False

    def test_training_config_defaults(self):
        c = Config()
        assert c.training_config.learning_rate == 0.001
        assert c.training_config.batch_size == 32
        assert c.training_config.optimizer == "adamw"


class TestModelConfig:
    def test_defaults(self):
        mc = ModelConfig()
        assert mc.model_name == "capibara-gpt"
        assert mc.get_parameter("temperature") == 0.7

    def test_custom_name(self):
        mc = ModelConfig("my-model")
        assert mc.model_name == "my-model"

    def test_set_and_get(self):
        mc = ModelConfig()
        mc.set_parameter("temperature", 1.0)
        assert mc.get_parameter("temperature") == 1.0

    def test_get_missing_default(self):
        mc = ModelConfig()
        assert mc.get_parameter("missing", 99) == 99


class TestRouterConfig:
    def test_defaults(self):
        rc = RouterConfig()
        assert rc.strategy == "default"
        assert rc.load_balancing is True
        assert rc.timeout == 30.0
        assert rc.max_retries == 3

    def test_to_dict(self):
        rc = RouterConfig()
        d = rc.to_dict()
        assert d["strategy"] == "default"
        assert "timeout" in d


class TestModularModelConfig:
    def test_add_and_get_module(self):
        mc = ModularModelConfig()
        mc.add_module("enc", {"layers": 6})
        assert mc.get_module_config("enc") == {"layers": 6}

    def test_get_missing_module(self):
        mc = ModularModelConfig()
        assert mc.get_module_config("nope") is None

    def test_defaults(self):
        mc = ModularModelConfig()
        assert mc.enable_caching is True
        assert mc.parallel_execution is False


class TestDistributedDecorators:
    def test_distributed_jit_passthrough(self):
        @distributed_jit
        def fn(x):
            return x + 1
        assert fn(1) == 2

    def test_model_sharded_jit_passthrough(self):
        @model_sharded_jit
        def fn(x):
            return x * 2
        assert fn(3) == 6

    def test_batch_sharded_jit_passthrough(self):
        @batch_sharded_jit
        def fn(x):
            return x - 1
        assert fn(5) == 4

    def test_create_unified_mesh(self):
        assert create_unified_mesh() is None


class TestLoadConfig:
    def test_load_from_file(self, tmp_path):
        path = str(tmp_path / "c.json")
        with open(path, "w") as f:
            json.dump({"k": "v"}, f)
        result = load_config(path)
        assert result == {"k": "v"}

    def test_load_nonexistent(self):
        result = load_config("/no/file.json")
        assert result == {}

    def test_load_none_returns_settings(self):
        result = load_config(None)
        assert isinstance(result, dict)
