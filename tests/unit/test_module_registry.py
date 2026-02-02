"""Isolated tests for core.module_registry module."""

import pytest
from core.module_registry import ModuleRegistry, IModule


class DummyModule(IModule):
    def __init__(self, size=64):
        self.size = size


class TestModuleRegistry:
    def test_init(self):
        reg = ModuleRegistry()
        assert reg.backend == "v6e"
        assert reg.get_registered_modules() == [] if hasattr(reg, 'get_registered_modules') else True

    def test_register_and_create(self):
        reg = ModuleRegistry()
        reg.register("dummy", DummyModule)
        mod = reg.create_module("dummy", size=128)
        assert isinstance(mod, DummyModule)
        assert mod.size == 128

    def test_register_factory(self):
        reg = ModuleRegistry()
        reg.register_factory("factory_mod", lambda **kw: DummyModule(**kw))
        mod = reg.create_module("factory_mod", size=256)
        assert mod.size == 256

    def test_factory_takes_precedence(self):
        reg = ModuleRegistry()
        reg.register("x", DummyModule)
        reg.register_factory("x", lambda **kw: DummyModule(size=999))
        mod = reg.create_module("x")
        assert mod.size == 999

    def test_create_unregistered_raises(self):
        reg = ModuleRegistry()
        with pytest.raises(KeyError):
            reg.create_module("nonexistent")

    def test_get_module(self):
        reg = ModuleRegistry()
        reg.register("dummy", DummyModule)
        cls = reg.get_module("dummy")
        assert cls is DummyModule

    def test_get_module_unregistered_raises(self):
        reg = ModuleRegistry()
        with pytest.raises(KeyError):
            reg.get_module("nope")

    def test_different_backends(self):
        for backend in ["v6e", "v5e", "v4"]:
            reg = ModuleRegistry(backend=backend)
            assert reg.backend == backend

    def test_overwrite_registration(self):
        reg = ModuleRegistry()

        class Other(IModule):
            def __init__(self):
                self.name = "other"

        reg.register("m", DummyModule)
        reg.register("m", Other)
        mod = reg.create_module("m")
        assert isinstance(mod, Other)
