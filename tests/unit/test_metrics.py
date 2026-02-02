"""Isolated tests for core.metrics module."""

import pytest
import numpy as np
from core.metrics import MetricsCollector


class TestMetricsCollector:
    def test_init_defaults(self):
        mc = MetricsCollector()
        assert mc.window_size == 1000
        assert mc.metrics == {}

    def test_init_custom_window(self):
        mc = MetricsCollector(window_size=50)
        assert mc.window_size == 50

    def test_record_and_get_stats(self):
        mc = MetricsCollector()
        mc.record("loss", 1.0)
        mc.record("loss", 2.0)
        mc.record("loss", 3.0)
        stats = mc.get_stats("loss")
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(3.0)
        assert stats["last"] == pytest.approx(3.0)
        assert stats["count"] == 3

    def test_get_stats_unknown(self):
        mc = MetricsCollector()
        assert mc.get_stats("nope") == {}

    def test_sliding_window(self):
        mc = MetricsCollector(window_size=3)
        for i in range(5):
            mc.record("x", float(i))
        stats = mc.get_stats("x")
        assert stats["count"] == 3
        assert stats["min"] == pytest.approx(2.0)

    def test_update_numeric(self):
        mc = MetricsCollector()
        mc.update({"a": 1.0, "b": 2})
        assert mc.get_stats("a")["last"] == pytest.approx(1.0)
        assert mc.get_stats("b")["last"] == pytest.approx(2.0)

    def test_update_non_numeric(self):
        mc = MetricsCollector()
        mc.update({"model": "test", "loss": 0.5})
        all_data = mc.get_all()
        assert all_data["model"] == "test"
        assert "mean" in all_data["loss"]

    def test_record_with_timestamp(self):
        mc = MetricsCollector()
        mc.record("x", 1.0, timestamp=100.0)
        assert list(mc.timestamps["x"]) == [100.0]

    def test_record_primitive_call(self):
        mc = MetricsCollector()
        mc.record_primitive_call("op1", success=True)
        mc.record_primitive_call("op1", success=True)
        mc.record_primitive_call("op1", success=False)
        assert mc.primitive_calls["op1"]["success"] == 2
        assert mc.primitive_calls["op1"]["failure"] == 1

    def test_get_all(self):
        mc = MetricsCollector()
        mc.record("a", 1.0)
        mc.record("b", 2.0)
        all_data = mc.get_all()
        assert "a" in all_data
        assert "b" in all_data

    def test_numpy_floating_in_update(self):
        mc = MetricsCollector()
        mc.update({"val": np.float32(3.14)})
        assert mc.get_stats("val")["last"] == pytest.approx(3.14, abs=1e-5)
