"""
MoE Control API Tests - Unit tests for Mixture of Experts control interface.

This module provides tests for the MoE Control API, validating expert
detection, health monitoring, and layer management functionality.

Author: Skydesk International Dev Team.
"""

import core.modular_model as modular_model
from core.experts import MoEControlAPI


def test_moe_control_api_detects_layers(monkeypatch):
    monkeypatch.setattr(modular_model, "TOML_AVAILABLE", False)
    monkeypatch.setattr(modular_model, "toml", None, raising=False)

    config = modular_model.ModularConfig()
    config.hidden_size = 16
    config.num_router_experts = 2
    config.enable_moe = True
    config.moe_num_layers = 1
    config.moe_num_experts = 2
    config.moe_num_active_experts = 1
    config.moe_hidden_size = 16

    model = modular_model.ModularCapibaraModel(config)
    assert model.moe_layers

    api = MoEControlAPI(model)
    health = api.get_system_health()

    assert health["status"] != "unavailable"
    assert health["total_layers"] == len(model.moe_layers)
