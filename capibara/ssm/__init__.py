"""
SSM (State Space Models) Module for CapibaraGPT.

This package contains various SSM implementations optimized for TPU:
- SSMBlock: Classic state space model with O(n) complexity
- SpikeSSM: Bio-inspired spiking variant with surrogate gradients
- AdaptiveSpikeSSM: Multi-timescale adaptive spiking model
"""

from .ssm_tpu import SSMBlock, SSMConfig, BidirectionalSSM, create_ssm_block
from .spike_ssm import SpikeSSM, AdaptiveSpikeSSM, create_spike_ssm

__all__ = [
    "SSMBlock",
    "SSMConfig",
    "BidirectionalSSM",
    "create_ssm_block",
    "SpikeSSM",
    "AdaptiveSpikeSSM",
    "create_spike_ssm",
]
