"""
TPU configuration module for CapibaraGPT.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class TpuConfig:
    """TPU configuration settings."""

    tpu_version: str = "v4"
    num_devices: int = 8
    mesh_shape: tuple = (2, 4)
    memory_fraction: float = 0.9
    use_bfloat16: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "tpu_version": self.tpu_version,
            "num_devices": self.num_devices,
            "mesh_shape": self.mesh_shape,
            "memory_fraction": self.memory_fraction,
            "use_bfloat16": self.use_bfloat16,
        }


# Global config instance
config = TpuConfig()
