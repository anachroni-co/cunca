"""
Configuration Loader for CapibaraGPT v3.

Loads configuration from YAML files with support for:
- Default config (config.yaml)
- Local overrides (config.local.yaml)
- Environment variable overrides
- Programmatic overrides
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import YAML parser
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False
    logger.warning("PyYAML not available, config loading limited to JSON")

try:
    import json
    JSON_AVAILABLE = True
except ImportError:
    JSON_AVAILABLE = False


CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"
LOCAL_CONFIG_PATH = CONFIG_DIR / "config.local.yaml"


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def get_env_override(key_path: str, default: Any = None) -> Optional[str]:
    """Get environment variable override for a config key.

    Config key 'model.hidden_size' maps to env var 'CAPIBARA_MODEL_HIDDEN_SIZE'
    """
    env_key = "CAPIBARA_" + key_path.upper().replace(".", "_")
    return os.environ.get(env_key, default)


def apply_env_overrides(config: Dict, prefix: str = "") -> Dict:
    """Apply environment variable overrides to config."""
    result = config.copy()

    for key, value in config.items():
        key_path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            result[key] = apply_env_overrides(value, key_path)
        else:
            env_value = get_env_override(key_path)
            if env_value is not None:
                # Try to parse as the same type as the original value
                try:
                    if isinstance(value, bool):
                        result[key] = env_value.lower() in ("true", "1", "yes")
                    elif isinstance(value, int):
                        result[key] = int(env_value)
                    elif isinstance(value, float):
                        result[key] = float(env_value)
                    elif isinstance(value, list):
                        result[key] = env_value.split(",")
                    else:
                        result[key] = env_value
                except (ValueError, TypeError):
                    result[key] = env_value

    return result


class ConfigLoader:
    """Loads and manages configuration for CapibaraGPT."""

    _instance: Optional["ConfigLoader"] = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._config = self.load()

    def load(
        self,
        config_path: Optional[Union[str, Path]] = None,
        local_path: Optional[Union[str, Path]] = None,
        apply_env: bool = True
    ) -> Dict[str, Any]:
        """Load configuration from files.

        Args:
            config_path: Path to main config file (default: config.yaml)
            local_path: Path to local overrides (default: config.local.yaml)
            apply_env: Whether to apply environment variable overrides

        Returns:
            Merged configuration dictionary
        """
        config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        local_path = Path(local_path) if local_path else LOCAL_CONFIG_PATH

        # Load default config
        config = {}
        if config_path.exists():
            config = self._load_file(config_path)
            logger.info(f"Loaded config from {config_path}")
        else:
            logger.warning(f"Default config not found: {config_path}")

        # Merge local overrides
        if local_path.exists():
            local_config = self._load_file(local_path)
            config = deep_merge(config, local_config)
            logger.info(f"Applied local overrides from {local_path}")

        # Apply environment variable overrides
        if apply_env:
            config = apply_env_overrides(config)

        self._config = config
        return config

    def _load_file(self, path: Path) -> Dict[str, Any]:
        """Load a single config file."""
        suffix = path.suffix.lower()

        try:
            content = path.read_text(encoding="utf-8")

            if suffix in (".yaml", ".yml"):
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML required to load YAML config")
                return yaml.safe_load(content) or {}

            elif suffix == ".json":
                return json.loads(content)

            else:
                # Try YAML first, then JSON
                if YAML_AVAILABLE:
                    try:
                        return yaml.safe_load(content) or {}
                    except Exception:
                        pass
                return json.loads(content)

        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return {}

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a config value by dot-separated path.

        Args:
            key_path: Dot-separated path (e.g., 'model.hidden_size')
            default: Default value if not found

        Returns:
            Config value or default
        """
        if self._config is None:
            self.load()

        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire config section."""
        return self.get(section, {})

    @property
    def model(self) -> Dict[str, Any]:
        """Model configuration section."""
        return self.get_section("model")

    @property
    def training(self) -> Dict[str, Any]:
        """Training configuration section."""
        return self.get_section("training")

    @property
    def data(self) -> Dict[str, Any]:
        """Data configuration section."""
        return self.get_section("data")

    @property
    def inference(self) -> Dict[str, Any]:
        """Inference configuration section."""
        return self.get_section("inference")

    @property
    def hardware(self) -> Dict[str, Any]:
        """Hardware configuration section."""
        return self.get_section("hardware")

    def reload(self) -> Dict[str, Any]:
        """Reload configuration from files."""
        self._config = None
        return self.load()

    def to_dict(self) -> Dict[str, Any]:
        """Return full config as dictionary."""
        if self._config is None:
            self.load()
        return self._config.copy()


# Global config instance
config = ConfigLoader()


def get_config() -> ConfigLoader:
    """Get the global config instance."""
    return config


def load_config(
    config_path: Optional[str] = None,
    local_path: Optional[str] = None
) -> Dict[str, Any]:
    """Load configuration from files.

    This is a convenience function that returns the config dictionary directly.
    """
    return config.load(config_path, local_path)


__all__ = [
    "ConfigLoader",
    "config",
    "get_config",
    "load_config",
    "deep_merge",
]
