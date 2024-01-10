"""
Configuration management for the RAG system.

Loads settings from YAML config file and environment variables,
with environment variables taking precedence.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

_CONFIG_PATH = os.getenv("RAG_CONFIG_PATH", "config/settings.yaml")


def _load_yaml(path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    config_file = Path(path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with config_file.open() as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_settings() -> dict[str, Any]:
    """
    Load and cache application settings.

    Returns:
        dict: Merged configuration from YAML and environment variables.
    """
    cfg = _load_yaml(_CONFIG_PATH)
    logger.debug(f"Loaded config from {_CONFIG_PATH}")
    return cfg


def get(section: str, key: str, default: Any = None) -> Any:
    """
    Retrieve a specific config value.

    Args:
        section: Top-level config section (e.g. 'retrieval').
        key: Key within the section.
        default: Fallback value if not found.

    Returns:
        Config value or default.
    """
    settings = get_settings()
    return settings.get(section, {}).get(key, default)
