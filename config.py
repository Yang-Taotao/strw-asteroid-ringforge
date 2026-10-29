"""
Config loader, reads in `./config.toml` to build configs.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

CONFIG_PATH: Path = Path(__file__).with_name("config.toml")


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Read `config.toml`; return an empty dict as fallback."""
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def value(
    config: dict[str, Any],
    section: str,
    key: str,
    default: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Return `config[section][key]` or `default`."""
    section_value = config.get(section, {})
    if not isinstance(section_value, dict):
        return default
    return section_value.get(key, default)
