"""Application configuration.

``AppConfig`` is a plain pydantic v2 ``BaseModel`` (not ``pydantic_settings``,
which is not a project dependency). Environment-variable overrides are read
lazily via ``default_factory`` callables so that constructing ``AppConfig()``
never prompts for input, never touches the network, and never requires
``data/`` to exist on disk.
"""

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _default_data_dir() -> Path:
    """Resolve the repo's ``data/`` directory relative to this package's
    location on disk (NOT the current working directory).

    This file lives at ``src/dnd_auto_dmg/config.py``. Walking up three
    parents from the resolved file path lands on the repo root:
        src/dnd_auto_dmg/config.py -> src/dnd_auto_dmg -> src -> <repo root>
    """
    return Path(__file__).resolve().parents[2] / "data"


def _model_default() -> str:
    return os.environ.get("DND_MODEL", "openai:gpt-4o")


def _data_dir_default() -> Path:
    override = os.environ.get("DND_DATA_DIR")
    return Path(override) if override else _default_data_dir()


class AppConfig(BaseModel):
    """Central, dependency-light application configuration.

    Construction never prompts, never hits the network, and never requires
    ``data_dir`` to exist.
    """

    model: str = Field(default_factory=_model_default)
    data_dir: Path = Field(default_factory=_data_dir_default)
    fuzzy_threshold_lookup: int = 40
    fuzzy_threshold_combatant: int = 70
    default_adhoc_hp: int = 10
    enable_narration: bool = True
    stream_nodes: set[str] = Field(default_factory=lambda: {"calculate_damage", "narrate"})
