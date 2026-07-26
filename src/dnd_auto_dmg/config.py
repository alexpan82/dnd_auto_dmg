"""Application configuration.

``AppConfig`` is a plain pydantic v2 ``BaseModel`` (not ``pydantic_settings``,
which is not a project dependency). Environment-variable overrides are read
lazily via ``default_factory`` callables so that constructing ``AppConfig()``
never prompts for input, never touches the network, and never requires
``data/`` to exist on disk.
"""

import json
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
    return os.environ.get("DND_MODEL", "ollama:minimax-m3:cloud")


def _data_dir_default() -> Path:
    override = os.environ.get("DND_DATA_DIR")
    return Path(override) if override else _default_data_dir()


def _temperature_default() -> float:
    override = os.environ.get("DND_TEMPERATURE")
    if override is None:
        return 0.0
    try:
        return float(override)
    except ValueError:
        return 0.0


def _num_ctx_default() -> int:
    override = os.environ.get("DND_NUM_CTX")
    if override is None:
        return 16384
    try:
        return int(override)
    except ValueError:
        return 16384


def _keep_alive_default() -> str:
    return os.environ.get("DND_KEEP_ALIVE", "10m")


def _model_kwargs_default() -> dict:
    override = os.environ.get("DND_MODEL_KWARGS")
    if not override:
        return {}
    try:
        parsed = json.loads(override)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class AppConfig(BaseModel):
    """Central, dependency-light application configuration.

    Construction never prompts, never hits the network, and never requires
    ``data_dir`` to exist. Malformed env var overrides (e.g. a non-numeric
    ``DND_NUM_CTX`` or invalid-JSON ``DND_MODEL_KWARGS``) degrade to the
    field's default rather than raising -- ``AppConfig()`` must never blow up
    on a junk environment.
    """

    model: str = Field(default_factory=_model_default)
    data_dir: Path = Field(default_factory=_data_dir_default)
    fuzzy_threshold_lookup: int = 60
    fuzzy_threshold_combatant: int = 80
    default_adhoc_hp: int = 10
    enable_narration: bool = True
    stream_nodes: set[str] = Field(default_factory=lambda: {"calculate_damage", "narrate"})

    #: Sampling temperature passed to the chat model. ``DND_TEMPERATURE``.
    temperature: float = Field(default_factory=_temperature_default)
    #: Context window size, ollama-only. ``DND_NUM_CTX``.
    num_ctx: int = Field(default_factory=_num_ctx_default)
    #: How long ollama keeps the model loaded after the last request.
    #: ``DND_KEEP_ALIVE``.
    keep_alive: str = Field(default_factory=_keep_alive_default)
    #: Extra provider kwargs, JSON-encoded via ``DND_MODEL_KWARGS``. Merged
    #: LAST (wins) over any other model kwargs assembled by the caller.
    model_kwargs: dict = Field(default_factory=_model_kwargs_default)
