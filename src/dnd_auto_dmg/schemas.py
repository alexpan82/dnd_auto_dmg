"""Pydantic models for dnd_auto_dmg.

This module holds two families of models:

* **Runtime models** (Spec §4) — the live combat state that flows through
  the graph: parsed LLM turns, resolved actions, combatants, and damage
  reports.
* **Data-file models** (Spec §8) — the shape of the JSON files under
  ``data/`` (characters, weapons, spells, features, monsters). These are
  intentionally kept in this same module (rather than split out) so that
  both runtime code and the data registry share one source of truth for
  validation.
"""

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------

#: Matches simple dice-notation strings like "2d6", "1d8+2", "2d6-1".
#: Rejects trailing operators with no number ("2d6+"), missing dice count
#: ("d6"), non-dice text ("abc"), and non-"d" separators ("2x6").
DICE_STRING_RE = re.compile(r"^\d+d\d+([+-]\d+)?$")


def _validate_dice_string(value: str) -> str:
    if not isinstance(value, str) or not DICE_STRING_RE.match(value):
        raise ValueError(
            f"invalid dice string {value!r}; expected format like '2d6' or '1d8+2'"
        )
    return value


def _validate_dice_dict(value: Dict[str, str]) -> Dict[str, str]:
    for dmg_type, dice in value.items():
        _validate_dice_string(dice)
    return value


# ---------------------------------------------------------------------------
# Spec §4 -- Runtime models
# ---------------------------------------------------------------------------


class StatusEffect(BaseModel):
    """A condition/buff/debuff currently applied to a combatant."""

    name: str  # "tensers_transformation", "unconscious"
    source: Optional[str] = None  # spell/feature id or actor id
    duration_rounds: Optional[int] = None  # None = until removed
    applied_round: Optional[int] = None
    notes: Optional[str] = None


class Combatant(BaseModel):
    """Live combat participant (PC, NPC, or monster instance)."""

    id: str  # slug: "avantor", "kobold_1"
    name: str
    kind: Literal["pc", "npc", "monster"]
    max_hp: int
    hp: int
    ac: Optional[int] = None
    attributes: Dict[str, int] = Field(default_factory=dict)
    features: List[str] = Field(default_factory=list)  # ids into features.json
    statuses: List[StatusEffect] = Field(default_factory=list)
    inventory: List[str] = Field(default_factory=list)  # ids into weapons.json
    origin: Literal["roster", "adhoc"] = "roster"
    is_alive: bool = True  # set False when hp reaches 0


class TargetRef(BaseModel):
    """A reference to a target (or group of targets) named in a turn."""

    name: str  # "kobold", "goblin", "self"
    count: int = 1  # "3 kobolds" -> count=3


class ParsedAction(BaseModel):
    """A single action extracted from free-text narration by the LLM."""

    actor: str
    action_type: Literal[
        "attack",
        "heal",
        "cast",
        "apply_status",
        "remove_status",
        "advance_round",
        "other",
    ]
    weapon_or_spell: Optional[str] = None
    targets: List[TargetRef] = Field(default_factory=list)
    spell_level: Optional[int] = None  # "5th level fireball"
    is_critical_hit: bool = False
    statuses_applied: List[str] = Field(default_factory=list)
    statuses_removed: List[str] = Field(default_factory=list)
    features_used: List[str] = Field(default_factory=list)


class ParsedTurn(BaseModel):
    """Structured-output wrapper for one or more parsed actions."""

    actions: List[ParsedAction] = Field(default_factory=list)


class ResolvedAction(BaseModel):
    """A ``ParsedAction`` after actor/target/item resolution against the
    live roster and data registry."""

    action: ParsedAction
    actor_id: str
    target_ids: List[str] = Field(default_factory=list)
    item: Optional[dict] = None  # matched weapon/spell def (model_dump) or None
    item_name_raw: Optional[str] = None  # unmatched name -> LLM 5e-knowledge fallback
    feature_texts: Dict[str, str] = Field(default_factory=dict)  # id -> effect text


class PerTargetDamage(BaseModel):
    """Damage/healing dealt to a single target as part of one action."""

    target_id: str
    damage: int = 0
    healing: int = 0


class DamageReport(BaseModel):
    """Final computed result of resolving one action's damage/healing."""

    action_index: int
    total_damage: int = 0
    total_healing: int = 0
    per_target: List[PerTargetDamage] = Field(default_factory=list)
    breakdown: Dict[str, Any] = Field(default_factory=dict)  # damage-type -> rolled amounts
    description: str = ""


# ---------------------------------------------------------------------------
# Spec §8 -- Data-file models (data/*.json)
# ---------------------------------------------------------------------------


class CharacterSheet(BaseModel):
    """A player-character roster entry (``data/characters.json``)."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    kind: Literal["pc"]
    class_: List[str] = Field(alias="class")
    subclass: str
    race: str
    level: int
    max_hp: int
    ac: int
    attributes: Dict[str, int]
    proficiency_bonus: int
    features: List[str] = Field(default_factory=list)  # ids
    inventory: List[str] = Field(default_factory=list)  # ids
    spells: List[str] = Field(default_factory=list)  # ids


class WeaponDef(BaseModel):
    """A weapon definition (``data/weapons.json``)."""

    name: str
    damage: Dict[str, str]  # dmg_type -> dice string
    magic_bonus: int = 0
    ability: str
    properties: List[str] = Field(default_factory=list)
    special_effects: str = ""

    @field_validator("damage")
    @classmethod
    def _validate_damage_dice(cls, value: Dict[str, str]) -> Dict[str, str]:
        return _validate_dice_dict(value)


class SpellSave(BaseModel):
    """The saving-throw component of a spell."""

    ability: str
    half_on_success: bool


class AppliedStatus(BaseModel):
    """A status effect a spell applies on cast/hit."""

    name: str
    duration_rounds: Optional[int] = None
    effect: str


class SpellDef(BaseModel):
    """A spell definition (``data/spells.json``)."""

    name: str
    level: int
    damage: Optional[Dict[str, str]] = None  # dmg_type -> dice string
    healing: Optional[str] = None  # dice string
    save: Optional[SpellSave] = None
    attack_roll: bool = False
    scaling_per_higher_level: Optional[Dict[str, str]] = None  # dmg_type -> dice string
    targeting: Literal["single", "area", "self"]
    applies_status: Optional[AppliedStatus] = None
    description: str = ""

    @field_validator("damage", "scaling_per_higher_level")
    @classmethod
    def _validate_dmg_dice(cls, value: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        if value is None:
            return value
        return _validate_dice_dict(value)

    @field_validator("healing")
    @classmethod
    def _validate_healing_dice(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        return _validate_dice_string(value)


class FeatureDef(BaseModel):
    """A class/racial feature definition (``data/features.json``)."""

    name: str
    description: str = ""
    effect: str = ""  # mechanical text consumed by the damage LLM


class MonsterDef(BaseModel):
    """A monster/statblock definition (``data/monsters.json``)."""

    name: str
    kind: Literal["monster"]
    max_hp: int
    ac: int
    attributes: Dict[str, int]
