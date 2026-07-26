"""``DataRegistry`` loads and validates ``data/*.json`` and exposes fuzzy
lookups over characters, weapons, spells, features, and monsters.

All fuzzy matching delegates to :func:`dnd_auto_dmg.tools.fuzzy_match`, using
``config.fuzzy_threshold_lookup`` as the score threshold. Matching is done
against a candidate list built from BOTH each entry's slug id and its
human-readable ``name``; whichever candidate wins is mapped back to its id.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from dnd_auto_dmg.config import AppConfig
from dnd_auto_dmg.schemas import (
    CharacterSheet,
    Combatant,
    FeatureDef,
    MonsterDef,
    SpellDef,
    WeaponDef,
)
from dnd_auto_dmg.tools import fuzzy_match

ModelT = TypeVar("ModelT", bound=BaseModel)


class DataRegistryError(ValueError):
    """Raised when a ``data/*.json`` file fails to load or validate.

    The message always identifies the offending file and (when applicable)
    the entry id, e.g. ``"weapons.json entry 'flametongue_greatsword': ..."``.
    """


def _slugify(name: str) -> str:
    """Turn an arbitrary display name into a lowercase snake_case id."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "combatant"


def _load_json_file(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise DataRegistryError(f"{path.name}: file not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataRegistryError(f"{path.name}: invalid JSON ({exc})") from exc


def _load_entries(path: Path, model: Type[ModelT]) -> Dict[str, ModelT]:
    raw = _load_json_file(path)
    if not isinstance(raw, dict):
        raise DataRegistryError(f"{path.name}: expected a JSON object keyed by entry id")

    entries: Dict[str, ModelT] = {}
    for entry_id, payload in raw.items():
        try:
            entries[entry_id] = model.model_validate(payload)
        except ValidationError as exc:
            raise DataRegistryError(f"{path.name} entry '{entry_id}': {exc}") from exc
    return entries


class DataRegistry:
    """Loads, validates, and exposes fuzzy lookups over the game data.

    Construction eagerly loads and validates all five ``data/*.json`` files
    against their pydantic models; any invalid entry raises
    ``DataRegistryError`` identifying the file and entry id.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        data_dir = Path(config.data_dir)

        self.characters: Dict[str, CharacterSheet] = _load_entries(
            data_dir / "characters.json", CharacterSheet
        )
        self.weapons: Dict[str, WeaponDef] = _load_entries(
            data_dir / "weapons.json", WeaponDef
        )
        self.spells: Dict[str, SpellDef] = _load_entries(
            data_dir / "spells.json", SpellDef
        )
        self.features: Dict[str, FeatureDef] = _load_entries(
            data_dir / "features.json", FeatureDef
        )
        self.monsters: Dict[str, MonsterDef] = _load_entries(
            data_dir / "monsters.json", MonsterDef
        )

    # -- fuzzy lookups ----------------------------------------------------

    def _fuzzy_lookup(
        self, query: str, entries: Dict[str, ModelT]
    ) -> Optional[Tuple[str, ModelT]]:
        """Fuzzy-match ``query`` against ids and names of ``entries``."""
        candidates = []
        candidate_to_id: Dict[str, str] = {}
        for entry_id, item in entries.items():
            for candidate in (entry_id, item.name):
                candidates.append(candidate)
                candidate_to_id[candidate] = entry_id

        if not candidates:
            return None

        match, _score = fuzzy_match(
            query, candidates, threshold=self.config.fuzzy_threshold_lookup
        )
        if match is None:
            return None

        entry_id = candidate_to_id[match]
        return entry_id, entries[entry_id]

    def find_character(self, name: str) -> Optional[Tuple[str, CharacterSheet]]:
        """Fuzzy match over character names+ids."""
        return self._fuzzy_lookup(name, self.characters)

    def find_monster(self, name: str) -> Optional[Tuple[str, MonsterDef]]:
        """Fuzzy match over monster names+ids."""
        return self._fuzzy_lookup(name, self.monsters)

    def find_feature(self, name: str) -> Optional[Tuple[str, FeatureDef]]:
        """Fuzzy match over feature names+ids."""
        return self._fuzzy_lookup(name, self.features)

    def find_item(
        self, name: str
    ) -> Optional[Tuple[str, str, Union[WeaponDef, SpellDef]]]:
        """Fuzzy match over the MERGED weapons+spells namespace.

        Returns ``(id, "weapon"|"spell", item_def)`` or ``None``.
        """
        candidates = []
        candidate_to_entry: Dict[str, Tuple[str, str, Union[WeaponDef, SpellDef]]] = {}

        for entry_id, item in self.weapons.items():
            for candidate in (entry_id, item.name):
                candidates.append(candidate)
                candidate_to_entry[candidate] = (entry_id, "weapon", item)
        for entry_id, item in self.spells.items():
            for candidate in (entry_id, item.name):
                candidates.append(candidate)
                candidate_to_entry[candidate] = (entry_id, "spell", item)

        if not candidates:
            return None

        match, _score = fuzzy_match(
            name, candidates, threshold=self.config.fuzzy_threshold_lookup
        )
        if match is None:
            return None

        return candidate_to_entry[match]

    # -- combatant builders -------------------------------------------------

    def combatant_from_character(self, char_id: str) -> Combatant:
        """Build a runtime ``Combatant`` from a roster ``CharacterSheet``."""
        try:
            sheet = self.characters[char_id]
        except KeyError as exc:
            raise KeyError(f"no character with id '{char_id}' in characters.json") from exc

        return Combatant(
            id=char_id,
            name=sheet.name,
            kind="pc",
            max_hp=sheet.max_hp,
            hp=sheet.max_hp,
            ac=sheet.ac,
            attributes=dict(sheet.attributes),
            features=list(sheet.features),
            inventory=list(sheet.inventory),
            origin="roster",
            level=sheet.level,
            proficiency_bonus=sheet.proficiency_bonus,
        )

    def combatant_from_monster(self, monster_id: str, suffix: Union[int, str]) -> Combatant:
        """Build a runtime ``Combatant`` instance from a ``MonsterDef``.

        The instance gets id ``f"{monster_id}_{suffix}"`` (e.g. ``kobold_1``)
        and a display name combining the monster's name with the suffix
        (e.g. "Kobold 1").
        """
        try:
            monster = self.monsters[monster_id]
        except KeyError as exc:
            raise KeyError(f"no monster with id '{monster_id}' in monsters.json") from exc

        return Combatant(
            id=f"{monster_id}_{suffix}",
            name=f"{monster.name} {suffix}",
            kind="monster",
            max_hp=monster.max_hp,
            hp=monster.max_hp,
            ac=monster.ac,
            attributes=dict(monster.attributes),
            origin="roster",
        )

    def adhoc_combatant(self, name: str, hp: int) -> Combatant:
        """Build an ad-hoc (non-roster) ``Combatant`` from a bare name+HP."""
        return Combatant(
            id=_slugify(name),
            name=name,
            kind="monster",
            max_hp=hp,
            hp=hp,
            origin="adhoc",
        )
