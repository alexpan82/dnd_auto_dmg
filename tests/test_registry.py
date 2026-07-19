import json
import shutil
from pathlib import Path

import pytest

from dnd_auto_dmg.config import AppConfig
from dnd_auto_dmg.registry import DataRegistry, DataRegistryError
from dnd_auto_dmg.schemas import Combatant

REPO_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def registry() -> DataRegistry:
    """Registry built against the REAL data/ files (also validates them)."""
    return DataRegistry(AppConfig())


# --------- construction / loading --------- #
def test_registry_constructs_from_real_data_dir_without_error(registry):
    assert len(registry.characters) >= 2
    assert len(registry.weapons) >= 1
    assert len(registry.spells) >= 2
    assert len(registry.features) >= 2
    assert len(registry.monsters) >= 6


# --------- find_character --------- #
def test_find_character_avantor_has_expected_sheet(registry):
    result = registry.find_character("Avantor")
    assert result is not None
    char_id, sheet = result
    assert char_id == "avantor"
    assert sheet.max_hp == 62
    assert sheet.ac == 15
    assert sheet.attributes == {
        "strength": 18,
        "dexterity": 16,
        "constitution": 14,
        "intelligence": 18,
        "wisdom": 12,
        "charisma": 10,
    }
    assert sheet.proficiency_bonus == 4


def test_find_character_lowercase_hits(registry):
    result = registry.find_character("avantor")
    assert result is not None
    char_id, sheet = result
    assert char_id == "avantor"
    assert sheet.name == "Avantor"


# --------- find_item --------- #
def test_find_item_fireball_hits_spell(registry):
    result = registry.find_item("fireball")
    assert result is not None
    item_id, kind, item_def = result
    assert item_id == "fireball"
    assert kind == "spell"
    assert item_def.name == "Fireball"


def test_find_item_flametongue_hits_weapon(registry):
    result = registry.find_item("flametongue")
    assert result is not None
    item_id, kind, item_def = result
    assert item_id == "flametongue_greatsword"
    assert kind == "weapon"


def test_find_item_fuzzy_partial_names_hit_weapon(registry):
    # Verified empirically against the real fuzzy_match implementation:
    # "flame sword" scores ~66.7 and "flametongue greatsword" scores ~95.5,
    # both well above the default lookup threshold (40).
    for query in ("flame sword", "flametongue greatsword"):
        result = registry.find_item(query)
        assert result is not None, f"expected a hit for {query!r}"
        item_id, kind, _item_def = result
        assert item_id == "flametongue_greatsword"
        assert kind == "weapon"


# --------- find_monster --------- #
def test_find_monster_kobold_hits(registry):
    result = registry.find_monster("kobold")
    assert result is not None
    monster_id, monster_def = result
    assert monster_id == "kobold"
    assert monster_def.max_hp == 5
    assert monster_def.ac == 12


def test_find_monster_nonsense_query_returns_none(registry):
    assert registry.find_monster("xyzzy") is None


# --------- combatant builders --------- #
def test_combatant_from_character_avantor(registry):
    combatant = registry.combatant_from_character("avantor")
    assert isinstance(combatant, Combatant)
    assert combatant.hp == combatant.max_hp == 62
    assert combatant.kind == "pc"
    assert combatant.id == "avantor"
    assert combatant.origin == "roster"


def test_combatant_from_monster_kobold_2(registry):
    combatant = registry.combatant_from_monster("kobold", 2)
    assert combatant.id == "kobold_2"
    assert combatant.hp == 5
    assert combatant.max_hp == 5
    assert combatant.kind == "monster"
    assert combatant.origin == "roster"


def test_adhoc_combatant_weird_beast(registry):
    combatant = registry.adhoc_combatant("Weird Beast", 10)
    assert combatant.origin == "adhoc"
    assert combatant.max_hp == combatant.hp == 10
    assert combatant.kind == "monster"
    assert combatant.name == "Weird Beast"


# --------- per-entry error context --------- #
def test_invalid_entry_error_identifies_file_and_entry_id(tmp_path):
    # Copy the real (valid) data files, then corrupt weapons.json with a
    # bad dice string on a specific entry id.
    for filename in ("characters.json", "spells.json", "features.json", "monsters.json"):
        shutil.copy(REPO_DATA_DIR / filename, tmp_path / filename)

    weapons = json.loads((REPO_DATA_DIR / "weapons.json").read_text())
    weapons["flametongue_greatsword"]["damage"]["slashing"] = "2d6+"  # invalid dice string
    (tmp_path / "weapons.json").write_text(json.dumps(weapons))

    config = AppConfig(data_dir=tmp_path)
    with pytest.raises(DataRegistryError) as exc_info:
        DataRegistry(config)

    message = str(exc_info.value)
    assert "weapons.json" in message
    assert "flametongue_greatsword" in message
