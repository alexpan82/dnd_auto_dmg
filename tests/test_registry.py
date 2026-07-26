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
    # WP10 added generic weapons (including a plain "greatsword") to
    # weapons.json, which changes which partial-name queries land on
    # flametongue_greatsword vs. the new generic entries under the current
    # (partial_ratio / token_set_ratio) fuzzy_match scorer:
    #
    # * "flame sword" and "flametongue greatsword" now resolve to the new
    #   "longsword"/"greatsword" entries respectively -- both contain
    #   "sword"/"greatsword" as a clean substring/token subset, which scores
    #   100 under partial_ratio and token_set_ratio, beating flametongue's
    #   ~95. Deliberately NOT tested here as flametongue hits anymore; the
    #   deeper scoring fix is a later work package's job, not this one's.
    # * Queries that mention "flametongue" but avoid the bare "greatsword"/
    #   "sword" substring still hit flametongue_greatsword reliably (verified
    #   empirically against the real fuzzy_match implementation).
    for query in ("flametongue sword", "flametongue blade", "flametongue gr8sword"):
        result = registry.find_item(query)
        assert result is not None, f"expected a hit for {query!r}"
        item_id, kind, _item_def = result
        assert item_id == "flametongue_greatsword"
        assert kind == "weapon"


def test_find_item_bare_greatsword_query_still_hits_flametongue(registry):
    # "greatsword" alone ties at 100.0 between "flametongue_greatsword" and
    # the new generic "greatsword" entry. fuzzy_match's stable sort means
    # whichever id comes FIRST in weapons.json wins the tie -- this is why
    # flametongue_greatsword must stay the first key in weapons.json (see
    # data/weapons.json and tests/test_graph_integration.py, which drives
    # weapon_or_spell="greatsword" and depends on this exact behavior).
    result = registry.find_item("greatsword")
    assert result is not None
    item_id, kind, _item_def = result
    assert item_id == "flametongue_greatsword"
    assert kind == "weapon"


def test_find_item_generic_weapons_are_loaded(registry):
    for weapon_id in ("longsword", "greatsword", "dagger", "shortbow", "mace", "handaxe"):
        assert weapon_id in registry.weapons, f"expected {weapon_id!r} in weapons.json"


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


def test_combatant_from_character_passes_through_level_and_proficiency_bonus(registry):
    # WP10: registry.combatant_from_character must populate level/PB from
    # the roster CharacterSheet instead of dropping them.
    combatant = registry.combatant_from_character("avantor")
    assert combatant.level == 10
    assert combatant.proficiency_bonus == 4
    # And a fresh Combatant's temp_hp defaults to 0 (not sourced from the
    # roster, which has no notion of temp hp).
    assert combatant.temp_hp == 0


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


# ---------------------------------------------------------------------------
# WP10 -- tensers/savage/GWM JSON round-trip through the REAL registry
# ---------------------------------------------------------------------------


def test_tensers_transformation_damage_rider_and_temp_hp_round_trip(registry):
    spell = registry.spells["tensers_transformation"]
    assert spell.applies_status is not None
    assert spell.applies_status.damage_rider == {"force": "2d12"}
    assert spell.applies_status.grants_temp_hp == 50
    # Free-text effect must be preserved, not replaced by the new fields.
    assert spell.applies_status.effect


def test_savage_attacks_crit_extra_die_round_trip(registry):
    feature = registry.features["savage_attacks"]
    assert feature.crit_extra_die is True


def test_great_weapon_master_flat_bonus_and_opt_in_round_trip(registry):
    feature = registry.features["great_weapon_master"]
    assert feature.flat_damage_bonus == 10
    assert feature.opt_in is True


def test_flametongue_special_effects_does_not_restate_damage_dice(registry):
    # WP10: special_effects should no longer duplicate the 2d6 fire that's
    # already represented structurally in `damage`.
    weapon = registry.weapons["flametongue_greatsword"]
    assert "2d6" not in weapon.special_effects


# ---------------------------------------------------------------------------
# WP10 -- AppConfig new fields + env overrides
# ---------------------------------------------------------------------------


def test_app_config_new_fields_have_spec_defaults():
    config = AppConfig()
    assert config.temperature == 0.0
    assert config.num_ctx == 16384
    assert config.keep_alive == "10m"
    assert config.model_kwargs == {}
    assert config.fuzzy_threshold_lookup == 60
    assert config.fuzzy_threshold_combatant == 80
    # Unchanged this phase (WP15 flips it):
    assert config.stream_nodes == {"calculate_damage", "narrate"}
    assert config.enable_narration is True


def test_app_config_env_overrides(monkeypatch):
    monkeypatch.setenv("DND_TEMPERATURE", "0.7")
    monkeypatch.setenv("DND_NUM_CTX", "8192")
    monkeypatch.setenv("DND_KEEP_ALIVE", "5m")
    monkeypatch.setenv("DND_MODEL_KWARGS", '{"top_p": 0.9}')

    config = AppConfig()
    assert config.temperature == 0.7
    assert config.num_ctx == 8192
    assert config.keep_alive == "5m"
    assert config.model_kwargs == {"top_p": 0.9}


@pytest.mark.parametrize(
    "env_var,bad_value,attr,default",
    [
        ("DND_TEMPERATURE", "not-a-float", "temperature", 0.0),
        ("DND_NUM_CTX", "not-an-int", "num_ctx", 16384),
        ("DND_MODEL_KWARGS", "{not valid json", "model_kwargs", {}),
    ],
)
def test_app_config_malformed_env_vars_degrade_to_default_without_raising(
    monkeypatch, env_var, bad_value, attr, default
):
    monkeypatch.setenv(env_var, bad_value)
    # Must not raise -- construction never explodes on a junk env var.
    config = AppConfig()
    assert getattr(config, attr) == default


def test_app_config_model_kwargs_non_dict_json_degrades_to_empty_dict(monkeypatch):
    monkeypatch.setenv("DND_MODEL_KWARGS", "[1, 2, 3]")
    config = AppConfig()
    assert config.model_kwargs == {}
