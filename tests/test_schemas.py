import pytest
from pydantic import ValidationError

from dnd_auto_dmg.schemas import (
    AppliedStatus,
    Combatant,
    CharacterSheet,
    DamageReport,
    FeatureDef,
    MonsterDef,
    ParsedAction,
    ParsedTurn,
    PerTargetDamage,
    SpellDef,
    SpellSave,
    StatusEffect,
    TargetRef,
    WeaponDef,
)


# --------- dice-string regex validation --------- #
@pytest.mark.parametrize("dice", ["2d6", "1d8+2", "2d6-1"])
def test_weapon_def_accepts_valid_dice_strings(dice):
    weapon = WeaponDef(
        name="Test Weapon",
        damage={"slashing": dice},
        ability="strength",
    )
    assert weapon.damage["slashing"] == dice


@pytest.mark.parametrize("dice", ["2d6+", "d6", "abc", "2x6"])
def test_weapon_def_rejects_invalid_dice_strings(dice):
    with pytest.raises(ValidationError):
        WeaponDef(
            name="Test Weapon",
            damage={"slashing": dice},
            ability="strength",
        )


# --------- SpellDef --------- #
def test_spell_def_valid_fireball_shaped_payload_validates():
    spell = SpellDef(
        name="Fireball",
        level=3,
        damage={"fire": "8d6"},
        healing=None,
        save={"ability": "dexterity", "half_on_success": True},
        attack_roll=False,
        scaling_per_higher_level={"fire": "1d6"},
        targeting="area",
        applies_status=None,
        description="Boom.",
    )
    assert spell.damage == {"fire": "8d6"}
    assert isinstance(spell.save, SpellSave)
    assert spell.save.ability == "dexterity"
    assert spell.save.half_on_success is True


def test_spell_def_healing_dice_regex_enforced():
    with pytest.raises(ValidationError):
        SpellDef(
            name="Cure Wounds",
            level=1,
            targeting="single",
            healing="not-a-dice-string",
        )

    spell = SpellDef(
        name="Cure Wounds",
        level=1,
        targeting="single",
        healing="1d8+3",
    )
    assert spell.healing == "1d8+3"


def test_spell_def_nested_save_and_applies_status_models_validate():
    spell = SpellDef(
        name="Tenser's Transformation",
        level=6,
        targeting="self",
        applies_status={
            "name": "tensers_transformation",
            "duration_rounds": 10,
            "effect": "Extra 2d12 force damage on hit.",
        },
    )
    assert isinstance(spell.applies_status, AppliedStatus)
    assert spell.applies_status.name == "tensers_transformation"
    assert spell.applies_status.duration_rounds == 10
    assert spell.save is None


# --------- Combatant defaults --------- #
def test_combatant_defaults():
    combatant = Combatant(
        id="kobold_1",
        name="Kobold 1",
        kind="monster",
        max_hp=5,
        hp=5,
    )
    assert combatant.origin == "roster"
    assert combatant.is_alive is True
    assert combatant.ac is None
    assert combatant.attributes == {}
    assert combatant.features == []
    assert combatant.statuses == []
    assert combatant.inventory == []


# --------- CharacterSheet "class" alias --------- #
def test_character_sheet_class_json_alias():
    payload = {
        "name": "Avantor",
        "kind": "pc",
        "class": ["Wizard"],
        "subclass": "Bladesinging",
        "race": "Half-Elf",
        "level": 10,
        "max_hp": 62,
        "ac": 15,
        "attributes": {
            "strength": 18,
            "dexterity": 16,
            "constitution": 14,
            "intelligence": 18,
            "wisdom": 12,
            "charisma": 10,
        },
        "proficiency_bonus": 4,
        "features": ["savage_attacks"],
        "inventory": ["flametongue_greatsword"],
        "spells": ["fireball"],
    }
    sheet = CharacterSheet.model_validate(payload)
    assert sheet.class_ == ["Wizard"]

    # populate_by_name means constructing with the python attribute name
    # (rather than the JSON alias) also works.
    by_name_payload = {k: v for k, v in payload.items() if k != "class"}
    by_name_payload["class_"] = ["Wizard"]
    sheet2 = CharacterSheet(**by_name_payload)
    assert sheet2.class_ == ["Wizard"]


# --------- DamageReport / ParsedAction / ParsedTurn round-trip --------- #
def test_damage_report_basic_construction():
    report = DamageReport(
        action_index=0,
        total_damage=14,
        total_healing=0,
        per_target=[PerTargetDamage(target_id="kobold_1", damage=14)],
        breakdown={"slashing": [8], "fire": [6]},
        description="Avantor slashes the kobold.",
    )
    assert report.total_damage == 14
    assert report.per_target[0].target_id == "kobold_1"
    assert report.breakdown["fire"] == [6]


def test_parsed_action_and_parsed_turn_round_trip():
    action = ParsedAction(
        actor="Avantor",
        action_type="attack",
        weapon_or_spell="Flametongue Greatsword",
        targets=[TargetRef(name="kobold", count=3)],
        is_critical_hit=True,
    )
    turn = ParsedTurn(actions=[action])

    assert turn.actions[0].actor == "Avantor"
    assert turn.actions[0].targets[0].count == 3
    assert turn.actions[0].statuses_applied == []

    # round-trip through dict/json
    dumped = turn.model_dump()
    restored = ParsedTurn.model_validate(dumped)
    assert restored == turn


def test_status_effect_and_feature_and_monster_def_construct():
    status = StatusEffect(name="unconscious")
    assert status.source is None
    assert status.duration_rounds is None

    feature = FeatureDef(name="Great Weapon Master", description="d", effect="e")
    assert feature.name == "Great Weapon Master"

    monster = MonsterDef(
        name="Kobold",
        kind="monster",
        max_hp=5,
        ac=12,
        attributes={
            "strength": 7,
            "dexterity": 15,
            "constitution": 9,
            "intelligence": 8,
            "wisdom": 7,
            "charisma": 8,
        },
    )
    assert monster.max_hp == 5
