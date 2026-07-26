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
    ResolvedAction,
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
    # WP10 additive fields -- must default cleanly for every existing caller
    # that never sets them.
    assert combatant.temp_hp == 0
    assert combatant.level is None
    assert combatant.proficiency_bonus is None


def test_combatant_temp_hp_level_pb_settable():
    combatant = Combatant(
        id="avantor",
        name="Avantor",
        kind="pc",
        max_hp=62,
        hp=62,
        temp_hp=50,
        level=10,
        proficiency_bonus=4,
    )
    assert combatant.temp_hp == 50
    assert combatant.level == 10
    assert combatant.proficiency_bonus == 4


# --------- StatusEffect.damage_rider --------- #
def test_status_effect_damage_rider_defaults_to_none_and_is_settable():
    status = StatusEffect(name="tensers_transformation")
    assert status.damage_rider is None

    status_with_rider = StatusEffect(
        name="tensers_transformation", damage_rider={"force": "2d12"}
    )
    assert status_with_rider.damage_rider == {"force": "2d12"}


# --------- AppliedStatus.damage_rider / grants_temp_hp --------- #
def test_applied_status_defaults_cleanly():
    status = AppliedStatus(name="blessed", effect="+1d4 to attacks and saves.")
    assert status.damage_rider is None
    assert status.grants_temp_hp is None


def test_applied_status_accepts_valid_damage_rider():
    status = AppliedStatus(
        name="tensers_transformation",
        effect="Your weapon attacks deal an extra 2d12 force damage on a hit.",
        damage_rider={"force": "2d12"},
        grants_temp_hp=50,
    )
    assert status.damage_rider == {"force": "2d12"}
    assert status.grants_temp_hp == 50


@pytest.mark.parametrize("bad_dice", ["2x6", "d6", "2d6+", "abc"])
def test_applied_status_rejects_invalid_damage_rider_dice(bad_dice):
    with pytest.raises(ValidationError):
        AppliedStatus(
            name="tensers_transformation",
            effect="Extra damage on a hit.",
            damage_rider={"force": bad_dice},
        )


# --------- FeatureDef additive fields --------- #
def test_feature_def_new_fields_default_cleanly():
    feature = FeatureDef(name="Some Feature")
    assert feature.crit_extra_die is False
    assert feature.flat_damage_bonus == 0
    assert feature.opt_in is False


def test_feature_def_new_fields_settable():
    savage = FeatureDef(name="Savage Attacks", crit_extra_die=True)
    assert savage.crit_extra_die is True

    gwm = FeatureDef(
        name="Great Weapon Master", flat_damage_bonus=10, opt_in=True
    )
    assert gwm.flat_damage_bonus == 10
    assert gwm.opt_in is True


# --------- ParsedAction.actor default / ParsedTurn.relevant default --------- #
def test_parsed_action_actor_defaults_to_empty_string():
    # actorless advance_round action -- actor is no longer required.
    action = ParsedAction(action_type="advance_round")
    assert action.actor == ""


def test_parsed_turn_relevant_defaults_to_true():
    turn = ParsedTurn(actions=[])
    assert turn.relevant is True

    irrelevant_turn = ParsedTurn(actions=[], relevant=False)
    assert irrelevant_turn.relevant is False


# --------- ResolvedAction additive fields --------- #
def test_resolved_action_new_fields_default_cleanly():
    resolved = ResolvedAction(
        action=ParsedAction(actor="Avantor", action_type="attack"),
        actor_id="avantor",
    )
    assert resolved.item_id is None
    assert resolved.item_kind is None
    assert resolved.feature_defs == {}
    assert resolved.features_invoked == []
    assert resolved.feature_texts == {}  # still present, untouched by the new fields


def test_resolved_action_new_fields_settable():
    resolved = ResolvedAction(
        action=ParsedAction(actor="Avantor", action_type="attack"),
        actor_id="avantor",
        item_id="flametongue_greatsword",
        item_kind="weapon",
        feature_defs={"savage_attacks": {"name": "Savage Attacks", "crit_extra_die": True}},
        features_invoked=["savage_attacks"],
    )
    assert resolved.item_id == "flametongue_greatsword"
    assert resolved.item_kind == "weapon"
    assert resolved.feature_defs["savage_attacks"]["crit_extra_die"] is True
    assert resolved.features_invoked == ["savage_attacks"]


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
