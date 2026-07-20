"""Tests for the WP4a deterministic graph nodes (Spec §6 / §10):
``begin_turn``, ``resolve_combatants``, ``apply_damage``.

No LLM, no network -- these three nodes are pure functions of ``CombatState``
(``apply_damage`` reads a final ``AIMessage`` out of ``state["messages"]``,
but that message is constructed by hand in these tests, never produced by a
real model call).

Uses the ``registry``/``app_config`` fixtures from ``tests/conftest.py``,
which load the REAL ``data/*.json`` files -- so these tests double as an
integration check that the node logic lines up with the actual roster
(Avantor: pc, 62 hp, ac 15, features ``savage_attacks``+
``great_weapon_master``, inventory ``flametongue_greatsword``, spells
``fireball``+``tensers_transformation``) and monster stat blocks (kobold 5
hp, goblin 7 hp, etc.).
"""

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from dnd_auto_dmg.config import AppConfig
from dnd_auto_dmg.nodes.apply import make_apply_damage
from dnd_auto_dmg.nodes.begin_turn import make_begin_turn
from dnd_auto_dmg.nodes.resolve import make_resolve_combatants
from dnd_auto_dmg.schemas import (
    Combatant,
    DamageReport,
    ParsedAction,
    ResolvedAction,
    StatusEffect,
    TargetRef,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _combatant(cid: str, **overrides) -> Combatant:
    defaults = dict(id=cid, name=cid.title(), kind="monster", max_hp=10, hp=10)
    defaults.update(overrides)
    return Combatant(**defaults)


def _parsed_action(actor: str, **overrides) -> ParsedAction:
    defaults = dict(actor=actor, action_type="attack", targets=[])
    defaults.update(overrides)
    return ParsedAction(**defaults)


def _resolved_action(actor_id: str, target_ids: list, **overrides) -> ResolvedAction:
    defaults = dict(
        action=_parsed_action(actor_id),
        actor_id=actor_id,
        target_ids=target_ids,
    )
    defaults.update(overrides)
    return ResolvedAction(**defaults)


def _scaffold_messages(final_content: str) -> list:
    """A typical tool-loop message scaffold for one action: a human turn,
    one tool-call round trip, and a final (no-tool-call) report AIMessage.
    Every message carries an explicit id so RemoveMessage assertions can
    pin down exactly which ones get deleted."""
    return [
        HumanMessage(content="Avantor attacks", id="human1"),
        AIMessage(
            content="",
            tool_calls=[{"name": "roll_dice", "args": {}, "id": "1"}],
            id="ai1",
        ),
        ToolMessage(content="12", tool_call_id="1", id="t1"),
        AIMessage(content=final_content, id="final"),
    ]


# ---------------------------------------------------------------------------
# begin_turn
# ---------------------------------------------------------------------------


def test_begin_turn_resets_all_five_scratch_fields():
    node = make_begin_turn()
    state = {
        "combatants": {"avantor": _combatant("avantor", kind="pc")},
        "round_number": 3,
        "relevant_query": True,
        "parsed_actions": [_parsed_action("Avantor")],
        "resolved_actions": [_resolved_action("avantor", [])],
        "current_action_index": 5,
        "damage_reports": [DamageReport(action_index=0, total_damage=3)],
    }
    update = node(state)

    assert update["relevant_query"] is False
    assert update["parsed_actions"] == []
    assert update["resolved_actions"] == []
    assert update["current_action_index"] == 0
    assert update["damage_reports"] == []


def test_begin_turn_first_turn_initializes_combatants_and_round():
    node = make_begin_turn()
    update = node({})  # no "combatants" key at all -> first turn ever

    assert update["combatants"] == {}
    assert update["round_number"] == 1


def test_begin_turn_does_not_clobber_existing_combatants_or_round():
    node = make_begin_turn()
    state = {
        "combatants": {"x": _combatant("x")},
        "round_number": 3,
    }
    update = node(state)

    # round_number is last-write-wins with no reducer -- must NOT be
    # touched on a turn where combatants already exists, or it would stomp
    # whatever round the encounter is actually on.
    assert "combatants" not in update
    assert "round_number" not in update


def test_begin_turn_returns_no_messages_key():
    node = make_begin_turn()
    update = node({})
    assert "messages" not in update


# ---------------------------------------------------------------------------
# resolve_combatants -- actor resolution
# ---------------------------------------------------------------------------


def test_resolve_actor_instantiates_from_character_sheet_when_not_in_combat(registry, app_config):
    node = make_resolve_combatants(registry, app_config)
    action = _parsed_action("Avantor")
    state = {"parsed_actions": [action], "combatants": {}, "round_number": 1}

    update = node(state)

    assert "avantor" in update["combatants"]
    avantor = update["combatants"]["avantor"]
    assert avantor.hp == 62
    assert avantor.kind == "pc"
    assert len(update["resolved_actions"]) == 1
    assert update["resolved_actions"][0].actor_id == "avantor"


def test_resolve_actor_reuses_live_combatant_no_duplicate_instantiation(registry, app_config):
    node = make_resolve_combatants(registry, app_config)
    live_avantor = registry.combatant_from_character("avantor").model_copy(update={"hp": 40})
    action = _parsed_action("avantor the wizard")
    state = {
        "parsed_actions": [action],
        "combatants": {"avantor": live_avantor},
        "round_number": 1,
    }

    update = node(state)

    assert update["resolved_actions"][0].actor_id == "avantor"
    # The already-live combatant must not be re-touched/re-instantiated --
    # nothing about this action should produce a "combatants" update for it.
    assert "avantor" not in update["combatants"]


def test_resolve_unknown_actor_drops_action_and_logs_event(registry, app_config):
    node = make_resolve_combatants(registry, app_config)
    action = _parsed_action("xyzzyplugh999")
    state = {"parsed_actions": [action], "combatants": {}, "round_number": 1}

    update = node(state)

    assert update["resolved_actions"] == []
    assert any("xyzzyplugh999" in line for line in update["event_log"])


# ---------------------------------------------------------------------------
# resolve_combatants -- target resolution
# ---------------------------------------------------------------------------


def test_resolve_targets_creates_multiple_kobolds(registry, app_config):
    node = make_resolve_combatants(registry, app_config)
    action = _parsed_action("Avantor", targets=[TargetRef(name="kobold", count=3)])
    state = {"parsed_actions": [action], "combatants": {}, "round_number": 1}

    update = node(state)

    resolved = update["resolved_actions"][0]
    assert resolved.target_ids == ["kobold_1", "kobold_2", "kobold_3"]
    for tid in resolved.target_ids:
        assert update["combatants"][tid].hp == 5
        assert update["combatants"][tid].kind == "monster"


def test_resolve_targets_reuses_living_kobolds_and_skips_dead(registry, app_config):
    node = make_resolve_combatants(registry, app_config)
    kobold_1 = registry.combatant_from_monster("kobold", 1)  # alive
    kobold_2 = registry.combatant_from_monster("kobold", 2).model_copy(
        update={"hp": 0, "is_alive": False}
    )
    action = _parsed_action("Avantor", targets=[TargetRef(name="kobold", count=3)])
    state = {
        "parsed_actions": [action],
        "combatants": {"kobold_1": kobold_1, "kobold_2": kobold_2},
        "round_number": 1,
    }

    update = node(state)

    resolved = update["resolved_actions"][0]
    assert len(resolved.target_ids) == 3
    assert "kobold_1" in resolved.target_ids  # living instance reused
    assert "kobold_2" not in resolved.target_ids  # dead instance skipped
    fresh_ids = set(resolved.target_ids) - {"kobold_1"}
    assert fresh_ids == {"kobold_3", "kobold_4"}


def test_resolve_unknown_monster_target_creates_adhoc_combatant(registry, app_config):
    node = make_resolve_combatants(registry, app_config)
    action = _parsed_action("Avantor", targets=[TargetRef(name="qzxjkvw482", count=1)])
    state = {"parsed_actions": [action], "combatants": {}, "round_number": 1}

    update = node(state)

    resolved = update["resolved_actions"][0]
    assert len(resolved.target_ids) == 1
    adhoc = update["combatants"][resolved.target_ids[0]]
    assert adhoc.origin == "adhoc"
    assert adhoc.hp == app_config.default_adhoc_hp
    assert adhoc.max_hp == app_config.default_adhoc_hp


def test_resolve_self_target_resolves_to_actor_id(registry, app_config):
    node = make_resolve_combatants(registry, app_config)
    action = _parsed_action(
        "Avantor",
        action_type="cast",
        weapon_or_spell="tensers transformation",
        targets=[TargetRef(name="self")],
    )
    state = {"parsed_actions": [action], "combatants": {}, "round_number": 1}

    update = node(state)

    resolved = update["resolved_actions"][0]
    assert resolved.target_ids == ["avantor"]


# ---------------------------------------------------------------------------
# resolve_combatants -- item resolution
# ---------------------------------------------------------------------------


def test_resolve_item_match_hit_populates_item_dict(registry, app_config):
    node = make_resolve_combatants(registry, app_config)
    action = _parsed_action("Avantor", weapon_or_spell="flametongue")
    state = {"parsed_actions": [action], "combatants": {}, "round_number": 1}

    update = node(state)

    resolved = update["resolved_actions"][0]
    assert resolved.item is not None
    assert resolved.item["name"] == "Flametongue Greatsword"
    assert resolved.item_name_raw is None


def test_resolve_item_match_miss_preserves_raw_name(registry, app_config):
    node = make_resolve_combatants(registry, app_config)
    action = _parsed_action("Avantor", weapon_or_spell="quantum blaster")
    state = {"parsed_actions": [action], "combatants": {}, "round_number": 1}

    update = node(state)

    resolved = update["resolved_actions"][0]
    assert resolved.item is None
    assert resolved.item_name_raw == "quantum blaster"


# ---------------------------------------------------------------------------
# resolve_combatants -- statuses (applied BEFORE damage) and advance_round
# ---------------------------------------------------------------------------


def test_resolve_applies_spell_status_to_actor_before_damage(registry, app_config):
    node = make_resolve_combatants(registry, app_config)
    action = _parsed_action(
        "Avantor",
        action_type="cast",
        weapon_or_spell="tensers transformation",
        targets=[TargetRef(name="self")],
        statuses_applied=["tensers_transformation"],
    )
    state = {"parsed_actions": [action], "combatants": {}, "round_number": 1}

    update = node(state)

    avantor = update["combatants"]["avantor"]
    matches = [s for s in avantor.statuses if s.name == "tensers_transformation"]
    assert len(matches) == 1
    assert matches[0].duration_rounds == 10


def test_resolve_removes_matching_status_from_actor(registry, app_config):
    node = make_resolve_combatants(registry, app_config)
    avantor = registry.combatant_from_character("avantor").model_copy(
        update={"statuses": [StatusEffect(name="tensers_transformation", duration_rounds=5)]}
    )
    action = _parsed_action(
        "Avantor", action_type="other", statuses_removed=["tensers_transformation"]
    )
    state = {
        "parsed_actions": [action],
        "combatants": {"avantor": avantor},
        "round_number": 1,
    }

    update = node(state)

    assert update["combatants"]["avantor"].statuses == []


def test_resolve_advance_round_decrements_and_expires_statuses(registry, app_config):
    node = make_resolve_combatants(registry, app_config)
    avantor = registry.combatant_from_character("avantor").model_copy(
        update={"statuses": [StatusEffect(name="short_buff", duration_rounds=1)]}
    )
    kobold = registry.combatant_from_monster("kobold", 1).model_copy(
        update={"statuses": [StatusEffect(name="long_curse", duration_rounds=5)]}
    )
    action = _parsed_action("Avantor", action_type="advance_round")
    state = {
        "parsed_actions": [action],
        "combatants": {"avantor": avantor, "kobold_1": kobold},
        "round_number": 1,
    }

    update = node(state)

    assert update["round_number"] == 2
    assert update["resolved_actions"] == []  # advance_round yields no ResolvedAction
    assert update["combatants"]["avantor"].statuses == []  # 1 -> expired
    assert update["combatants"]["kobold_1"].statuses[0].duration_rounds == 4
    assert any("expires" in line and "Avantor" in line for line in update["event_log"])


# ---------------------------------------------------------------------------
# resolve_combatants -- feature texts
# ---------------------------------------------------------------------------


def test_resolve_collects_feature_texts_from_actor_features(registry, app_config):
    node = make_resolve_combatants(registry, app_config)
    action = _parsed_action("Avantor", weapon_or_spell="flametongue")
    state = {"parsed_actions": [action], "combatants": {}, "round_number": 1}

    update = node(state)

    feature_texts = update["resolved_actions"][0].feature_texts
    assert feature_texts["savage_attacks"] == registry.features["savage_attacks"].effect
    assert feature_texts["great_weapon_master"] == registry.features["great_weapon_master"].effect


# ---------------------------------------------------------------------------
# apply_damage
# ---------------------------------------------------------------------------


def test_apply_damage_valid_report_kills_monster(registry, app_config):
    node = make_apply_damage(app_config)
    kobold_1 = registry.combatant_from_monster("kobold", 1)  # 5 hp
    avantor = registry.combatant_from_character("avantor")
    resolved = _resolved_action("avantor", ["kobold_1"])
    final_content = (
        '```json\n{"action_index": 0, "total_damage": 12, '
        '"per_target": [{"target_id": "kobold_1", "damage": 12}], '
        '"breakdown": {}, "description": "Avantor cleaves the kobold."}\n```'
    )
    state = {
        "combatants": {"avantor": avantor, "kobold_1": kobold_1},
        "resolved_actions": [resolved],
        "current_action_index": 0,
        "damage_reports": [],
        "messages": _scaffold_messages(final_content),
    }

    update = node(state)

    kobold_updated = update["combatants"]["kobold_1"]
    assert kobold_updated.hp == 0
    assert kobold_updated.is_alive is False
    assert any(s.name == "dead" for s in kobold_updated.statuses)
    assert any("kobold_1" in line and "dead" in line for line in update["event_log"])
    assert len(update["damage_reports"]) == 1
    assert update["current_action_index"] == 1


def test_apply_damage_pc_downed_gets_unconscious_not_dead(registry, app_config):
    node = make_apply_damage(app_config)
    generic_pc = registry.combatant_from_character("generic").model_copy(update={"hp": 5})
    kobold_1 = registry.combatant_from_monster("kobold", 1)
    resolved = _resolved_action("kobold_1", ["generic"])
    final_content = (
        '{"action_index": 0, "total_damage": 5, '
        '"per_target": [{"target_id": "generic", "damage": 5}], "description": "bite"}'
    )
    state = {
        "combatants": {"kobold_1": kobold_1, "generic": generic_pc},
        "resolved_actions": [resolved],
        "current_action_index": 0,
        "damage_reports": [],
        "messages": _scaffold_messages(final_content),
    }

    update = node(state)

    generic_updated = update["combatants"]["generic"]
    assert generic_updated.hp == 0
    assert generic_updated.is_alive is False
    assert any(s.name == "unconscious" for s in generic_updated.statuses)
    assert not any(s.name == "dead" for s in generic_updated.statuses)


def test_apply_damage_clamps_hp_at_zero_never_negative(registry, app_config):
    node = make_apply_damage(app_config)
    kobold_1 = registry.combatant_from_monster("kobold", 1)  # 5 hp
    avantor = registry.combatant_from_character("avantor")
    resolved = _resolved_action("avantor", ["kobold_1"])
    final_content = (
        '{"action_index": 0, "total_damage": 999, '
        '"per_target": [{"target_id": "kobold_1", "damage": 999}]}'
    )
    state = {
        "combatants": {"avantor": avantor, "kobold_1": kobold_1},
        "resolved_actions": [resolved],
        "current_action_index": 0,
        "damage_reports": [],
        "messages": _scaffold_messages(final_content),
    }

    update = node(state)

    assert update["combatants"]["kobold_1"].hp == 0


def test_apply_damage_healing_clamps_at_max_hp(registry, app_config):
    node = make_apply_damage(app_config)
    wounded_avantor = registry.combatant_from_character("avantor").model_copy(
        update={"hp": 55}
    )
    resolved = _resolved_action("avantor", ["avantor"])
    final_content = (
        '{"action_index": 0, "total_healing": 20, '
        '"per_target": [{"target_id": "avantor", "damage": 0, "healing": 20}]}'
    )
    state = {
        "combatants": {"avantor": wounded_avantor},
        "resolved_actions": [resolved],
        "current_action_index": 0,
        "damage_reports": [],
        "messages": _scaffold_messages(final_content),
    }

    update = node(state)

    assert update["combatants"]["avantor"].hp == 62  # clamped at max_hp
    assert any("heals" in line for line in update["event_log"])


def test_apply_damage_malformed_report_splits_total_damage_evenly(registry, app_config):
    node = make_apply_damage(app_config)
    kobold_1 = registry.combatant_from_monster("kobold", 1)
    kobold_2 = registry.combatant_from_monster("kobold", 2)
    avantor = registry.combatant_from_character("avantor")
    resolved = _resolved_action("avantor", ["kobold_1", "kobold_2"])
    final_content = '{"total_damage": 10}'  # no per_target at all
    state = {
        "combatants": {"avantor": avantor, "kobold_1": kobold_1, "kobold_2": kobold_2},
        "resolved_actions": [resolved],
        "current_action_index": 0,
        "damage_reports": [],
        "messages": _scaffold_messages(final_content),
    }

    update = node(state)

    report = update["damage_reports"][0]
    assert report.action_index == 0
    assert sorted((pt.target_id, pt.damage) for pt in report.per_target) == [
        ("kobold_1", 5),
        ("kobold_2", 5),
    ]
    assert update["combatants"]["kobold_1"].hp == 0
    assert update["combatants"]["kobold_2"].hp == 0


def test_apply_damage_no_json_produces_zero_damage_report(registry, app_config):
    node = make_apply_damage(app_config)
    kobold_1 = registry.combatant_from_monster("kobold", 1)
    avantor = registry.combatant_from_character("avantor")
    resolved = _resolved_action("avantor", ["kobold_1"])
    final_content = "The attack narrowly misses."
    state = {
        "combatants": {"avantor": avantor, "kobold_1": kobold_1},
        "resolved_actions": [resolved],
        "current_action_index": 0,
        "damage_reports": [],
        "messages": _scaffold_messages(final_content),
    }

    update = node(state)

    assert update["combatants"] == {}  # no per_target entries -> nothing touched
    report = update["damage_reports"][0]
    assert report.total_damage == 0
    assert report.per_target == []
    assert update["current_action_index"] == 1


def test_apply_damage_emits_remove_messages_for_scaffolding_only(registry, app_config):
    node = make_apply_damage(app_config)
    kobold_1 = registry.combatant_from_monster("kobold", 1)
    avantor = registry.combatant_from_character("avantor")
    resolved = _resolved_action("avantor", ["kobold_1"])
    final_content = (
        '{"action_index": 0, "total_damage": 3, '
        '"per_target": [{"target_id": "kobold_1", "damage": 3}]}'
    )
    state = {
        "combatants": {"avantor": avantor, "kobold_1": kobold_1},
        "resolved_actions": [resolved],
        "current_action_index": 0,
        "damage_reports": [],
        "messages": _scaffold_messages(final_content),
    }

    update = node(state)

    assert all(isinstance(m, RemoveMessage) for m in update["messages"])
    removed_ids = {m.id for m in update["messages"]}
    assert removed_ids == {"ai1", "t1"}
    assert "final" not in removed_ids
    assert "human1" not in removed_ids


def test_apply_damage_out_of_range_index_is_defensive_noop(app_config):
    node = make_apply_damage(app_config)
    state = {"current_action_index": 2, "resolved_actions": []}

    update = node(state)

    assert update == {"current_action_index": 2}


def test_apply_damage_cursor_always_increments_by_one(registry, app_config):
    node = make_apply_damage(app_config)
    kobold_1 = registry.combatant_from_monster("kobold", 1)
    avantor = registry.combatant_from_character("avantor")
    resolved = _resolved_action("avantor", ["kobold_1"])
    state = {
        "combatants": {"avantor": avantor, "kobold_1": kobold_1},
        "resolved_actions": [resolved],
        "current_action_index": 0,
        "damage_reports": [],
        "messages": _scaffold_messages('{"total_damage": 1, "per_target": []}'),
    }

    update = node(state)

    assert update["current_action_index"] == 1
