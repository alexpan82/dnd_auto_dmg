"""Tests for the LLM-backed graph nodes (WP4b, Spec §6/§10).

All tests use the ``make_scripted_llm`` fixture from ``tests/conftest.py``
(``ScriptedChatModel``) -- no network access, no API key, fully
deterministic. Factories are imported directly from their owning submodules
(``dnd_auto_dmg.nodes.relevance`` etc.), NOT from ``dnd_auto_dmg.nodes``
(that package's ``__init__.py`` is owned by a different work package).
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from dnd_auto_dmg.nodes.damage import make_calculate_damage
from dnd_auto_dmg.nodes.narrate import make_narrate
from dnd_auto_dmg.nodes.parse import make_parse_actions
from dnd_auto_dmg.nodes.relevance import make_check_relevance
from dnd_auto_dmg.schemas import (
    Combatant,
    ParsedAction,
    ParsedTurn,
    ResolvedAction,
    TargetRef,
)
from dnd_auto_dmg.tools import roll_dice

# ---------------------------------------------------------------------------
# check_relevance
# ---------------------------------------------------------------------------


def test_check_relevance_yes_exact_update(make_scripted_llm):
    llm = make_scripted_llm([AIMessage(content="Yes")])
    node = make_check_relevance(llm)

    update = node({"messages": [HumanMessage(content="Avantor attacks the goblin")]})

    assert update == {"relevant_query": True}
    assert "messages" not in update


def test_check_relevance_no_case_insensitive(make_scripted_llm):
    llm = make_scripted_llm([AIMessage(content="no.")])
    node = make_check_relevance(llm)

    update = node({"messages": [HumanMessage(content="What's the weather like?")]})

    assert update == {"relevant_query": False}


def test_check_relevance_yes_with_trailing_text(make_scripted_llm):
    llm = make_scripted_llm([AIMessage(content="YES, definitely")])
    node = make_check_relevance(llm)

    update = node({"messages": [HumanMessage(content="Avantor casts fireball")]})

    assert update == {"relevant_query": True}


def test_check_relevance_sends_system_message_first(make_scripted_llm):
    llm = make_scripted_llm([AIMessage(content="Yes")])
    node = make_check_relevance(llm)
    human = HumanMessage(content="Avantor attacks the goblin")

    node({"messages": [human]})

    assert len(llm.calls) == 1
    sent = llm.calls[0]
    assert isinstance(sent[0], SystemMessage)
    assert sent[1:] == [human]


# ---------------------------------------------------------------------------
# parse_actions
# ---------------------------------------------------------------------------


def test_parse_actions_single_action(make_scripted_llm):
    turn = ParsedTurn(
        actions=[
            ParsedAction(
                actor="Avantor",
                action_type="attack",
                weapon_or_spell="flametongue",
                targets=[TargetRef(name="goblin")],
            )
        ]
    )
    llm = make_scripted_llm([turn])
    node = make_parse_actions(llm)

    update = node(
        {"messages": [HumanMessage(content="Avantor attacks the goblin with his flametongue")]}
    )

    assert "messages" not in update
    actions = update["parsed_actions"]
    assert len(actions) == 1
    assert actions[0].actor == "Avantor"
    assert actions[0].action_type == "attack"
    assert actions[0].weapon_or_spell == "flametongue"


def test_parse_actions_multi_action(make_scripted_llm):
    turn = ParsedTurn(
        actions=[
            ParsedAction(actor="Avantor", action_type="attack", targets=[TargetRef(name="goblin")]),
            ParsedAction(actor="Mira", action_type="heal", targets=[TargetRef(name="Avantor")]),
        ]
    )
    llm = make_scripted_llm([turn])
    node = make_parse_actions(llm)

    update = node({"messages": [HumanMessage(content="Avantor attacks; Mira heals him")]})

    actions = update["parsed_actions"]
    assert len(actions) == 2
    assert actions[0].actor == "Avantor"
    assert actions[0].action_type == "attack"
    assert actions[1].actor == "Mira"
    assert actions[1].action_type == "heal"


def test_parse_actions_structured_output_failure_falls_back_to_extract_json(
    make_scripted_llm, monkeypatch
):
    content = (
        'Sure: {"actions": [{"actor": "Avantor", "action_type": "attack", '
        '"targets": [{"name": "kobold", "count": 3}]}]}'
    )
    llm = make_scripted_llm([AIMessage(content=content)])

    def _raise_no_structured_output(self, schema, **kwargs):
        raise RuntimeError("no structured output")

    monkeypatch.setattr(type(llm), "with_structured_output", _raise_no_structured_output)

    node = make_parse_actions(llm)
    update = node({"messages": [HumanMessage(content="They do it again, 3 kobolds this time")]})

    actions = update["parsed_actions"]
    assert len(actions) == 1
    assert actions[0].actor == "Avantor"
    assert actions[0].targets[0].name == "kobold"
    assert actions[0].targets[0].count == 3


def test_parse_actions_garbage_fallback_yields_empty_actions(make_scripted_llm, monkeypatch):
    llm = make_scripted_llm([AIMessage(content="This is just prose, no JSON anywhere in here.")])

    def _raise_no_structured_output(self, schema, **kwargs):
        raise RuntimeError("no structured output")

    monkeypatch.setattr(type(llm), "with_structured_output", _raise_no_structured_output)

    node = make_parse_actions(llm)
    update = node({"messages": [HumanMessage(content="blah blah blah")]})

    assert update["parsed_actions"] == []
    assert "messages" not in update


# ---------------------------------------------------------------------------
# calculate_damage
# ---------------------------------------------------------------------------


def _damage_test_state():
    avantor = Combatant(
        id="avantor",
        name="Avantor",
        kind="pc",
        max_hp=45,
        hp=40,
        ac=18,
        attributes={"str": 18, "dex": 12},
    )
    kobold = Combatant(
        id="kobold_1",
        name="Kobold",
        kind="monster",
        max_hp=5,
        hp=5,
        ac=12,
    )
    resolved = ResolvedAction(
        action=ParsedAction(
            actor="Avantor",
            action_type="attack",
            weapon_or_spell="flametongue",
            is_critical_hit=True,
        ),
        actor_id="avantor",
        target_ids=["kobold_1"],
        item={"name": "flametongue", "damage": {"fire": "2d6"}, "magic_bonus": 1, "ability": "str"},
        feature_texts={"savage_attacks": "On a crit roll an extra die"},
    )
    return {
        "combatants": {"avantor": avantor, "kobold_1": kobold},
        "resolved_actions": [resolved],
        "current_action_index": 0,
        "messages": [HumanMessage(content="Avantor crits the kobold with his flametongue")],
    }


def test_calculate_damage_returns_only_messages(make_scripted_llm):
    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "roll_dice",
                "args": {"num_dice": 2, "num_sides": 6, "modifier": 1},
                "id": "call_1",
            }
        ],
    )
    llm = make_scripted_llm([tool_call_response])
    node = make_calculate_damage(llm)

    update = node(_damage_test_state())

    assert set(update.keys()) == {"messages"}
    assert update["messages"] == [tool_call_response]
    assert update["messages"][0].tool_calls == tool_call_response.tool_calls


def test_calculate_damage_prompt_contents(make_scripted_llm):
    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "roll_dice",
                "args": {"num_dice": 2, "num_sides": 6, "modifier": 1},
                "id": "call_1",
            }
        ],
    )
    llm = make_scripted_llm([tool_call_response])
    node = make_calculate_damage(llm)

    update = node(_damage_test_state())

    assert len(llm.calls) == 1
    sent = llm.calls[0]
    system_msg = sent[0]
    assert isinstance(system_msg, SystemMessage)
    content = system_msg.content

    # Real tool name + real description rendered (fixes old bug 1).
    assert "roll_dice" in content
    assert roll_dice.description in content

    # The literal, never-formatted placeholders from the old buggy prompt
    # must NOT appear.
    assert "{tools}" not in content
    assert "{roll_dice}" not in content

    # Actual target id, DamageReport shape marker, actor, feature text,
    # and a critical-hit indication.
    assert "kobold_1" in content
    assert "action_index" in content
    assert "Avantor" in content or "avantor" in content
    assert "On a crit roll an extra die" in content
    assert "crit" in content.lower()

    # System prompt itself is ephemeral -- never returned in the update.
    assert all(not isinstance(m, SystemMessage) for m in update["messages"])


# ---------------------------------------------------------------------------
# narrate
# ---------------------------------------------------------------------------


def test_narrate_returns_only_response_message(make_scripted_llm):
    response = AIMessage(content="The blade sings as it cleaves through the kobold's hide.")
    llm = make_scripted_llm([response])
    node = make_narrate(llm)

    state = {
        "event_log": [
            "Round 1 begins",
            "Avantor deals 12 fire damage to kobold_1 (0/5 HP, dead)",
        ],
        "messages": [HumanMessage(content="continue")],
    }

    update = node(state)

    assert update == {"messages": [response]}


def test_narrate_prompt_includes_event_log(make_scripted_llm):
    response = AIMessage(content="The blade sings...")
    llm = make_scripted_llm([response])
    node = make_narrate(llm)

    state = {
        "event_log": [
            "Round 1 begins",
            "Avantor deals 12 fire damage to kobold_1 (0/5 HP, dead)",
        ],
        "messages": [HumanMessage(content="continue")],
    }

    node(state)

    sent = llm.calls[0]
    system_msg = sent[0]
    assert isinstance(system_msg, SystemMessage)
    assert "Avantor deals 12 fire damage to kobold_1 (0/5 HP, dead)" in system_msg.content
