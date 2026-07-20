"""Integration tests for the assembled combat graph (Spec §10).

Exercises ``dnd_auto_dmg.graph.build_graph`` end-to-end -- START through
every real node (``begin_turn``, ``check_relevance``, ``parse_actions``,
``resolve_combatants``, ``no_actions``, ``calculate_damage``, the
``roll_dice`` ``ToolNode``, ``apply_damage``, ``narrate``) -- against a
``ScriptedChatModel`` (fake LLM) and the REAL data registry, compiled with a
real ``SqliteSaver`` checkpointer.

The scripted five-turn conversation mirrors ``src/demo.py``'s manual smoke
test. Per relevant turn the scripted queue is consumed in this order:

    1. check_relevance   -> AIMessage("Yes")
    2. parse_actions      -> a ParsedTurn pydantic object (structured output)
    3. calculate_damage   -> (repeated once per resolved action; a tool-call
                              AIMessage optionally precedes the final report
                              AIMessage, looping through the real ToolNode)
    4. narrate             -> AIMessage (only when narration is enabled,
                              which is the default)

For a non-relevant turn, only step 1 is consumed.
"""

import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from dnd_auto_dmg.graph import RECURSION_LIMIT, build_graph
from dnd_auto_dmg.schemas import DamageReport, ParsedAction, ParsedTurn, PerTargetDamage, TargetRef

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _report_message(report: DamageReport) -> AIMessage:
    """A ``calculate_damage``-shaped final AIMessage: a fenced ```json block
    (parsed by ``dnd_auto_dmg.tools.extract_json``) followed by prose."""
    return AIMessage(content=f"```json\n{report.model_dump_json()}\n```\n{report.description}")


def _tool_call_message(call_id: str = "call1") -> AIMessage:
    """An AIMessage carrying a ``roll_dice`` tool call, to exercise the real
    ``ToolNode`` loop (``calculate_damage`` -> ``roll_dice`` -> ``calculate_damage``)."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "roll_dice",
                "args": {"num_dice": 2, "num_sides": 6, "modifier": 4},
                "id": call_id,
            }
        ],
    )


def _compile_app(llm, registry, tmp_path, filename="ckpt.sqlite"):
    graph = build_graph(llm=llm, registry=registry)
    conn = sqlite3.connect(str(tmp_path / filename), check_same_thread=False)
    saver = SqliteSaver(conn)
    return graph.compile(checkpointer=saver)


def _invoke(app, thread_id, text):
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    return app.invoke({"messages": [HumanMessage(text)]}, config), config


# ---------------------------------------------------------------------------
# The §10 five-turn scenario
# ---------------------------------------------------------------------------


def test_full_five_turn_scenario(registry, make_scripted_llm, tmp_path):
    tensers_report = DamageReport(
        action_index=0,
        total_damage=0,
        per_target=[PerTargetDamage(target_id="avantor", damage=0)],
        description="Avantor is wreathed in magical might.",
    )
    goblin_hit_1 = DamageReport(
        action_index=0,
        total_damage=6,
        per_target=[PerTargetDamage(target_id="goblin_1", damage=6)],
        description="The greatsword bites into the goblin.",
    )
    goblin_hit_2 = DamageReport(
        action_index=0,
        total_damage=6,
        per_target=[PerTargetDamage(target_id="goblin_1", damage=6)],
        description="Another vicious slash finishes the goblin.",
    )
    fireball_report = DamageReport(
        action_index=0,
        total_damage=60,
        per_target=[
            PerTargetDamage(target_id="kobold_1", damage=20),
            PerTargetDamage(target_id="kobold_2", damage=20),
            PerTargetDamage(target_id="kobold_3", damage=20),
        ],
        description="The fireball engulfs the kobolds.",
    )

    llm = make_scripted_llm(
        [
            # Turn 1: Tenser's Transformation (buff, self-target, no tool loop).
            AIMessage("Yes"),
            ParsedTurn(
                actions=[
                    ParsedAction(
                        actor="Avantor",
                        action_type="cast",
                        weapon_or_spell="tensers transformation",
                        targets=[TargetRef(name="self")],
                        statuses_applied=["tensers_transformation"],
                    )
                ]
            ),
            _report_message(tensers_report),
            AIMessage("Avantor's muscles swell with magical power."),
            # Turn 2: greatsword attack on the goblin (exercises the tool loop).
            AIMessage("Yes"),
            ParsedTurn(
                actions=[
                    ParsedAction(
                        actor="Avantor",
                        action_type="attack",
                        weapon_or_spell="greatsword",
                        targets=[TargetRef(name="goblin")],
                    )
                ]
            ),
            _tool_call_message(),
            _report_message(goblin_hit_1),
            AIMessage("The goblin reels from the blow."),
            # Turn 3: "They do it again" -- fake resolves the pronoun itself.
            AIMessage("Yes"),
            ParsedTurn(
                actions=[
                    ParsedAction(
                        actor="Avantor",
                        action_type="attack",
                        weapon_or_spell="greatsword",
                        targets=[TargetRef(name="goblin")],
                    )
                ]
            ),
            _report_message(goblin_hit_2),
            AIMessage("The goblin collapses."),
            # Turn 4: 5th-level fireball at a group of 3 kobolds.
            AIMessage("Yes"),
            ParsedTurn(
                actions=[
                    ParsedAction(
                        actor="Avantor",
                        action_type="cast",
                        weapon_or_spell="fireball",
                        spell_level=5,
                        targets=[TargetRef(name="kobold", count=3)],
                    )
                ]
            ),
            _report_message(fireball_report),
            AIMessage("The kobolds are incinerated."),
            # Turn 5: irrelevant message -- only the relevance check fires.
            AIMessage("No"),
        ]
    )

    app = _compile_app(llm, registry, tmp_path)
    thread_id = "scenario-1"

    # -- Turn 1 -----------------------------------------------------------
    result, config = _invoke(app, thread_id, "Avantor casts Tenser's Transformation on themself")
    avantor = result["combatants"]["avantor"]
    assert avantor.hp == 62
    status_names = [s.name for s in avantor.statuses]
    assert "tensers_transformation" in status_names
    tt_status = next(s for s in avantor.statuses if s.name == "tensers_transformation")
    assert tt_status.duration_rounds == 10
    assert result["current_action_index"] == 1 == len(result["resolved_actions"])

    # -- Turn 2 -----------------------------------------------------------
    result, _ = _invoke(app, thread_id, "Avantor attacks the goblin with his greatsword")
    assert "goblin_1" in result["combatants"]
    goblin = result["combatants"]["goblin_1"]
    assert goblin.max_hp == 7
    assert goblin.hp == 1
    assert goblin.is_alive is True
    assert result["current_action_index"] == 1 == len(result["resolved_actions"])

    # Message hygiene end-to-end (Spec §5): inspect the PERSISTED state.
    persisted_messages = app.get_state(config).values["messages"]
    assert not any(isinstance(m, SystemMessage) for m in persisted_messages)
    assert not any(isinstance(m, ToolMessage) for m in persisted_messages)
    assert not any(
        isinstance(m, AIMessage) and getattr(m, "tool_calls", None) for m in persisted_messages
    )
    # The tool-loop scaffolding is gone; the final report AIMessage and the
    # narrate AIMessage from turn 2 are the last two messages.
    assert "```json" in persisted_messages[-2].content
    assert persisted_messages[-1].content == "The goblin reels from the blow."

    # -- Turn 3 -----------------------------------------------------------
    result, _ = _invoke(app, thread_id, "They do it again")
    goblin = result["combatants"]["goblin_1"]
    assert goblin.hp == 0
    assert goblin.is_alive is False
    assert any(s.name == "dead" for s in goblin.statuses)
    # Target reuse: still exactly one "goblin_" combatant (goblin_1, not goblin_2).
    goblin_ids = [cid for cid in result["combatants"] if cid.startswith("goblin_")]
    assert goblin_ids == ["goblin_1"]

    # -- Turn 4 -----------------------------------------------------------
    result, config = _invoke(
        app, thread_id, "He then casts 5th level fireball at a group of 3 kobolds"
    )
    for kid in ("kobold_1", "kobold_2", "kobold_3"):
        assert kid in result["combatants"]
        kobold = result["combatants"][kid]
        assert kobold.max_hp == 5
        assert kobold.hp == 0
        assert kobold.is_alive is False
        assert any(s.name == "dead" for s in kobold.statuses)

    # Explicit termination assertion (Spec §12 risk 3), via get_state.
    snapshot = app.get_state(config).values
    assert snapshot["current_action_index"] == 1 == len(snapshot["resolved_actions"])

    combatants_before_turn5 = {
        cid: (c.hp, c.is_alive) for cid, c in result["combatants"].items()
    }

    # -- Turn 5 -----------------------------------------------------------
    result, _ = _invoke(app, thread_id, "Literal nonsense")
    combatants_after_turn5 = {
        cid: (c.hp, c.is_alive) for cid, c in result["combatants"].items()
    }
    assert combatants_after_turn5 == combatants_before_turn5
    assert result["damage_reports"] == []
    # Every scripted response was consumed -- proves no extra LLM calls
    # happened on the irrelevant turn.
    assert llm.responses == []


# ---------------------------------------------------------------------------
# Multi-action single message: the per-action cursor loop
# ---------------------------------------------------------------------------


def test_multi_action_cursor_loop(registry, make_scripted_llm, tmp_path):
    goblin_report = DamageReport(
        action_index=0,
        total_damage=6,
        per_target=[PerTargetDamage(target_id="goblin_1", damage=6)],
        description="The greatsword connects with the goblin.",
    )
    kobold_report = DamageReport(
        action_index=1,
        total_damage=5,
        per_target=[PerTargetDamage(target_id="kobold_1", damage=5)],
        description="A follow-up strike drops the kobold.",
    )

    llm = make_scripted_llm(
        [
            AIMessage("Yes"),
            ParsedTurn(
                actions=[
                    ParsedAction(
                        actor="Avantor",
                        action_type="attack",
                        weapon_or_spell="greatsword",
                        targets=[TargetRef(name="goblin")],
                    ),
                    ParsedAction(
                        actor="Avantor",
                        action_type="attack",
                        weapon_or_spell="greatsword",
                        targets=[TargetRef(name="kobold")],
                    ),
                ]
            ),
            _report_message(goblin_report),
            _report_message(kobold_report),
            AIMessage("Avantor cuts down both foes in a single flurry."),
        ]
    )

    app = _compile_app(llm, registry, tmp_path)
    result, config = _invoke(
        app, "multi-action", "Avantor attacks the goblin and the kobold with his greatsword"
    )

    assert result["combatants"]["goblin_1"].hp == 1
    assert result["combatants"]["kobold_1"].hp == 0
    assert result["current_action_index"] == 2 == len(result["resolved_actions"])
    assert len(result["damage_reports"]) == 2
    assert llm.responses == []


# ---------------------------------------------------------------------------
# No resolvable actions: route_actions -> no_actions -> END
# ---------------------------------------------------------------------------


def test_no_actions_route(registry, make_scripted_llm, tmp_path):
    llm = make_scripted_llm(
        [
            AIMessage("Yes"),
            ParsedTurn(actions=[]),
        ]
    )

    app = _compile_app(llm, registry, tmp_path)
    result, _ = _invoke(app, "no-actions", "Something combat-flavored but unparseable")

    assert result["resolved_actions"] == []
    assert "No resolvable actions this turn." in result["event_log"]
    # No calculate_damage/narrate calls were made -- queue fully consumed.
    assert llm.responses == []
