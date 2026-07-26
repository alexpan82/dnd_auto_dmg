"""Tests for ``dnd_auto_dmg.state``.

Covers:

* ``merge_combatants`` reducer semantics in isolation (Spec §5).
* The SqliteSaver round-trip lock-in for pydantic values in state channels
  (Spec §12 risk 1) -- CONFIRMED PASS per the WP3 de-risk experiment; these
  tests pin that behavior down so a future langgraph/langgraph-checkpoint
  upgrade can't silently regress it.
* A demonstration that per-turn scratch fields persist across invokes on the
  same checkpointed thread (Spec §12 risk 2) -- this is exactly why a
  ``begin_turn`` node (WP4a) must explicitly reset them every turn.
* Self-tests for the ``ScriptedChatModel`` test double defined in
  ``tests/conftest.py``, so later work packages don't have to debug the
  fixture itself.
"""

import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel

from dnd_auto_dmg.schemas import Combatant, DamageReport, StatusEffect
from dnd_auto_dmg.state import CombatState, merge_combatants

# ---------------------------------------------------------------------------
# merge_combatants -- reducer unit tests
# ---------------------------------------------------------------------------


def _combatant(cid: str, **overrides) -> Combatant:
    defaults = dict(id=cid, name=cid.title(), kind="monster", max_hp=10, hp=10)
    defaults.update(overrides)
    return Combatant(**defaults)


def test_merge_combatants_insert_into_none_existing():
    kobold = _combatant("kobold_1")
    result = merge_combatants(None, {"kobold_1": kobold})
    assert result == {"kobold_1": kobold}


def test_merge_combatants_insert_into_empty_existing():
    kobold = _combatant("kobold_1")
    result = merge_combatants({}, {"kobold_1": kobold})
    assert result == {"kobold_1": kobold}


def test_merge_combatants_replace_is_wholesale_not_deep_merge():
    old = _combatant(
        "avantor",
        kind="pc",
        statuses=[StatusEffect(name="tensers_transformation", duration_rounds=10)],
    )
    new = _combatant("avantor", kind="pc", hp=5)  # no statuses at all

    existing = {"avantor": old}
    result = merge_combatants(existing, {"avantor": new})

    assert result["avantor"] is new
    assert result["avantor"].statuses == []  # old status is GONE, not merged in
    assert result["avantor"].hp == 5


def test_merge_combatants_delete_via_none_value():
    kobold = _combatant("kobold_1")
    existing = {"kobold_1": kobold, "kobold_2": _combatant("kobold_2")}
    result = merge_combatants(existing, {"kobold_1": None})
    assert result == {"kobold_2": existing["kobold_2"]}


def test_merge_combatants_delete_of_missing_key_is_silent_noop():
    existing = {"kobold_1": _combatant("kobold_1")}
    result = merge_combatants(existing, {"nonexistent": None})
    assert result == existing


def test_merge_combatants_empty_update_returns_equal_but_not_same_dict():
    existing = {"kobold_1": _combatant("kobold_1")}
    result = merge_combatants(existing, {})
    assert result == existing
    assert result is not existing


def test_merge_combatants_does_not_mutate_existing():
    kobold_1 = _combatant("kobold_1")
    kobold_2 = _combatant("kobold_2")
    existing = {"kobold_1": kobold_1, "kobold_2": kobold_2}
    existing_snapshot = dict(existing)

    new_kobold_1 = _combatant("kobold_1", hp=1)
    merge_combatants(existing, {"kobold_1": new_kobold_1, "kobold_2": None})

    # existing must be untouched after the call
    assert existing == existing_snapshot
    assert existing["kobold_1"] is kobold_1
    assert existing["kobold_2"] is kobold_2


def test_merge_combatants_mixed_insert_and_delete_in_same_update():
    kobold_1 = _combatant("kobold_1")
    kobold_2 = _combatant("kobold_2")
    existing = {"kobold_1": kobold_1, "kobold_2": kobold_2}

    goblin = _combatant("goblin_1")
    result = merge_combatants(existing, {"goblin_1": goblin, "kobold_2": None})

    assert result == {"kobold_1": kobold_1, "goblin_1": goblin}
    assert "kobold_2" not in result


def test_merge_combatants_none_update_returns_copy_of_existing():
    existing = {"kobold_1": _combatant("kobold_1")}
    result = merge_combatants(existing, None)
    assert result == existing
    assert result is not existing


# ---------------------------------------------------------------------------
# SqliteSaver round-trip lock-in (Spec §12 risk 1)
# ---------------------------------------------------------------------------


def _make_graph():
    """A minimal single-node StateGraph over CombatState, used across the
    checkpoint tests. Rebuilt fresh each time a "new process" needs to be
    simulated (same node logic, freshly compiled graph object)."""
    g = StateGraph(CombatState)

    def write_avantor(state: CombatState) -> dict:
        return {
            "combatants": {
                "avantor": Combatant(
                    id="avantor",
                    name="Avantor",
                    kind="pc",
                    max_hp=62,
                    hp=62,
                    statuses=[
                        StatusEffect(name="tensers_transformation", duration_rounds=10)
                    ],
                )
            },
            "round_number": 1,
            "event_log": ["x"],
        }

    g.add_node("write_avantor", write_avantor)
    g.add_edge(START, "write_avantor")
    g.add_edge("write_avantor", END)
    return g


def test_sqlite_saver_round_trips_pydantic_combatant(tmp_path):
    db_path = tmp_path / "ckpt.sqlite"
    thread_cfg = {"configurable": {"thread_id": "t1"}}

    # --- "process 1": write ---
    conn1 = sqlite3.connect(str(db_path), check_same_thread=False)
    saver1 = SqliteSaver(conn1)
    graph1 = _make_graph().compile(checkpointer=saver1)
    graph1.invoke({}, thread_cfg)
    conn1.close()

    # --- "process 2": fresh connection/saver/graph, read back ---
    conn2 = sqlite3.connect(str(db_path), check_same_thread=False)
    saver2 = SqliteSaver(conn2)
    graph2 = _make_graph().compile(checkpointer=saver2)

    snapshot = graph2.get_state(thread_cfg)
    values = snapshot.values

    avantor = values["combatants"]["avantor"]
    assert isinstance(avantor, Combatant)
    assert isinstance(avantor.statuses[0], StatusEffect)
    assert avantor.hp == 62
    assert avantor.statuses[0].duration_rounds == 10
    assert values["round_number"] == 1
    assert values["event_log"] == ["x"]

    conn2.close()


def test_sqlite_saver_resumed_thread_node_receives_live_model(tmp_path):
    db_path = tmp_path / "ckpt.sqlite"
    thread_cfg = {"configurable": {"thread_id": "t1"}}

    conn1 = sqlite3.connect(str(db_path), check_same_thread=False)
    saver1 = SqliteSaver(conn1)
    graph1 = _make_graph().compile(checkpointer=saver1)
    graph1.invoke({}, thread_cfg)
    conn1.close()

    # Fresh compile over the same DB/thread; a node that reads the persisted
    # Combatant back out of state and calls .model_copy(update=...) on it --
    # proving nodes on a RESUMED thread receive live pydantic models, not
    # plain dicts.
    conn2 = sqlite3.connect(str(db_path), check_same_thread=False)
    saver2 = SqliteSaver(conn2)

    g2 = StateGraph(CombatState)

    def damage_avantor(state: CombatState) -> dict:
        current = state["combatants"]["avantor"]
        updated = current.model_copy(update={"hp": current.hp - 10})
        return {"combatants": {"avantor": updated}}

    g2.add_node("damage_avantor", damage_avantor)
    g2.add_edge(START, "damage_avantor")
    g2.add_edge("damage_avantor", END)
    graph2 = g2.compile(checkpointer=saver2)

    result = graph2.invoke({}, thread_cfg)
    avantor = result["combatants"]["avantor"]
    assert isinstance(avantor, Combatant)
    assert avantor.hp == 52

    conn2.close()


# ---------------------------------------------------------------------------
# Scratch-field persistence across invokes (Spec §12 risk 2)
# ---------------------------------------------------------------------------


def test_scratch_fields_persist_across_invokes_without_explicit_reset(tmp_path):
    """Demonstrates WHY begin_turn (WP4a) must reset scratch fields itself:
    plain TypedDict fields with no reducer are last-write-wins WITHIN an
    invoke, but across separate invokes on the same checkpointed thread the
    checkpointer restores whatever was last written -- nothing clears it for
    you between turns."""
    db_path = tmp_path / "ckpt.sqlite"
    thread_cfg = {"configurable": {"thread_id": "t1"}}

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)

    g = StateGraph(CombatState)

    def set_index(state: CombatState) -> dict:
        return {"current_action_index": 3}

    g.add_node("set_index", set_index)
    g.add_edge(START, "set_index")
    g.add_edge("set_index", END)
    graph = g.compile(checkpointer=saver)

    graph.invoke({}, thread_cfg)
    snapshot = graph.get_state(thread_cfg)
    assert snapshot.values["current_action_index"] == 3

    # Second invoke on the SAME thread, with a node that does NOT touch
    # current_action_index at all.
    g2 = StateGraph(CombatState)

    def noop(state: CombatState) -> dict:
        return {"event_log": ["noop ran"]}

    g2.add_node("noop", noop)
    g2.add_edge(START, "noop")
    g2.add_edge("noop", END)
    graph2 = g2.compile(checkpointer=saver)

    graph2.invoke({}, thread_cfg)
    snapshot2 = graph2.get_state(thread_cfg)

    # Still 3 -- nothing reset it. This is the exact hazard begin_turn exists
    # to fix.
    assert snapshot2.values["current_action_index"] == 3

    conn.close()


# ---------------------------------------------------------------------------
# ScriptedChatModel self-tests (from tests/conftest.py)
# ---------------------------------------------------------------------------


class _Dummy(BaseModel):
    value: int = 0


def test_scripted_llm_returns_queued_messages_in_order(make_scripted_llm):
    llm = make_scripted_llm([AIMessage(content="first"), AIMessage(content="second")])
    r1 = llm.invoke([HumanMessage(content="hi")])
    r2 = llm.invoke([HumanMessage(content="hi again")])
    assert r1.content == "first"
    assert r2.content == "second"


def test_scripted_llm_preserves_tool_calls(make_scripted_llm):
    tool_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "roll_dice", "args": {"num_dice": 1, "num_sides": 6, "modifier": 0}, "id": "1"}
        ],
    )
    llm = make_scripted_llm([tool_msg])
    result = llm.invoke([HumanMessage(content="roll it")])
    assert result.tool_calls
    assert result.tool_calls[0]["name"] == "roll_dice"
    assert result.tool_calls[0]["args"] == {"num_dice": 1, "num_sides": 6, "modifier": 0}


def test_scripted_llm_with_structured_output_returns_pydantic_object_directly(make_scripted_llm):
    llm = make_scripted_llm([_Dummy(value=42)])
    structured = llm.with_structured_output(_Dummy)
    result = structured.invoke([HumanMessage(content="give me structured data")])
    assert isinstance(result, _Dummy)
    assert result.value == 42


def test_scripted_llm_exhausted_queue_raises_index_error(make_scripted_llm):
    llm = make_scripted_llm([])
    with pytest.raises(IndexError):
        llm.invoke([HumanMessage(content="anything")])


def test_scripted_llm_bind_tools_keeps_popping_same_queue(make_scripted_llm):
    llm = make_scripted_llm([AIMessage(content="bound response")])
    bound = llm.bind_tools([])
    result = bound.invoke([HumanMessage(content="hi")])
    assert result.content == "bound response"


def test_scripted_llm_tracks_calls(make_scripted_llm):
    llm = make_scripted_llm([AIMessage(content="ok")])
    messages = [HumanMessage(content="track me")]
    llm.invoke(messages)
    assert len(llm.calls) == 1


def test_scripted_llm_two_fixtures_do_not_share_queue_state(make_scripted_llm):
    llm_a = make_scripted_llm([AIMessage(content="a")])
    llm_b = make_scripted_llm([AIMessage(content="b")])
    assert llm_a.responses is not llm_b.responses
    assert llm_a.invoke([HumanMessage(content="x")]).content == "a"
    assert llm_b.invoke([HumanMessage(content="x")]).content == "b"


# ---------------------------------------------------------------------------
# WP10 -- new CombatState scratch channels
# ---------------------------------------------------------------------------


def test_combat_state_declares_new_wp10_scratch_channels():
    """All new §16 scratch fields are present on CombatState, and the
    pre-existing ``relevant_query`` field (kept until WP15) is untouched."""
    annotations = CombatState.__annotations__
    for field in (
        "relevant_query",
        "turn_status",
        "turn_event_start",
        "pending_report",
        "damage_messages",
    ):
        assert field in annotations, f"expected {field!r} in CombatState.__annotations__"


def test_damage_messages_channel_uses_add_messages_reducer(tmp_path):
    """``damage_messages`` is an isolated ``add_messages`` channel: writes
    append, and ``RemoveMessage(id=REMOVE_ALL_MESSAGES)`` clears it -- exactly
    like ``messages`` -- without touching the main ``messages`` channel."""
    db_path = tmp_path / "ckpt.sqlite"
    thread_cfg = {"configurable": {"thread_id": "t1"}}

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)

    g = StateGraph(CombatState)

    def write_damage_messages(state: CombatState) -> dict:
        return {
            "messages": [HumanMessage(content="main channel")],
            "damage_messages": [AIMessage(content="tool loop message")],
        }

    g.add_node("write", write_damage_messages)
    g.add_edge(START, "write")
    g.add_edge("write", END)
    graph = g.compile(checkpointer=saver)

    result = graph.invoke({}, thread_cfg)
    assert len(result["damage_messages"]) == 1
    assert result["damage_messages"][0].content == "tool loop message"
    assert len(result["messages"]) == 1

    # Clear damage_messages only -- messages must survive untouched.
    g2 = StateGraph(CombatState)

    def clear_damage_messages(state: CombatState) -> dict:
        return {"damage_messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

    g2.add_node("clear", clear_damage_messages)
    g2.add_edge(START, "clear")
    g2.add_edge("clear", END)
    graph2 = g2.compile(checkpointer=saver)

    result2 = graph2.invoke({}, thread_cfg)
    assert result2["damage_messages"] == []
    assert len(result2["messages"]) == 1


def test_turn_status_and_turn_event_start_and_pending_report_are_last_write_wins():
    """These three fields have no reducer registered -- a node's returned
    value simply overwrites the channel, same as ``round_number``."""
    g = StateGraph(CombatState)

    def set_scratch(state: CombatState) -> dict:
        return {
            "turn_status": "combat",
            "turn_event_start": 3,
            "pending_report": DamageReport(action_index=0, total_damage=7),
        }

    g.add_node("set_scratch", set_scratch)
    g.add_edge(START, "set_scratch")
    g.add_edge("set_scratch", END)
    graph = g.compile()

    result = graph.invoke({})
    assert result["turn_status"] == "combat"
    assert result["turn_event_start"] == 3
    assert isinstance(result["pending_report"], DamageReport)
    assert result["pending_report"].total_damage == 7
