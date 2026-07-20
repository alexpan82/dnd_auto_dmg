"""Graph assembly for the combat turn-processing pipeline (Spec §6).

``build_graph`` wires the seven node factories from ``dnd_auto_dmg.nodes``
(plus one tiny inline ``no_actions`` node -- see below) into a single
``StateGraph``. It returns the graph **uncompiled**: callers compile it with
their own checkpointer, mirroring the legacy pattern::

    from dnd_auto_dmg.graph import build_graph, RECURSION_LIMIT

    app = build_graph().compile(checkpointer=memory)
    app.invoke(
        {"messages": [...]},
        config={"configurable": {"thread_id": "1"}, "recursion_limit": RECURSION_LIMIT},
    )

Wiring (Spec §6)
-----------------
::

    START -> begin_turn -> check_relevance
    check_relevance --(not relevant)--> END
    check_relevance --(relevant)-----> parse_actions -> resolve_combatants -> route_actions
    route_actions --(no valid actions)--> no_actions -> END   (event_log entry)
    route_actions --(actions)----------> calculate_damage
    calculate_damage --(tool_calls)--> roll_dice (ToolNode) -> calculate_damage
    calculate_damage --(no tool_calls)--> apply_damage
    apply_damage --(current_action_index < len(resolved_actions))--> calculate_damage
    apply_damage --(done, narration enabled)--> narrate -> END
    apply_damage --(done, narration disabled)--> END

Lazy LLM defaults
------------------
``get_llm``/``get_llm_with_tools`` (``dnd_auto_dmg.llm``) construct a
real provider-backed chat model, which raises if no API key is configured.
``build_graph`` therefore only calls them when the caller does NOT inject an
``llm`` -- ``from dnd_auto_dmg.graph import build_graph`` and
``build_graph(llm=<fake>, registry=<real>)`` must both work with no API key
present. Only a bare ``build_graph()`` (``llm=None``) touches ``get_llm``.

When an ``llm`` IS injected, this module binds the dice/math tools onto it
itself (``llm.bind_tools([...], parallel_tool_calls=False)``). Fakes such as
``tests.conftest.ScriptedChatModel`` implement ``bind_tools`` by returning
``self`` (ignoring kwargs), so they flow straight through. If an injected
model's ``bind_tools`` does not accept ``parallel_tool_calls`` at all, the
call is retried without it.

Recursion guard (Spec §12 risk 3)
-----------------------------------
``RECURSION_LIMIT`` is a module constant that callers should pass as
``config={"configurable": {...}, "recursion_limit": RECURSION_LIMIT}`` on
every ``invoke``/``stream`` call. The real protection against a stuck
per-action cursor is ``apply_damage``'s unconditional
``current_action_index`` increment (tested in ``tests/test_nodes_deterministic.py``);
this limit is a backstop, not the primary mechanism.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from dnd_auto_dmg.config import AppConfig
from dnd_auto_dmg.llm import get_llm, get_llm_with_tools
from dnd_auto_dmg.nodes import (
    make_apply_damage,
    make_begin_turn,
    make_calculate_damage,
    make_check_relevance,
    make_narrate,
    make_parse_actions,
    make_resolve_combatants,
)
from dnd_auto_dmg.registry import DataRegistry
from dnd_auto_dmg.state import CombatState
from dnd_auto_dmg.tools import add, divide, multiply, roll_dice, subtract

#: Backstop recursion limit for graph invokes (Spec §12 risk 3). Callers pass
#: this via ``config={"recursion_limit": RECURSION_LIMIT, ...}``.
RECURSION_LIMIT = 100

#: The dice/math tool set bound onto the damage-calculation LLM and wired
#: into the ``roll_dice`` ``ToolNode``.
_TOOLS = [roll_dice, add, subtract, multiply, divide]


def _bind_tools(llm):
    """Bind ``_TOOLS`` onto an injected ``llm``.

    Real ``BaseChatModel.bind_tools`` implementations accept
    ``parallel_tool_calls``; some fakes/providers may not, so retry without
    it on ``TypeError``.
    """
    try:
        return llm.bind_tools(_TOOLS, parallel_tool_calls=False)
    except TypeError:
        return llm.bind_tools(_TOOLS)


def build_graph(llm=None, registry: DataRegistry | None = None, config: AppConfig | None = None) -> StateGraph:
    """Assemble the combat pipeline graph (Spec §6). Returns it UNCOMPILED.

    Args:
        llm: Optional chat model to use for every LLM node (relevance,
            parsing, damage calculation, narration). If ``None`` (the
            default), real provider-backed models are constructed via
            ``dnd_auto_dmg.llm.get_llm``/``get_llm_with_tools`` -- this is
            the ONLY case that touches those functions, so a missing API key
            only surfaces when ``build_graph()`` is called with no ``llm``.
            When an ``llm`` IS supplied, it is used directly for the plain
            nodes and tool-bound (via this module's own ``bind_tools`` call)
            for ``calculate_damage``.
        registry: Optional ``DataRegistry``. Defaults to
            ``DataRegistry(config)``.
        config: Optional ``AppConfig``. Defaults to ``AppConfig()``.
    """
    config = config or AppConfig()
    registry = registry or DataRegistry(config)

    if llm is None:
        base_llm = get_llm(config)
        llm_with_tools = get_llm_with_tools(config)
    else:
        base_llm = llm
        llm_with_tools = _bind_tools(llm)

    graph = StateGraph(CombatState)

    def no_actions(state: CombatState) -> dict:
        """Tiny inline node: conditional edges can't write state, so
        ``route_actions``'s "no valid actions" branch lands here to record
        the event_log entry before ending the turn."""
        return {"event_log": ["No resolvable actions this turn."]}

    graph.add_node("begin_turn", make_begin_turn())
    graph.add_node("check_relevance", make_check_relevance(base_llm))
    graph.add_node("parse_actions", make_parse_actions(base_llm))
    graph.add_node("resolve_combatants", make_resolve_combatants(registry, config))
    graph.add_node("no_actions", no_actions)
    graph.add_node("calculate_damage", make_calculate_damage(llm_with_tools))
    graph.add_node("roll_dice", ToolNode(_TOOLS))
    graph.add_node("apply_damage", make_apply_damage(config))
    graph.add_node("narrate", make_narrate(base_llm))

    graph.add_edge(START, "begin_turn")
    graph.add_edge("begin_turn", "check_relevance")

    def route_relevance(state: CombatState) -> str:
        return "parse_actions" if state.get("relevant_query", False) else END

    graph.add_conditional_edges(
        "check_relevance",
        route_relevance,
        {"parse_actions": "parse_actions", END: END},
    )

    graph.add_edge("parse_actions", "resolve_combatants")

    def route_actions(state: CombatState) -> str:
        return "calculate_damage" if state.get("resolved_actions") else "no_actions"

    graph.add_conditional_edges(
        "resolve_combatants",
        route_actions,
        {"calculate_damage": "calculate_damage", "no_actions": "no_actions"},
    )
    graph.add_edge("no_actions", END)

    graph.add_conditional_edges(
        "calculate_damage",
        tools_condition,
        {"tools": "roll_dice", END: "apply_damage"},
    )
    graph.add_edge("roll_dice", "calculate_damage")

    def route_apply(state: CombatState) -> str:
        idx = state.get("current_action_index", 0)
        n = len(state.get("resolved_actions") or [])
        if idx < n:
            return "calculate_damage"
        return "narrate" if config.enable_narration else END

    graph.add_conditional_edges(
        "apply_damage",
        route_apply,
        {"calculate_damage": "calculate_damage", "narrate": "narrate", END: END},
    )
    graph.add_edge("narrate", END)

    return graph
