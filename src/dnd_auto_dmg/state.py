"""LangGraph state definition for the combat graph (Spec §5).

State shape
-----------
``CombatState`` is a ``TypedDict`` (not a pydantic model) whose VALUES may
still be pydantic models — the graph channels are typed dicts, but nothing
stops a channel from holding a pydantic object, and LangGraph's checkpoint
serializer (``JsonPlusSerializer``, used by ``SqliteSaver``) round-trips
pydantic v2 models transparently. This was explicitly de-risked: a
``Combatant`` (including nested ``StatusEffect`` values) written into the
``combatants`` channel, persisted via ``SqliteSaver``, and re-hydrated from a
FRESH connection/saver/graph comes back as a real ``Combatant`` instance
(``isinstance`` holds, ``.model_copy(update=...)`` works), not a plain dict.
See ``tests/test_state.py`` for the lock-in test. Conclusion: store
``Combatant`` objects directly in state — no dict fallback needed.

``combatants`` merge semantics — per-key REPLACE, not deep merge
------------------------------------------------------------------
The combat graph is linear (parse -> resolve -> calculate -> apply -> narrate
-> ...), so nodes never need to race to merge partial updates to the same
combatant. The convention is: a node reads the CURRENT ``Combatant`` for a
key out of state, builds a new one via ``combatant.model_copy(update={...})``,
and returns ``{"combatants": {cid: new_combatant}}``. ``merge_combatants``
(the reducer registered on the channel) then replaces the WHOLE entry for
that key -- it does NOT merge individual fields of the old and new
``Combatant`` together. This keeps the reducer trivial and keeps "what does
this combatant look like right now" entirely in the hands of the node that
last touched it.

Per-turn scratch fields and checkpointing
------------------------------------------
``relevant_query``, ``parsed_actions``, ``resolved_actions``,
``current_action_index``, and ``damage_reports`` are scratch space for a
single turn's processing pipeline. Because the graph runs under a
checkpointer, LangGraph persists EVERY channel's value at each step and
restores it on the next ``invoke`` against the same ``thread_id`` -- these
scratch fields are NOT automatically cleared between turns just because a
new user message comes in. A WP4a node named ``begin_turn`` is responsible
for explicitly resetting all of these fields at the START of every
invocation (see ``tests/test_state.py`` for a demonstration that, absent
such a reset, a scratch field set on turn N is still visible on turn N+1).

Frozen UI props contract (Spec §9)
-----------------------------------
This is the FROZEN contract between the graph state and the UI layer
(``public/elements/LanggraphStateDisplay.jsx`` on the frontend, surfaced by
``src/app.py`` on the backend). WP6-JSX may build against this shape without
waiting on the rest of the graph. Do not change these three top-level key
names (``combatants``, ``round``, ``log``) or their types without updating
both consumers.

.. code-block:: json

    {"langgraphState": {
        "combatants": {"<id>": {"...Combatant.model_dump()..."}},
        "round": 1,
        "log": ["last", "10", "event_log", "lines"]
    }}

Field notes:

* ``combatants`` -- a dict keyed by combatant id, each value the FULL
  ``Combatant.model_dump()`` for that id (i.e. straight from
  ``CombatState["combatants"]``, dumped).
* ``round`` -- ``CombatState["round_number"]``.
* ``log`` -- the LAST 10 entries of ``CombatState["event_log"]`` (oldest of
  the ten first, most recent last), not the full log.
"""

import operator
from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from dnd_auto_dmg.schemas import Combatant, DamageReport, ParsedAction, ResolvedAction


def merge_combatants(
    existing: Optional[Dict[str, Combatant]],
    update: Optional[Dict[str, Combatant]],
) -> Dict[str, Combatant]:
    """Reducer for the ``combatants`` channel.

    Per-key semantics ONLY -- there is no deep merge of ``Combatant`` fields:

    * ``existing=None`` is treated as ``{}`` (LangGraph may call the reducer
      with ``None`` on the very first update to a channel).
    * ``update=None`` is treated as ``{}`` -- returns a shallow copy of
      ``existing`` unchanged.
    * For each ``key: value`` pair in ``update``:
        - ``value is None`` -> delete ``key`` from the result. Deleting a
          key that isn't present is a silent no-op (no ``KeyError``).
        - ``value`` is a ``Combatant`` -> insert/replace the WHOLE entry for
          ``key`` (nodes are expected to return full, already-updated
          ``Combatant`` objects, built via ``.model_copy(update=...)``).

    Never mutates ``existing`` in place; always returns a new dict.
    """
    result: Dict[str, Combatant] = dict(existing) if existing else {}

    if not update:
        return result

    for key, value in update.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value

    return result


class CombatState(TypedDict, total=False):
    """Graph state for the combat turn-processing pipeline (Spec §5)."""

    #: Chat history, accumulated via LangGraph's standard ``add_messages``
    #: reducer (append semantics, with de-dup on message ``id``).
    messages: Annotated[list[BaseMessage], add_messages]

    #: Live combat participants keyed by combatant id. See
    #: ``merge_combatants`` above for the per-key replace semantics.
    combatants: Annotated[Dict[str, Combatant], merge_combatants]

    #: Current combat round. Last-write-wins (plain ``TypedDict`` field, no
    #: reducer registered -- a node's returned value simply overwrites it).
    round_number: int

    #: Free-text narration/debug trail. Accumulates via ``operator.add``
    #: (i.e. nodes return a list of new lines and it's concatenated onto the
    #: existing log -- never replaced or truncated in state itself; only the
    #: UI props contract truncates it, to the last 10 lines, for display).
    event_log: Annotated[list[str], operator.add]

    # -- Per-turn scratch space -------------------------------------------
    # Reset to their "empty" values by begin_turn (a WP4a node, not defined
    # in this module) at the START of every invocation. These fields have
    # NO reducer, so without an explicit reset they would persist across
    # invokes under a checkpointer (see module docstring and
    # tests/test_state.py for a demonstration of that persistence).

    #: Whether the current user message is relevant to combat processing
    #: (set by an early routing node; gates the rest of the turn pipeline).
    relevant_query: bool

    #: Actions extracted from the current turn's free-text narration.
    parsed_actions: list[ParsedAction]

    #: ``parsed_actions`` after actor/target/item resolution.
    resolved_actions: list[ResolvedAction]

    #: Index into ``resolved_actions``/``damage_reports`` of the action
    #: currently being processed by the per-action loop.
    current_action_index: int

    #: Computed damage/healing results, one per resolved action.
    damage_reports: list[DamageReport]
