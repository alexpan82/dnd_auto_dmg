"""``begin_turn`` -- deterministic graph node (Spec §6).

Resets the per-turn scratch fields (``relevant_query``, ``parsed_actions``,
``resolved_actions``, ``current_action_index``, ``damage_reports``) at the
START of every graph invocation.

This reset is necessary because those fields have no reducer registered on
``CombatState`` (see ``dnd_auto_dmg.state`` module docstring): under a
checkpointer, LangGraph restores whatever was last written to a channel on
the previous invoke of the same thread, so without an explicit reset here a
scratch value set on turn N would still be visible -- and wrongly acted on --
on turn N+1.

``combatants``/``round_number`` are handled differently: they are NOT
per-turn scratch, they are the persistent combat state that must survive
across turns within the same encounter. This node only initializes them
(``combatants={}``, ``round_number=1``) the very first time it runs on a
thread (detected via ``"combatants" not in state``) and otherwise leaves
them untouched.
"""

from dnd_auto_dmg.state import CombatState


def make_begin_turn():
    """Factory for the ``begin_turn`` node. Takes no dependencies."""

    def begin_turn(state: CombatState) -> dict:
        update: dict = {
            "relevant_query": False,
            "parsed_actions": [],
            "resolved_actions": [],
            "current_action_index": 0,
            "damage_reports": [],
        }

        if "combatants" not in state:
            # First turn ever on this thread -- initialize the persistent
            # combat state. round_number is last-write-wins (no reducer), so
            # it must only be set here, never on subsequent turns, or it
            # would clobber whatever round the encounter is actually on.
            update["combatants"] = {}
            update["round_number"] = 1

        return update

    return begin_turn
