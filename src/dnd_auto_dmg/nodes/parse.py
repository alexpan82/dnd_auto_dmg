"""``parse_actions`` node (Spec §6): turns the recent chat history into a
``ParsedTurn`` of structured ``ParsedAction`` objects.

Primary path is ``llm.with_structured_output(ParsedTurn)``. Because
structured-output support varies across providers (Spec §12 risk 7), a
fallback path asks the plain ``llm`` for JSON in prose and recovers it with
``dnd_auto_dmg.tools.extract_json``. Any failure anywhere in the fallback
(bad JSON, JSON that doesn't validate as ``ParsedTurn``) degrades to
``ParsedTurn(actions=[])`` rather than raising -- an empty ``actions`` list
*is* the parse-failure signal that downstream routing (WP5) checks for.

Message hygiene (Spec §5): both prompts built here are ephemeral -- this
node returns no ``messages`` key at all.
"""

from langchain_core.messages import SystemMessage

from dnd_auto_dmg.schemas import ParsedTurn
from dnd_auto_dmg.state import CombatState
from dnd_auto_dmg.tools import extract_json

STRUCTURED_SYSTEM_PROMPT = (
    "You are the action parser for a D&D combat tracker. Read the "
    "conversation so far and extract every combat action described in the "
    "MOST RECENT user message as a list of structured actions.\n\n"
    "Rules:\n"
    "- A single message may describe multiple actions (e.g. two attacks, "
    "or an attack followed by a spell) -- emit one entry per action, in "
    "the order they happened.\n"
    "- Target groups carry a count: 'a group of 3 kobolds' becomes a "
    "target with name='kobold' and count=3. A single named target has "
    "count=1.\n"
    "- Resolve pronouns and elided actors/weapons from earlier messages in "
    "the conversation: if the latest message says 'They do it again' (or "
    "similar), reuse the actor and weapon/spell from the most recent "
    "matching prior action.\n"
    "- Spells cast at a higher level than their base level ('5th level "
    "fireball') set spell_level to that level (5).\n"
    "- Record any statuses/conditions applied to or removed from a "
    "combatant in statuses_applied / statuses_removed.\n"
    "- If the message expresses intent to move on ('next round', 'end "
    "the round'), emit an action with action_type='advance_round'.\n"
    "- If nothing in the latest message describes a combat action, return "
    "an empty actions list."
)

FALLBACK_SYSTEM_PROMPT = (
    STRUCTURED_SYSTEM_PROMPT
    + "\n\nRespond with ONLY a JSON object of the shape "
    '{"actions": [{"actor": str, "action_type": '
    '"attack"|"heal"|"cast"|"apply_status"|"remove_status"|"advance_round"|"other", '
    '"weapon_or_spell": str|null, "targets": [{"name": str, "count": int}], '
    '"spell_level": int|null, "is_critical_hit": bool, '
    '"statuses_applied": [str], "statuses_removed": [str], '
    '"features_used": [str]}]}. No prose, no markdown fences.'
)


def make_parse_actions(llm):
    """Factory for the ``parse_actions`` node."""

    def parse_actions(state: CombatState) -> dict:
        messages = list(state.get("messages", []))

        parsed_turn = None
        try:
            structured_llm = llm.with_structured_output(ParsedTurn)
            result = structured_llm.invoke(
                [SystemMessage(STRUCTURED_SYSTEM_PROMPT)] + messages
            )
            if isinstance(result, ParsedTurn):
                parsed_turn = result
        except Exception:
            parsed_turn = None

        if parsed_turn is None:
            parsed_turn = ParsedTurn(actions=[])
            try:
                response = llm.invoke(
                    [SystemMessage(FALLBACK_SYSTEM_PROMPT)] + messages
                )
                data = extract_json(response.content)
                if data is not None:
                    parsed_turn = ParsedTurn.model_validate(data)
            except Exception:
                parsed_turn = ParsedTurn(actions=[])

        return {"parsed_actions": parsed_turn.actions}

    return parse_actions
