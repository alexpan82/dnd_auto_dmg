"""``calculate_damage`` node (Spec §6): the LLM + tool-loop step that turns
one resolved action into a ``DamageReport``.

Fixes old bug 1: the pre-refactor system prompt was a plain string with
literal, never-formatted ``{tools}`` / ``{roll_dice}`` placeholders. Here the
prompt is built by rendering the ACTUAL tool objects' ``.name`` /
``.description`` into the text (f-strings/concatenation only -- nothing here
is run through ``str.format()``, so stray braces in the JSON example below
are inert, not template placeholders).

This node does not wire the tool-calling loop itself (``tools_condition`` ->
``ToolNode`` -> back is WP5's graph-wiring job). It only builds the prompt,
invokes the already-tool-bound ``llm_with_tools`` once, and returns the
response message (which may or may not carry ``tool_calls``).

Message hygiene (Spec §5): the system prompt is ephemeral -- the update
returns ONLY ``{"messages": [response]}``, never the ``SystemMessage``.
"""

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage

from dnd_auto_dmg.schemas import Combatant, ResolvedAction
from dnd_auto_dmg.state import CombatState
from dnd_auto_dmg.tools import add, divide, multiply, roll_dice, subtract

#: Default tool set whose names/descriptions are rendered into the prompt.
#: NOTE: this is for prompt TEXT only -- the caller is responsible for
#: actually binding tools onto the ``llm_with_tools`` passed to the factory
#: (that binding happens upstream; see ``dnd_auto_dmg.llm.get_llm_with_tools``).
DEFAULT_TOOLS = [roll_dice, add, subtract, multiply, divide]


def _render_tool_list(tools: List[Any]) -> str:
    lines = []
    for t in tools:
        name = getattr(t, "name", str(t))
        description = getattr(t, "description", "")
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)


def _damage_report_shape(target_ids: List[str]) -> str:
    """A ``DamageReport``-shaped JSON example using the ACTUAL target ids."""
    example = {
        "action_index": 0,
        "total_damage": 0,
        "total_healing": 0,
        "per_target": [
            {"target_id": tid, "damage": 0, "healing": 0} for tid in target_ids
        ],
        "breakdown": {"<damage_type>": "<rolled amounts / notes>"},
        "description": "<one or two sentences of prose>",
    }
    return json.dumps(example, indent=2)


def _target_context(
    target_ids: List[str], combatants: Dict[str, Combatant]
) -> List[Dict[str, Any]]:
    info = []
    for tid in target_ids:
        target = combatants.get(tid)
        info.append(
            {
                "id": tid,
                "name": target.name if target is not None else None,
                "ac": target.ac if target is not None else None,
                "hp": target.hp if target is not None else None,
            }
        )
    return info


def _build_system_prompt(
    resolved: ResolvedAction,
    actor_sheet: Dict[str, Any],
    actor_statuses: List[Dict[str, Any]],
    target_info: List[Dict[str, Any]],
    tools: List[Any],
) -> str:
    action = resolved.action
    target_ids = [t["id"] for t in target_info]

    item_block: Dict[str, Any]
    if resolved.item is not None:
        item_block = resolved.item
    else:
        item_block = {"raw_name": resolved.item_name_raw}

    crit_note = "YES -- this is a CRITICAL HIT" if action.is_critical_hit else "no"

    tool_list = _render_tool_list(tools)
    shape = _damage_report_shape(target_ids)

    return f"""You are the damage-calculation engine for a D&D combat tracker.

You may call the following tools to roll dice and do arithmetic before
producing your final answer:
{tool_list}

Context for this action:
- Actor id: {resolved.actor_id}
- Actor sheet: {json.dumps(actor_sheet, default=str)}
- Actor's current statuses: {json.dumps(actor_statuses, default=str)}
- Feature effect texts in play: {json.dumps(resolved.feature_texts)}
- Weapon/spell used: {json.dumps(item_block, default=str)}
- Critical hit: {crit_note}
- Spell level (if a spell was upcast, else null): {json.dumps(action.spell_level)}
- Targets (id, name, ac, hp): {json.dumps(target_info, default=str)}

Roll whatever dice you need using the tools above. When you have the final
numbers, respond with a fenced ```json code block containing an object of
EXACTLY this shape (fill in real numbers, and include one per_target entry
for every target id listed above -- action_index is the index of this
action within the turn):

```json
{shape}
```

After the JSON code block, add 1-2 sentences of prose describing what
happened.
"""


def make_calculate_damage(llm_with_tools, tools: Optional[List[Any]] = None):
    """Factory for the ``calculate_damage`` node.

    ``llm_with_tools`` must already have tools bound (``.bind_tools(...)``)
    -- this factory does not bind anything itself. ``tools`` defaults to
    ``[roll_dice, add, subtract, multiply, divide]`` and is used ONLY to
    render tool names/descriptions into the prompt text.
    """
    if tools is None:
        tools = DEFAULT_TOOLS

    def calculate_damage(state: CombatState) -> dict:
        resolved_actions = state.get("resolved_actions") or []
        idx = state.get("current_action_index", 0)
        resolved = resolved_actions[idx]

        combatants: Dict[str, Combatant] = state.get("combatants") or {}
        actor = combatants.get(resolved.actor_id)
        actor_sheet = actor.model_dump() if actor is not None else {"id": resolved.actor_id}
        actor_statuses = (
            [s.model_dump() for s in actor.statuses] if actor is not None else []
        )
        target_info = _target_context(resolved.target_ids, combatants)

        prompt = _build_system_prompt(
            resolved, actor_sheet, actor_statuses, target_info, tools
        )

        messages = list(state.get("messages", []))
        response = llm_with_tools.invoke([SystemMessage(prompt)] + messages)
        return {"messages": [response]}

    return calculate_damage
