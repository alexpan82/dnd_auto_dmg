"""``apply_damage`` -- deterministic graph node (Spec §6).

Fixes old bug 9 ("no HP tracking"): extracts the final ``DamageReport`` JSON
produced by the damage-calculation LLM loop, applies it to the live
combatants (HP loss/gain, unconscious/dead statuses), appends the resulting
``DamageReport`` to ``state["damage_reports"]``, advances
``current_action_index``, and -- per the Spec §5 message-hygiene rule --
deletes the current action's tool-loop scaffolding (the intermediate
tool-call AIMessages and ToolMessages) via ``RemoveMessage``, keeping only
the final report AIMessage.

Tolerant JSON handling: the damage LLM is expected to emit a ``DamageReport``
as JSON (possibly fenced in a code block), but this node degrades gracefully
if that JSON is malformed or missing ``per_target`` entries (splitting
``total_damage`` evenly across ``resolved.target_ids``), or missing
altogether (a zero-damage report whose description is the raw AIMessage
text).
"""

from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, ToolMessage
from pydantic import ValidationError

from dnd_auto_dmg.config import AppConfig
from dnd_auto_dmg.schemas import Combatant, DamageReport, PerTargetDamage, StatusEffect
from dnd_auto_dmg.state import CombatState
from dnd_auto_dmg.tools import extract_json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_final_ai_message(messages: List[BaseMessage]) -> Optional[AIMessage]:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _split_damage_evenly(total_damage: int, target_ids: List[str]) -> List[PerTargetDamage]:
    """Integer-divide ``total_damage`` across ``target_ids``, remainder to
    the first target."""
    if not target_ids:
        return []
    count = len(target_ids)
    base = total_damage // count
    remainder = total_damage - base * count
    per_target: List[PerTargetDamage] = []
    for i, target_id in enumerate(target_ids):
        damage = base + (remainder if i == 0 else 0)
        per_target.append(PerTargetDamage(target_id=target_id, damage=damage, healing=0))
    return per_target


def _extract_damage_report(messages: List[BaseMessage], idx: int, target_ids: List[str]) -> DamageReport:
    final_ai = _find_final_ai_message(messages)
    if final_ai is None:
        return DamageReport(action_index=idx)

    content = final_ai.content if isinstance(final_ai.content, str) else str(final_ai.content)
    data = extract_json(content)

    if isinstance(data, dict):
        try:
            report = DamageReport.model_validate({**data, "action_index": idx})
        except ValidationError:
            report = None

        if report is not None and report.per_target:
            return report

        # Tolerant fallback: salvage total_damage and split it evenly.
        total_damage = int(data.get("total_damage", 0) or 0)
        total_healing = int(data.get("total_healing", 0) or 0)
        breakdown = data.get("breakdown") if isinstance(data.get("breakdown"), dict) else {}
        description = data.get("description") if isinstance(data.get("description"), str) else ""
        return DamageReport(
            action_index=idx,
            total_damage=total_damage,
            total_healing=total_healing,
            per_target=_split_damage_evenly(total_damage, target_ids),
            breakdown=breakdown,
            description=description,
        )

    # No JSON at all -- zero-damage report, raw text preserved for narration.
    return DamageReport(action_index=idx, description=content)


def _scaffolding_ids_to_remove(messages: List[BaseMessage]) -> List[str]:
    """Ids of the current action's tool-loop scaffolding: walking backwards
    from the end of ``messages`` over the contiguous run of
    (AIMessage | ToolMessage) instances, collect every ToolMessage id and
    every tool-call-bearing AIMessage id. Stops at the first message that is
    neither (e.g. a HumanMessage). The final report AIMessage (no
    tool_calls) is walked over but NOT collected, so it is kept."""
    ids: List[str] = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            if message.id is not None:
                ids.append(message.id)
        elif isinstance(message, AIMessage):
            if getattr(message, "tool_calls", None) and message.id is not None:
                ids.append(message.id)
        else:
            break
    return ids


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def make_apply_damage(config: AppConfig):
    """Factory for the ``apply_damage`` node."""

    def apply_damage(state: CombatState) -> dict:
        idx = state.get("current_action_index", 0)
        resolved_actions = state.get("resolved_actions") or []

        if idx < 0 or idx >= len(resolved_actions):
            # Defensive no-op: nothing to apply against.
            return {"current_action_index": idx}

        resolved = resolved_actions[idx]
        messages: List[BaseMessage] = state.get("messages") or []

        report = _extract_damage_report(messages, idx, resolved.target_ids)

        original_combatants: Dict[str, Combatant] = state.get("combatants") or {}
        changed: Dict[str, Combatant] = {}
        event_log: List[str] = []

        actor = original_combatants.get(resolved.actor_id)
        actor_name = actor.name if actor is not None else resolved.actor_id

        for per_target in report.per_target:
            target = changed.get(per_target.target_id) or original_combatants.get(
                per_target.target_id
            )
            if target is None:
                continue

            was_alive = target.is_alive
            new_hp = max(0, target.hp - per_target.damage)
            if per_target.healing:
                new_hp = min(target.max_hp, new_hp + per_target.healing)

            statuses: List[StatusEffect] = list(target.statuses)
            is_alive = target.is_alive
            newly_downed_status: Optional[str] = None
            if new_hp == 0 and was_alive:
                newly_downed_status = "unconscious" if target.kind == "pc" else "dead"
                statuses = statuses + [
                    StatusEffect(name=newly_downed_status, source=resolved.actor_id)
                ]
                is_alive = False

            updated = target.model_copy(
                update={"hp": new_hp, "statuses": statuses, "is_alive": is_alive}
            )
            changed[per_target.target_id] = updated

            suffix = f", {newly_downed_status}" if newly_downed_status else ""
            if per_target.damage:
                event_log.append(
                    f"{actor_name} deals {per_target.damage} damage to "
                    f"{per_target.target_id} ({new_hp}/{target.max_hp} HP{suffix})"
                )
            if per_target.healing:
                event_log.append(
                    f"{actor_name} heals {per_target.target_id} for "
                    f"{per_target.healing} ({new_hp}/{target.max_hp} HP{suffix})"
                )

        removal_ids = _scaffolding_ids_to_remove(messages)

        update: dict = {
            "combatants": changed,
            "damage_reports": (state.get("damage_reports") or [])[:] + [report],
            "current_action_index": idx + 1,
        }
        if event_log:
            update["event_log"] = event_log
        if removal_ids:
            update["messages"] = [RemoveMessage(id=mid) for mid in removal_ids]
        return update

    return apply_damage
