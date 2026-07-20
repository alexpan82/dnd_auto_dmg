"""``resolve_combatants`` -- deterministic graph node (Spec §6).

Resolves each ``ParsedAction`` in ``state["parsed_actions"]`` against the
live combatant roster and the static data registry:

1. **Actor** -- fuzzy-matched against live combatants first, then the
   registry's character roster (instantiating a fresh ``Combatant`` on a
   roster hit that isn't in combat yet). No match -> the action is dropped
   with an ``event_log`` entry.
2. **Targets** -- ``"self"`` resolves to the actor. Otherwise fuzzy-matched
   against live combatants (at the stricter ``fuzzy_threshold_combatant``),
   then ``monsters.json`` (instantiating ``count`` instances, reusing living
   instances first), then finally an ad-hoc combatant as a last resort.
3. **Item** -- fuzzy-matched against the merged weapons+spells namespace; a
   miss preserves the raw name in ``item_name_raw`` so a downstream LLM node
   can fall back on its own 5e knowledge.
4. **Statuses** -- ``statuses_applied``/``statuses_removed`` are applied to
   the actor HERE, before any damage is calculated, so a buff cast this turn
   affects this turn's rolls.

``advance_round`` actions are handled entirely inside this node (round
counter increment, per-combatant status duration decrement/expiry) and
produce NO ``ResolvedAction`` -- downstream damage-calculation nodes never
see them, which keeps that per-action loop simple and avoids emitting
zero-damage reports for a "damage-free" action type.

The node works on a LOCAL copy of the combatants dict so that resolving
multiple actions in one turn sees earlier instantiations (e.g. two actions
in the same turn both targeting "the goblin" reuse the same instance), and
returns only the entries it added/modified (the ``combatants`` channel does
a per-key replace merge, not a deep merge).
"""

from typing import Dict, List, Optional, Tuple

from dnd_auto_dmg.config import AppConfig
from dnd_auto_dmg.registry import DataRegistry
from dnd_auto_dmg.schemas import Combatant, ParsedAction, ResolvedAction, StatusEffect, TargetRef
from dnd_auto_dmg.state import CombatState
from dnd_auto_dmg.tools import fuzzy_match

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _combatant_candidates(combatants: Dict[str, Combatant]) -> Dict[str, str]:
    """Map every fuzzy-matchable candidate string (name or id) -> combatant id."""
    candidate_to_id: Dict[str, str] = {}
    for cid, combatant in combatants.items():
        candidate_to_id[combatant.name] = cid
        candidate_to_id[cid] = cid
    return candidate_to_id


def _fuzzy_match_working(
    query: str, working: Dict[str, Combatant], threshold: int
) -> Optional[str]:
    """Fuzzy-match ``query`` against live combatant names+ids; None on miss."""
    candidates = _combatant_candidates(working)
    if not candidates:
        return None
    match, _score = fuzzy_match(query, list(candidates.keys()), threshold=threshold)
    if match is None:
        return None
    return candidates[match]


def _resolve_actor(
    actor_name: str,
    working: Dict[str, Combatant],
    registry: DataRegistry,
    config: AppConfig,
    changed: Dict[str, Combatant],
) -> Optional[str]:
    """Resolve an action's actor name to a combatant id, instantiating a
    fresh ``Combatant`` from the roster on a not-yet-in-combat hit."""
    matched = _fuzzy_match_working(actor_name, working, config.fuzzy_threshold_combatant)
    if matched is not None:
        return matched

    found = registry.find_character(actor_name)
    if found is not None:
        char_id, _sheet = found
        if char_id not in working:
            new_combatant = registry.combatant_from_character(char_id)
            working[char_id] = new_combatant
            changed[char_id] = new_combatant
        return char_id

    return None


def _resolve_monster_targets(
    monster_id: str,
    count: int,
    working: Dict[str, Combatant],
    changed: Dict[str, Combatant],
    registry: DataRegistry,
) -> List[str]:
    """Resolve ``count`` instances of ``monster_id``, reusing existing LIVING
    instances first (in id/suffix order) and creating the remainder with the
    next unused numeric suffix."""
    prefix = f"{monster_id}_"
    existing_suffixes: set = set()
    living: List[Tuple[str, str]] = []
    for cid, combatant in working.items():
        if cid.startswith(prefix):
            suffix = cid[len(prefix):]
            existing_suffixes.add(suffix)
            if combatant.is_alive:
                living.append((cid, suffix))

    def _suffix_sort_key(item: Tuple[str, str]):
        _cid, suffix = item
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, suffix)

    living.sort(key=_suffix_sort_key)

    target_ids: List[str] = [cid for cid, _suffix in living[:count]]

    next_suffix = 1
    while len(target_ids) < count:
        while str(next_suffix) in existing_suffixes:
            next_suffix += 1
        new_id = f"{prefix}{next_suffix}"
        new_combatant = registry.combatant_from_monster(monster_id, next_suffix)
        working[new_id] = new_combatant
        changed[new_id] = new_combatant
        existing_suffixes.add(str(next_suffix))
        target_ids.append(new_id)
        next_suffix += 1

    return target_ids


def _resolve_targets(
    targets: List[TargetRef],
    actor_id: str,
    working: Dict[str, Combatant],
    registry: DataRegistry,
    config: AppConfig,
    changed: Dict[str, Combatant],
) -> List[str]:
    target_ids: List[str] = []

    for target in targets:
        if target.name.strip().lower() == "self":
            target_ids.append(actor_id)
            continue

        matched_id = _fuzzy_match_working(target.name, working, config.fuzzy_threshold_combatant)
        if matched_id is not None and target.count == 1:
            target_ids.append(matched_id)
            continue

        found_monster = registry.find_monster(target.name)
        if found_monster is not None:
            monster_id, _monster_def = found_monster
            target_ids.extend(
                _resolve_monster_targets(monster_id, target.count, working, changed, registry)
            )
            continue

        # Last resort: no roster/monster match at all -> ad-hoc combatant(s).
        if target.count == 1:
            adhoc = registry.adhoc_combatant(target.name, config.default_adhoc_hp)
            working[adhoc.id] = adhoc
            changed[adhoc.id] = adhoc
            target_ids.append(adhoc.id)
        else:
            for k in range(1, target.count + 1):
                adhoc = registry.adhoc_combatant(f"{target.name} {k}", config.default_adhoc_hp)
                working[adhoc.id] = adhoc
                changed[adhoc.id] = adhoc
                target_ids.append(adhoc.id)

    return target_ids


def _status_matches_spell(raw_name: str, spell_status_name: str, config: AppConfig) -> bool:
    if raw_name.strip().lower() == spell_status_name.strip().lower():
        return True
    match, _score = fuzzy_match(
        raw_name, [spell_status_name], threshold=config.fuzzy_threshold_lookup
    )
    return match is not None


def _apply_statuses(
    actor_id: str,
    action: ParsedAction,
    item_kind: Optional[str],
    item_def,
    working: Dict[str, Combatant],
    changed: Dict[str, Combatant],
    round_number: int,
    config: AppConfig,
) -> None:
    """Apply/remove statuses on the ACTOR, before any damage calculation."""
    if not action.statuses_applied and not action.statuses_removed:
        return

    actor = working[actor_id]
    statuses: List[StatusEffect] = list(actor.statuses)

    for raw_name in action.statuses_applied:
        status_name = raw_name
        duration_rounds = None
        if item_kind == "spell" and item_def is not None and item_def.applies_status is not None:
            applied = item_def.applies_status
            if _status_matches_spell(raw_name, applied.name, config):
                status_name = applied.name
                duration_rounds = applied.duration_rounds
        statuses.append(
            StatusEffect(
                name=status_name,
                source=actor_id,
                duration_rounds=duration_rounds,
                applied_round=round_number,
            )
        )

    for raw_name in action.statuses_removed:
        statuses = [
            s for s in statuses if s.name.strip().lower() != raw_name.strip().lower()
        ]

    updated = actor.model_copy(update={"statuses": statuses})
    working[actor_id] = updated
    changed[actor_id] = updated


def _collect_feature_texts(
    action: ParsedAction, actor: Combatant, registry: DataRegistry
) -> Dict[str, str]:
    feature_texts: Dict[str, str] = {}
    for name in list(action.features_used) + list(actor.features):
        found = registry.find_feature(name)
        if found is not None:
            feat_id, feat_def = found
            feature_texts[feat_id] = feat_def.effect
    return feature_texts


def _advance_round(
    round_number: int,
    working: Dict[str, Combatant],
    changed: Dict[str, Combatant],
    event_log: List[str],
) -> int:
    """Increment the round counter and decrement/expire every working
    combatant's timed statuses. Returns the new round number."""
    new_round = round_number + 1
    event_log.append(f"Round advances to {new_round}.")

    for cid, combatant in list(working.items()):
        new_statuses: List[StatusEffect] = []
        expired_names: List[str] = []
        anything_changed = False

        for status in combatant.statuses:
            if status.duration_rounds is None:
                new_statuses.append(status)
                continue
            remaining = status.duration_rounds - 1
            anything_changed = True
            if remaining <= 0:
                expired_names.append(status.name)
            else:
                new_statuses.append(status.model_copy(update={"duration_rounds": remaining}))

        if anything_changed:
            updated = combatant.model_copy(update={"statuses": new_statuses})
            working[cid] = updated
            changed[cid] = updated
            for name in expired_names:
                event_log.append(f"{name} expires on {combatant.name}.")

    return new_round


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def make_resolve_combatants(registry: DataRegistry, config: AppConfig):
    """Factory for the ``resolve_combatants`` node."""

    def resolve_combatants(state: CombatState) -> dict:
        parsed_actions: List[ParsedAction] = state.get("parsed_actions") or []
        working: Dict[str, Combatant] = dict(state.get("combatants") or {})
        changed: Dict[str, Combatant] = {}
        round_number: int = state.get("round_number", 1)
        event_log: List[str] = []
        resolved_actions: List[ResolvedAction] = []

        for action in parsed_actions:
            actor_id = _resolve_actor(action.actor, working, registry, config, changed)
            if actor_id is None:
                event_log.append(
                    f"Could not resolve actor '{action.actor}'; action dropped."
                )
                continue

            if action.action_type == "advance_round":
                round_number = _advance_round(round_number, working, changed, event_log)
                continue

            target_ids = _resolve_targets(
                action.targets, actor_id, working, registry, config, changed
            )

            item: Optional[dict] = None
            item_name_raw: Optional[str] = None
            item_kind: Optional[str] = None
            item_def = None
            if action.weapon_or_spell:
                found_item = registry.find_item(action.weapon_or_spell)
                if found_item is not None:
                    _item_id, item_kind, item_def = found_item
                    item = item_def.model_dump()
                else:
                    item_name_raw = action.weapon_or_spell

            _apply_statuses(
                actor_id, action, item_kind, item_def, working, changed, round_number, config
            )

            feature_texts = _collect_feature_texts(action, working[actor_id], registry)

            resolved_actions.append(
                ResolvedAction(
                    action=action,
                    actor_id=actor_id,
                    target_ids=target_ids,
                    item=item,
                    item_name_raw=item_name_raw,
                    feature_texts=feature_texts,
                )
            )

        update: dict = {
            "resolved_actions": resolved_actions,
            "combatants": changed,
            "round_number": round_number,
        }
        if event_log:
            update["event_log"] = event_log
        return update

    return resolve_combatants
