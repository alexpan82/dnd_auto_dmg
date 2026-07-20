"""Graph nodes for the combat pipeline (Spec §6).

Deterministic nodes: begin_turn, resolve_combatants, apply_damage.
LLM nodes: check_relevance, parse_actions, calculate_damage, narrate.
All are factories taking their dependencies explicitly; ``build_graph``
(WP5, ``dnd_auto_dmg.graph``) wires real defaults and tests inject fakes.
"""

from dnd_auto_dmg.nodes.apply import make_apply_damage
from dnd_auto_dmg.nodes.begin_turn import make_begin_turn
from dnd_auto_dmg.nodes.damage import make_calculate_damage
from dnd_auto_dmg.nodes.narrate import make_narrate
from dnd_auto_dmg.nodes.parse import make_parse_actions
from dnd_auto_dmg.nodes.relevance import make_check_relevance
from dnd_auto_dmg.nodes.resolve import make_resolve_combatants

__all__ = [
    "make_begin_turn",
    "make_check_relevance",
    "make_parse_actions",
    "make_resolve_combatants",
    "make_calculate_damage",
    "make_apply_damage",
    "make_narrate",
]
