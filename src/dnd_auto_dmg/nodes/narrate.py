"""``narrate`` node (Spec §6): the user-facing turn summary.

Fixes old bug 6 (broken narrator): asks the plain ``llm`` to summarize the
turn in 2-3 vivid sentences, grounded in the actual ``event_log`` entries
written by ``apply_damage`` (WP4a) rather than inventing details.

The ``config.enable_narration`` gate and whether this node even runs on a
given turn is graph wiring (WP5) -- this node just does the summarization
when called.

Message hygiene (Spec §5): the system prompt is ephemeral -- the update
returns ONLY ``{"messages": [response]}``, never the ``SystemMessage``.
"""

from langchain_core.messages import SystemMessage

from dnd_auto_dmg.state import CombatState

#: How many of the most recent event_log lines to include in the prompt.
RECENT_EVENT_LOG_LINES = 20

SYSTEM_PROMPT_TEMPLATE = (
    "You are the narrator for a D&D combat tracker. Summarize this combat "
    "turn in 2-3 vivid sentences, using ONLY the event log below as your "
    "source of truth -- do not invent damage, targets, or outcomes that "
    "aren't reflected in it.\n\n"
    "Event log for this turn:\n{log_block}"
)


def make_narrate(llm):
    """Factory for the ``narrate`` node."""

    def narrate(state: CombatState) -> dict:
        event_log = state.get("event_log") or []
        recent = event_log[-RECENT_EVENT_LOG_LINES:]
        log_block = "\n".join(recent) if recent else "(no events recorded this turn)"
        prompt = SYSTEM_PROMPT_TEMPLATE.format(log_block=log_block)

        messages = list(state.get("messages", []))
        response = llm.invoke([SystemMessage(prompt)] + messages)
        return {"messages": [response]}

    return narrate
