"""``check_relevance`` node (Spec §6): the yes/no gate at the top of the
combat pipeline.

Decides whether the latest user message is worth running the rest of the
turn-processing pipeline on. Uses the **plain** ``llm`` -- no tools bound,
unlike the pre-refactor code, which bound tools here for no reason (nothing
in this node ever needs to call a tool).

Message hygiene (Spec §5): the system prompt built here is ephemeral. It is
sent as the first message of the invocation but is never added to
``state["messages"]`` -- this node returns no ``messages`` key at all.
"""

from langchain_core.messages import SystemMessage

from dnd_auto_dmg.state import CombatState

SYSTEM_PROMPT = (
    "You are the relevance gate for a D&D combat-tracking assistant. Decide "
    "whether the latest user message describes D&D combat actions -- "
    "attacks, spells, damage, healing, statuses being applied or removed, "
    "or advancing to the next round.\n\n"
    "Respond with exactly one word: 'yes' if the message is about D&D "
    "combat actions, or 'no' if it is not. Do not add any other text."
)


def make_check_relevance(llm):
    """Factory for the ``check_relevance`` node.

    ``llm`` is the plain chat model (no ``bind_tools``/``with_structured_output``
    wrapping -- this node just needs a yes/no answer).
    """

    def check_relevance(state: CombatState) -> dict:
        messages = list(state.get("messages", []))
        response = llm.invoke([SystemMessage(SYSTEM_PROMPT)] + messages)
        content = str(response.content or "")
        relevant = content.strip().lower().startswith("yes")
        return {"relevant_query": relevant}

    return check_relevance
