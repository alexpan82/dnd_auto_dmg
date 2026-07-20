"""Thin manual-smoke CLI entry point for the combat graph (Spec §6/§7).

Runs the SAME five-turn scripted conversation as the legacy ``src/agent.py``
demo, but against the rebuilt graph (``dnd_auto_dmg.graph.build_graph``) and
the REAL LLM configured by ``dnd_auto_dmg.config.AppConfig``. This is a
manual smoke test, not part of CI -- it needs a real ``OPENAI_API_KEY`` and
makes live network calls, so it is never imported by the test suite.

Runs from any working directory: ``dnd_auto_dmg`` is an editable install, so
``import dnd_auto_dmg`` works without ``sys.path`` hacks.

Interactive API-key prompting is the ONE place this is allowed (Spec §7),
and it only happens under ``if __name__ == "__main__":`` -- importing this
module (e.g. ``python -c "import demo"``) never prompts and never requires
``OPENAI_API_KEY`` to already be set.
"""

import getpass
import os
import sqlite3

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from dnd_auto_dmg.graph import RECURSION_LIMIT, build_graph

#: The scripted conversation (same five messages, same order, as the legacy
#: ``src/agent.py`` demo). The dead ``"log": []`` key from the first invoke
#: is dropped -- it was a leftover from the pre-refactor state shape.
TURNS = [
    "Avantor casts Tenser's Transformation on themself",
    "Avantor attacks the goblin with his greatsword",
    "They do it again",
    "He then casts 5th level fireball at a group of 3 kobolds",
    "Literal nonsense",
]

#: How many trailing event_log lines to print at the very end.
TAIL_EVENT_LOG_LINES = 10


def _ensure_api_key() -> None:
    """Prompt for ``OPENAI_API_KEY`` if it isn't already set in the
    environment. The only place in this codebase interactive key prompting
    is allowed (Spec §7)."""
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OPENAI_API_KEY: ")


def _format_combatant(combatant) -> str:
    statuses = ", ".join(
        f"{s.name}({s.duration_rounds}r)" if s.duration_rounds is not None else s.name
        for s in combatant.statuses
    )
    status_block = f" [{statuses}]" if statuses else ""

    marker = ""
    if not combatant.is_alive:
        marker = " (unconscious)" if combatant.kind == "pc" else " (dead)"

    return f"  {combatant.id}: {combatant.hp}/{combatant.max_hp} HP{status_block}{marker}"


def _print_roster(state: dict) -> None:
    combatants = state.get("combatants") or {}
    round_number = state.get("round_number", 1)
    print(f"-- Round {round_number} --")
    for combatant_id in sorted(combatants):
        print(_format_combatant(combatants[combatant_id]))


def main() -> None:
    _ensure_api_key()

    graph = build_graph()
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    memory = SqliteSaver(conn)
    app = graph.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "1"}, "recursion_limit": RECURSION_LIMIT}

    result: dict = {}
    last_printed_message_id = None

    for turn_text in TURNS:
        print(f"\n> {turn_text}")
        result = app.invoke({"messages": [HumanMessage(turn_text)]}, config)

        messages = result.get("messages") or []
        last_message = messages[-1] if messages else None
        if (
            isinstance(last_message, AIMessage)
            and last_message.content
            and last_message.id != last_printed_message_id
        ):
            print(last_message.content)
            last_printed_message_id = last_message.id

        _print_roster(result)

    print("\n-- Recent event log --")
    for line in (result.get("event_log") or [])[-TAIL_EVENT_LOG_LINES:]:
        print(line)


if __name__ == "__main__":
    main()
