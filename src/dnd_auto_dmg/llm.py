"""Provider-agnostic LLM construction.

Nothing in this module runs at import time: no ``getpass`` prompts, no
network calls, no model instantiation. A missing API key only surfaces as a
runtime error when the returned model is actually invoked. Interactive key
prompting belongs in entry points (``app.py`` / ``demo.py``), not here.
"""

from langchain.chat_models import init_chat_model

from dnd_auto_dmg.config import AppConfig
from dnd_auto_dmg.tools import add, divide, multiply, roll_dice, subtract


def get_llm(config: AppConfig | None = None):
    """Construct a chat model for ``config.model`` (provider-agnostic).

    Does not prompt for credentials and does not make a network call at
    construction time.
    """
    if config is None:
        config = AppConfig()
    return init_chat_model(config.model)


def get_llm_with_tools(config: AppConfig | None = None):
    """Return the base chat model bound to the dice/math tools."""
    if config is None:
        config = AppConfig()
    llm = get_llm(config)
    tools = [roll_dice, add, subtract, multiply, divide]
    return llm.bind_tools(tools, parallel_tool_calls=False)
