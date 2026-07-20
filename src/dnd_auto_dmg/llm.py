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


#: Providers whose ``bind_tools`` accepts the ``parallel_tool_calls`` kwarg.
_PARALLEL_TOOL_CALLS_PROVIDERS = frozenset({"openai", "azure_openai"})


def get_llm_with_tools(config: AppConfig | None = None):
    """Return the base chat model bound to the dice/math tools.

    ``parallel_tool_calls=False`` is an OpenAI-only ``bind_tools`` kwarg. It is
    NOT validated at bind time: ``bind_tools`` simply stashes it into the
    ``RunnableBinding`` and forwards it to the provider client at INVOKE time,
    where non-OpenAI clients (e.g. ``ollama.Client.chat``) raise
    ``TypeError: ... got an unexpected keyword argument 'parallel_tool_calls'``.
    A ``try/except`` around the bind therefore never fires — the failure only
    surfaces on the first ``.invoke``. So gate the kwarg on the configured
    provider instead of relying on a bind-time exception.
    """
    if config is None:
        config = AppConfig()
    llm = get_llm(config)
    tools = [roll_dice, add, subtract, multiply, divide]
    provider = config.model.split(":", 1)[0] if ":" in config.model else ""
    if provider in _PARALLEL_TOOL_CALLS_PROVIDERS:
        return llm.bind_tools(tools, parallel_tool_calls=False)
    return llm.bind_tools(tools)
