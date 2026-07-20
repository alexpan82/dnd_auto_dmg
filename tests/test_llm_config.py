"""WP9: Ollama is the default chat model.

``AppConfig.model`` reads ``DND_MODEL`` via a ``default_factory``, which is
evaluated at ``AppConfig()`` construction time -- so each test below must
set/unset the env var BEFORE constructing ``AppConfig()``, not after.

These tests never construct a real chat model (no ``get_llm()`` calls) and
never require a running Ollama server -- they only assert on the plain
string ``AppConfig().model`` resolves to.
"""

from dnd_auto_dmg.config import AppConfig
from dnd_auto_dmg.llm import get_llm_with_tools


def test_default_model_is_ollama(monkeypatch):
    monkeypatch.delenv("DND_MODEL", raising=False)
    config = AppConfig()
    assert config.model == "ollama:minimax-m3:cloud"


def test_default_model_provider_is_ollama(monkeypatch):
    monkeypatch.delenv("DND_MODEL", raising=False)
    config = AppConfig()
    provider, _, model_name = config.model.partition(":")
    assert provider == "ollama"
    assert model_name == "minimax-m3:cloud"
    # init_chat_model splits on the FIRST colon only, so the model name
    # itself keeps its embedded colon (the ":cloud" suffix).
    assert config.model.split(":", 1) == ["ollama", "minimax-m3:cloud"]


def test_dnd_model_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("DND_MODEL", "openai:gpt-4o")
    config = AppConfig()
    assert config.model == "openai:gpt-4o"
    assert config.model.split(":", 1)[0] == "openai"


# --- get_llm_with_tools cross-provider binding -------------------------------
# These assert on the BOUND kwargs, not just that binding succeeds. The
# original bug was that `parallel_tool_calls=False` is not validated at bind
# time -- it is stashed in RunnableBinding.kwargs and only rejected by the
# ollama client at INVOKE time. So a test that merely calls get_llm_with_tools
# (as the earlier verification did) passes while a real Ollama run crashes.
# Asserting on the bound kwargs catches the regression without a live server.


def test_ollama_binding_omits_parallel_tool_calls(monkeypatch):
    monkeypatch.delenv("DND_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    bound = get_llm_with_tools(AppConfig())
    # tools are bound, but the OpenAI-only kwarg must NOT be, or ollama's
    # Client.chat() raises TypeError at invoke time.
    assert "tools" in bound.kwargs
    assert "parallel_tool_calls" not in bound.kwargs


def test_openai_binding_includes_parallel_tool_calls(monkeypatch):
    monkeypatch.setenv("DND_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    bound = get_llm_with_tools(AppConfig())
    assert "tools" in bound.kwargs
    assert bound.kwargs.get("parallel_tool_calls") is False
