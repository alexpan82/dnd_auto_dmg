"""Shared pytest fixtures for the ``dnd_auto_dmg`` test suite.

This module MUST import cleanly with no API key present and no network
access: nothing at import time constructs a real provider-backed chat model
(no ``get_llm()`` calls, no ``init_chat_model(...)`` at module scope).
"""

import shutil
from pathlib import Path
from typing import Any, List, Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from dnd_auto_dmg.config import AppConfig
from dnd_auto_dmg.registry import DataRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_DATA_DIR = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# ScriptedChatModel (Spec §10) -- fake chat model for injecting into nodes.
# ---------------------------------------------------------------------------


class ScriptedChatModel(BaseChatModel):
    """A fake ``BaseChatModel`` that replays a queue of scripted responses.

    Each queued response is either:

    * an ``AIMessage`` (optionally carrying ``tool_calls``) -- returned as-is
      by ``.invoke()`` / ``.generate()``, OR
    * a pydantic ``BaseModel`` instance -- intended for
      ``with_structured_output(...).invoke()``, which returns it directly
      (unwrapped, not stuffed into an ``AIMessage``). If a pydantic response
      is popped via a plain (non-structured-output) ``.invoke()`` instead, it
      is wrapped in an ``AIMessage`` (JSON-dumped content) so the normal
      chat-model contract still holds.

    Responses are consumed FIFO. Popping from an empty queue raises
    ``IndexError`` with a clear message.

    ``bind_tools(...)`` returns ``self`` unchanged -- node code that does
    ``llm.bind_tools([...])`` and then calls the result keeps popping from
    the SAME queue.

    Every call to ``_generate`` (i.e. every plain ``.invoke()``/``.generate()``)
    and every call through the ``with_structured_output`` wrapper appends the
    input messages to ``self.calls``, so tests can assert on what was sent to
    the "model".

    Because ``BaseChatModel`` is itself a pydantic model, the response queue
    and call log are declared as real fields (not plain instance attributes)
    -- this also means each ``ScriptedChatModel(...)`` instance gets its own
    independent list (no shared mutable default across fixtures/tests).
    """

    responses: List[Any] = Field(default_factory=list)
    calls: List[Any] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _pop(self) -> Any:
        if not self.responses:
            raise IndexError("ScriptedChatModel: no more scripted responses")
        return self.responses.pop(0)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(messages)
        response = self._pop()
        if isinstance(response, PydanticBaseModel) and not isinstance(response, BaseMessage):
            # A structured-output-shaped object was queued but popped via a
            # plain invoke() -- wrap it so _generate's contract (a message)
            # still holds.
            message: BaseMessage = AIMessage(content=response.model_dump_json())
        else:
            message = response
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        """Return a runnable whose ``.invoke()`` pops the next queued
        response and returns it AS-IS (expected to be a pydantic instance,
        per Spec §10 -- "the fake returns the pydantic object directly")."""
        model = self

        def _invoke(model_input: Any, config: Any = None, **_kwargs: Any) -> Any:
            model.calls.append(model_input)
            return model._pop()

        return RunnableLambda(_invoke)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        """Return self so node code that binds tools keeps using the same
        scripted queue and call log."""
        return self


@pytest.fixture
def make_scripted_llm():
    """Factory fixture: ``make_scripted_llm([...responses...]) -> ScriptedChatModel``.

    Each call returns a brand-new ``ScriptedChatModel`` with its own queue.
    """

    def _make(responses: Optional[List[Any]] = None) -> ScriptedChatModel:
        return ScriptedChatModel(responses=list(responses) if responses else [])

    return _make


# ---------------------------------------------------------------------------
# Config / registry / data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_config() -> AppConfig:
    """A real ``AppConfig`` pointed at the repo's real ``data/`` dir."""
    return AppConfig()


@pytest.fixture(scope="module")
def registry() -> DataRegistry:
    """A ``DataRegistry`` built over the REAL ``data/*.json`` files.

    Deliberately does NOT depend on the (function-scoped) ``app_config``
    fixture -- a module-scoped fixture can't depend on a function-scoped one
    (pytest raises ``ScopeMismatch``). Registry is read-only, so a fresh
    ``AppConfig()`` constructed here is equivalent and safe to share across
    a whole test module (mirrors the pattern in tests/test_registry.py).
    """
    return DataRegistry(AppConfig())


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Copy the real ``data/*.json`` files into a fresh ``tmp_path`` dir.

    For tests that need to mutate data files without touching the repo's
    real ``data/`` directory. Returns the path to the copy.
    """
    dest = tmp_path / "data"
    dest.mkdir()
    for filename in (
        "characters.json",
        "weapons.json",
        "spells.json",
        "features.json",
        "monsters.json",
    ):
        shutil.copy(REPO_DATA_DIR / filename, dest / filename)
    return dest
