"""The loop's half of the reply room and of a refused prompt.

Each request carries a ``max_tokens`` sized to the room beside its prompt once
the model's whole ceiling no longer fits there; a refusal is learned from and
offered to the history strategy before the loop elides anything; every
answered request is reported to the engine so a strategy can replay it; and the
in-turn window passes leave what a strategy marked alone.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from raven.agent.loop import AgentLoop
from raven.agent.loop.bundles import EngineWiring, ToolWiring, TurnPolicy
from raven.agent.window import compaction, shrink
from raven.config.raven import CheckpointConfig, RuntimeConfig
from raven.context_engine import headroom
from raven.context_engine.history.markers import HISTORY_CHECKPOINT_KEY, HISTORY_PINNED_KEY
from raven.contracts.context import OverflowOutcome
from raven.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from raven.utils.tokens import estimate_prompt_tokens

_WINDOW = 16_000
_CEILING = 8_192


class _Recording(LLMProvider):
    def __init__(self, script: list[LLMResponse]):
        super().__init__(api_key="test")
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7, **kw):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools, "max_tokens": max_tokens})
        return self._script.pop(0)

    def get_default_model(self) -> str:
        return "stub"


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture(autouse=True)
def _forget_learned_headroom():
    headroom.reset()
    yield
    headroom.reset()


def _agent(workspace: Path, provider: LLMProvider, monkeypatch, iterations: int = 4) -> AgentLoop:
    agent = AgentLoop(
        provider=provider,
        workspace=workspace,
        model="stub",
        policy=TurnPolicy(max_iterations=iterations),
        tools=ToolWiring(restrict_to_workspace=True),
        engine=EngineWiring(
            context_window_tokens=_WINDOW,
            runtime_config=RuntimeConfig(checkpoint=CheckpointConfig(policy="never")),
        ),
    )
    monkeypatch.setattr(agent, "_wire_output_ceiling", lambda model=None: _CEILING)
    return agent


def _prompt(words: int) -> list[dict[str, Any]]:
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": "word " * words}]


@pytest.mark.asyncio
async def test_a_short_prompt_goes_out_with_the_ceiling_it_always_had(workspace, monkeypatch):
    from raven.providers.base import send_max_tokens

    provider = _Recording([LLMResponse(content="ok", finish_reason="stop")])
    agent = _agent(workspace, provider, monkeypatch)

    await agent._run_agent_loop(_prompt(10))

    assert provider.calls[0]["max_tokens"] == send_max_tokens(provider.generation, "stub", allow_fetch=False)


@pytest.mark.asyncio
async def test_a_prompt_near_the_window_carries_the_room_beside_it(workspace, monkeypatch):
    provider = _Recording([LLMResponse(content="ok", finish_reason="stop")])
    agent = _agent(workspace, provider, monkeypatch)

    await agent._run_agent_loop(_prompt(10_000))

    call = provider.calls[0]
    estimate = estimate_prompt_tokens(call["messages"], call["tools"])
    expected = max(4_096, _WINDOW - int(estimate * 1.05))
    assert expected < _CEILING, "the prompt has to be near enough the window for this to test anything"
    assert call["max_tokens"] == expected


@pytest.mark.asyncio
async def test_every_answered_request_is_reported_to_the_engine(workspace, monkeypatch):
    provider = _Recording(
        [
            LLMResponse(content="", tool_calls=[ToolCallRequest(id="c1", name="no_such_tool", arguments={})]),
            LLMResponse(content="done", finish_reason="stop"),
        ]
    )
    agent = _agent(workspace, provider, monkeypatch)
    heard: list[tuple[str, int, str]] = []
    monkeypatch.setattr(
        agent.context_engine,
        "note_request",
        lambda key, messages, *, model: heard.append((key, len(messages), model)),
        raising=False,
    )

    await agent._run_agent_loop(_prompt(10), session_key="s1")

    assert [(k, n) for k, n, _ in heard] == [
        ("s1", len(provider.calls[0]["messages"])),
        ("s1", len(provider.calls[1]["messages"])),
    ]
    assert all(model for _, _, model in heard)


def _overflow() -> LLMResponse:
    return LLMResponse(content="This model's maximum context length (8192 tokens) was exceeded", finish_reason="error")


@pytest.mark.asyncio
async def test_a_refusal_is_learned_and_the_strategys_reclaim_is_retried(workspace, monkeypatch):
    provider = _Recording([_overflow(), LLMResponse(content="answered", finish_reason="stop")])
    agent = _agent(workspace, provider, monkeypatch)
    asked: list[tuple[str, int]] = []

    async def reclaim(key, messages, *, attempt, window):
        asked.append((key, attempt))
        return OverflowOutcome(messages=[messages[0], messages[-1]], reclaimed_tokens=100, advanced=True)

    monkeypatch.setattr(agent.context_engine, "on_context_overflow", reclaim)
    initial = [*_prompt(10)[:1], {"role": "user", "content": "old"}, {"role": "assistant", "content": "old answer"}]
    initial.append({"role": "user", "content": "now"})

    final, _, _, _ = await agent._run_agent_loop(initial, session_key="s1")

    assert final == "answered"
    assert asked == [("s1", 1)]
    assert headroom.required_headroom("stub") > 0, "the refusal taught the reply room"
    assert [m["content"] for m in provider.calls[1]["messages"]] == ["sys", "now"]


@pytest.mark.asyncio
async def test_a_strategy_that_declines_leaves_the_refusal_to_the_loops_elision(workspace, monkeypatch):
    provider = _Recording(
        [
            *[
                LLMResponse(content="", tool_calls=[ToolCallRequest(id=f"c{i}", name="no_such_tool", arguments={})])
                for i in range(4)
            ],
            _overflow(),
            LLMResponse(content="answered", finish_reason="stop"),
        ]
    )
    agent = _agent(workspace, provider, monkeypatch, iterations=8)
    asked: list[int] = []

    async def decline(key, messages, *, attempt, window):
        asked.append(attempt)
        return None

    monkeypatch.setattr(agent.context_engine, "on_context_overflow", decline)

    final, _, _, _ = await agent._run_agent_loop(_prompt(10), session_key="s1")

    assert final == "answered"
    assert asked == [1]
    assert shrink.ELIDED_TOOL_OUTPUT in [m.get("content") for m in provider.calls[-1]["messages"]]


def test_the_head_summary_never_takes_what_a_strategy_marked():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "user", "content": "the checkpoint", HISTORY_CHECKPOINT_KEY: True},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "p"}], HISTORY_PINNED_KEY: True},
        {"role": "tool", "tool_call_id": "p", "content": "pinned body", HISTORY_PINNED_KEY: True},
        {"role": "assistant", "content": "later"},
    ]
    assert compaction.protected_prefix_end(messages) == 5
    unmarked = [{k: v for k, v in m.items() if not k.startswith("_history")} for m in messages]
    assert compaction.protected_prefix_end(unmarked) == 2


def test_the_overflow_elision_keeps_a_pinned_result():
    messages: list[dict[str, Any]] = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    messages.append({"role": "tool", "tool_call_id": "p", "content": "pinned body", HISTORY_PINNED_KEY: True})
    for i in range(4):
        messages.append({"role": "tool", "tool_call_id": f"t{i}", "content": f"body {i}"})

    shrunk, elided = shrink.emergency_shrink(messages)

    assert elided == 1
    assert shrunk[2]["content"] == "pinned body"
    assert shrunk[3]["content"] == shrink.ELIDED_TOOL_OUTPUT
