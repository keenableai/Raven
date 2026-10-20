"""A turn is filed from what it said, not from the window its last call saw.

The in-turn window passes summarise the head, elide older tool bodies and
withdraw pictures so the next request fits. Filing the turn from that window
put the summary into the session as a user message, lost the stretch of the
turn the summary covered, and stored placeholders where the tool output was.
These drive the real ``_process_message`` path and read the session back.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import pytest

from raven.agent.loop import AgentLoop
from raven.agent.loop.bundles import EngineWiring, ToolWiring, TurnPolicy
from raven.agent.window import compaction, shrink
from raven.agent.window.turn_log import TURN_LOG_KEY, TurnLog
from raven.config.raven import CheckpointConfig, RuntimeConfig
from raven.config.schema import CompactionConfig
from raven.context_engine.history_trimmer import HistoryTrimmer
from raven.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from raven.spine.message import ChatType, Source
from raven.spine.turn import Origin, TurnRequest

_KEY = "test:c1"
_HUGE_USAGE = {"prompt_tokens": 4990, "completion_tokens": 10}
_BIG_ASSISTANT = "Working through the accumulated evidence. " * 200
_SUMMARY_TEXT = "Compacted handoff brief. " * 60


class _ScriptProvider(LLMProvider):
    """Scripted main-loop answers (an exception entry is raised); head-summary
    requests and calls made beside the turn are answered apart and never
    consume the script."""

    def __init__(self, script: list[Any]):
        super().__init__(api_key="test")
        self._script = list(script)
        self.summary_calls = 0
        self.main_calls: list[list[dict]] = []

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ):
        if messages and messages[0].get("content") == compaction.SUMMARY_INSTRUCTIONS:
            self.summary_calls += 1
            return LLMResponse(content=_SUMMARY_TEXT, finish_reason="stop")
        if tools is None:
            # A call from beside the turn (the session title, recall), not one of its iterations.
            return LLMResponse(content="aside", finish_reason="stop")
        self.main_calls.append([dict(m) for m in messages])
        step = self._script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    def get_default_model(self) -> str:
        return "stub"


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def _agent(workspace: Path, provider: LLMProvider, *, compact: bool = True) -> AgentLoop:
    cfg = CompactionConfig(enabled=True, reserved_tokens=199_100, preserve_recent_tokens=300) if compact else None
    return AgentLoop(
        provider=provider,
        workspace=workspace,
        model="stub",
        policy=TurnPolicy(max_iterations=12),
        tools=ToolWiring(restrict_to_workspace=True),
        engine=EngineWiring(
            # Wide enough that the history slot has room and the Curator takes
            # its fast path; the reserve puts the in-turn trigger at 900 tokens.
            context_window_tokens=200_000,
            compaction_config=cfg,
            runtime_config=RuntimeConfig(checkpoint=CheckpointConfig(policy="never")),
        ),
    )


def _tool_step(i: int, usage: dict | None = None, content: str = "") -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[ToolCallRequest(id=f"c{i}", name="no_such_tool", arguments={"n": i})],
        finish_reason="tool_calls",
        usage=dict(usage or {}),
    )


def _turn(text: str) -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel="test", chat_id="c1", sender_id="user", chat_type=ChatType.DM),
        text=text,
    )


def _reloaded(agent: AgentLoop) -> list[dict[str, Any]]:
    agent.sessions.invalidate(_KEY)
    return list(agent.sessions.get_or_create(_KEY).messages)


def _texts(messages: list[dict[str, Any]]) -> list[str]:
    return [str(m.get("content") or "") for m in messages]


def _no_window_artifacts(messages: list[dict[str, Any]]) -> None:
    assert not any(t.startswith(compaction.SUMMARY_MARKER) for t in _texts(messages))
    assert shrink.ELIDED_TOOL_OUTPUT not in _texts(messages)
    assert not any(TURN_LOG_KEY in m for m in messages)
    assert HistoryTrimmer.structural_errors(messages) == []


def _seed_history(agent: AgentLoop) -> None:
    session = agent.sessions.get_or_create(_KEY)
    session.record({"role": "user", "content": "the earlier question"})
    session.record(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "s0", "type": "function", "function": {"name": "no_such_tool", "arguments": "{}"}}],
        }
    )
    session.record({"role": "tool", "tool_call_id": "s0", "name": "no_such_tool", "content": "earlier evidence " * 60})
    session.record({"role": "assistant", "content": "the earlier answer"})
    agent.sessions.save(session)


@pytest.mark.asyncio
async def test_a_head_summary_on_the_first_turn_is_not_filed(workspace):
    provider = _ScriptProvider(
        [
            _tool_step(1, _HUGE_USAGE, _BIG_ASSISTANT),
            _tool_step(2, _HUGE_USAGE, _BIG_ASSISTANT),
            LLMResponse(content="final answer", finish_reason="stop"),
        ]
    )
    agent = _agent(workspace, provider)

    await agent._process_message(_turn("the task statement"), session_key=_KEY)

    assert provider.summary_calls >= 1, "the turn has to have been summarised for this to test anything"
    assert any(t.startswith(compaction.SUMMARY_MARKER) for t in _texts(provider.main_calls[-1]))
    filed = _reloaded(agent)
    _no_window_artifacts(filed)
    assert [m["role"] for m in filed] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert [m.get("tool_call_id") for m in filed if m["role"] == "tool"] == ["c1", "c2"]
    assert filed[0]["content"] == "the task statement"
    assert filed[-1]["content"] == "final answer"


@pytest.mark.asyncio
async def test_a_summary_over_earlier_turns_leaves_the_session_whole(workspace):
    provider = _ScriptProvider(
        [
            _tool_step(1, _HUGE_USAGE, _BIG_ASSISTANT),
            _tool_step(2, _HUGE_USAGE, _BIG_ASSISTANT),
            LLMResponse(content="final answer", finish_reason="stop"),
        ]
    )
    agent = _agent(workspace, provider)
    _seed_history(agent)

    await agent._process_message(_turn("the follow-up"), session_key=_KEY)

    assert provider.summary_calls >= 1
    filed = _reloaded(agent)
    _no_window_artifacts(filed)
    assert filed[0]["content"] == "the earlier question"
    follow_up = _texts(filed).index("the follow-up")
    assert follow_up == 4
    assert [m["role"] for m in filed[follow_up:]] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_an_elided_tool_body_is_filed_as_the_tool_said_it(workspace):
    script: list[Any] = [_tool_step(i) for i in range(5)]
    script.append(
        LLMResponse(content="This model's maximum context length (8192 tokens) was exceeded", finish_reason="error")
    )
    script.append(LLMResponse(content="answer after the elision", finish_reason="stop"))
    provider = _ScriptProvider(script)
    agent = _agent(workspace, provider, compact=False)

    await agent._process_message(_turn("go"), session_key=_KEY)

    assert shrink.ELIDED_TOOL_OUTPUT in _texts(provider.main_calls[-1]), "the window really was elided"
    filed = _reloaded(agent)
    _no_window_artifacts(filed)
    results = [m for m in filed if m["role"] == "tool"]
    assert len(results) == 5
    assert all(r["content"] for r in results)


@pytest.mark.asyncio
async def test_a_turn_cancelled_after_a_summary_keeps_what_came_after_it(workspace):
    provider = _ScriptProvider(
        [
            _tool_step(1, _HUGE_USAGE, _BIG_ASSISTANT),
            _tool_step(2, _HUGE_USAGE, _BIG_ASSISTANT),
            _tool_step(3),
            asyncio.CancelledError(),
        ]
    )
    agent = _agent(workspace, provider)

    with pytest.raises(asyncio.CancelledError):
        await agent._process_message(_turn("the task statement"), session_key=_KEY)

    assert provider.summary_calls >= 1
    filed = _reloaded(agent)
    _no_window_artifacts(filed)
    assert [m.get("tool_call_id") for m in filed if m["role"] == "tool"] == ["c1", "c2", "c3"]
    assert filed[-1].get("turn_ended", {}).get("status") == "cancelled"


def test_the_log_keeps_a_withdrawn_pictures_note_and_drops_what_the_window_made_up():
    question = {"role": "user", "content": "q"}
    base = [{"role": "system", "content": "s"}, question]
    opening = list(base)
    log = TurnLog(base, len(base))
    # The loop appends to the list it opened the turn with, as here.
    window = base
    picture = {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "shot"}]}
    elided = {"role": "tool", "tool_call_id": "b", "content": "the body"}
    window.extend([picture, elided])
    log.follow(window)

    noted = dict(picture, content=[{"type": "text", "text": "[picture withdrawn]"}])
    stand_in = dict(elided, content=shrink.ELIDED_TOOL_OUTPUT)
    made_up = {"role": "user", "content": f"{compaction.SUMMARY_MARKER}\n\nbrief"}
    window = [opening[0], made_up, noted, stand_in]
    log.follow(window)
    window.append({"role": "assistant", "content": "done"})

    turn = log.messages()
    assert turn[: len(opening)] == opening
    assert [m["content"] for m in turn[len(opening) :]] == [
        [{"type": "text", "text": "[picture withdrawn]"}],
        "the body",
        "done",
    ]


def test_a_rerun_seed_copied_from_another_attempt_is_recorded_afresh():
    base = [{"role": "user", "content": "q"}]
    first = TurnLog(base, 1)
    base.append({"role": "user", "content": "a correction", "_mid_turn_user": True})
    first.follow(base)

    seed = [dict(m) for m in first.messages()]
    second = TurnLog(seed, 1)

    assert [m["content"] for m in second.messages()] == ["q", "a correction"]
