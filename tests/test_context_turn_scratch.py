"""Turn scratch: the query-conditioned segments ride in the user tail, never the log.

Covers the block helpers, where the assembler puts ``stable = False`` segments
with the switch on and off, the normalised segment order, the session write
that must never record the block, the replay comparison that must not read it
as a divergence, the hook-facing question that must not include it, and the
cache-breakpoint budget with the block in place.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from raven.agent.context import ContextBuilder
from raven.context_engine import ContextAssembler, TurnContext
from raven.context_engine.scratch import (
    TURN_SCRATCH_TAG,
    append_turn_scratch,
    render_turn_scratch,
    strip_turn_scratch,
)
from raven.context_engine.segments import (
    ActiveSkillsSegmentBuilder,
    BootstrapSegmentBuilder,
    IdentitySegmentBuilder,
    MemorySegmentBuilder,
    SkillsSegmentBuilder,
)
from raven.contracts.assembled import TokenBudget
from raven.contracts.context import AssemblyContext, Segment
from raven.contracts.memory import Memory
from raven.memory_engine.skill_forge import RouterHit, SkillForgeRouter
from raven.providers.base import LLMProvider, LLMResponse

_SEP = "\n\n---\n\n"
_MEMORY_HEADING = re.compile(r"(?m)^# Memory$")
_SKILLS_HEADING = re.compile(r"(?m)^# Skills$")


def _budget() -> TokenBudget:
    return TokenBudget(100_000, 4_000, 2_000, 1_000, 93_000)


def _texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    return [str(b.get("text", "")) for b in content or [] if isinstance(b, dict)]


def _whole_prompt(messages: list[dict[str, Any]]) -> str:
    return "\n".join(t for m in messages for t in _texts(m.get("content")))


class _EchoBackend:
    async def start(self):
        pass

    async def stop(self):
        pass

    async def store(self, session_id, messages, *, metadata=None):
        pass

    async def feedback(self, signals):
        pass

    async def recall(self, query, *, user_id=None, agent_id=None, top_k):
        if user_id is None:
            return []
        return [Memory(text=f"recalled for {query}", score=1.0)]


class _EchoSource:
    name = "local"
    weight = 1.0

    async def search(self, query, history, k):
        return [RouterHit(qualified_id="local/e", name="e", content=f"skill for {query}", score=0.9)]


class _HistoryCurator:
    name = "curator"
    order = 6
    needs_prefix = True
    stable = False

    async def build(self, ctx: AssemblyContext) -> Segment:
        assert ctx.prefix is not None
        return Segment(text="", history=list(ctx.session_messages))


@pytest.fixture
def builder(tmp_path: Path) -> ContextBuilder:
    (tmp_path / "TOOLS.md").write_text("bootstrap rules", encoding="utf-8")
    always = tmp_path / "skills" / "always-on"
    always.mkdir(parents=True)
    (always / "SKILL.md").write_text(
        '---\nname: always-on\ndescription: d\nmetadata: {"raven": {"always": true}}\n---\nalways body\n',
        encoding="utf-8",
    )
    return ContextBuilder(workspace=tmp_path)


def _builders(builder: ContextBuilder) -> list[Any]:
    return [
        IdentitySegmentBuilder(builder.workspace),
        BootstrapSegmentBuilder(builder.workspace),
        MemorySegmentBuilder(builder.memory, _EchoBackend()),
        ActiveSkillsSegmentBuilder(builder.skills),
        SkillsSegmentBuilder(SkillForgeRouter([_EchoSource()])),
        _HistoryCurator(),
    ]


def _engine(builder: ContextBuilder, **kw) -> ContextAssembler:
    return ContextAssembler(_builders(builder), lambda: [], **kw)


# ---------------------------------------------------------------------------
# The block helpers
# ---------------------------------------------------------------------------


class TestBlockHelpers:
    def test_nothing_to_render_is_the_empty_string(self) -> None:
        assert render_turn_scratch([]) == ""
        assert render_turn_scratch(["", ""]) == ""

    def test_parts_join_under_one_tag(self) -> None:
        block = render_turn_scratch(["# Memory\n\na", "", "# Skills\n\nb"])
        assert block == f"{TURN_SCRATCH_TAG}\n\n# Memory\n\na{_SEP}# Skills\n\nb"

    def test_a_string_stays_a_string(self) -> None:
        assert append_turn_scratch("hi", "S") == "hi\n\nS"

    def test_a_block_list_gains_exactly_one_text_block(self) -> None:
        content = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "x"}}]
        out = append_turn_scratch(content, "S")
        assert out[:-1] == content
        assert out[-1] == {"type": "text", "text": "S"}

    def test_an_empty_scratch_leaves_the_content_alone(self) -> None:
        content = [{"type": "text", "text": "hi"}]
        assert append_turn_scratch(content, "") is content

    def test_strip_undoes_append_for_both_shapes(self) -> None:
        block = render_turn_scratch(["# Memory\n\nrecalled"])
        assert strip_turn_scratch(append_turn_scratch("hi", block)) == "hi"
        content = [{"type": "text", "text": "hi"}]
        assert strip_turn_scratch(append_turn_scratch(content, block)) == content

    def test_strip_returns_the_same_object_when_there_is_no_scratch(self) -> None:
        content = [{"type": "text", "text": "hi"}]
        assert strip_turn_scratch(content) is content
        assert strip_turn_scratch("hi") == "hi"
        assert strip_turn_scratch(None) is None

    def test_strip_of_scratch_alone_is_none(self) -> None:
        block = render_turn_scratch(["x"])
        assert strip_turn_scratch(block) is None
        assert strip_turn_scratch([{"type": "text", "text": block}]) is None


# ---------------------------------------------------------------------------
# Where the assembler puts the segments
# ---------------------------------------------------------------------------


class TestPlacement:
    async def test_memory_is_in_the_user_tail_exactly_once(self, builder: ContextBuilder) -> None:
        ac = await _engine(builder).assemble("s", [], _budget(), turn=TurnContext(current_message="espresso"))

        system = ac.messages[0]["content"]
        user = ac.messages[-1]["content"]
        assert not _MEMORY_HEADING.search(system)
        assert not _SKILLS_HEADING.search(system)
        assert "espresso" not in system
        assert isinstance(user, str)
        tail = user[user.index(TURN_SCRATCH_TAG) :]
        assert len(_MEMORY_HEADING.findall(_whole_prompt(ac.messages))) == 1
        assert len(_MEMORY_HEADING.findall(tail)) == 1
        assert "recalled for espresso" in tail
        assert "skill for espresso" in tail
        assert tail.index("# Memory") < tail.index("# Skills")
        assert user.count(TURN_SCRATCH_TAG) == 1

    async def test_the_stable_segments_stay_in_system(self, builder: ContextBuilder) -> None:
        ac = await _engine(builder).assemble("s", [], _budget(), turn=TurnContext(current_message="espresso"))

        system = ac.messages[0]["content"]
        assert "bootstrap rules" in system
        assert "# Active Skills" in system
        assert "always body" in system

    async def test_a_block_list_message_gains_one_scratch_block(self, builder: ContextBuilder, tmp_path: Path) -> None:
        from PIL import Image

        picture = tmp_path / "p.png"
        Image.new("RGB", (8, 8), "red").save(picture)
        turn = TurnContext(current_message="espresso", media=[str(picture)], can_see_images=True)
        plain = await _engine(builder, turn_scratch=False).assemble("s", [], _budget(), turn=turn)
        ac = await _engine(builder).assemble("s", [], _budget(), turn=turn)

        content = ac.messages[-1]["content"]
        assert isinstance(content, list)
        assert len(content) == len(plain.messages[-1]["content"]) + 1
        assert content[-1]["type"] == "text"
        assert content[-1]["text"].startswith(TURN_SCRATCH_TAG)
        assert sum(TURN_SCRATCH_TAG in t for t in _texts(content)) == 1
        assert len(_MEMORY_HEADING.findall(_whole_prompt(ac.messages))) == 1

    async def test_system_is_byte_identical_across_two_turns(self, builder: ContextBuilder) -> None:
        eng = _engine(builder)
        history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "sure"}]

        first = await eng.assemble("s", history, _budget(), turn=TurnContext(current_message="one"))
        second = await eng.assemble("s", history, _budget(), turn=TurnContext(current_message="two"))

        assert first.messages[0] == second.messages[0]
        assert first.messages[:-1] == second.messages[:-1]
        assert "recalled for one" in _whole_prompt(first.messages[-1:])
        assert "recalled for two" in _whole_prompt(second.messages[-1:])

    async def test_an_undeclared_builder_stays_in_system(self, builder: ContextBuilder) -> None:
        class _Plugin:
            name, order, needs_prefix = "plugin", 7, False

            async def build(self, ctx: AssemblyContext) -> Segment:
                return Segment(text="# Plugin\n\nplugin text")

        eng = ContextAssembler([*_builders(builder), _Plugin()], lambda: [])
        ac = await eng.assemble("s", [], _budget(), turn=TurnContext(current_message="hi"))

        assert "plugin text" in ac.messages[0]["content"]
        assert "plugin text" not in _whole_prompt(ac.messages[-1:])


class TestTurnScratchOff:
    async def test_the_system_prompt_is_the_segments_in_the_new_order(self, builder: ContextBuilder) -> None:
        """Off means the pre-scratch layout, but in the normalised order:
        identity, bootstrap, active skills, memory, skills."""
        ac = await _engine(builder, turn_scratch=False).assemble(
            "s", [], _budget(), turn=TurnContext(current_message="espresso")
        )

        ctx = AssemblyContext(
            session_key="s",
            current_message="espresso",
            media=None,
            channel=None,
            chat_id=None,
            session_messages=[],
            budget=_budget(),
        )
        expected = []
        for b in sorted(_builders(builder)[:5], key=lambda b: b.order):
            seg = await b.build(ctx)
            expected.append(seg.text if seg is not None else "")
        assert [b.name for b in sorted(_builders(builder)[:5], key=lambda b: b.order)] == [
            "identity",
            "bootstrap",
            "active_skills",
            "memory",
            "skills",
        ]
        # The recall fence carries a fresh nonce per render.
        nonce = re.compile(r"#[0-9a-f]{8}\b")
        assert nonce.sub("#N", ac.messages[0]["content"]) == nonce.sub("#N", _SEP.join(t for t in expected if t))
        assert TURN_SCRATCH_TAG not in _whole_prompt(ac.messages)
        assert ac.metadata["turn_scratch"] is False

    def test_the_factory_reads_the_switch(self, tmp_path: Path) -> None:
        from raven.config.raven import ContextCacheConfig, ContextConfig
        from raven.context_engine import build_context_engine

        class _Provider:
            def get_default_model(self) -> str:
                return "stub"

        def _build(turn_scratch: bool) -> ContextAssembler:
            return build_context_engine(
                workspace=tmp_path,
                builder=ContextBuilder(workspace=tmp_path),
                config=ContextConfig(cache=ContextCacheConfig(turn_scratch=turn_scratch)),
                provider=_Provider(),
                model="stub",
                context_window_tokens=100_000,
                get_tool_definitions=lambda: [],
            )

        assert _build(True)._turn_scratch is True
        assert _build(False)._turn_scratch is False


# ---------------------------------------------------------------------------
# The session write
# ---------------------------------------------------------------------------


def _loop_stub():
    from datetime import datetime

    from raven.agent.loop.main import AgentLoop

    loop = object.__new__(AgentLoop)
    loop._now_fn = lambda: datetime(2026, 10, 20, 12, 0)
    return loop


class TestSessionNeverRecordsTheScratch:
    def _save(self, tmp_path: Path, messages: list[dict], **kw) -> tuple[list[dict], str]:
        from raven.agent.loop.main import AgentLoop
        from raven.session.manager import SessionManager

        manager = SessionManager(tmp_path)
        session = manager.get_or_create("cli:x")
        AgentLoop._save_turn(_loop_stub(), session, messages, 0, **kw)
        manager.save(session)
        return session.messages, manager.session_path("cli:x").read_text(encoding="utf-8")

    def test_both_shapes_reach_disk_without_the_block(self, tmp_path: Path) -> None:
        runtime = f"{ContextBuilder._RUNTIME_CONTEXT_TAG}\nCurrent Time: now"
        scratch = render_turn_scratch(["# Memory\n\nrecalled: espresso"])
        plain = append_turn_scratch(f"{runtime}\n\nwhat did I say?", scratch)
        blocks = append_turn_scratch(
            [{"type": "text", "text": runtime}, {"type": "text", "text": "and this?"}],
            scratch,
        )

        stored, on_disk = self._save(
            tmp_path,
            [
                {"role": "user", "content": plain},
                {"role": "user", "content": blocks},
                {"role": "assistant", "content": "answer"},
            ],
        )

        assert TURN_SCRATCH_TAG not in on_disk
        assert "recalled: espresso" not in on_disk
        assert [m["content"] for m in stored] == ["what did I say?", [{"type": "text", "text": "and this?"}], "answer"]

    def test_the_inbound_original_path_records_the_words_sent(self, tmp_path: Path) -> None:
        runtime = f"{ContextBuilder._RUNTIME_CONTEXT_TAG}\nCurrent Time: now"
        scratch = render_turn_scratch(["# Memory\n\nrecalled: espresso"])

        stored, on_disk = self._save(
            tmp_path,
            [{"role": "user", "content": append_turn_scratch(f"{runtime}\n\nhello", scratch)}],
            inbound_original="hello",
            received_at="2026-10-20T11:59:00",
        )

        assert TURN_SCRATCH_TAG not in on_disk
        assert stored[0]["content"] == "hello"

    def test_a_block_list_inbound_drops_only_the_scratch_block(self, tmp_path: Path) -> None:
        runtime = f"{ContextBuilder._RUNTIME_CONTEXT_TAG}\nCurrent Time: now"
        scratch = render_turn_scratch(["# Memory\n\nrecalled: espresso"])
        content = append_turn_scratch(
            [
                {"type": "text", "text": runtime},
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
            scratch,
        )

        stored, on_disk = self._save(tmp_path, [{"role": "user", "content": content}], inbound_original="look")

        assert TURN_SCRATCH_TAG not in on_disk
        assert "recalled: espresso" not in on_disk
        texts = [b.get("text") for b in stored[0]["content"] if b.get("type") == "text"]
        assert "look" in texts
        assert not any(TURN_SCRATCH_TAG in (t or "") for t in texts)


# ---------------------------------------------------------------------------
# Replay and the hook-facing question
# ---------------------------------------------------------------------------


class TestReplayIgnoresTheScratch:
    def _recorded(self, user_content: Any) -> dict[str, Any]:
        return {
            "model": "m",
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": user_content}],
            "tools": [],
        }

    def test_a_string_message_with_scratch_matches_the_recording_without(self) -> None:
        from raven.trajectory.replay import compare_llm_request

        live = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": append_turn_scratch("hi", render_turn_scratch(["# Memory\n\nx"]))},
        ]

        assert compare_llm_request(self._recorded("hi"), live, [], "m") is None

    def test_a_block_list_message_with_scratch_matches_the_recording_without(self) -> None:
        from raven.trajectory.replay import compare_llm_request

        content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        live = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": append_turn_scratch(content, render_turn_scratch(["# Memory\n\nx"]))},
        ]

        assert compare_llm_request(self._recorded(content), live, [], "m") is None

    def test_a_different_question_still_diverges(self) -> None:
        from raven.trajectory.replay import compare_llm_request

        live = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": append_turn_scratch("bye", render_turn_scratch(["# Memory\n\nx"]))},
        ]

        mismatch = compare_llm_request(self._recorded("hi"), live, [], "m")
        assert mismatch is not None and mismatch.field == "messages[1]"


def test_the_turn_question_leaves_the_scratch_out() -> None:
    from raven.agent.loop._shared import turn_question

    runtime = f"{ContextBuilder._RUNTIME_CONTEXT_TAG}\nCurrent Time: now"
    scratch = render_turn_scratch(["# Memory\n\nrecalled"])

    assert turn_question([{"role": "user", "content": append_turn_scratch(f"{runtime}\n\nwhy?", scratch)}]) == "why?"
    blocks = append_turn_scratch([{"type": "text", "text": "why?"}], scratch)
    assert turn_question([{"role": "user", "content": blocks}]) == "why?"


# ---------------------------------------------------------------------------
# Cache breakpoints with the scratch in place
# ---------------------------------------------------------------------------


def _marks(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
    def count(value: Any) -> int:
        if isinstance(value, dict):
            return ("cache_control" in value) + sum(count(v) for v in value.values())
        if isinstance(value, list):
            return sum(count(v) for v in value)
        return 0

    return count(messages) + count(tools or [])


class TestCacheOptimizerBudget:
    async def test_at_most_four_breakpoints_on_the_first_and_second_iteration(self, builder: ContextBuilder) -> None:
        from raven.token_wise.cache_optimizer import CacheOptimizer

        optimizer = CacheOptimizer(supports_caching=lambda _model: True)
        tools = [{"type": "function", "function": {"name": "exec", "parameters": {}}}]
        history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "sure"}]
        ac = await _engine(builder).assemble("s", history, _budget(), turn=TurnContext(current_message="espresso"))

        first, first_tools, _ = await optimizer.before_llm_call(list(ac.messages), tools, "m")
        assert _marks(first, first_tools) <= 4
        assert "cache_control" in str(first[0])

        second_in = [
            *ac.messages,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "exec", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "exec", "content": "ok"},
        ]
        second, second_tools, _ = await optimizer.before_llm_call(second_in, tools, "m")
        assert _marks(second, second_tools) <= 4
        assert second[0] == first[0]

    async def test_the_system_mark_reads_the_same_on_the_next_turn(self, builder: ContextBuilder) -> None:
        from raven.token_wise.cache_optimizer import CacheOptimizer

        optimizer = CacheOptimizer(supports_caching=lambda _model: True)
        eng = _engine(builder)
        one = await eng.assemble("s", [], _budget(), turn=TurnContext(current_message="one"))
        two = await eng.assemble("s", [], _budget(), turn=TurnContext(current_message="two"))

        marked_one, _, _ = await optimizer.before_llm_call(list(one.messages), None, "m")
        marked_two, _, _ = await optimizer.before_llm_call(list(two.messages), None, "m")

        assert marked_one[0] == marked_two[0]


# ---------------------------------------------------------------------------
# A real loop over two turns
# ---------------------------------------------------------------------------


class _RecordingProvider(LLMProvider):
    """Answers every call and keeps each request's messages."""

    def __init__(self) -> None:
        super().__init__(api_key="test")
        self.requests: list[list[dict]] = []

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7, **_kw):
        self.requests.append(json.loads(json.dumps(messages, default=str)))
        return LLMResponse(content="ok", finish_reason="stop")

    def get_default_model(self) -> str:
        return "stub"


def _agent(workspace: Path, *, turn_scratch: bool = True):
    from raven.agent.loop import AgentLoop
    from raven.agent.loop.bundles import EngineWiring, ToolWiring, TurnPolicy
    from raven.config.raven import ContextCacheConfig, ContextConfig

    return AgentLoop(
        provider=_RecordingProvider(),
        workspace=workspace,
        model="stub",
        policy=TurnPolicy(max_iterations=2),
        tools=ToolWiring(restrict_to_workspace=True),
        engine=EngineWiring(
            backend=_EchoBackend(),
            context_config=ContextConfig(cache=ContextCacheConfig(turn_scratch=turn_scratch)),
        ),
    )


def _request(text: str):
    from raven.spine.message import ChatType, Source
    from raven.spine.turn import Origin, TurnRequest

    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel="mock", chat_id="c1", sender_id="user", chat_type=ChatType.DM),
        text=text,
    )


class TestTwoTurnsThroughTheLoop:
    async def test_system_is_identical_and_memory_rides_the_tail(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)

        await agent._process_message(_request("first question about espresso"))
        await agent._process_message(_request("second question about tea"))

        first, second = agent.provider.requests[0], agent.provider.requests[-1]
        assert len(agent.provider.requests) == 2
        assert first[0]["role"] == "system"
        assert first[0] == second[0]
        for request, query in ((first, "espresso"), (second, "tea")):
            assert not _MEMORY_HEADING.search(_whole_prompt(request[:1]))
            assert len(_MEMORY_HEADING.findall(_whole_prompt(request))) == 1
            tail = _whole_prompt(request[-1:])
            assert len(_MEMORY_HEADING.findall(tail)) == 1
            assert "recalled for" not in _whole_prompt(request[:-1])
            assert query in tail[tail.index(TURN_SCRATCH_TAG) :]
        # The first turn's scratch is gone from the second turn's history.
        assert TURN_SCRATCH_TAG not in _whole_prompt(second[:-1])

        await agent.drain_backend_stores(timeout=5.0)
        on_disk = agent.sessions.session_path("mock:c1").read_text(encoding="utf-8")
        assert TURN_SCRATCH_TAG not in on_disk
        assert "recalled for" not in on_disk
        assert "first question about espresso" in on_disk

    async def test_turn_scratch_off_keeps_memory_in_system(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path, turn_scratch=False)

        await agent._process_message(_request("question about espresso"))

        request = agent.provider.requests[0]
        assert _MEMORY_HEADING.search(request[0]["content"])
        assert TURN_SCRATCH_TAG not in _whole_prompt(request)
