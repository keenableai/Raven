"""The history strategy registry and the factory's use of it.

``context.history.strategy`` picks the builder that fills the history slot. A
name the registry does not know stops the engine from being built at all, the
strategy's declared capabilities reach the assembler, and a strategy that is
not selected is never imported.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from raven.agent.context import ContextBuilder
from raven.config.raven import ContextConfig, HubSourceConfig, MemoryConfig, SkillForgeRouterConfig
from raven.context_engine import build_context_engine
from raven.context_engine import history as strategies
from raven.context_engine.assembler import ContextAssembler
from raven.providers.base import LLMProvider, LLMResponse

REPO = Path(__file__).resolve().parent.parent


class _StubProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test")

    async def chat(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(content="", finish_reason="stop")

    def get_default_model(self) -> str:
        return "stub"


def _engine(tmp_path: Path, config: ContextConfig) -> ContextAssembler:
    return build_context_engine(
        workspace=tmp_path,
        config=config,
        builder=ContextBuilder(workspace=tmp_path),
        provider=_StubProvider(),
        model="stub",
        context_window_tokens=8192,
        get_tool_definitions=list,
        memory_config=MemoryConfig(),
        skill_forge_router_config=SkillForgeRouterConfig(hub=HubSourceConfig(endpoint=None)),
    )


def test_the_registry_names_both_strategies() -> None:
    assert strategies.registered() == ["compacting", "curator"]
    compacting = strategies.resolve("compacting")
    assert (compacting.wants_full_history, compacting.owns_compaction, compacting.mints_source_refs) == (
        True,
        True,
        True,
    )
    curator = strategies.resolve("curator")
    assert (curator.wants_full_history, curator.owns_compaction, curator.mints_source_refs) == (True, True, False)


def test_a_name_is_read_without_case_or_padding() -> None:
    assert strategies.resolve("  Curator ").name == "curator"


@pytest.mark.parametrize("name", ["", "passthrough", "unified", "curat0r", None])
def test_an_unknown_strategy_is_refused_naming_the_ones_there_are(name) -> None:
    with pytest.raises(ValueError) as raised:
        strategies.resolve(name)
    assert "compacting" in str(raised.value) and "curator" in str(raised.value)


def test_the_factory_refuses_to_build_on_an_unknown_strategy(tmp_path: Path) -> None:
    config = ContextConfig.model_validate({"history": {"strategy": "passthrough"}})
    with pytest.raises(ValueError, match="context.history.strategy"):
        _engine(tmp_path, config)


def test_the_curator_strategy_builds_the_engine_it_always_did(tmp_path: Path) -> None:
    engine = _engine(tmp_path, ContextConfig.model_validate({"history": {"strategy": "curator"}}))
    assert [b.name for b in engine._phase_b] == ["curator"]
    assert engine.wants_full_history and engine.owns_compaction
    assert not engine.mints_source_refs


def test_the_flat_curator_keys_still_reach_the_curator(tmp_path: Path) -> None:
    config = ContextConfig.model_validate({"history": {"strategy": "curator"}, "fastPathThreshold": 0.33})
    engine = _engine(tmp_path, config)
    assert engine._phase_b[0].config.fast_path_threshold == 0.33


def test_a_strategy_that_is_not_selected_is_never_imported(tmp_path: Path) -> None:
    script = (
        "import sys\n"
        "from raven.context_engine import history\n"
        "history.resolve('curator')\n"
        "print('raven.context_engine.history.compacting' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=120, check=True
    )
    assert out.stdout.strip() == "False"


@pytest.mark.asyncio
async def test_an_assembler_with_no_strategy_hook_leaves_overflow_to_the_loop(tmp_path: Path) -> None:
    engine = _engine(tmp_path, ContextConfig.model_validate({"history": {"strategy": "curator"}}))
    assert await engine.on_context_overflow("k", [{"role": "user", "content": "x"}], attempt=1, window=8192) is None
