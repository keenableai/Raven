"""The compacting strategy's history builder, and what it keeps between turns.

Fills the history slot by measuring the whole assembled prompt against an
adaptive trigger and, when it is over, emptying the middle of the conversation
until it is back under target. Deterministic in what it removes: the same
session and the same window always produce the same selection, and no model is
asked which messages matter.

Over the trigger it works down a ladder and stops at the first rung that is
enough. First the pruner: tool-result bodies are replaced by a record of the
call and a ref, which reclaims most of a large session without any message
leaving the window. The prompt is then measured again, and a turn that is back
under the line ships with every exchange still present. Only what the pruner
could not buy is paid for by removing messages -- and what is removed leaves a
checkpoint behind: eight sections written by the turn's own model, or, when no
model answers, a fold saying how much went and where to ask for it.

Three properties hold the history together across turns:

- **What was replaced stays replaced.** The ordinals a checkpoint covers are
  retired, so a later turn selects around them instead of selecting them
  again. Without that, a session that compacted once would compact again on
  every turn after it -- the log only grows -- and the history the provider
  cached would shrink and grow underneath it.
- **What was pruned stays pruned.** The ordinals the pruner rewrote are kept
  and rewritten again on every later turn, including the ones under the
  trigger, so a body never flows back in once its record line has been sent.
- **It survives the process.** Both sets, the compaction count and the
  standing checkpoint are read back from the workspace every turn, for the
  session's current revision, so a restarted process sends the same bytes
  without recompacting -- and a ``/new`` or an undo, which change the
  revision, start from nothing rather than from ordinals that now name other
  messages.

The checkpoint **sits where its messages sat**, between the protected head and
the tail, so the next compaction finds it at the front of the range it is about
to remove and folds it forward without any bookkeeping about where the last
summary went. It carries :data:`HISTORY_CHECKPOINT_KEY`, and every message of a
pinned exchange carries :data:`HISTORY_PINNED_KEY`, so in-turn window
management can leave both alone; neither key may reach a provider.

Two entry points, one pipeline. ``build`` runs it against the strategy's own
reading of the prompt size; ``on_context_overflow`` runs it again after a
provider has refused an assembled prompt, by which point the shared headroom
table has learned what that provider really needs and the same trigger
arithmetic produces a tighter line on its own. The overflow path writes no
checkpoint and does not count as a compaction: what arrives there is an
assembled prompt, where no position is a session ordinal, so nothing minted
from one would point anywhere, and counting it would decay the head of a
history that was never compacted.

The strategy needs the whole append-only log (``wants_full_history``) and puts
refs where the model can read them (``mints_source_refs``); the engine that
hosts it has to answer both with ``True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from loguru import logger

from raven.context_engine.headroom import reply_headroom
from raven.context_engine.history.compacting.checkpoint import CompactionCheckpoint, fold_checkpoint, timestamp
from raven.context_engine.history.compacting.ledger import MessageLedger
from raven.context_engine.history.compacting.pruner import PruneOutcome, ToolResultPruner
from raven.context_engine.history.compacting.retention import Ranges, RetentionPlanner, Selection
from raven.context_engine.history.compacting.store import CHECKPOINT_DIR, CheckpointStore, CompactingState
from raven.context_engine.history.compacting.trigger import CompactionPlan, CompactionTrigger, TriggerPolicy
from raven.context_engine.history.compacting.writer import (
    SYSTEM_KEY,
    USER_KEY,
    CheckpointWriter,
    MessageKey,
    RequestRecord,
    checkpoint_key,
    session_message_key,
)
from raven.context_engine.history.markers import HISTORY_CHECKPOINT_KEY, HISTORY_PINNED_KEY, HISTORY_PRIVATE_KEYS
from raven.context_engine.history_trimmer import HistoryTrimmer, TrimOutcome
from raven.contracts.context import AssemblyContext, OverflowOutcome, Segment
from raven.providers import prompt_cache
from raven.providers.binding import ModelBinding, active_window, resolve
from raven.session.refs import session_ref
from raven.utils.tokens import estimate_message_tokens, estimate_prompt_tokens

if TYPE_CHECKING:
    from raven.config.raven import ContextConfig
    from raven.contracts.context import AssembledPrefix, SessionRevision
    from raven.contracts.llm_provider import LLMProvider

SEGMENT_PATH = "compacting"
"""The ``path`` this builder reports in segment metadata."""

_MAX_TRACKED_SESSIONS = 64
"""Sessions whose in-process memo is kept, most recent first.

Bounded because the memo holds the last request verbatim, which keeps that
session's messages alive for as long as the entry does. Evicting one costs one
cache reuse on the next summary and nothing else: everything that decides the
history is on disk.
"""


@dataclass
class _Memo:
    """What this process remembers about a session between turns.

    A cache, never the authority: the history is decided from the state on
    disk. What lives only here is what losing costs at most one prefix reuse --
    the last request as it was sent, and the key list that names its messages.
    """

    scope: str = ""
    generation: int = 0
    checkpoint_message: dict[str, Any] | None = None
    keys: tuple[MessageKey, ...] = ()
    """What each message of the list this turn produced stands for, in order:
    the system message, every history message, the user message."""

    view: tuple[dict[str, Any], ...] = ()
    """The history this turn produced, normalised the way :func:`_comparable`
    reads a sent message, so :meth:`CompactingHistoryBuilder.note_request` can
    tell a request that carried it unchanged from one that did not."""

    record: RequestRecord | None = None
    warned: bool = False


class CompactingHistoryBuilder:
    """Selects ``*history`` by compacting it; contributes no system text."""

    name = "compacting"
    order = 6
    needs_prefix = True
    stable = False

    def __init__(
        self,
        workspace: Path,
        config: "ContextConfig",
        provider: "LLMProvider",
        model: str,
        context_window_tokens: int,
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.cfg = config.history.compacting
        self._fallback = ModelBinding(provider, model)
        self._fallback_window = int(context_window_tokens)
        self.get_tool_definitions = get_tool_definitions
        self.trigger = TriggerPolicy(self.cfg)
        self.planner = RetentionPlanner(self.cfg, list(getattr(config, "pinned_skill_ids", None) or []))
        self.pruner = ToolResultPruner(self.cfg.prune)
        self.trimmer = HistoryTrimmer(provider, model, get_tool_definitions, context_window_tokens)
        self.writer = CheckpointWriter(self.cfg.summary, now_fn=now_fn)
        self._now_fn = now_fn
        self.store = CheckpointStore(workspace / CHECKPOINT_DIR)
        self._sessions: dict[str, _Memo] = {}

    # ------------------------------------------------------------------
    # The turn's model and window
    # ------------------------------------------------------------------

    @property
    def provider(self) -> "LLMProvider":
        """The provider of the turn's binding; the build-time one outside a turn."""
        return resolve(None, self._fallback).provider

    @property
    def model(self) -> str:
        """The model of the turn's binding; the build-time one outside a turn."""
        return resolve(None, self._fallback).model

    @property
    def context_window_tokens(self) -> int:
        """The running turn's window; the one built with, outside a turn."""
        return active_window(self._fallback_window)

    @context_window_tokens.setter
    def context_window_tokens(self, tokens: int) -> None:
        self._fallback_window = int(tokens)

    def set_provider(self, provider: "LLMProvider", model: str) -> None:
        """Adopt the provider a live ``/model`` switch just built."""
        self._fallback = ModelBinding(provider, model)
        self.trimmer.set_provider(provider, model)

    def set_context_window(self, tokens: int) -> None:
        """Follow a ``/model`` switch down into the trimmer this builder owns."""
        self.context_window_tokens = tokens
        self.trimmer.context_window_tokens = tokens

    # ------------------------------------------------------------------
    # The request the loop actually sent
    # ------------------------------------------------------------------

    def note_request(self, session_key: str, messages: list[dict[str, Any]], model: str = "") -> None:
        """Keep a request the loop is sending, for the next summary to reuse.

        The next compaction is built out of these bytes rather than out of a
        freshly derived equivalent, which is what makes it a prefix the provider
        can serve from cache. Derivation is not enough: the pruner rewrites a
        body on the turn it fires, a cache breakpoint is marked on the way out,
        and a cache matches on bytes.

        Kept only when its leading messages are still this turn's history. The
        assembler may have merged two adjacent assistant messages, and in-turn
        window management may have rewritten a body; either way the keys no
        longer name what was sent, and the record is left as it was -- reusing
        nothing beats reusing one message's bytes under another's name. Later
        iterations of a turn append after the history, so the latest request
        that still carries it is the one kept.

        ``model`` is the model the request went out under. Losing a record --
        to a restart or an eviction -- costs one summary its cache reuse.
        """
        memo = self._sessions.get(session_key)
        if memo is None or not memo.keys:
            return
        count = len(memo.keys)
        if len(messages) < count:
            return
        head = [dict(message) for message in messages[:count]]
        if head[0].get("role") != "system" or head[-1].get("role") != "user":
            return
        if tuple(_comparable(message) for message in head[1:-1]) != memo.view:
            return
        memo.record = RequestRecord(messages=tuple(head), keys=memo.keys, model=model or "")

    # ------------------------------------------------------------------
    # One turn
    # ------------------------------------------------------------------

    async def build(self, ctx: AssemblyContext) -> Segment:
        if ctx.prefix is None:
            raise RuntimeError("CompactingHistoryBuilder requires ctx.prefix (phase B)")
        session = list(ctx.session_messages or [])
        if not session:
            return Segment(text="", history=[], meta={"path": SEGMENT_PATH, "compaction_action": "pass"})
        revision = ctx.session_revision
        if revision is None:
            return self._unscoped(ctx, ctx.prefix, session)

        memo = self._memo(ctx.session_key, revision.scope)
        state = self.store.load(ctx.session_key, revision, message_count=len(session))
        binding = resolve(None, self._fallback)
        window = self.context_window_tokens
        headroom_tokens = ctx.budget.reserved_output

        prefix = ctx.prefix
        system_message = {"role": "system", "content": prefix.system_prefix}
        fixed = estimate_prompt_tokens([system_message, prefix.user_message], prefix.tool_defs)

        ledger = MessageLedger(session)
        pinned = self.planner.pinned_ids(ledger)
        sticky = self.pruner.reapply(session, revision=revision, ordinals=state.pruned, protected_ids=pinned)
        if sticky.ordinals:
            session = sticky.messages
            ledger = MessageLedger(session)

        retired = state.retired
        checkpoint = state.checkpoint
        checkpoint_message = self._render(checkpoint)
        standing = estimate_message_tokens(checkpoint_message) if checkpoint_message is not None else 0
        generation = state.generation

        plan = self._evaluate(ledger, retired, fixed + standing, headroom_tokens, window)
        ranges = self.planner.split(ledger, plan, generation=generation)

        pruned = self._prune(revision, session, ranges, plan, pinned | retired)
        if pruned.changed:
            session = pruned.messages
            ledger = MessageLedger(session)
            plan = self._evaluate(ledger, retired, fixed + standing, headroom_tokens, window)
            ranges = self.planner.split(ledger, plan, generation=generation)
            self._report_prune(ctx.session_key, pruned, plan)

        selection = self._select(ledger, ranges, plan, fixed_tokens=fixed + standing, retired=retired)
        removed = frozenset(selection.dropped) - retired
        written: CompactionCheckpoint | None = None
        if removed:
            generation += 1
            written = await self._checkpoint_for(
                ctx.session_key,
                prefix,
                binding,
                revision,
                session,
                ledger,
                memo,
                checkpoint,
                checkpoint_message,
                selection,
                removed,
                retired,
                generation,
            )
            if written is not None:
                # A generation that produced none leaves the previous one
                # standing: it is still true about its own span, and dropping it
                # would take the only account of everything compacted before
                # this turn with it.
                checkpoint = written
                checkpoint_message = self._render(written)
            self._report(ctx.session_key, plan, selection, removed, first=not memo.warned)
            memo.warned = True

        estimate_at = _checkpoint_position(selection.kept, selection, checkpoint_message)
        _messages, outcome = self.trimmer.trim(
            session_messages=session,
            ids=list(selection.kept),
            protected_ids=set(ranges.hole_ids),
            reserved_output=headroom_tokens,
            # Position inside this callback is arbitrary: the trimmer builds
            # the list only to price it, and a token estimate does not depend
            # on where in the history a message sits. The list that ships is
            # spliced below, against what actually survived the budget.
            build_messages=lambda history: [
                system_message,
                *_splice(history, checkpoint_message, estimate_at),
                prefix.user_message,
            ],
        )
        checkpoint_at = _checkpoint_position(outcome.included_ids, selection, checkpoint_message)
        if written is not None and checkpoint is written:
            # Retirement is what a checkpoint buys: these messages are spoken
            # for, so later turns select around them instead of removing them
            # again. With nothing standing in for them there is nothing to
            # justify it, and re-deciding every turn is the honest behaviour --
            # which is also what makes turning summaries and folds both off a
            # true rollback rather than a different scheme.
            retiring = frozenset(range(len(session))) - frozenset(outcome.included_ids)
            checkpoint = dataclass_replace(written, retired=tuple(sorted(retiring)))
            self.store.write(checkpoint)

        self._persist(
            ctx.session_key,
            state,
            CompactingState(
                scope=revision.scope,
                generation=generation,
                pruned=state.pruned | pruned.ordinals,
                checkpoint=checkpoint,
            ),
        )

        plain = _splice(outcome.history, checkpoint_message, checkpoint_at)
        memo.generation = generation
        memo.checkpoint_message = checkpoint_message
        memo.keys = self._keys(outcome, checkpoint, checkpoint_at)
        memo.view = tuple(_comparable(message) for message in plain) if memo.keys else ()
        return Segment(
            text="",
            history=self._marked(outcome, checkpoint_message, checkpoint_at, pinned),
            meta=self._meta(
                generation,
                checkpoint,
                plan,
                selection,
                outcome,
                pruned,
                ranges,
                total=len(session),
                pruned_total=len(state.pruned | pruned.ordinals),
            ),
        )

    async def on_context_overflow(
        self,
        session_key: str,
        messages: list[dict[str, Any]],
        *,
        attempt: int,
        window: int,
    ) -> OverflowOutcome | None:
        """Compact a prompt the provider just refused, or decline.

        Declining is ``None`` rather than a non-advancing outcome, and the
        difference matters: ``None`` lets the loop's emergency shrink try the
        lever this path does not have -- eliding tool-result bodies without a
        ref. Neither the pruner nor the checkpoint writer can run here, because
        what arrives is an assembled prompt rather than the log: most of it is
        *this* turn's tool results, which the loop has not saved yet, so no
        position in this list is a session ordinal and any ref minted from one
        would resolve to a different message. Answering "nothing left to give"
        while an unlabelled elision still has something to give would end a
        recoverable turn on an error.

        By the time this runs the loop has already fed the refusal to the
        shared headroom table, so the trigger below is computed against what
        the provider actually demonstrated it needs, not against the floor that
        was optimistic enough to get refused.

        Not a compaction, so the compaction count does not move: counting it
        would stop protecting the head of a history no checkpoint stands for.
        """
        if len(messages) < 2 or messages[0].get("role") != "system":
            return None
        memo = self._sessions.get(session_key) or _Memo()
        system_message = messages[0]
        history = messages[1:]
        tools = self.get_tool_definitions()
        fixed = estimate_prompt_tokens([system_message], tools)
        ledger = MessageLedger(history)
        binding = resolve(None, self._fallback)
        plan = self.trigger.evaluate(
            prompt_tokens=fixed + ledger.total_tokens,
            window=window,
            reply_headroom=reply_headroom(
                _wire_model(binding.provider, binding.model),
                window=window,
                minimum=self.config.headroom.min_tokens,
            ),
            trigger=CompactionTrigger.OVERFLOW,
        )
        if plan.action == "pass":
            return None

        ranges = self._hold_marked(self.planner.split(ledger, plan, generation=memo.generation), history, memo)
        selection = self.planner.keep_within(ledger, ranges, fixed_tokens=fixed, budget_tokens=plan.target_tokens)
        # Priced from what survived the adjacency closure rather than from what
        # the planner chose: closure can pull a parent back in, and reporting a
        # reclaim the caller did not get is how a retry loop stops converging.
        ids = HistoryTrimmer.canonical_ids(history, list(selection.kept))
        kept_history = _carry_private_keys(history, ids, HistoryTrimmer.history_from_ids(history, ids))
        if not kept_history:
            # The closure found no user message to anchor on and discarded
            # everything. That reads as a total reclaim and is really a lost
            # conversation, so it is declined rather than shipped.
            return None
        reclaimed = ledger.total_tokens - sum(estimate_message_tokens(message) for message in kept_history)
        if reclaimed <= 0:
            return None

        reclaimed_messages = [system_message, *kept_history]
        errors = HistoryTrimmer.structural_errors(reclaimed_messages)
        if errors:
            logger.warning("Compaction declined an overflow reclaim that would not close: {}", errors)
            return None

        logger.warning(
            "Compaction reclaimed ~{} token(s) after a refused prompt (attempt {}, window {})",
            reclaimed,
            attempt,
            window,
        )
        return OverflowOutcome(messages=reclaimed_messages, reclaimed_tokens=reclaimed, advanced=True)

    # ------------------------------------------------------------------

    def _memo(self, session_key: str, scope: str) -> _Memo:
        memo = self._sessions.pop(session_key, None)
        if memo is not None and memo.scope != scope:
            memo = None
        if memo is None:
            memo = _Memo(scope=scope)
        self._sessions[session_key] = memo
        while len(self._sessions) > _MAX_TRACKED_SESSIONS:
            self._sessions.pop(next(iter(self._sessions)))
        return memo

    def _render(self, checkpoint: CompactionCheckpoint | None) -> dict[str, Any] | None:
        if checkpoint is None:
            return None
        return checkpoint.message(inline_refs=self.cfg.summary.refs_inline)

    def _persist(self, session_key: str, before: CompactingState, after: CompactingState) -> None:
        """Write the state when this turn changed it, and only then.

        A turn that changed nothing writes nothing, so a session that never
        crossed the trigger leaves no trace on disk.
        """
        before_id = before.checkpoint.checkpoint_id if before.checkpoint is not None else None
        after_id = after.checkpoint.checkpoint_id if after.checkpoint is not None else None
        unchanged = before.generation == after.generation and before.pruned == after.pruned and before_id == after_id
        if unchanged and not before.stale:
            return
        self.store.save(session_key, after)

    def _evaluate(
        self,
        ledger: MessageLedger,
        retired: frozenset[int],
        fixed_tokens: int,
        headroom_tokens: int,
        window: int,
    ) -> CompactionPlan:
        """Measure what would go out, not what the log holds.

        Retired messages are represented by the checkpoint and are not in the
        prompt, so counting them would keep a compacted session permanently over
        its own trigger.
        """
        live = frozenset(range(len(ledger))) - retired
        return self.trigger.evaluate(
            prompt_tokens=fixed_tokens + ledger.tokens_for(live),
            window=window,
            reply_headroom=headroom_tokens,
        )

    def _prune(
        self,
        revision: "SessionRevision",
        session: list[dict[str, Any]],
        ranges: Ranges,
        plan: CompactionPlan,
        protected: frozenset[int],
    ) -> PruneOutcome:
        """Shorten what can be shortened, before anything is removed.

        Only under pressure. A turn below the trigger is handed over exactly as
        it was the turn before -- rewriting a message that did not need
        rewriting would invalidate the cached prefix for nothing.

        The tail is held back, so the exchanges the model is currently working
        through keep their bodies; pinned bodies are held back for the stronger
        reason that a ref is not a substitute for text whose absence is
        silently wrong. Pinned here means every one of them, not only the
        holes: a hole is the pinned set narrowed to the range removal operates
        on, and a body sitting in the protected head is in none of them.
        Retired messages are held back because rewriting one buys nothing --
        they are already out of the window.
        """
        if plan.action != "compact":
            return PruneOutcome(messages=session, reclaimed_tokens=0, pruned_count=0, refs=())
        return self.pruner.run(
            session,
            revision=revision,
            protect_from=ranges.tail.start,
            protected_ids=protected,
        )

    def _report_prune(self, session_key: str, pruned: PruneOutcome, plan: CompactionPlan) -> None:
        """Say what pruning bought, and whether it was enough on its own."""
        logger.info(
            "Pruned {} tool message(s) for {}, reclaiming ~{} token(s){}",
            pruned.pruned_count,
            session_key,
            pruned.reclaimed_tokens,
            "; the turn is back under the trigger with nothing removed"
            if plan.action == "pass"
            else "; still over the trigger, so messages go next",
        )

    def _select(
        self,
        ledger: MessageLedger,
        ranges: Ranges,
        plan: CompactionPlan,
        *,
        fixed_tokens: int,
        retired: frozenset[int],
    ) -> Selection:
        if plan.action == "pass":
            kept = frozenset(range(len(ledger))) - retired
            return Selection(
                kept=tuple(sorted(kept)),
                dropped=tuple(sorted(retired)),
                kept_tokens=ledger.tokens_for(kept),
                exhausted=False,
            )
        return self.planner.keep_within(
            ledger,
            ranges,
            fixed_tokens=fixed_tokens,
            budget_tokens=plan.target_tokens,
            retired=retired,
        )

    async def _checkpoint_for(
        self,
        session_key: str,
        prefix: "AssembledPrefix",
        binding: ModelBinding,
        revision: "SessionRevision",
        session: list[dict[str, Any]],
        ledger: MessageLedger,
        memo: _Memo,
        previous: CompactionCheckpoint | None,
        previous_message: dict[str, Any] | None,
        selection: Selection,
        removed: frozenset[int],
        retired: frozenset[int],
        generation: int,
    ) -> CompactionCheckpoint | None:
        """What stands in the window where the removed messages were.

        What the checkpoint stands for is everything compaction has taken out
        by the end of this turn, not only this turn's share -- it replaces the
        one before it rather than sitting beside it. The rung is chosen on this
        turn's removal, which is what the model call would be paying for.
        """
        covered = removed | retired
        covers = session_ref(revision, min(covered), max(covered))
        folded_messages = len(covered)
        folded_tokens = ledger.tokens_for(covered)
        wants_summary = self.writer.prefers_summary(
            reached_target=not selection.exhausted,
            folded_messages=len(removed),
        )
        if wants_summary:
            keys, fresh = self._summary_input(
                prefix, session, ledger, previous, previous_message, retired, through=max(removed)
            )
        else:
            keys, fresh = ((), ())
        if wants_summary and keys:
            return await self.writer.compact(
                provider=binding.provider,
                model=binding.model,
                session_key=session_key,
                generation=generation,
                covers=covers,
                previous=previous,
                keys=keys,
                fresh=fresh,
                record=memo.record,
                tool_defs=prefix.tool_defs,
                folded_messages=folded_messages,
                folded_tokens=folded_tokens,
            )
        if self.cfg.summary.fold:
            return fold_checkpoint(
                session_key=session_key,
                generation=generation,
                covers=covers,
                previous=previous,
                folded_messages=folded_messages,
                folded_tokens=folded_tokens,
                created_at=timestamp(self._now_fn),
            )
        return None

    def _summary_input(
        self,
        prefix: "AssembledPrefix",
        session: list[dict[str, Any]],
        ledger: MessageLedger,
        previous: CompactionCheckpoint | None,
        previous_message: dict[str, Any] | None,
        retired: frozenset[int],
        *,
        through: int,
    ) -> tuple[tuple[MessageKey, ...], tuple[dict[str, Any], ...]]:
        """The leading messages a summarising request is built from.

        Everything the conversation has said up to and including the last
        message being removed, in order, with the standing checkpoint where the
        messages it replaced used to be. The cut runs forward to the next place
        no tool call is left open, so the request never ends on a call whose
        result was going to arrive two messages later.

        These are the *fresh* versions. Whichever of them the recorded request
        also holds is substituted for the recorded bytes before the request goes
        out; this is what the rest is built from, and what the whole request
        falls back to when there is no record to reuse.
        """
        end = ledger.align_forward(through + 1, self.cfg.boundary)
        ordinals = [index for index in range(end) if index not in retired]
        messages = HistoryTrimmer.history_from_ids(session, ordinals)
        if len(messages) != len(ordinals):
            return ((), ())
        keys: list[MessageKey] = [session_message_key(index) for index in ordinals]
        if previous is not None and previous_message is not None and retired:
            at = sum(1 for index in ordinals if index < min(retired))
            keys = _splice(keys, checkpoint_key(previous.checkpoint_id), at)
            messages = _splice(messages, previous_message, at)
        leading = [{"role": "system", "content": prefix.system_prefix}, *messages]
        errors = HistoryTrimmer.structural_errors(leading)
        if errors:
            logger.warning("Compaction skipped a summary whose request would not close: {}", errors)
            return ((), ())
        return (SYSTEM_KEY, *keys), tuple(leading)

    @staticmethod
    def _keys(
        outcome: TrimOutcome,
        checkpoint: CompactionCheckpoint | None,
        checkpoint_at: int,
    ) -> tuple[MessageKey, ...]:
        """What each message of the list this turn produced stands for.

        Empty when the history and the ids it was projected from do not line up,
        which is the honest answer: an unaligned key list would make the next
        compaction reuse one message's bytes under another message's name.
        """
        if len(outcome.history) != len(outcome.included_ids):
            return ()
        keys: list[MessageKey] = [session_message_key(index) for index in outcome.included_ids]
        if checkpoint is not None:
            keys = _splice(keys, checkpoint_key(checkpoint.checkpoint_id), checkpoint_at)
        return (SYSTEM_KEY, *keys, USER_KEY)

    @staticmethod
    def _marked(
        outcome: TrimOutcome,
        checkpoint_message: dict[str, Any] | None,
        checkpoint_at: int,
        pinned: frozenset[int],
    ) -> list[dict[str, Any]]:
        """The history as it ships: private keys on what the loop must not touch."""
        aligned = len(outcome.history) == len(outcome.included_ids)
        history = [
            {**message, HISTORY_PINNED_KEY: True} if aligned and outcome.included_ids[pos] in pinned else message
            for pos, message in enumerate(outcome.history)
        ]
        if checkpoint_message is None:
            return history
        return _splice(history, {**checkpoint_message, HISTORY_CHECKPOINT_KEY: True}, checkpoint_at)

    @staticmethod
    def _hold_marked(ranges: Ranges, history: list[dict[str, Any]], memo: _Memo) -> Ranges:
        """Make the standing checkpoint, and anything marked pinned, holes on the overflow path.

        Positions there are positions in an assembled prompt, so the planner
        cannot recognise the checkpoint by ordinal. It is found by its private
        key, or by its bytes when the key was already stripped, and held: it is
        the only thing standing for everything the session has already
        compacted away, and dropping it would lose that outright rather than
        move it.
        """
        wanted = (memo.checkpoint_message or {}).get("content")
        found = [
            index
            for index, message in enumerate(history)
            if message.get(HISTORY_CHECKPOINT_KEY)
            or message.get(HISTORY_PINNED_KEY)
            or (wanted is not None and message.get("content") == wanted)
        ]
        if not found:
            return ranges
        return dataclass_replace(ranges, holes=(*ranges.holes, *(range(index, index + 1) for index in found)))

    def _unscoped(
        self,
        ctx: AssemblyContext,
        prefix: "AssembledPrefix",
        session: list[dict[str, Any]],
    ) -> Segment:
        """A turn with no log revision behind it: budget it, mint nothing.

        Every rung above the trimmer names messages by ordinal in a revision of
        the log, and without one there is nothing those names could resolve
        against -- a ref minted here would promise bytes nobody can produce.
        """
        system_message = {"role": "system", "content": prefix.system_prefix}
        pinned = self.planner.pinned_ids(MessageLedger(session))
        _messages, outcome = self.trimmer.trim(
            session_messages=session,
            ids=list(range(len(session))),
            protected_ids=set(pinned),
            reserved_output=ctx.budget.reserved_output,
            build_messages=lambda history: [system_message, *history, prefix.user_message],
        )
        return Segment(
            text="",
            history=self._marked(outcome, None, 0, pinned),
            meta={
                "path": SEGMENT_PATH,
                "compaction_action": "unscoped",
                "history_kept": len(outcome.included_ids),
                "history_total": len(session),
                "history_tokens": outcome.estimated_tokens,
                "budget_trimmed": len(outcome.warnings),
            },
        )

    def _meta(
        self,
        generation: int,
        checkpoint: CompactionCheckpoint | None,
        plan: CompactionPlan,
        selection: Selection,
        outcome: TrimOutcome,
        pruned: PruneOutcome,
        ranges: Ranges,
        *,
        total: int,
        pruned_total: int,
    ) -> dict[str, Any]:
        return {
            "path": SEGMENT_PATH,
            "compaction_action": plan.action,
            "compaction_trigger": str(plan.trigger),
            "compaction_generation": generation,
            "prompt_tokens": plan.prompt_tokens,
            "trigger_tokens": plan.trigger_tokens,
            "target_tokens": plan.target_tokens,
            "headroom_clamped": plan.headroom_clamped,
            "history_kept": len(outcome.included_ids),
            "history_total": total,
            "history_tokens": outcome.estimated_tokens,
            "pruned_messages": pruned.pruned_count,
            "pruned_total": pruned_total,
            "prune_reclaimed_tokens": pruned.reclaimed_tokens,
            "compacted_away": len(selection.dropped),
            "pinned_held": sum(len(hole) for hole in ranges.holes),
            "budget_trimmed": len(outcome.warnings),
            "summary_mode": self.cfg.summary.mode,
            "checkpoint_kind": checkpoint.kind if checkpoint is not None else "",
            "checkpoint_id": checkpoint.checkpoint_id if checkpoint is not None else "",
            "checkpoint_supersedes": (checkpoint.supersedes or "") if checkpoint is not None else "",
            "checkpoint_covers": checkpoint.covers.render() if checkpoint is not None else "",
            "byte_prefix_reused": checkpoint.byte_prefix_reused if checkpoint is not None else False,
            "cache_status": checkpoint.cache_status if checkpoint is not None else "unobservable",
            "cache_read_tokens": checkpoint.cache_read_tokens if checkpoint is not None else 0,
            "summary_input_tokens": checkpoint.summary_input_tokens if checkpoint is not None else 0,
            "summary_output_tokens": checkpoint.summary_output_tokens if checkpoint is not None else 0,
        }

    def _report(
        self,
        session_key: str,
        plan: CompactionPlan,
        selection: Selection,
        removed: frozenset[int],
        *,
        first: bool,
    ) -> None:
        """Say what left the window, once loudly and then quietly.

        Compaction is the designed path, so a session that has crossed the
        trigger compacts again later and a warning each time would be noise.
        The first one is worth the volume: it is the turn on which a reader
        stops being able to see the earlier conversation directly.
        """
        say = logger.warning if first else logger.info
        say(
            "Compacted {} of {} message(s) for {} at {}/{} prompt tokens",
            len(removed),
            len(removed) + len(selection.kept),
            session_key,
            plan.prompt_tokens,
            plan.trigger_tokens,
        )
        if selection.exhausted:
            logger.warning(
                "Compaction could not reach its target of {} tokens: the protected head and tail already "
                "exceed it, so this window cannot host the configured retention policy",
                plan.target_tokens,
            )


def _checkpoint_position(
    included: Sequence[int],
    selection: Selection,
    checkpoint_message: dict[str, Any] | None,
) -> int:
    """Where in the history the checkpoint goes: where its messages were.

    Between the protected head and the tail, which is also what puts it at
    the front of the range the next compaction removes -- so folding the
    last summary into the next one needs no record of where it went.
    """
    if checkpoint_message is None or not selection.dropped:
        return 0
    return sum(1 for index in included if index < min(selection.dropped))


def _splice(items: Sequence[Any], item: Any, at: int) -> list[Any]:
    """``items`` with ``item`` inserted at ``at``, clamped into range."""
    listed = list(items)
    if item is None:
        return listed
    position = max(0, min(at, len(listed)))
    return [*listed[:position], item, *listed[position:]]


def _comparable(message: dict[str, Any]) -> dict[str, Any]:
    """A message as the provider-facing request carries it.

    Cache breakpoints and the history markers are both added or removed on
    the way out, so neither is evidence that a message changed.
    """
    stripped, _tools = prompt_cache.strip([message], None)
    return {key: value for key, value in stripped[0].items() if key not in HISTORY_PRIVATE_KEYS}


def _carry_private_keys(
    source: list[dict[str, Any]],
    ids: list[int],
    projected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """``projected`` with the history markers of the messages it came from.

    Projection keeps provider-safe keys only, and a reclaimed prompt still has
    to tell in-turn window management which messages are not its to touch.
    """
    if len(projected) != len(ids):
        return projected
    out: list[dict[str, Any]] = []
    for index, message in zip(ids, projected):
        marks = {key: source[index][key] for key in HISTORY_PRIVATE_KEYS if source[index].get(key)}
        out.append({**message, **marks} if marks else message)
    return out


def _wire_model(provider: Any, model: str) -> str:
    """The id the request will go out under -- what the headroom table keys on."""
    resolve_wire = getattr(provider, "wire_model_id", None)
    return resolve_wire(model) if callable(resolve_wire) else model


__all__ = ["SEGMENT_PATH", "CompactingHistoryBuilder"]
