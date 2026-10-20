"""The session log's side of the ``[[ref ...]]`` contract.

:mod:`raven.utils.refs` owns the token grammar; this module owns what the
``session`` store puts inside one. Two parties need to agree on it and neither
may import the other: the tool-result pruner mints a ref when it replaces a
result body with a one-line call record, and ``context_retrieve`` resolves that
ref back to the verbatim original. Both import this.

The scope is the log's :class:`SessionRevision` scope, ``<log_id>@<generation>``.
The grammar's third convention asks that "a store whose sequence can restart must
fold the generation into ``scope``, so a stale ref resolves to nothing rather
than to the wrong entry". ``session.messages`` is append-only within a
generation, so an ordinal is a stable name for a message; ``/new`` starts a log
with a fresh ``log_id`` and an undo bumps ``generation``, so a ref minted before
either names a scope that no longer exists -- a miss rather than a wrong answer.
"""

from __future__ import annotations

from typing import Any

from raven.contracts.context import SessionRevision
from raven.utils.refs import SourceRef

SESSION_REF_STORE = "session"
"""The ``store`` name for refs into ``session.messages``."""

TOOL_RESULT_TRUNCATION_MARKER = "\n... (truncated)"
"""What the loop's ``_save_turn`` appends to a tool result it had to cut.

Machine-recognisable on purpose: ``context.toolResultMaxChars`` defaults high
enough that nothing written today is cut, but history written under the old
16000-character cap is not recoverable, and retrieval has to say so instead of
returning a prefix as though it were the whole thing.
"""


def session_ref(revision: SessionRevision, lo: int, hi: int | None = None) -> SourceRef:
    """A ref into ``revision`` of the log, covering ordinals ``lo``..``hi``."""
    return SourceRef(store=SESSION_REF_STORE, scope=revision.scope, lo=lo, hi=lo if hi is None else hi)


def belongs_to(ref: SourceRef, revision: SessionRevision) -> bool:
    """Whether ``ref`` names entries of this revision of the log."""
    return ref.store == SESSION_REF_STORE and ref.scope == revision.scope


def is_truncated(content: Any) -> bool:
    """Whether a stored tool result was cut on the way to disk."""
    return isinstance(content, str) and content.endswith(TOOL_RESULT_TRUNCATION_MARKER)


__all__ = [
    "SESSION_REF_STORE",
    "TOOL_RESULT_TRUNCATION_MARKER",
    "belongs_to",
    "is_truncated",
    "session_ref",
]
