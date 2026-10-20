"""What a turn produced, kept apart from the window its model calls see.

The loop appends every message of a turn to the list it sends, and the window
passes in :mod:`raven.agent.window.shrink` make that list smaller mid-turn: a
head summary replaces a stretch of it with one user message, an elision swaps
older tool bodies for a placeholder, a picture is withdrawn into a note. Those
are decisions about what the next request can carry. They are not decisions
about what the turn said, and persisting the turn from the shrunken list filed
the summary as a user message, lost whatever the summary covered, and stored
placeholders in place of the output they stood for.

:class:`TurnLog` follows the live list and keeps the turn's own record. Each
message the turn adds is stamped when the log first sees it; the window
passes copy a message with ``dict(m)`` when they rewrite it, so the stamp rides
along and the log can tell a rewrite of a turn message from a message the
window made up. A rewrite replaces the logged version when it is still the
message (a withdrawn picture's note is what the session keeps), and is ignored
when it is a stand-in (an elided body). Messages the window made up -- the
head summary, the user message carrying demoted pictures -- carry no stamp
and never reach the log.

The log has to see every list the loop works on before a window pass reads it
and after the pass hands one back: ``follow`` is called on both sides, so the
pass's copies of a turn message already carry the stamp, and the list it
returns is adopted with nothing appended to the old one lost. Messages removed
by a hook rollback were appended after the last ``follow`` and are never seen,
which is what a rollback means.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from raven.agent.window.shrink import ELIDED_TOOL_OUTPUT

TURN_LOG_KEY = "_turn_log_seq"
"""The stamp. Underscored so no provider payload carries it (every transport
sends an allowlist of keys) and dropped by the loop before a message is filed."""


class TurnLog:
    """The record of one attempt at a turn.

    ``base[:start]`` is what the turn opened with and is already filed (the
    history and the turn's question); ``base[start:]`` -- the mid-turn messages
    a rerun carries forward -- is the first thing the log records. The stamp's
    value is unique to this log, so a message copied from another attempt is
    recorded afresh rather than mistaken for one this log already holds.
    """

    def __init__(self, base: list[dict[str, Any]], start: int) -> None:
        self._head = list(base[:start])
        self._token = uuid.uuid4().hex
        self._entries: list[dict[str, Any]] = []
        self._index: dict[str, int] = {}
        self._live: list[dict[str, Any]] = base
        self._mark = start
        self._sync(base)

    @property
    def start(self) -> int:
        return len(self._head)

    def follow(self, messages: list[dict[str, Any]]) -> None:
        """Adopt ``messages`` as the list the loop works on from here.

        The list being replaced is read first, so what was appended to it since
        the last call is recorded. The new list's own unstamped messages are not
        the turn's -- they are what the window pass just wrote -- so only its
        stamped ones are read, for rewrites.
        """
        if messages is not self._live:
            self._sync(self._live)
            self._live = messages
            self._mark = len(messages)
        self._sync(messages)

    def messages(self, transient_keys: Iterable[str] = ()) -> list[dict[str, Any]]:
        """The turn as it happened: the opening messages, then every message the
        turn added, in order, each in its latest version that is still itself.

        ``transient_keys`` names marks of loop scaffolding to leave out, the way
        the loop's own return value leaves them out.
        """
        self._sync(self._live)
        keys = tuple(transient_keys)
        kept = [m for m in self._entries if not any(m.get(k) for k in keys)]
        return [*self._head, *kept]

    def _sync(self, messages: list[dict[str, Any]]) -> None:
        for i, m in enumerate(messages):
            seq = m.get(TURN_LOG_KEY)
            if isinstance(seq, str) and seq.startswith(self._token):
                at = self._index.get(seq)
                if at is not None and self._entries[at] is not m and not _stands_in(m):
                    self._entries[at] = m
                continue
            if i < self._mark:
                continue
            seq = f"{self._token}:{len(self._entries)}"
            m[TURN_LOG_KEY] = seq
            self._index[seq] = len(self._entries)
            self._entries.append(m)
        self._mark = len(messages)


def _stands_in(message: dict[str, Any]) -> bool:
    """A rewrite that replaces what the message said rather than restating it."""
    return message.get("role") == "tool" and message.get("content") == ELIDED_TOOL_OUTPUT


__all__ = ["TURN_LOG_KEY", "TurnLog"]
