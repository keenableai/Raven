"""Turn scratch — the per-turn block that reaches the model but not the log.

Recalled memory and routed skills change with every query. Rendered into the
system prompt they sit ahead of the whole conversation in every provider's
cache prefix, so each turn discards the cached history behind them; rendered at
the end of the turn's user message they sit behind it, and the history stays
cacheable.

The block must not be written back to ``session.messages``. Persisting it would
turn the log from append-only into rewritten -- next turn's prefix would differ
from this turn's, losing the cache anyway, and the recalled text would
accumulate in history forever. The loop's ``_save_turn`` strips it on the way
in, the same way it already strips the runtime-context header.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Same bracketed-sentence shape as ContextBuilder._RUNTIME_CONTEXT_TAG, and for
# the same two reasons: it tells the model what it is reading, and it gives the
# persistence path a literal to cut on.
TURN_SCRATCH_TAG = "[Turn Scratch — recalled for this turn only, not conversation history]"


def render_turn_scratch(parts: Sequence[str]) -> str:
    """Join segment texts into one tagged block, or ``""`` when there is none."""
    body = "\n\n---\n\n".join(part for part in parts if part)
    return f"{TURN_SCRATCH_TAG}\n\n{body}" if body else ""


def append_turn_scratch(content: Any, scratch: str) -> Any:
    """Append ``scratch`` to a user message's content, keeping its shape.

    A string stays a string. Converting it to a one-element block list would be
    irreversible downstream: ``prompt_cache.strip`` only unwraps content that is
    exactly one text block, and "content must be a string" is among the
    commonest ways an OpenAI-compatible endpoint refuses a request.
    """
    if not scratch:
        return content
    if isinstance(content, str):
        return f"{content}\n\n{scratch}"
    if isinstance(content, list):
        return [*content, {"type": "text", "text": scratch}]
    return content


def strip_turn_scratch(content: Any) -> Any | None:
    """Content with the scratch removed, or ``None`` when nothing else is left.

    ``None`` is the caller's cue to skip the message entirely rather than record
    an empty one.
    """
    if isinstance(content, str):
        head, sep, _ = content.partition(TURN_SCRATCH_TAG)
        if not sep:
            return content
        return head.rstrip() or None
    if isinstance(content, list):
        kept = [block for block in content if not _is_scratch_block(block)]
        if len(kept) == len(content):
            return content
        return kept or None
    return content


def _is_scratch_block(block: Any) -> bool:
    return isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].startswith(TURN_SCRATCH_TAG)


__all__ = [
    "TURN_SCRATCH_TAG",
    "append_turn_scratch",
    "render_turn_scratch",
    "strip_turn_scratch",
]
