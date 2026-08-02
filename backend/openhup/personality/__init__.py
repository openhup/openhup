"""The personality gamble: a voice drawn at setup, kept a mystery until revealed (ADR-014).

On a fresh install with `personality.gamble` enabled, one personality is drawn from the pool at
random and becomes the effective default without being announced. The household discovers the
voice by living with it; Settings is the reveal surface, and the ADR and the preset file are the
documentation. The draw is a single row in `personality_draw`, so it survives restarts, can be
re-drawn any number of times, and can be deleted to return to the configured default.

Two invariants keep the gamble honest:

* **A draw is never clamped in secret.** The pool is restricted to personalities at intensity 3
  or below, so the default `humor_ceiling: 3` / `roast_consent: false` can never silently tone
  down what was drawn. If an operator raises the pool, the draw validates against the loaded
  personalities and refuses ids that would be clamped rather than drawing them anyway.
* **The draw is an override, not a replacement.** `default_personality` in config stays the
  operator's choice; the draw shadows it in memory. Deleting the draw restores it exactly.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from ..db import PersonalityDrawRow

#: The shipped pool. Every id is a preset in examples/personalities/personalities.yaml, all at
#: intensity <= 3, so a draw works under the default humor_ceiling without roast_consent.
GAMBLE_POOL: tuple[str, ...] = ("friendly", "shy", "sassy", "sarcastic", "angry")

#: Single-row key. There is only ever one draw per install.
DRAW_ID = "default"


async def load_draw(session: AsyncSession) -> PersonalityDrawRow | None:
    """The current draw, or None when no gamble has happened."""
    return await session.get(PersonalityDrawRow, DRAW_ID)


def effective_default_id(configured: str, draw: PersonalityDrawRow | None) -> str:
    """The personality that actually speaks: the draw when one exists, else the config default."""
    if draw is None:
        return configured
    return draw.personality_id


async def draw(
    session: AsyncSession,
    *,
    pool: Sequence[str],
    available: Sequence[str],
) -> PersonalityDrawRow:
    """Draw (or re-draw) the mystery personality.

    `pool` is the operator's configured gamble pool; `available` is the set of personalities that
    actually loaded. Ids that are not loaded are skipped silently, and a drawn id is refused if
    it does not exist - an unknown id would silently fall back to the config default, which
    defeats the point of a gamble. Raises `ValueError` when the pool is empty after filtering.
    """
    valid = [personality_id for personality_id in pool if personality_id in set(available)]
    if not valid:
        raise ValueError(
            "the personality gamble pool is empty or none of its ids are loaded personalities. "
            f"Pool: {list(pool)}"
        )
    chosen = secrets.choice(valid)
    row = await load_draw(session)
    if row is None:
        row = PersonalityDrawRow(id=DRAW_ID, personality_id=chosen)
        session.add(row)
    else:
        row.personality_id = chosen
        row.reroll_count += 1
    await session.flush()
    return row


async def clear(session: AsyncSession) -> bool:
    """Delete the draw; returns True when there was one. The config default takes over."""
    row = await load_draw(session)
    if row is None:
        return False
    await session.delete(row)
    await session.flush()
    return True


__all__ = ["DRAW_ID", "GAMBLE_POOL", "clear", "draw", "effective_default_id", "load_draw"]
