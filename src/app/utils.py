from __future__ import annotations

from datetime import datetime, timezone


def resolve_reference_now(now: datetime | None) -> datetime:
    """Resolve a timezone-aware 'now' for local-date comparisons.

    Defaults to the local now when `now` is omitted; naive values are
    assumed UTC.
    """
    reference_now = now if now is not None else datetime.now().astimezone()
    if reference_now.tzinfo is None:
        reference_now = reference_now.replace(tzinfo=timezone.utc)
    return reference_now
