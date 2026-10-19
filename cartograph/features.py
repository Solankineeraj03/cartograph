"""Reference as-of lookup. Production Feast/export integration is future work."""

from datetime import datetime


def _timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return timestamp


def latest_available(rows: list[dict], product_id: str, cutoff: str) -> dict | None:
    """Choose latest event known by cutoff, resolving corrections by availability."""
    cutoff_time = _timestamp(cutoff)
    candidates = []
    for row in rows:
        if row["product_id"] != product_id:
            continue
        event = _timestamp(row["event_time"])
        available = _timestamp(row["available_at"])
        if event <= cutoff_time and available <= cutoff_time:
            candidates.append((event, available, row))
    if not candidates:
        return None
    best_key = max((event, available) for event, available, _ in candidates)
    winners = [row for event, available, row in candidates if (event, available) == best_key]
    if any(row != winners[0] for row in winners[1:]):
        raise ValueError("Conflicting feature rows at the same event/availability time")
    return winners[0].copy()
