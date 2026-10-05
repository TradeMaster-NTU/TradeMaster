"""Generate a macro-event feature for TradeMaster experiments.

FXMacroData's public release-calendar endpoint can be used as a simple event
risk feature. The helper below returns tier-one USD announcement dates that can
be merged into a research dataset as a binary ``macro_blackout`` column.
"""

from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import json
import os
from typing import Any, Dict, Iterable, List, Set
from urllib.parse import urlencode
from urllib.request import Request, urlopen


FXMD_CALENDAR_URL = "https://api.fxmacrodata.com/v1/calendar/{currency}"


def fetch_release_events(
    currency: str,
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    params = urlencode({"start_date": start_date, "end_date": end_date})
    url = f"{FXMD_CALENDAR_URL.format(currency=currency.upper())}?{params}"
    api_key = (os.getenv("FXMACRODATA_API_KEY") or "").strip()
    if any(ch.isspace() or not ch.isprintable() for ch in api_key):
        raise ValueError("FXMACRODATA_API_KEY contains whitespace or control characters")

    request = Request(url)
    if api_key:
        # Unredirected headers are not copied onto a redirected request.
        request.add_unredirected_header("X-API-Key", api_key)
    with urlopen(request, timeout=20) as response:
        payload = json.load(response)

    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        detail = payload.get("detail") if isinstance(payload, dict) else None
        raise ValueError(f"Unexpected FXMacroData calendar response: {detail or 'missing data'}")
    return [row for row in rows if isinstance(row, dict)]


def build_blackout_dates(
    events: Iterable[Dict[str, Any]],
    min_market_tier: int = 1,
    window_days: int = 0,
) -> Set[str]:
    blackout_dates: Set[str] = set()

    for event in events:
        if not event.get("release_date_confirmed"):
            continue

        market_tier = event.get("market_tier")
        if market_tier is None or int(market_tier) > min_market_tier:
            continue

        announcement_ts = event.get("announcement_datetime")
        if announcement_ts is None:
            continue

        event_date = datetime.fromtimestamp(
            int(announcement_ts),
            tz=timezone.utc,
        ).date()
        for offset in range(-window_days, window_days + 1):
            blackout_dates.add((event_date + timedelta(days=offset)).isoformat())

    return blackout_dates


def macro_blackout_feature(trading_date: date, blackout_dates: Set[str]) -> int:
    return int(trading_date.isoformat() in blackout_dates)


# Example:
#
# events = fetch_release_events("USD", "2026-07-01", "2026-07-31")
# blackout_dates = build_blackout_dates(events, min_market_tier=1, window_days=0)
# feature_value = macro_blackout_feature(datetime(2026, 7, 14).date(), blackout_dates)
