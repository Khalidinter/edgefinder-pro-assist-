"""Helpers shared by the live Odds API fetchers (paper_trade.py and
rebound_paper_trade.py). Pure stdlib so they can be unit-tested without the
ML stack installed.

Quota rules for api.the-odds-api.com/v4 that matter to this pipeline:
  * GET /sports/{sport}/events is free.
  * GET /sports/{sport}/events/{id}/odds costs [markets] x [regions] credits per
    call, and a `bookmakers` list of up to 10 books counts as one region. Our
    calls (one market, DraftKings only) therefore cost 1 credit per event.
  * Every response carries x-requests-remaining / x-requests-used headers.

The events endpoint lists every upcoming game it knows about, often several
days out, while DraftKings posts player props roughly a day ahead. Fetching
props for games 3+ days away mostly returns nothing but still burns credits, so
the fetchers only query events tipping off within ODDS_LOOKAHEAD_HOURS.
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

from lib.config import logger

DEFAULT_LOOKAHEAD_HOURS = 48.0


def lookahead_hours() -> Optional[float]:
    """Window from ODDS_LOOKAHEAD_HOURS. Unset -> 48h. 0 or negative -> no limit."""
    raw = os.getenv("ODDS_LOOKAHEAD_HOURS", "").strip()
    if not raw:
        return DEFAULT_LOOKAHEAD_HOURS
    try:
        hours = float(raw)
    except ValueError:
        logger.warning("ODDS_LOOKAHEAD_HOURS=%r is not a number; using %.0fh", raw, DEFAULT_LOOKAHEAD_HOURS)
        return DEFAULT_LOOKAHEAD_HOURS
    return hours if hours > 0 else None


def parse_commence_time(value) -> Optional[datetime]:
    """Parse the API's ISO-8601 commence_time ('2026-10-21T23:30:00Z') to aware UTC."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def filter_events_by_lookahead(events: Iterable[Dict], hours: Optional[float] = None,
                               now: Optional[datetime] = None) -> List[Dict]:
    """Drop events that tip off more than `hours` after `now`.

    Events with a missing or unparseable commence_time are kept (fail open), as
    are games that have already started — the resolver decides what to do with
    those. `hours=None` disables filtering.
    """
    events = list(events)
    if hours is None:
        hours = lookahead_hours()
    if hours is None:
        return events
    now = now or datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours)
    kept, dropped = [], 0
    for ev in events:
        ct = parse_commence_time(ev.get("commence_time"))
        if ct is not None and ct > cutoff:
            dropped += 1
            continue
        kept.append(ev)
    if dropped:
        logger.info("Skipping %d event(s) tipping off more than %.0fh out (saves %d credit(s) per market)",
                    dropped, hours, dropped)
    return kept


def log_quota(response, context: str = "") -> None:
    """Log the Odds API quota headers from the most recent response."""
    if response is None:
        return
    remaining = response.headers.get("x-requests-remaining")
    used = response.headers.get("x-requests-used")
    if remaining is None and used is None:
        return
    suffix = f" ({context})" if context else ""
    logger.info("Odds API quota%s: %s credits remaining, %s used this billing period",
                suffix, remaining, used)
