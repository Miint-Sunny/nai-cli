"""stats: the ledger's requests, images and spend over a range of days."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from nai_cli import errors, ledger
from nai_cli.commands.context import Context

DEFAULT_DAYS = 30


@dataclass(frozen=True)
class StatsSettings:
    since: date
    until: date
    by: str


def resolve(opts: dict[str, Any], ctx: Context) -> StatsSettings:
    if opts.get("days") is not None and opts.get("since"):
        raise errors.NaiError(errors.BAD_ARGS, "give --days or --since, not both")
    until = _day(opts.get("until"), "--until") or ctx.now().date()
    days = DEFAULT_DAYS if opts.get("days") is None else opts["days"]
    if days < 1:
        raise errors.NaiError(errors.BAD_ARGS, f"--days needs at least 1: {days}")
    since = _day(opts.get("since"), "--since") or _days_before(until, days - 1)
    if since > until:
        raise errors.NaiError(errors.BAD_ARGS, f"--since {since} comes after --until {until}")
    return StatsSettings(since, until, opts.get("by") or "day")


def _days_before(day: date, days: int) -> date:
    # A window reaching before year 1 starts there
    try:
        return day - timedelta(days=days)
    except OverflowError:
        return date.min


def _day(text: str | None, option: str) -> date | None:
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise errors.NaiError(errors.BAD_ARGS, f"{option} takes YYYY-MM-DD: {text!r}") from None


def run(s: StatsSettings, ctx: Context) -> dict[str, Any]:
    try:
        return {"ok": True, **ledger.summary(s.since, s.until, s.by)}
    except OSError as e:
        raise errors.NaiError(errors.LEDGER, f"cannot read the ledger: {e}") from e
