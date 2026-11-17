"""The spend ledger: one line after each paid request, read back for stats and for the request that saved a file."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from nai_cli import config, errors, fs, values


def ledger_path() -> Path:
    return config.data_dir() / "ledger.jsonl"


def record(entry: dict[str, Any], at: datetime) -> None:
    line = {"time": at.astimezone().isoformat(timespec="seconds"), **entry}
    fs.append(ledger_path(), values.to_json(line) + "\n")


def entries() -> Iterator[dict[str, Any]]:
    """Each request recorded, in order; a line that is not one, such as one cut short, is skipped."""
    path = ledger_path()
    if not fs.is_file(path):
        return
    for raw in path.read_bytes().split(b"\n"):
        try:
            line = values.parse_json(raw.decode("utf-8"))
        except ValueError:
            continue
        if isinstance(line, dict) and isinstance(line.get("time"), str):
            yield line


def spent(entry: dict[str, Any]) -> int | None:
    value = entry.get("anlas_spent")
    return value if values.is_whole(value) else None


def images(entry: dict[str, Any]) -> int:
    value = entry.get("images_requested")
    return max(value, 0) if values.is_whole(value) else len(files(entry))


def files(entry: dict[str, Any]) -> list[str]:
    value = entry.get("files")
    return [f for f in value if isinstance(f, str)] if isinstance(value, list) else []


GROUPINGS = ("day", "command", "model")


def summary(since: date, until: date, by: str) -> dict[str, Any]:
    """Totals of the requests between two local days, inclusive, grouped by day, command or model."""
    totals = _Totals()
    groups: dict[str, _Totals] = {}
    for entry in entries():
        day = _day_of(entry)
        if day is None or not since <= day <= until:
            continue
        key = {"day": day.isoformat(), "command": entry.get("command"), "model": entry.get("model")}[by] or "-"
        totals.add(entry)
        groups.setdefault(str(key), _Totals()).add(entry)
    return {
        "since": since.isoformat(),
        "until": until.isoformat(),
        **totals.to_json(),
        "by": by,
        "groups": {key: groups[key].to_json() for key in sorted(groups)},
    }


def _day_of(entry: dict[str, Any]) -> date | None:
    """The day on this computer's clock, whatever time zone it was recorded in."""
    try:
        moment = datetime.fromisoformat(entry["time"])
        return (moment.astimezone() if moment.tzinfo else moment).date()
    except (OSError, ValueError, OverflowError):
        return None


@dataclass
class _Totals:
    requests: int = 0
    images: int = 0
    anlas_spent: int = 0
    # Requests whose closing balance went unread
    unknown: int = 0
    free_images: int = 0

    def add(self, entry: dict[str, Any]) -> None:
        self.requests += 1
        self.images += images(entry)
        value = spent(entry)
        if value is None:
            self.unknown += 1
            return
        self.anlas_spent += value
        if value == 0:
            self.free_images += images(entry)

    def to_json(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "images": self.images,
            "anlas_spent": self.anlas_spent,
            "free_images": self.free_images,
            "requests_with_unknown_spend": self.unknown,
        }


def by_file() -> dict[Path, dict[str, Any]]:
    """The record that saved each file; a later record of the same path wins."""
    found: dict[Path, dict[str, Any]] = {}
    for e in entries():
        for f in files(e):
            path = _resolved(f)
            if path is not None:
                found[path] = e
    return found


def _resolved(path: str | Path) -> Path | None:
    # Entries may name drives that are gone or paths this system cannot represent
    try:
        return Path(path).resolve()
    except (OSError, ValueError):
        return None


@contextmanager
def generation_lock(timeout: float = 600) -> Iterator[None]:
    """An account runs one generation at a time; concurrent processes queue here."""
    path = config.data_dir() / "generate.lock"
    try:
        held = fs.acquire(path, timeout, poll=1)
    except TimeoutError:
        raise errors.NaiError(errors.BUSY, f"another generation is still running after {int(timeout)} s") from None
    except OSError as e:
        # Taken before the request is priced, so nothing has been sent yet
        raise errors.NaiError(errors.LEDGER, f"cannot use the generation lock {path}: {e}") from e
    try:
        yield
    finally:
        fs.release(held)
