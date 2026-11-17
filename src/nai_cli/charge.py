"""Paid requests: priced against the balance, confirmed above the per-request limit, and written to the ledger even
when they fail."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, NoReturn, TypeVar

from nai_cli import api, config, cost, errors, fs, ledger

T = TypeVar("T")


@dataclass
class Charge:
    """Paid requests made under the generation lock, measured by the balance before and after them."""

    client: api.Client
    # Written to the ledger with the spend; callers add what they know before sending
    entry: dict[str, Any]
    warnings: list[str]
    now: Callable[[], datetime]
    after: api.Account | None = None
    # Set once a paid request ran or may have run; one refused or never sent costs nothing
    paid: bool = False
    # Images go to the ledger; saved lists every file written, for the error output
    files: list[str] = field(default_factory=list)
    saved: list[str] = field(default_factory=list)

    @cached_property
    def before(self) -> api.Account:
        # Read when first needed, so a run that finds nothing to pay for sends nothing
        return self.client.account()

    @property
    def spent(self) -> int | None:
        if not self.paid:
            return 0
        if self.after is None or not (self.before.measured and self.after.measured):
            return None
        return self.before.anlas - self.after.anlas

    def pay(self, request: Callable[..., T], *args: Any) -> T:
        # The spend is measured from here, so the opening balance must be known first
        _ = self.before
        try:
            result = request(*args)
        except errors.NaiError as e:
            self.paid = self.paid or bool(e.detail.get("maybe_processed"))
            raise
        except BaseException:
            self.paid = True
            raise
        self.paid = True
        return result

    def keep(self, data: bytes | str, path: Path, image: bool = True) -> str:
        """Save a paid-for result under path or the next free number; a failure is save_failed, since the request
        has already run."""
        try:
            written = fs.create_numbered(path, data)
        except (OSError, ValueError) as e:
            raise errors.NaiError(errors.SAVE_FAILED, f"the request ran, but its result could not be saved: {e}") from e
        self.saved.append(str(written))
        if image:
            self.files.append(str(written))
        return str(written)

    def settle(self, error: BaseException | None) -> bool:
        """Read the closing balance and write the ledger; returns whether Ctrl+C arrived meanwhile, which
        neither step lets lose the spend."""
        interrupted = False
        try:
            self.after = self.client.account()
        except (errors.NaiError, KeyboardInterrupt) as e:
            # Ctrl+C while waiting to retry the read arrives as an interrupted error
            interrupted = isinstance(e, KeyboardInterrupt) or e.code == errors.INTERRUPTED
            reason = "interrupted" if interrupted else e.message
            self.warnings.append(f"the balance could not be read after the request, so the spend is unknown: {reason}")
        else:
            if self.paid and self.spent is None:
                self.warnings.append("the account does not report its balance, so the spend is unknown")
        try:
            ledger.record(self._record(error), self.now())
        except OSError as e:
            self.warnings.append(f"the ledger could not be written: {e}")
        except KeyboardInterrupt:
            interrupted = True
            self.warnings.append("interrupted while writing the ledger, so this request may be missing from it")
        return interrupted

    def _record(self, error: BaseException | None) -> dict[str, Any]:
        record = {
            **self.entry,
            "files": self.files,
            "anlas_spent": self.spent,
            "quota_percent": self.after.quota_percent if self.after else None,
        }
        if error is not None:
            record["error"] = error_code(error)
        return record

    def report(self) -> dict[str, Any]:
        # The request ran, so sending it again may pay twice
        detail: dict[str, Any] = {"cost": balance_json(None, self), "maybe_processed": True}
        if self.saved:
            detail["saved"] = list(self.saved)
        if self.warnings:
            detail["warnings"] = list(self.warnings)
        return detail


def error_code(error: BaseException) -> str:
    if isinstance(error, errors.NaiError):
        return error.code
    return errors.INTERRUPTED if isinstance(error, KeyboardInterrupt) else errors.INTERNAL


@contextmanager
def charged(
    client: api.Client,
    warnings: list[str],
    entry: dict[str, Any],
    now: Callable[[], datetime],
    on_interrupt: Callable[[], None],
) -> Iterator[Charge]:
    """Hold the generation lock around paid requests so other processes cannot skew the spend. The opening balance
    prices and confirms them; once one has run, the closing balance is read and the ledger written even if a later
    step fails or is interrupted, and the error then reports the spend and the files already saved."""
    with ledger.generation_lock():
        charge = Charge(client, dict(entry), warnings, now)
        try:
            yield charge
        except (Exception, KeyboardInterrupt) as e:
            if not charge.paid:
                raise
            # The first failure keeps its code; Ctrl+C while settling stops the batch after it
            if charge.settle(e):
                on_interrupt()
            detail = charge.report()
            if isinstance(e, errors.NaiError):
                e.detail.update(detail)
                raise
            message = "interrupted" if isinstance(e, KeyboardInterrupt) else f"{type(e).__name__}: {e}"
            raise errors.NaiError(error_code(e), message, **detail) from e
        if charge.paid and charge.settle(None):
            raise errors.NaiError(errors.INTERRUPTED, "interrupted", **charge.report())


def authorize(
    price: int, images: int, charge: Charge, cfg: config.Config, yes: bool, cost_json: dict[str, Any]
) -> None:
    """Before anything is sent: the balance must cover the estimate, so a Vibe is not encoded for a generation
    that would be refused, and a large estimate needs --yes."""
    charge.entry["anlas_estimate"] = price
    charge.entry["images_requested"] = images
    _allow(price, charge.before, cfg, yes, cost_json)


def refusal(
    price: int, account: api.Account | None, cfg: config.Config, yes: bool, cost_json: dict[str, Any]
) -> dict[str, Any] | None:
    """For a dry run: the refusal the real run would meet before sending, by the balance read now if any."""
    try:
        _allow(price, account, cfg, yes, cost_json)
    except errors.NaiError as e:
        return {"code": e.code, "message": e.message}
    return None


def _allow(price: int, account: api.Account | None, cfg: config.Config, yes: bool, cost_json: dict[str, Any]) -> None:
    if account is not None and account.measured and price > account.anlas:
        raise errors.NaiError(
            errors.INSUFFICIENT_ANLAS,
            f"estimated {price} Anlas, but the balance is {account.anlas}",
            cost=cost_json,
        )
    if price > cfg.max_per_request and not yes:
        raise errors.NaiError(
            errors.NEED_CONFIRM,
            f"estimated {price} Anlas exceeds max_per_request ({cfg.max_per_request})",
            cost=cost_json,
        )


def keep_usable(error: api.DamagedImages, save: Callable[[list[bytes | None]], Any], warnings: list[str]) -> NoReturn:
    """Save the images of a partly damaged reply, then report the damage, or a save failure with the damage
    noted in warnings."""
    try:
        save(error.images)
    except errors.NaiError:
        warnings.append(error.message)
        raise
    raise error


def balance_json(account: api.Account | None, charge: Charge | None, quota: bool = False) -> dict[str, Any]:
    """After a paid request: the spend and the closing balance, either of which may be unknown."""
    out: dict[str, Any] = {}
    if charge is not None:
        out["spent"] = charge.spent
        account = charge.after
        if account is None:
            return out
    if account is None:
        out["assumed_opus"] = True
        return out
    if not account.measured:
        out["balance_known"] = False
        return out
    out["anlas_left"] = account.anlas
    if quota:
        out["quota_percent"] = account.quota_percent
        if account.quota_negative:
            out["quota_exhausted"] = True
            out["warning"] = "the V5 free quota is exhausted, so normally free images are charged"
    return out


def cost_json(estimate: cost.Estimate, account: api.Account | None, charge: Charge | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"estimate": estimate.anlas, "free_by_opus": estimate.free_by_opus}
    if estimate.free_by_opus and estimate.if_quota_exhausted != estimate.anlas:
        out["if_quota_exhausted"] = estimate.if_quota_exhausted
    return {**out, **balance_json(account, charge, quota=True)}


def price_json(price: int, account: api.Account | None, charge: Charge | None = None) -> dict[str, Any]:
    return {"estimate": price, **balance_json(account, charge)}
