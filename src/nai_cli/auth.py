"""Accounts: the config's tokens, and email logins kept as an access key and a token renewed from it."""

import base64
import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from argon2.low_level import Type, hash_secret_raw

from nai_cli import api, config, errors, fs, values

DEFAULT = config.DEFAULT_ACCOUNT
TOKEN_LIFETIME = timedelta(days=30)
# Renewed this long before it runs out, so a long batch does not lose it halfway
RENEW_BEFORE = timedelta(days=1)


@dataclass(frozen=True)
class Credentials:
    name: str
    token: str
    endpoint: api.Endpoint


def access_key(email: str, password: str) -> str:
    """As the official login derives it: Argon2id over the password, salted with a BLAKE2b hash of the start of
    the password and the lowercased email."""
    salt = hashlib.blake2b(f"{_js_prefix(password, 6)}{email.lower()}novelai_data_access_key".encode(), digest_size=16)
    raw = hash_secret_raw(
        password.encode(),
        salt.digest(),
        time_cost=2,
        memory_cost=2_000_000 // 1024,
        parallelism=1,
        hash_len=64,
        type=Type.ID,
    )
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")[:64]


def _js_prefix(text: str, units: int) -> str:
    """The first units UTF-16 code units, as String.slice cuts them; half a surrogate pair encodes as U+FFFD."""
    out, used = [], 0
    for ch in text:
        size = 2 if ord(ch) > 0xFFFF else 1
        if used + size > units:
            if used < units:
                out.append("�")
            break
        out.append(ch)
        used += size
    return "".join(out)


def credentials_path() -> Path:
    return config.data_dir() / "accounts.json"


def login(name: str, email: str, password: str, cfg: config.Config, now: datetime) -> dict[str, Any]:
    """Sign in to the official service and keep the access key and token; the password is not kept."""
    if name == DEFAULT or name in cfg.accounts:
        raise errors.NaiError(errors.BAD_ARGS, f"{name} is an account of the config file; choose another name")
    key = access_key(email.strip(), password)
    token = api.login(api.OFFICIAL, key)
    saved = {"email": email.strip().lower(), "access_key": key, **_token(token, now)}
    with _editing() as store:
        store["logins"][name] = saved
    return {"account": name, "email": saved["email"], "expires": saved["expires"]}


def logout(name: str, dry_run: bool = False) -> bool:
    """With dry_run, only whether there is such a login to sign out of."""
    if dry_run:
        return name in _load()["logins"]
    with _editing() as store:
        if name not in _valid(store["logins"]):
            return False
        del store["logins"][name]
        if store.get("current") == name:
            store["current"] = None
    return True


def use(name: str, cfg: config.Config, dry_run: bool = False) -> None:
    if name not in names(cfg):
        raise errors.NaiError(errors.BAD_ARGS, f"no account named {name}", choices=names(cfg))
    if dry_run:
        return
    with _editing() as store:
        # Checked again under the lock, as another nai may have signed out of it meanwhile
        if name != DEFAULT and name not in cfg.accounts and name not in _valid(store["logins"]):
            raise errors.NaiError(errors.BAD_ARGS, f"no account named {name}", choices=names(cfg))
        store["current"] = None if name == DEFAULT else name


def current() -> str:
    return _load()["current"] or DEFAULT


def names(cfg: config.Config) -> list[str]:
    return [DEFAULT, *cfg.accounts, *(n for n in _load()["logins"] if n not in cfg.accounts)]


def listing(cfg: config.Config) -> list[dict[str, Any]]:
    """Each account and how it signs in, never its token."""
    logins = _load()["logins"]
    active = current()
    out = [{"account": DEFAULT, "kind": "token", "token_set": bool(cfg.novelai_token), "source": cfg.token_source}]
    for name, account in cfg.accounts.items():
        kind = "token" if account.endpoint.official else "relay"
        entry: dict[str, Any] = {"account": name, "kind": kind}
        if not account.endpoint.official:
            entry.update(api_url=account.endpoint.api_base, image_url=account.endpoint.image_base)
        out.append(entry)
    for name, saved in logins.items():
        if name not in cfg.accounts:
            out.append({"account": name, "kind": "login", "email": saved.get("email"), "expires": saved.get("expires")})
    for entry in out:
        entry["current"] = entry["account"] == active
    return out


def has_token(cfg: config.Config, name: str | None) -> bool:
    chosen = name or current()
    return bool(cfg.novelai_token) if chosen == DEFAULT else chosen in names(cfg)


def credentials(cfg: config.Config, name: str | None, now: datetime, renew: bool = True) -> Credentials:
    """The token and addresses of an account, --account first, then the one chosen with nai account use; a
    login's token is renewed from its access key when it is about to run out. Without renew, a token still valid
    is used as it is and one run out is refused, so nothing is sent or written."""
    chosen = name or current()
    if chosen == DEFAULT:
        return Credentials(DEFAULT, cfg.require_novelai(), api.OFFICIAL)
    if chosen in cfg.accounts:
        account = cfg.accounts[chosen]
        return Credentials(chosen, account.token, account.endpoint)
    saved = _load()["logins"].get(chosen)
    if saved is None:
        raise errors.NaiError(errors.BAD_ARGS, f"no account named {chosen}", choices=names(cfg))
    left = _expires(saved) - now.astimezone(UTC)
    if left >= RENEW_BEFORE:
        return Credentials(chosen, saved["token"], api.OFFICIAL)
    if not renew:
        if left > timedelta(0):
            return Credentials(chosen, saved["token"], api.OFFICIAL)
        raise errors.NaiError(
            errors.AUTH, f"the login of {chosen} has run out; a run that sends nothing paid does not renew it"
        )
    try:
        token = api.login(api.OFFICIAL, saved["access_key"])
    except errors.NaiError as e:
        if e.code != errors.AUTH and left > timedelta(0):
            # A token still valid is used while the network or server is down; the next command renews it
            return Credentials(chosen, saved["token"], api.OFFICIAL)
        hint = "; run nai account login" if e.code == errors.AUTH else ""
        raise errors.NaiError(e.code, f"the login of {chosen} could not be renewed{hint}: {e}") from e
    with _editing() as store:
        entry = store["logins"].get(chosen)
        # Signed out or signed in again meanwhile: that newer state stands, and this run still has its token
        if isinstance(entry, dict) and entry.get("access_key") == saved["access_key"]:
            entry.update(_token(token, now))
    return Credentials(chosen, token, api.OFFICIAL)


def _token(token: str, now: datetime) -> dict[str, str]:
    return {"token": token, "expires": (now.astimezone(UTC) + TOKEN_LIFETIME).isoformat(timespec="seconds")}


def _expires(saved: dict[str, Any]) -> datetime:
    try:
        expires = datetime.fromisoformat(saved["expires"])
    except (KeyError, TypeError, ValueError):
        expires = None
    # Renew a login whose expiry cannot be read, or has no time zone to compare by
    if expires is None or expires.tzinfo is None:
        return datetime.min.replace(tzinfo=UTC)
    return expires


def _load() -> dict[str, Any]:
    """The logins that can be used and the one chosen."""
    store = _read()
    chosen = store.get("current")
    return {"current": chosen if isinstance(chosen, str) else None, "logins": _valid(store["logins"])}


def _valid(logins: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: saved
        for name, saved in logins.items()
        if isinstance(saved, dict) and isinstance(saved.get("access_key"), str) and isinstance(saved.get("token"), str)
    }


def _read() -> dict[str, Any]:
    """The store as written, entries this version cannot use included, so that an edit keeps them."""
    path = credentials_path()
    try:
        data = values.parse_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as e:
        raise errors.NaiError(errors.CONFIG_MISSING, f"{path}: cannot read: {e}") from e
    # Refused rather than read as empty, which the next save would write over the logins
    if not isinstance(data, dict) or not isinstance(data.setdefault("logins", {}), dict):
        raise errors.NaiError(errors.CONFIG_MISSING, f"{path} is not a login store; fix or delete it")
    return data


@contextmanager
def _editing() -> Iterator[dict[str, Any]]:
    """The store read and written back under a lock, so two processes editing it at once lose neither edit. Nothing
    is sent while the lock is held."""
    lock = credentials_path().with_name("accounts.lock")
    try:
        held = fs.acquire(lock, timeout=30, poll=0.1)
    except TimeoutError:
        raise errors.NaiError(errors.BUSY, f"another nai is still changing {credentials_path()}") from None
    except OSError as e:
        raise errors.NaiError(errors.CONFIG_MISSING, f"cannot lock {lock}: {e}") from e
    try:
        store = _read()
        yield store
        _save(store)
    finally:
        fs.release(held)


def _save(store: dict[str, Any]) -> None:
    # Holds the access key, which signs in without the password
    path = credentials_path()
    try:
        fs.replace(path, values.to_json(store, indent=2), private=True)
    except OSError as e:
        raise errors.NaiError(errors.CONFIG_MISSING, f"cannot write {path}: {e}") from e
