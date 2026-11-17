"""NovelAI's HTTP API: the requests, their retries, and the replies read back as images and accounts."""

import base64
import io
import re
import struct
import time
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.parse import SplitResult, quote, urlsplit, urlunsplit

import httpx

from nai_cli import __version__, decode, errors, values

API = "https://api.novelai.net"
# Generation, tools, login and account endpoints of the official service all live on this host
IMAGE = "https://image.novelai.net"
WEB = "https://novelai.net"
RETRY_429 = [5, 10, 20, 40, 60]
RETRY_5XX = [5]
TIMEOUT = httpx.Timeout(connect=30, read=180, write=60, pool=30)
# Token counts fall back to estimates, so an unreachable host should not hold up lint for long
TOKENIZER_TIMEOUT = httpx.Timeout(connect=10, read=30, write=10, pool=10)
# The larger tokenizer file is under 4 MB
_MAX_TOKENIZER = 32 * 1024 * 1024
# Server messages are cut to this length after credentials are hidden
_MAX_MESSAGE = 500
# Characters encodeURIComponent leaves as they are
_URI_SAFE = "-_.!~*'()"
# The connection was never made, so nothing reached NovelAI and even a paid request can be sent again
_NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
_BEARER = re.compile(r"Bearer\s+\S+")
# A 4096x4096 RGBA PNG stored without compression is 64 MiB
_MAX_ENTRY = 128 * 1024 * 1024
# More than any reply holds: eight images, or the three of background removal
_MAX_ENTRIES = 32
_MAX_REPLY = 4 * _MAX_ENTRY
# What a truncated or foreign archive raises while reading
_BAD_ARCHIVE = (
    zipfile.BadZipFile,
    zlib.error,
    EOFError,
    KeyError,
    ValueError,
    NotImplementedError,
    RuntimeError,
    OSError,
    struct.error,
)
# Account types the official client counts as subscribed whatever the expiry: B2B, service, support, admin
_STAFF_ACCOUNTS = {1, 2, 3, 4}
# What a JSON reply of the wrong shape raises while reading
_BAD_REPLY = (ValueError, TypeError, KeyError, AttributeError)
T = TypeVar("T")


def fetch_tokenizer(name: str, version: int) -> bytes:
    """The tokenizer file the official client loads for counting; free and needs no token. ValueError for one
    larger than any tokenizer, httpx.HTTPError when it cannot be fetched."""
    url = f"{WEB}/tokenizer/compressed/{name}?v={version}&static=true"
    data = _download(url, _MAX_TOKENIZER, TOKENIZER_TIMEOUT, follow_redirects=False)
    if len(data) > _MAX_TOKENIZER:
        raise ValueError(f"{url} sends more than {_MAX_TOKENIZER} bytes")
    return data


def fetch_file(url: str, size: int) -> bytes:
    """A public file of a known size, such as a local model; anything of another size is refused."""
    try:
        data = _download(url, size, TIMEOUT, follow_redirects=True)
    except httpx.HTTPError as e:
        raise errors.NaiError(errors.NETWORK, f"cannot download {url}: {e}") from e
    if len(data) != size:
        raise errors.NaiError(errors.NETWORK, f"{url} sent {len(data)} bytes where {size} were expected")
    return data


def _download(url: str, limit: int, timeout: httpx.Timeout, follow_redirects: bool) -> bytes:
    """At most limit + 1 bytes, so a reply too large is known without holding it all."""
    with httpx.stream(
        "GET", url, headers={"User-Agent": f"nai-cli/{__version__}"}, timeout=timeout, follow_redirects=follow_redirects
    ) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_bytes():
            data += chunk
            if len(data) > limit:
                break
    return bytes(data)


@dataclass(frozen=True)
class Endpoint:
    """Where requests go: the official hosts, or a NovelAI-compatible relay that may serve both from one URL."""

    api_base: str = API
    image_base: str = IMAGE

    @property
    def official(self) -> bool:
        return self == OFFICIAL

    def image(self, path: str) -> str:
        return self.image_base + path

    def user(self, path: str) -> str:
        # The official service answers user endpoints on the image host; relays keep their own layout
        return self.image(path) if self.official else self.api_base + path


OFFICIAL = Endpoint()


def endpoint(api_url: str, image_url: str = "") -> Endpoint:
    """A relay's addresses as typed: https:// is assumed, and http:// for this machine; image_url defaults to
    api_url."""
    main = _base_url(api_url)
    return Endpoint(main, _base_url(image_url) if image_url.strip() else main)


def _base_url(value: str) -> str:
    parts = _address(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _address(value: str) -> SplitResult:
    """https:// is assumed, and http:// for this machine. ValueError for anything but an http or https address
    of a host without credentials in it."""
    text = value.strip()
    if "://" not in text:
        host = text.split("/", 1)[0].lower()
        local = host.split(":", 1)[0] in ("localhost", "127.0.0.1") or host.startswith("[::1]")
        text = ("http://" if local else "https://") + text
    try:
        parts = urlsplit(text)
        hostname = parts.hostname
    except ValueError:
        hostname = None
    # The address is not repeated, since a key or password put in it by mistake would be printed
    if hostname is None or parts.scheme not in ("http", "https"):
        raise ValueError("the address is not an http or https address")
    if parts.query or parts.fragment or parts.username or parts.password:
        raise ValueError("the address takes no query, fragment, user or password; the token goes in token")
    return parts


def login(endpoint: Endpoint, access_key: str, transport: httpx.BaseTransport | None = None) -> str:
    """Exchange an access key for a token, as the official login does; the token lasts 30 days."""
    with httpx.Client(headers={"User-Agent": f"nai-cli/{__version__}"}, timeout=TIMEOUT, transport=transport) as http:
        try:
            response = http.post(endpoint.user("/user/login"), json={"key": access_key})
        except httpx.TransportError as e:
            raise errors.NaiError(errors.NETWORK, f"cannot reach NovelAI: {scrub(str(e), (access_key,))}") from e
    if response.status_code in (401, 403):
        raise errors.NaiError(errors.AUTH, "NovelAI refused the email or password", status=response.status_code)
    if response.status_code >= 400:
        raise _status_error(response, (access_key,))
    return _parsed(response, "login reply", lambda data: _text_field(data, "accessToken"), (access_key,))


def _text_field(data: dict[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} is not a text but {type(value).__name__}")
    return value


@dataclass
class Account:
    tier: int
    active: bool
    anlas_fixed: int
    anlas_purchased: int
    quota_percent: float | None
    quota_negative: bool
    quota_next_percent_seconds: float | None
    # A relay without the subscription endpoint: prices are standard and the spend cannot be measured
    measured: bool = True

    @property
    def anlas(self) -> int:
        return self.anlas_fixed + self.anlas_purchased

    @property
    def opus(self) -> bool:
        return self.tier >= 3 and self.active

    def to_json(self) -> dict[str, Any]:
        if not self.measured:
            return {"measured": False}
        return {
            "tier": {0: "Paper", 1: "Tablet", 2: "Scroll", 3: "Opus"}.get(self.tier, str(self.tier)),
            "active": self.active,
            "anlas": self.anlas,
            "anlas_fixed": self.anlas_fixed,
            "anlas_purchased": self.anlas_purchased,
            "quota_percent": self.quota_percent,
            "quota_exhausted": self.quota_negative,
            "quota_next_percent_seconds": self.quota_next_percent_seconds,
        }


def _multipart(image: bytes, request: dict[str, Any], mask: bytes | None = None) -> dict[str, tuple[str, bytes, str]]:
    """The image, then the mask, then the request, whose fields name the parts; a mask equal to the image is sent
    once, as the official client sends it."""
    parts = {"image": ("blob", image, "image/png")}
    if mask is not None and mask != image:
        parts["mask"] = ("blob", mask, "image/png")
    parts["request"] = ("blob", values.to_json(values.as_js(request)).encode(), "application/json")
    return parts


class DamagedImages(errors.NaiError):
    """Some images of a paid reply do not decode. The error carries every image in order, None where damaged,
    so the others can still be saved under their own numbers."""

    def __init__(self, message: str, images: list[bytes | None]) -> None:
        super().__init__(errors.API, message, maybe_processed=True)
        self.images = images


def unzip_images(data: bytes) -> list[bytes | None]:
    """Each image in a reply, None for an entry that cannot be read, so one damaged entry spares the rest."""
    if data[:2] != b"PK":
        return [data] if data[:8] == b"\x89PNG\r\n\x1a\n" else []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith((".png", ".jpg", ".webp")))
        out: list[bytes | None] = []
        room = _MAX_REPLY
        for name in names[:_MAX_ENTRIES]:
            entry = _entry(z, name, min(_MAX_ENTRY, room))
            room -= len(entry) if entry is not None else 0
            out.append(entry)
        return out


def _entry(z: zipfile.ZipFile, name: str, limit: int) -> bytes | None:
    # An entry past the limit is not read, so a small reply cannot fill the memory
    try:
        with z.open(name) as f:
            data = f.read(limit + 1)
    except _BAD_ARCHIVE:
        return None
    return data if len(data) <= limit else None


def _parsed(response: httpx.Response, what: str, read: Callable[[Any], T], secrets: tuple[str, ...]) -> T:
    try:
        return read(values.parse_json(response.content))
    except _BAD_REPLY as e:
        # The error may quote the reply, which a relay may have filled with what was sent
        reason = _short(scrub(f"{type(e).__name__}: {e}", secrets))
        raise errors.NaiError(errors.API, f"unexpected {what} from NovelAI: {reason}") from e


def _number(value: Any, name: str) -> float:
    if not values.is_number(value):
        raise ValueError(f"{name} is {value!r}")
    return value


def _optional_number(value: Any, name: str) -> float | None:
    return None if value is None else _number(value, name)


def _account(data: dict[str, Any]) -> Account:
    # As in the official client: two counts or a single number, and anything else is an error rather than 0
    steps = data["trainingStepsLeft"]
    if isinstance(steps, dict):
        fixed = _number(steps["fixedTrainingStepsLeft"], "fixedTrainingStepsLeft")
        purchased = _number(steps["purchasedTrainingSteps"], "purchasedTrainingSteps")
    else:
        fixed, purchased = _number(steps, "trainingStepsLeft"), 0
    tier = int(_number(data["tier"], "tier"))
    expires = _optional_number(data.get("expiresAt"), "expiresAt")
    # Decided as the official client does; the reply's own active flag only drives its billing pages
    active = _optional_number(data.get("accountType"), "accountType") in _STAFF_ACCOUNTS or (
        expires is not None and expires > time.time() and tier > 0
    )
    usage = data.get("usage") or {}
    return Account(
        tier=tier,
        active=active,
        anlas_fixed=int(fixed),
        anlas_purchased=int(purchased),
        quota_percent=_optional_number(usage.get("percent"), "usage.percent"),
        quota_negative=bool(usage.get("isNegative", False)),
        quota_next_percent_seconds=_optional_number(usage.get("timeUntilNextPercent"), "usage.timeUntilNextPercent"),
    )


UNMEASURED = Account(0, False, 0, 0, None, False, None, measured=False)


def _tags(data: Any) -> list[dict[str, Any]]:
    # Japanese lookups return a bare list with jp_tag, en_tag and power
    if isinstance(data, list):
        return [{"tag": t["en_tag"], "jp": t["jp_tag"], "count": t["power"]} for t in data]
    return [{"tag": t["tag"], "count": t["count"]} for t in data.get("tags", [])]


def _decodes(data: bytes | None) -> bool:
    if data is None:
        return False
    try:
        decode.verify(data)
    except decode.BAD_IMAGE:
        return False
    return True


def _error_message(response: httpx.Response, secrets: tuple[str, ...]) -> str:
    """What the server says went wrong, short, with the credentials sent hidden in case a relay repeats them."""
    try:
        payload = values.parse_json(response.content)
    except ValueError:
        text = response.text or response.reason_phrase
    else:
        found = payload.get("message") or payload.get("error") if isinstance(payload, dict) else None
        text = str(found or payload)
    # Hidden before cutting, so a credential cut in half cannot show its first part
    return _short(scrub(text, secrets))


def _short(text: str) -> str:
    return text if len(text) <= _MAX_MESSAGE else text[:_MAX_MESSAGE] + "…"


def _status_error(response: httpx.Response, secrets: tuple[str, ...]) -> errors.NaiError:
    """The error of a request NovelAI refused or failed, by its status."""
    status = response.status_code
    code = {
        401: errors.AUTH,
        402: errors.INSUFFICIENT_ANLAS,
        429: errors.RATE_LIMITED,
    }.get(status, errors.SERVER if status >= 500 else errors.API)
    return errors.NaiError(code, f"NovelAI returned {status}: {_error_message(response, secrets)}", status=status)


def scrub(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<hidden>")
    return _BEARER.sub("Bearer <hidden>", text)


class Client:
    def __init__(
        self,
        token: str,
        endpoint: Endpoint = OFFICIAL,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
        before_send: Callable[[], None] | None = None,
    ):
        self.endpoint = endpoint
        # Called as each request goes on the wire, which paces a batch
        self._before_send = before_send or (lambda: None)
        self._secrets = (token,)
        self._sleep = sleep
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {token}", "User-Agent": f"nai-cli/{__version__}"},
            timeout=TIMEOUT,
            transport=transport,
        )
        self.retries: list[dict[str, Any]] = []

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _wait(self, seconds: float) -> None:
        # Nothing is in flight between attempts, and a paid request is only retried when it cannot have run
        try:
            self._sleep(seconds)
        except KeyboardInterrupt:
            raise errors.NaiError(errors.INTERRUPTED, "interrupted") from None

    def _send(self, method: str, url: str, paid: bool = False, **kwargs: Any) -> httpx.Response:
        """Paid requests are sent again only when they cannot have run, as the official client never resends them;
        429 means the request was refused."""
        attempt_429 = attempt_5xx = attempt_net = 0
        while True:
            self._before_send()
            try:
                response = self._http.request(method, url, **kwargs)
            except httpx.TransportError as e:
                unsent = isinstance(e, _NOT_SENT)
                if attempt_net < 1 and (unsent or not paid):
                    attempt_net += 1
                    self.retries.append({"reason": "network", "error": scrub(str(e), self._secrets)})
                    self._wait(5)
                    continue
                if paid and not unsent:
                    raise errors.NaiError(
                        errors.NETWORK,
                        "lost the connection to NovelAI after sending the request, which may have run: "
                        + scrub(str(e), self._secrets),
                        maybe_processed=True,
                    ) from e
                raise errors.NaiError(errors.NETWORK, f"cannot reach NovelAI: {scrub(str(e), self._secrets)}") from e
            status = response.status_code
            if status < 400:
                return response
            message = _error_message(response, self._secrets)
            if status == 429 and attempt_429 < len(RETRY_429):
                wait = RETRY_429[attempt_429]
                attempt_429 += 1
                self.retries.append({"reason": "429", "wait": wait, "message": message})
                self._wait(wait)
                continue
            if status >= 500 and paid:
                raise errors.NaiError(
                    errors.SERVER,
                    f"NovelAI returned {status}, and the request may have run: {message}",
                    status=status,
                    maybe_processed=True,
                )
            if status >= 500 and attempt_5xx < len(RETRY_5XX):
                wait = RETRY_5XX[attempt_5xx]
                attempt_5xx += 1
                self.retries.append({"reason": str(status), "wait": wait, "message": message})
                self._wait(wait)
                continue
            raise _status_error(response, self._secrets)

    @staticmethod
    def _images(response: httpx.Response, what: str, expected: int = 1) -> list[bytes]:
        """A paid request answered with 200 has run, so an unusable reply may still have been charged. Images
        missing from the reply count as damaged, after the ones that came."""
        try:
            entries = unzip_images(response.content)
        except _BAD_ARCHIVE as e:
            raise errors.NaiError(
                errors.API, f"NovelAI sent a damaged {what}: {type(e).__name__}: {e}", maybe_processed=True
            ) from e
        if not entries:
            raise errors.NaiError(errors.API, f"no image in the {what}", maybe_processed=True)
        decoded: list[bytes | None] = [p if _decodes(p) else None for p in entries]
        decoded += [None] * (expected - len(decoded))
        damaged = decoded.count(None)
        if damaged:
            raise DamagedImages(f"{damaged} of {len(decoded)} images in the {what} are damaged or missing", decoded)
        return [p for p in decoded if p is not None]

    def generate(self, body: dict[str, Any]) -> list[bytes]:
        response = self._send(
            "POST",
            self.endpoint.image("/ai/generate-image"),
            paid=True,
            content=values.to_json(values.as_js(body)).encode(),
            headers={"Content-Type": "application/json"},
        )
        return self._images(response, "response", body.get("parameters", {}).get("n_samples", 1))

    def upscale(self, image: bytes) -> list[bytes]:
        request = {"image": "image", "model": "nai-diffusion-5-curated", "declared_blur_sigma": 0}
        response = self._send("POST", self.endpoint.image("/ai/upscale"), paid=True, files=_multipart(image, request))
        return self._images(response, "upscale response")

    def augment(
        self,
        image: bytes,
        req_type: str,
        width: int,
        height: int,
        prompt: str = "",
        defry: int = 0,
        expected: int = 1,
    ) -> list[bytes]:
        """Every image of the reply: background removal returns three, as presets.TOOL_OUTPUTS names them."""
        request: dict[str, Any] = {"req_type": req_type, "use_new_shared_trial": True}
        if req_type in ("colorize", "emotion"):
            request["prompt"] = prompt
            request["defry"] = defry
        request.update({"width": width, "height": height, "image": "image"})
        response = self._send(
            "POST", self.endpoint.image("/ai/augment-image"), paid=True, files=_multipart(image, request)
        )
        return self._images(response, "Director tool response", expected)

    def encode_vibe(self, image: bytes, model: str, information_extracted: float = 1.0, mask: str | None = None) -> str:
        """mask is the base64 PNG a Vibe file keeps."""
        request: dict[str, Any] = {"image": "image", "information_extracted": information_extracted}
        # An empty mask stays in the request but sends no part, as the official client only uploads a non-empty one
        mask_png = base64.b64decode(mask) if mask else None
        if mask is not None:
            request["mask"] = mask if mask_png is None else "image" if mask_png == image else "mask"
        request["model"] = model
        response = self._send(
            "POST", self.endpoint.image("/ai/encode-vibe"), paid=True, files=_multipart(image, request, mask_png)
        )
        if not response.content:
            raise errors.NaiError(errors.API, "NovelAI sent back an empty Vibe encoding", maybe_processed=True)
        return base64.b64encode(response.content).decode()

    def suggest_tags(self, prompt: str, model: str, lexicon: str | None, lang: str | None) -> list[dict[str, Any]]:
        query = f"model={quote(model, safe=_URI_SAFE)}&prompt={quote(prompt, safe=_URI_SAFE)}"
        if lexicon:
            query += f"&type={lexicon}"
        if lang:
            query += f"&lang={lang}"
        response = self._send(
            "GET",
            self.endpoint.image(f"/ai/generate-image/suggest-tags?{query}"),
            headers={"Content-Type": "application/json"},
        )
        return _parsed(response, "tag suggestions", _tags, self._secrets)

    def account(self) -> Account:
        try:
            response = self._send("GET", self.endpoint.user("/user/subscription"))
        except errors.NaiError as e:
            if self.endpoint.official or e.detail.get("status") != 404:
                raise
            return UNMEASURED
        # A relay may answer unknown paths with a web page
        if not self.endpoint.official and "json" not in response.headers.get("content-type", ""):
            return UNMEASURED
        return _parsed(response, "account details", _account, self._secrets)
