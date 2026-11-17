import base64
import io
import json
import re
import time
import zipfile

import httpx
import pytest
from PIL import Image
from samples import CUT, bomb_png

from nai_cli import api, errors


def zipped(*pngs: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, data in enumerate(pngs):
            z.writestr(f"image_{i}.png", data)
    return buf.getvalue()


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return buf.getvalue()


PNG = png()


def client(handler):
    sleeps = []
    c = api.Client("pst-x", sleep=sleeps.append, transport=httpx.MockTransport(handler))
    return c, sleeps


def test_generate_retries_429_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(429, json={"message": "Concurrent generation is locked"})
        return httpx.Response(200, content=zipped(PNG, PNG))

    c, sleeps = client(handler)
    assert c.generate({"input": "x"}) == [PNG, PNG]
    assert sleeps == [5, 10] and len(c.retries) == 2
    assert calls[0].headers["Authorization"] == "Bearer pst-x"
    assert calls[0].url.host == "image.novelai.net"


@pytest.mark.parametrize("status,code", [(401, errors.AUTH), (402, errors.INSUFFICIENT_ANLAS), (400, errors.API)])
def test_error_codes(status, code):
    c, _ = client(lambda r: httpx.Response(status, json={"message": "nope"}))
    with pytest.raises(errors.NaiError) as e:
        c.generate({})
    assert e.value.code == code and e.value.detail["status"] == status


def test_5xx_retried_once_for_free_requests_only():
    n = []

    def handler(request):
        n.append(request.url.path)
        return httpx.Response(503, text="busy")

    c, _ = client(handler)
    with pytest.raises(errors.NaiError) as e:
        c.account()
    assert e.value.code == errors.SERVER and len(n) == 2 and "maybe_processed" not in e.value.detail
    n.clear()
    # A paid request may have run before the error, so sending it again could charge twice
    with pytest.raises(errors.NaiError) as e:
        c.generate({})
    assert e.value.code == errors.SERVER and e.value.detail["maybe_processed"] is True and len(n) == 1


@pytest.mark.parametrize(
    ("failure", "paid_attempts", "maybe_processed"),
    [(httpx.ConnectError, 2, None), (httpx.ConnectTimeout, 2, None), (httpx.ReadTimeout, 1, True)],
)
def test_network_errors_resend_paid_requests_only_when_unsent(failure, paid_attempts, maybe_processed):
    n = []

    def handler(request):
        n.append(request.method)
        raise failure("down", request=request)

    c, _ = client(handler)
    with pytest.raises(errors.NaiError) as e:
        c.generate({})
    assert e.value.code == errors.NETWORK and len(n) == paid_attempts
    assert e.value.detail.get("maybe_processed") is maybe_processed
    n.clear()
    with pytest.raises(errors.NaiError):
        c.account()
    assert n == ["GET", "GET"]


def test_paid_request_sent_again_after_connect_failure():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, content=zipped(PNG))

    c, sleeps = client(handler)
    assert c.generate({}) == [PNG] and sleeps == [5] and c.retries[0]["reason"] == "network"


def _multipart_request(request) -> dict:
    """JSON of the request field in a multipart body."""
    body = request.content
    start = body.index(b'name="request"')
    start = body.index(b"\r\n\r\n", start) + 4
    return json.loads(body[start : body.index(b"\r\n--", start)])


def test_upscale_uses_official_endpoint_only():
    seen = []

    def handler(request):
        seen.append((request.url.host, request.url.path))
        assert request.headers.get("content-type", "").startswith("multipart/form-data")
        assert _multipart_request(request) == {
            "image": "image",
            "model": "nai-diffusion-5-curated",
            "declared_blur_sigma": 0,
        }
        return httpx.Response(200, content=zipped(PNG))

    c, _ = client(handler)
    assert c.upscale(PNG) == [PNG]
    assert seen == [("image.novelai.net", "/ai/upscale")]


def test_upscale_errors_are_not_retried_elsewhere():
    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        return httpx.Response(402, json={"message": "no anlas"})

    c, _ = client(handler)
    with pytest.raises(errors.NaiError):
        c.upscale(PNG)
    assert hosts == ["image.novelai.net"]


def test_augment_sends_prompt_only_for_colorize_and_emotion():
    bodies = []

    def handler(request):
        bodies.append(_multipart_request(request))
        return httpx.Response(200, content=zipped(PNG))

    c, _ = client(handler)
    c.augment(PNG, "lineart", 1024, 1024, "ignored", 3)
    c.augment(PNG, "emotion", 1024, 1024, "happy;;smile", 2)
    assert bodies[0] == {
        "req_type": "lineart",
        "use_new_shared_trial": True,
        "width": 1024,
        "height": 1024,
        "image": "image",
    }
    assert list(bodies[1]) == ["req_type", "use_new_shared_trial", "prompt", "defry", "width", "height", "image"]
    assert bodies[1]["prompt"] == "happy;;smile" and bodies[1]["defry"] == 2


def test_account_uses_image_host():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json={"tier": 0, "active": False, "trainingStepsLeft": 0})

    c, _ = client(handler)
    c.account()
    assert urls == ["https://image.novelai.net/user/subscription"]


def test_account_parsing():
    c, _ = client(
        lambda r: httpx.Response(
            200,
            json={
                "tier": 3,
                "active": True,
                "expiresAt": int(time.time()) + 86400,
                "trainingStepsLeft": {"fixedTrainingStepsLeft": 9000, "purchasedTrainingSteps": 50},
                "usage": {"percent": 83.5, "isNegative": False, "timeUntilNextPercent": 120},
            },
        )
    )
    acc = c.account()
    assert acc.opus and acc.anlas == 9050 and acc.quota_percent == 83.5


def test_raw_png_response_accepted():
    assert api.unzip_images(PNG) == [PNG]
    assert api.unzip_images(b"garbage") == []


def test_import_size_like_official():
    from nai_cli import images

    portrait = (832, 1216)
    # Enhance results above the limit come back to about 1 MP, within the free tier
    assert images.import_size(1664, 2432, portrait) == (832, 1216)
    assert images.import_size(2432, 1664, portrait) == (1216, 832)
    assert images.import_size(1000, 1500, portrait) == (896, 1344)
    # Same aspect ratio as the current size and fits: keep the current size
    assert images.import_size(416, 608, portrait) == (832, 1216)
    # Multiples of 64 within the limit: used as is
    assert images.import_size(1024, 1024, portrait) == (1024, 1024)
    assert images.import_size(1536, 2048, portrait) == (1536, 2048)


def test_mask_request_is_full_size_black_and_white(tmp_path):
    from PIL import Image

    from nai_cli import images

    src = Image.new("L", (64, 128), 0)
    src.paste(255, (8, 8, 40, 40))
    latent = images.latent_mask(src, 64, 128)
    assert latent.size == (8, 16) and latent.getpixel((2, 2)) == 255 and latent.getpixel((7, 15)) == 0
    sent = Image.open(__import__("io").BytesIO(images.mask_request(latent, 64, 128)))
    assert sent.size == (64, 128) and sent.mode == "RGBA"
    assert sent.getpixel((10, 10)) == (255, 255, 255, 255) and sent.getpixel((60, 120)) == (0, 0, 0, 255)


def test_suggest_tags_query_like_official():
    seen = []

    def handler(request):
        seen.append(request)
        if "lang=jp" in str(request.url):
            return httpx.Response(200, json=[{"jp_tag": "青い目", "en_tag": "blue eyes", "power": 10000}])
        return httpx.Response(
            200, json={"tags": [{"tag": "hand on another's shoulder", "count": 9, "confidence": 0.0}]}
        )

    c, _ = client(handler)
    assert c.suggest_tags("hand on another's (sh", "nai-diffusion-5-full", "animev5", None) == [
        {"tag": "hand on another's shoulder", "count": 9}
    ]
    assert c.suggest_tags("青い", "nai-diffusion-4-5-full", None, "jp") == [
        {"tag": "blue eyes", "jp": "青い目", "count": 10000}
    ]
    # Encoded as encodeURIComponent does, with parameters in the official order
    assert seen[0].url.raw_path.decode() == (
        "/ai/generate-image/suggest-tags?model=nai-diffusion-5-full&prompt=hand%20on%20another's%20(sh&type=animev5"
    )
    assert seen[1].url.raw_path.decode() == (
        "/ai/generate-image/suggest-tags?model=nai-diffusion-4-5-full&prompt=%E9%9D%92%E3%81%84&lang=jp"
    )
    assert seen[0].method == "GET" and seen[0].headers["Content-Type"] == "application/json"


def test_encode_vibe_sends_mask_only_when_set():
    seen = []

    parts, bodies = [], []

    def handler(request):
        seen.append(_multipart_request(request))
        parts.append(re.findall(r'; name="(\w+)"', request.content.decode("latin-1")))
        bodies.append(request.content)
        return httpx.Response(200, content=b"enc")

    c, _ = client(handler)
    c.encode_vibe(PNG, "nai-diffusion-4-5-full", 0.7)
    c.encode_vibe(PNG, "nai-diffusion-4-5-full", 0.7, base64.b64encode(b"MASK").decode())
    c.encode_vibe(PNG, "nai-diffusion-4-5-full", 0.7, base64.b64encode(PNG).decode())
    assert [list(r) for r in seen] == [
        ["image", "information_extracted", "model"],
        ["image", "information_extracted", "mask", "model"],
        ["image", "information_extracted", "mask", "model"],
    ]
    # The mask is a part of its own, which the request names; one equal to the image shares its part
    assert seen[1]["mask"] == "mask" and seen[2]["mask"] == "image"
    assert parts[1] == ["image", "mask", "request"] and parts[2] == ["image", "request"]
    assert b"MASK" in bodies[1]
    # Empty, it is named in the request with no part; wrapped over lines, it is the same mask
    c.encode_vibe(PNG, "nai-diffusion-4-5-full", 0.7, "")
    wrapped = base64.b64encode(b"MASK" * 30).decode()
    c.encode_vibe(
        PNG, "nai-diffusion-4-5-full", 0.7, "\n".join(wrapped[i : i + 76] for i in range(0, len(wrapped), 76))
    )
    assert seen[3]["mask"] == "" and parts[3] == ["image", "request"]
    assert seen[4]["mask"] == "mask" and b"MASK" * 30 in bodies[4]


@pytest.mark.parametrize(
    "content",
    [b"PK\x03\x04truncated", zipped(), b"", zipped(b""), zipped(CUT), CUT, b"\x89PNG\r\n\x1a\nx", bomb_png()],
)
def test_unusable_paid_reply_may_have_been_charged(content):
    c, _ = client(lambda request: httpx.Response(200, content=content))
    for call in (lambda: c.generate({}), lambda: c.upscale(PNG), lambda: c.augment(PNG, "lineart", 64, 64)):
        with pytest.raises(errors.NaiError) as e:
            call()
        assert e.value.code == errors.API and e.value.detail["maybe_processed"] is True


def test_empty_vibe_encoding_refused():
    c, _ = client(lambda request: httpx.Response(200, content=b""))
    with pytest.raises(errors.NaiError) as e:
        c.encode_vibe(PNG, "nai-diffusion-4-5-full")
    assert e.value.code == errors.API and e.value.detail["maybe_processed"] is True


@pytest.mark.parametrize(
    "content",
    [
        b"<html>gateway</html>",
        b"[]",
        b"{}",
        b'{"tier": "x", "trainingStepsLeft": 5}',
        b'{"tier": 3, "trainingStepsLeft": {"fixedTrainingStepsLeft": 5}}',
        b'{"tier": 3, "trainingStepsLeft": true}',
        b'{"tier": 3, "trainingStepsLeft": 5, "usage": {"percent": NaN}}',
    ],
)
def test_unexpected_account_reply_is_an_api_error(content):
    c, _ = client(lambda request: httpx.Response(200, content=content))
    with pytest.raises(errors.NaiError) as e:
        c.account()
    assert e.value.code == errors.API and "maybe_processed" not in e.value.detail


@pytest.mark.parametrize("content", [b"<html>gateway</html>", b'{"tags": [{"tag": "x"}]}', b'[{"en_tag": "x"}]', b"1"])
def test_unexpected_tag_reply_is_an_api_error(content):
    c, _ = client(lambda request: httpx.Response(200, content=content))
    with pytest.raises(errors.NaiError) as e:
        c.suggest_tags("blue", "nai-diffusion-5-full", None, None)
    assert e.value.code == errors.API


@pytest.mark.parametrize("content", [b'"token expired"', b'["token expired"]'])
def test_error_reply_of_any_json_shape_keeps_its_message(content):
    c, _ = client(lambda request: httpx.Response(401, content=content))
    with pytest.raises(errors.NaiError) as e:
        c.account()
    assert e.value.code == errors.AUTH and "token expired" in e.value.message


def test_account_as_a_single_count_like_official():
    c, _ = client(lambda r: httpx.Response(200, json={"tier": 1, "trainingStepsLeft": 120}))
    acc = c.account()
    assert acc.anlas == 120 and acc.quota_percent is None


@pytest.mark.parametrize(
    ("fields", "opus"),
    [
        ({"expiresAt": 4102444800}, True),
        # Renewal cancelled but still paid up: the reply says inactive, the official client still counts it
        ({"expiresAt": 4102444800, "active": False}, True),
        ({"expiresAt": 1, "active": True}, False),
        ({"active": True}, False),
        ({"expiresAt": 1, "accountType": 2}, True),
        ({"expiresAt": 4102444800, "tier": 2}, False),
    ],
)
def test_opus_decided_like_official(fields, opus):
    c, _ = client(lambda r: httpx.Response(200, json={"tier": 3, "trainingStepsLeft": 0, **fields}))
    assert c.account().opus is opus


def test_one_damaged_image_spares_the_rest():
    c, _ = client(lambda r: httpx.Response(200, content=zipped(PNG, CUT, PNG)))
    with pytest.raises(api.DamagedImages) as e:
        c.generate({})
    assert e.value.images == [PNG, None, PNG] and e.value.code == errors.API and e.value.detail["maybe_processed"]


@pytest.mark.parametrize("refused", [True, False])
def test_interrupt_between_attempts_is_not_a_paid_request(refused):
    def handler(request):
        if refused:
            return httpx.Response(429, json={"message": "busy"})
        raise httpx.ConnectError("refused", request=request)

    def interrupted(seconds):
        raise KeyboardInterrupt

    c = api.Client("pst-x", sleep=interrupted, transport=httpx.MockTransport(handler))
    with pytest.raises(errors.NaiError) as e:
        c.generate({})
    assert e.value.code == errors.INTERRUPTED and "maybe_processed" not in e.value.detail


def test_background_removal_returns_every_image():
    c, _ = client(lambda r: httpx.Response(200, content=zipped(PNG, PNG, PNG)))
    assert c.augment(PNG, "bg-removal", 64, 64) == [PNG, PNG, PNG]


def test_a_reply_short_of_images_reports_the_missing_ones():
    c, _ = client(lambda r: httpx.Response(200, content=zipped(PNG)))
    with pytest.raises(api.DamagedImages) as e:
        c.generate({"parameters": {"n_samples": 2}})
    assert e.value.images == [PNG, None] and e.value.detail["maybe_processed"] is True
    with pytest.raises(api.DamagedImages):
        c.augment(PNG, "bg-removal", 64, 64, expected=3)


def test_credentials_never_come_back_in_errors():
    secret = "pst-SECRETTOKEN123456"
    for value in (f"https://user:{secret}@relay.example", f"https://relay.example/?key={secret}"):
        with pytest.raises(ValueError) as e:
            api.endpoint(value)
        assert secret not in str(e.value)
    echoed = f"bad header Authorization: Bearer {secret}"
    seen = []
    c = api.Client(
        secret,
        sleep=seen.append,
        transport=httpx.MockTransport(lambda r: httpx.Response(503, json={"message": echoed})),
    )
    with pytest.raises(errors.NaiError) as e:
        c.account()
    assert secret not in e.value.message and secret not in json.dumps(c.retries)


SECRET = "pst-SECRETTOKEN123456"


def _refusing(status, message):
    return httpx.MockTransport(lambda r: httpx.Response(status, json={"message": message}))


def _unreachable(request):
    raise httpx.ConnectError(f"proxy said: Bearer {SECRET} key={SECRET}", request=request)


@pytest.mark.parametrize(
    "transport",
    [
        # Cut at the message limit, where only the start of the credential would remain
        _refusing(400, "x" * 490 + SECRET),
        _refusing(401, f"token {SECRET} expired"),
        _refusing(429, f"slow down {SECRET}"),
        httpx.MockTransport(_unreachable),
        httpx.MockTransport(lambda r: httpx.Response(200, content=f'{{"echo": "{SECRET}"'.encode())),
    ],
)
@pytest.mark.parametrize("call", ["account", "tags", "login"])
def test_no_error_path_shows_the_credential_sent(transport, call):
    def run_call():
        if call == "login":
            return api.login(api.OFFICIAL, SECRET, transport=transport)
        with api.Client(SECRET, sleep=lambda s: None, transport=transport) as c:
            return c.account() if call == "account" else c.suggest_tags("1gi", "nai-diffusion-4-5-full", None, None)

    with pytest.raises(errors.NaiError) as e:
        run_call()
    shown = json.dumps(e.value.to_json())
    # Any part of the credential past its common prefix
    assert "SECRET" not in shown, shown


def test_a_reply_of_the_wrong_shape_is_reported_without_what_was_sent():
    def echoes(data):
        raise ValueError(f"no field in {data!r}")

    response = httpx.Response(200, json={"token": SECRET})
    with pytest.raises(errors.NaiError) as e:
        api._parsed(response, "reply", echoes, (SECRET,))
    assert e.value.code == errors.API and SECRET not in e.value.message


@pytest.mark.parametrize(("status", "code"), [(429, errors.RATE_LIMITED), (500, errors.SERVER), (400, errors.API)])
def test_login_errors_take_the_codes_of_every_other_request(status, code):
    with pytest.raises(errors.NaiError) as e:
        api.login(api.OFFICIAL, "KEY", transport=_refusing(status, "no"))
    assert e.value.code == code


def test_a_reply_entry_larger_than_any_image_is_not_read(monkeypatch):
    monkeypatch.setattr(api, "_MAX_ENTRY", 16)
    assert api.unzip_images(zipped(PNG, b"\x89PNG" + b"\0" * 8)) == [None, b"\x89PNG" + b"\0" * 8]


def test_a_reply_holds_no_more_than_any_request_returns(monkeypatch):
    small = b"\x89PNG" + b"\0" * 8
    assert len(api.unzip_images(zipped(*[small] * 40))) == api._MAX_ENTRIES
    # Entries past the total are not read, so many entries of the largest size cannot fill the memory
    monkeypatch.setattr(api, "_MAX_REPLY", len(small) * 2)
    assert api.unzip_images(zipped(small, small, small)) == [small, small, None]


def test_whole_numbers_are_sent_as_the_browser_writes_them():
    sent = []

    def handler(request):
        sent.append(request.content)
        return httpx.Response(200, content=zipped(PNG))

    c = api.Client("pst-x", sleep=lambda s: None, transport=httpx.MockTransport(handler))
    c.generate({"input": "x", "model": "m", "parameters": {"scale": 4.0, "cfg_rescale": 0.5, "seed": 3}})
    assert b'"scale": 4,' in sent[0] and b'"cfg_rescale": 0.5' in sent[0]
